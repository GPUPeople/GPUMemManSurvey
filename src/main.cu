#include <iostream>

#include "UtilityFunctions.cuh"

// ########################
#ifdef TEST_CUDA
#include "cuda/Instance.cuh"
#elif TEST_XMALLOC
#include "xmalloc/Instance.cuh"
#elif TEST_HALLOC
#include "halloc/Instance.cuh"
#elif TEST_SCATTERALLOC
#include "scatteralloc/Instance.cuh"
#elif TEST_OUROBOROS
#include "ouroboros/Instance.cuh"
#elif TEST_FDG
#include "fdg/Instance.cuh"
#elif TEST_REGEFF
#include "regeff/Instance.cuh"
#endif

template <typename MemoryManagerType>
__global__ void d_testAllocation(MemoryManagerType mm, int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;

	verification_ptr[tid] = reinterpret_cast<int*>(mm.malloc(allocation_size));
}

__global__ void d_testWriteToMemory(int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	auto ptr = verification_ptr[tid];

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		ptr[i] = tid;
	}
}

__global__ void d_testReadFromMemory(int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	auto ptr = verification_ptr[tid];

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		if(ptr[i] != tid)
		{
			printf("%d | We got a wrong value here! %d vs %d\n", tid, ptr[i], tid);
		}
	}
}

template <typename MemoryManagerType>
__global__ void d_testFree(MemoryManagerType mm, int** verification_ptr, int num_allocations)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;

	mm.free(verification_ptr[tid]);
}

int main(int argc, char* argv[])
{
	int device{0};
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";

	int num_allocations{10000};
	int allocation_size_byte{16};
	int num_iterations {10};
	if(argc >= 2)
	{
		num_allocations = atoi(argv[1]);
		if(argc >= 3)
		{
			allocation_size_byte = atoi(argv[2]);
			if(argc >= 4)
			{
				num_iterations = atoi(argv[3]);
			}
		}
	}
	allocation_size_byte = Utils::alignment(allocation_size_byte, sizeof(int));
	std::cout << "Number of Allocations: " << num_allocations << " | Allocation Size: " << allocation_size_byte << " | Iterations: " << num_iterations << std::endl;

#ifdef TEST_CUDA
	std::cout << "--- CUDA ---\n";
	MemoryManagerCUDA memory_manager;
#elif TEST_XMALLOC
	std::cout << "--- XMalloc ---\n";
	MemoryManagerXMalloc memory_manager;
#elif TEST_HALLOC
	std::cout << "--- Halloc ---\n";
	MemoryManagerHalloc memory_manager;
#elif TEST_SCATTERALLOC
	std::cout << "--- ScatterAlloc ---\n";
	MemoryManagerScatterAlloc memory_manager;
#elif TEST_OUROBOROS
	std::cout << "--- Ouroboros ---";
	#ifdef TEST_PAGES
	#ifdef TEST_VIRTUALIZED_ARRAY
	std::cout << " Page --- Virtualized Array ---\n";
	MemoryManagerOuroboros<OuroVAPQ> memory_manager;
	#elif TEST_VIRTUALIZED_LIST
	std::cout << " Page --- Virtualized List ---\n";
	MemoryManagerOuroboros<OuroVLPQ> memory_manager;
	#else
	std::cout << " Page --- Standard ---\n";
	MemoryManagerOuroboros<OuroPQ> memory_manager;
	#endif
	#endif
	#ifdef TEST_CHUNKS
	#ifdef TEST_VIRTUALIZED_ARRAY
	std::cout << " Chunk --- Virtualized Array ---\n";
	MemoryManagerOuroboros<OuroVACQ> memory_manager;
	#elif TEST_VIRTUALIZED_LIST
	std::cout << " Chunk --- Virtualized List ---\n";
	MemoryManagerOuroboros<OuroVLCQ> memory_manager;
	#else
	std::cout << " Chunk --- Standard ---\n";
	MemoryManagerOuroboros<OuroCQ> memory_manager;
	#endif
	#endif
#elif TEST_FDG
	std::cout << "--- FDGMalloc ---\n";
	MemoryManagerFDG memory_manager;
#elif TEST_REGEFF
	std::cout << "--- RegEff ---";
	#ifdef TEST_ATOMIC
	std::cout << " Atomic\n";
	MemoryManagerRegEff<RegEffVariants::AtomicMalloc> memory_manager;
	#elif TEST_ATOMIC_WRAP
	std::cout << " Atomic Wrap\n";
	MemoryManagerRegEff<RegEffVariants::AWMalloc> memory_manager;
	#elif TEST_CIRCULAR
	std::cout << " Circular\n";
	MemoryManagerRegEff<RegEffVariants::CMalloc> memory_manager;
	#elif TEST_CIRCULAR_FUSED
	std::cout << " Circular Fused\n";
	MemoryManagerRegEff<RegEffVariants::CFMalloc> memory_manager;
	#elif TEST_CIRCULAR_MULTI
	std::cout << " Circular Multi\n";
	MemoryManagerRegEff<RegEffVariants::CMMalloc> memory_manager;
	#elif TEST_CIRCULAR_FUSED_MULTI
	std::cout << " Circular Fused Multi\n";
	MemoryManagerRegEff<RegEffVariants::CFMMalloc> memory_manager;
	#endif
#endif

	int** d_memory{nullptr};
	CHECK_ERROR(cudaMalloc(&d_memory, sizeof(int*) * num_allocations));

	int blockSize {256};
	int gridSize {Utils::divup(num_allocations, blockSize)};
	float timing_allocation{0.0f};
	float timing_free{0.0f};
	cudaEvent_t start, end;
	for(auto i = 0; i < num_iterations; ++i)
	{
		std::cout << "Iteration " << i + 1 << " / " << num_iterations << std::endl;
		Utils::start_clock(start, end);
		d_testAllocation <decltype(memory_manager)> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, allocation_size_byte);
		timing_allocation += Utils::end_clock(start, end);

		CHECK_ERROR(cudaDeviceSynchronize());

		d_testWriteToMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);

		CHECK_ERROR(cudaDeviceSynchronize());

		d_testReadFromMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);

		CHECK_ERROR(cudaDeviceSynchronize());

		Utils::start_clock(start, end);
		d_testFree <decltype(memory_manager)> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations);
		timing_free += Utils::end_clock(start, end);

		CHECK_ERROR(cudaDeviceSynchronize());
	}
	timing_allocation /= num_iterations;
	timing_free /= num_iterations;

	std::cout << "Timing Allocation: " << timing_allocation << "ms" << std::endl;
	std::cout << "Timing       Free: " << timing_free << "ms" << std::endl;

	printf("Testcase done!\n");

	return 0;
}