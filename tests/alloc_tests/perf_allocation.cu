#include <iostream>
#include <fstream>

#include "UtilityFunctions.cuh"

// ########################
#ifdef TEST_CUDA
#include "cuda/Instance.cuh"
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

template <typename MemoryManagerType, bool warp_based>
__global__ void d_testAllocation(MemoryManagerType mm, int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid{0};
	if(warp_based)
	{
		tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
		if(threadIdx.x % 32 != 0)
			return;
	}
	else
	{
		tid = threadIdx.x + blockIdx.x * blockDim.x;
	}
	if(tid >= num_allocations)
		return;

	verification_ptr[tid] = reinterpret_cast<int*>(mm.malloc(allocation_size));
}

template <typename MemoryManagerType, bool warp_based>
__global__ void d_testFree(MemoryManagerType mm, int** verification_ptr, int num_allocations)
{
	int tid{0};
	if(warp_based)
	{
		tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
		if(threadIdx.x % 32 != 0)
			return;
	}
	else
	{
		tid = threadIdx.x + blockIdx.x * blockDim.x;
	}
	if(tid >= num_allocations)
		return;

	mm.free(verification_ptr[tid]);
}

int main(int argc, char* argv[])
{
	// Usage: num_allocations size_of_allocation_in_byte print_output
	unsigned int num_allocations{10000};
	unsigned int allocation_size_byte{16};
	int num_iterations {25};
	bool warp_based{false};
	bool print_output{true};
	bool free_memory{true};
	if(argc >= 2)
	{
		num_allocations = atoi(argv[1]);
		if(argc >= 3)
		{
			allocation_size_byte = atoi(argv[2]);
			if(argc >= 4)
			{
				num_iterations = atoi(argv[3]);
				if(argc >= 5)
				{
					warp_based = static_cast<bool>(atoi(argv[4]));
					if(argc >= 6)
					{
						print_output = static_cast<bool>(atoi(argv[5]));
						if(argc >= 7)
							free_memory = static_cast<bool>(atoi(argv[6]));
					}
				}
			}
		}
	}
	allocation_size_byte = Utils::alignment(allocation_size_byte, sizeof(int));
	if(print_output)
		std::cout << "Number of Allocations: " << num_allocations << " | Allocation Size: " << allocation_size_byte << std::endl;

	int device{0};
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);

#ifdef TEST_CUDA
	std::cout << "--- CUDA ---\n";
	MemoryManagerCUDA memory_manager;
	std::string mem_name("CUDA");
#elif TEST_HALLOC
	std::cout << "--- Halloc ---\n";
	MemoryManagerHalloc memory_manager;
	std::string mem_name("Halloc");
#elif TEST_SCATTERALLOC
	std::cout << "--- ScatterAlloc ---\n";
	MemoryManagerScatterAlloc memory_manager;
	std::string mem_name("ScatterAlloc");
#elif TEST_OUROBOROS
	std::cout << "--- Ouroboros ---";
	#ifdef TEST_PAGES
	#ifdef TEST_VIRTUALIZED_ARRAY
	std::cout << " Page --- Virtualized Array ---\n";
	MemoryManagerOuroboros<OuroVAPQ> memory_manager;
	std::string mem_name("Ouroboros-P-VA");
	#elif TEST_VIRTUALIZED_LIST
	std::cout << " Page --- Virtualized List ---\n";
	MemoryManagerOuroboros<OuroVLPQ> memory_manager;
	std::string mem_name("Ouroboros-P-VL");
	#else
	std::cout << " Page --- Standard ---\n";
	MemoryManagerOuroboros<OuroPQ> memory_manager;
	std::string mem_name("Ouroboros-P-S");
	#endif
	#endif
	#ifdef TEST_CHUNKS
	#ifdef TEST_VIRTUALIZED_ARRAY
	std::cout << " Chunk --- Virtualized Array ---\n";
	MemoryManagerOuroboros<OuroVACQ> memory_manager;
	std::string mem_name("Ouroboros-C-VA");
	#elif TEST_VIRTUALIZED_LIST
	std::cout << " Chunk --- Virtualized List ---\n";
	MemoryManagerOuroboros<OuroVLCQ> memory_manager;
	std::string mem_name("Ouroboros-C-VL");
	#else
	std::cout << " Chunk --- Standard ---\n";
	MemoryManagerOuroboros<OuroCQ> memory_manager;
	std::string mem_name("Ouroboros-C-S");
	#endif
	#endif
#elif TEST_FDG
	std::cout << "--- FDGMalloc ---\n";
	MemoryManagerFDG memory_manager;
	std::string mem_name("FDGMalloc");
#elif TEST_REGEFF
	std::cout << "--- RegEff ---";
	#ifdef TEST_ATOMIC
	std::cout << " Atomic\n";
	MemoryManagerRegEff<RegEffVariants::AtomicMalloc> memory_manager;
	std::string mem_name("RegEff-A");
	#elif TEST_ATOMIC_WRAP
	std::cout << " Atomic Wrap\n";
	MemoryManagerRegEff<RegEffVariants::AWMalloc> memory_manager;
	std::string mem_name("RegEff-AW");
	#elif TEST_CIRCULAR
	std::cout << " Circular\n";
	MemoryManagerRegEff<RegEffVariants::CMalloc> memory_manager;
	std::string mem_name("RegEff-C");
	#elif TEST_CIRCULAR_FUSED
	std::cout << " Circular Fused\n";
	MemoryManagerRegEff<RegEffVariants::CFMalloc> memory_manager;
	std::string mem_name("RegEff-CF");
	#elif TEST_CIRCULAR_MULTI
	std::cout << " Circular Multi\n";
	MemoryManagerRegEff<RegEffVariants::CMMalloc> memory_manager;
	std::string mem_name("RegEff-CM");
	#elif TEST_CIRCULAR_FUSED_MULTI
	std::cout << " Circular Fused Multi\n";
	MemoryManagerRegEff<RegEffVariants::CFMMalloc> memory_manager;
	std::string mem_name("RegEff-CFM");
	#endif
#endif

	memory_manager.init();

	int** d_memory{nullptr};
	CHECK_ERROR(cudaMalloc(&d_memory, sizeof(int*) * num_allocations));

	std::ofstream results_alloc, results_free;
	results_alloc.open((std::string("../results/alloc_") + prop.name  + "_" + mem_name + "_" + std::to_string(num_allocations) + ".csv").c_str(), std::ios_base::app);
	results_free.open((std::string("../results/free_") + prop.name + "_" + mem_name + "_" + std::to_string(num_allocations) + ".csv").c_str(), std::ios_base::app);
	results_alloc << "\n" << allocation_size_byte << ",";
	results_free << "\n" << allocation_size_byte << ",";

	int blockSize {256};
	int gridSize {Utils::divup<int>(num_allocations, blockSize)};
	float timing_allocation{0.0f};
	float timing_free{0.0f};
	cudaEvent_t start, end;
	for(auto i = 0; i < num_iterations; ++i)
	{
		Utils::start_clock(start, end);
		if(warp_based)
			d_testAllocation <decltype(memory_manager), true> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, allocation_size_byte);
		else
			d_testAllocation <decltype(memory_manager), false> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, allocation_size_byte);
		timing_allocation += Utils::end_clock(start, end);

		CHECK_ERROR(cudaDeviceSynchronize());

		if(free_memory)
		{
			Utils::start_clock(start, end);
			if(warp_based)
				d_testFree <decltype(memory_manager), true> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations);
			else
				d_testFree <decltype(memory_manager), false> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations);
			timing_free += Utils::end_clock(start, end);
	
			CHECK_ERROR(cudaDeviceSynchronize());
		}
	}
	timing_allocation /= num_iterations;
	timing_free /= num_iterations;

	if(print_output)
	{
		std::cout << "Timing Allocation: " << timing_allocation << "ms" << std::endl;
		std::cout << "Timing       Free: " << timing_free << "ms" << std::endl;

		std::cout << "Testcase DONE!\n";
	}
	
	results_alloc << timing_allocation;
	results_free << timing_free;

	return 0;
}