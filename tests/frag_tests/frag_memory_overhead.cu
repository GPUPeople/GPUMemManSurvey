#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <algorithm> 

#include "UtilityFunctions.cuh"

// ########################
#ifdef TEST_CUDA
#include "cuda/Instance.cuh"
using MemoryManager = MemoryManagerCUDA;
const std::string mem_name("CUDA");
#elif TEST_HALLOC
#include "halloc/Instance.cuh"
using MemoryManager = MemoryManagerHalloc;
const std::string mem_name("HALLOC");
#elif TEST_XMALLOC
#include "xmalloc/Instance.cuh"
using MemoryManager = MemoryManagerXMalloc;
const std::string mem_name("XMALLOC");
#elif TEST_SCATTERALLOC
#include "scatteralloc/Instance.cuh"
using MemoryManager = MemoryManagerScatterAlloc;
const std::string mem_name("ScatterAlloc");
#elif TEST_FDG
#include "fdg/Instance.cuh"
using MemoryManager = MemoryManagerFDG;
const std::string mem_name("FDGMalloc");
#elif TEST_OUROBOROS
#include "ouroboros/Instance.cuh"
	#ifdef TEST_PAGES
	#ifdef TEST_VIRTUALIZED_ARRAY
	using MemoryManager = MemoryManagerOuroboros<OuroVAPQ>;
	const std::string mem_name("Ouroboros-P-VA");
	#elif TEST_VIRTUALIZED_LIST
	using MemoryManager = MemoryManagerOuroboros<OuroVLPQ>;
	const std::string mem_name("Ouroboros-P-VL");
	#else
	using MemoryManager = MemoryManagerOuroboros<OuroPQ>;
	const std::string mem_name("Ouroboros-P-S");
	#endif
	#endif
	#ifdef TEST_CHUNKS
	#ifdef TEST_VIRTUALIZED_ARRAY
	using MemoryManager = MemoryManagerOuroboros<OuroVACQ>;
	const std::string mem_name("Ouroboros-C-VA");
	#elif TEST_VIRTUALIZED_LIST
	using MemoryManager = MemoryManagerOuroboros<OuroVLCQ>;
	const std::string mem_name("Ouroboros-C-VL");
	#else
	using MemoryManager = MemoryManagerOuroboros<OuroCQ>;
	const std::string mem_name("Ouroboros-C-S");
	#endif
	#endif
#elif TEST_REGEFF
#include "regeff/Instance.cuh"
	#ifdef TEST_ATOMIC
	using MemoryManager = MemoryManagerRegEff<RegEffVariants::AtomicMalloc>;
	const std::string mem_name("RegEff-A");
	#elif TEST_ATOMIC_WRAP
	using MemoryManager = MemoryManagerRegEff<RegEffVariants::AWMalloc>;
	const std::string mem_name("RegEff-AW");
	#elif TEST_CIRCULAR
	using MemoryManager = MemoryManagerRegEff<RegEffVariants::CMalloc>;
	const std::string mem_name("RegEff-C");
	#elif TEST_CIRCULAR_FUSED
	using MemoryManager = MemoryManagerRegEff<RegEffVariants::CFMalloc>;
	const std::string mem_name("RegEff-CF");
	#elif TEST_CIRCULAR_MULTI
	using MemoryManager = MemoryManagerRegEff<RegEffVariants::CMMalloc>;
	const std::string mem_name("RegEff-CM");
	#elif TEST_CIRCULAR_FUSED_MULTI
	using MemoryManager = MemoryManagerRegEff<RegEffVariants::CFMMalloc>;
	const std::string mem_name("RegEff-CFM");
	#endif
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
			__trap();
		}
	}
}

int main(int argc, char* argv[])
{
	// Usage: num_allocations size_of_allocation_in_byte print_output
	unsigned int num_allocations{10000};
	unsigned int allocation_size_byte{16};
	int num_iterations {25};
	bool warp_based{false};
	bool print_output{true};
	bool test_oom{false};
	int allocSizeinGB{8};
	std::string csv_path{"../results/tmp/"};
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
					test_oom = static_cast<bool>(atoi(argv[4]));
					if(argc >= 6)
					{
						csv_path = std::string(argv[5]);
						if(argc >= 7)
						{
							allocSizeinGB = atoi(argv[6]);
						}
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
	
	MemoryManager memory_manager(allocSizeinGB * 1024ULL * 1024ULL * 1024ULL);

	int** d_memory{nullptr};
	CHECK_ERROR(cudaMalloc(&d_memory, sizeof(int*) * num_allocations));

	std::ofstream results_frag;
	results_frag.open(csv_path.c_str(), std::ios_base::app);

	int blockSize {256};
	int gridSize {Utils::divup<int>(num_allocations, blockSize)};

	for(auto i = 0; i < num_iterations; ++i)
	{
		if(test_oom)
		{
			d_testAllocation <decltype(memory_manager), false> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, allocation_size_byte);
			CHECK_ERROR(cudaDeviceSynchronize());
			results_frag << "," << i;
			std::cout << "Round " << i + 1 << " done" << std::endl;
		}
		else
		{
			d_testAllocation <decltype(memory_manager), false> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, allocation_size_byte);
			CHECK_ERROR(cudaDeviceSynchronize());

			d_testWriteToMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);

			CHECK_ERROR(cudaDeviceSynchronize());
	
			d_testReadFromMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);
	
			CHECK_ERROR(cudaDeviceSynchronize());

			// Look at address range
			static int* static_min_ptr{reinterpret_cast<int*>(0xFFFFFFFFFFFFFFFFULL)};
			static int* static_max_ptr{nullptr};
			std::vector<int*> verification_pointers(num_allocations);
			CHECK_ERROR(cudaMemcpy(verification_pointers.data(), d_memory, sizeof(int*) * verification_pointers.size(), cudaMemcpyDeviceToHost));
			auto min_ptr = *min_element(verification_pointers.begin(), verification_pointers.end());
			auto max_ptr = *max_element(verification_pointers.begin(), verification_pointers.end());
			static_min_ptr = std::min(static_min_ptr, min_ptr);
			static_max_ptr = std::max(static_max_ptr, max_ptr);
			printf("%llu | %llu | %llu MB | %llu | %llu | %llu B\n", 
			reinterpret_cast<unsigned long long>(min_ptr), 
			reinterpret_cast<unsigned long long>(max_ptr), 
			(reinterpret_cast<unsigned long long>(max_ptr) - reinterpret_cast<unsigned long long>(min_ptr)) / (1024*1024),
			reinterpret_cast<unsigned long long>(static_min_ptr), 
			reinterpret_cast<unsigned long long>(static_max_ptr), 
			(reinterpret_cast<unsigned long long>(static_max_ptr) - reinterpret_cast<unsigned long long>(static_min_ptr)));
			results_frag << (reinterpret_cast<unsigned long long>(max_ptr) - reinterpret_cast<unsigned long long>(min_ptr)) 
				<< "," 
				<<(reinterpret_cast<unsigned long long>(static_max_ptr) - reinterpret_cast<unsigned long long>(static_min_ptr));
			if(num_iterations != 1)
				results_frag << ",";

			d_testFree <decltype(memory_manager), false> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations);
			CHECK_ERROR(cudaDeviceSynchronize());
		}
	}
	
	return 0;
}