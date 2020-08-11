#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <algorithm> 
#include <random>

#include "UtilityFunctions.cuh"
#include "CudaUniquePointer.cuh"
#include "PerformanceMeasure.cuh"

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


__global__ void d_baseline_requirements(const int* __restrict allocation_size, int num_allocations, int* __restrict requirements)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
		
	requirements[tid] = allocation_size[tid] / sizeof(int);
}

__global__ void d_write_assignment(int* __restrict assignment, int* __restrict pos, int num_allocations, const int* __restrict requirements)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	const auto offset = requirements[tid];
	const auto elements = requirements[tid + 1] - offset;
	for(auto i = 0; i < elements; ++i)
	{
		assignment[offset + i] = tid;
		pos[offset + i] = i;
	}
}

__global__ void d_write(const int* __restrict assignment, const int* __restrict position,  const int* __restrict requirements, int** __restrict verification_ptr, int num_values)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_values)
		return;
	
	auto ptr = verification_ptr[assignment[tid]];
	ptr[position[tid]] = tid;
}

#ifdef TEST_BASELINE

__global__ void d_baseline_write(const int* __restrict requirements, int num_allocations, int* __restrict storage_area, int** __restrict verification_ptr)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	verification_ptr[tid] = storage_area + requirements[tid];
}

#else

__global__ void d_memorymanager_alloc(MemoryManager mm, const int* __restrict allocation_size, int num_allocations, int** __restrict verification_ptr)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	verification_ptr[tid] = reinterpret_cast<int*>(mm.malloc(allocation_size[tid]));
}

__global__ void d_memorymanager_free(MemoryManager mm, const int* __restrict allocation_size, int num_allocations, int** __restrict verification_ptr)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	mm.free(verification_ptr[tid]);
}

#endif

int main(int argc, char* argv[])
{
	// Usage: num_allocations size_of_allocation_in_byte print_output
	unsigned int num_allocations{10000};
	unsigned int allocation_size_range_lower{4};
	unsigned int allocation_size_range_upper{64};
	int num_iterations {25};
	int allocSizeinGB{8};
	std::string csv_path{"../results/tmp/"};
	bool writeToMemory{false};
	if(argc >= 2)
	{
		num_allocations = atoi(argv[1]);
		if(argc >= 3)
		{
			allocation_size_range_lower = atoi(argv[2]);
			if(argc >= 4)
			{
				allocation_size_range_upper = atoi(argv[3]);
				if(argc >= 5)
				{
					num_iterations = atoi(argv[4]);
					if(argc >= 6)
					{
						writeToMemory = static_cast<bool>(atoi(argv[5]));
						if(argc >= 7)
						{
							csv_path = std::string(argv[6]);
							
							if(argc >= 8)
							{
								allocSizeinGB = atoi(argv[7]);
							}
						}
					}
				}
			}
		}
	}

	allocation_size_range_lower = Utils::alignment(allocation_size_range_lower, sizeof(int));
	allocation_size_range_upper = Utils::alignment(allocation_size_range_upper, sizeof(int));
	std::cout << "Number of Allocations: " << num_allocations << " | Allocation Range: " << allocation_size_range_lower << " - " << allocation_size_range_upper << std::endl;

	int device{0};
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	
	#ifndef TEST_BASELINE
	MemoryManager memory_manager(allocSizeinGB * 1024ULL * 1024ULL * 1024ULL);
	#endif

	std::ofstream results;
	results.open(csv_path.c_str(), std::ios_base::app);

	CudaUniquePointer<int> d_allocation_sizes(num_allocations);
	CudaUniquePointer<int*> d_verification_ptrs(num_allocations);
	int allocation_sizes[num_allocations];
	auto range = allocation_size_range_upper - allocation_size_range_lower;
	auto offset = allocation_size_range_lower;

	// PerfMeasure timing;

	// for(auto i = 0; i < num_iterations; ++i)
	// {
	// 	std::mt19937 gen(i); //Standard mersenne_twister_engine seeded with iteration
    // 	std::uniform_real_distribution<> dis(0.0, 1.0);
	// 	// Generate sizes
	// 	srand(i);
	// 	int num_integers{0};
	// 	for(auto i = 0; i < num_allocations; ++i)
	// 	{
	// 		int val = Utils::alignment(offset + dis(gen) * range, sizeof(int));
	// 		allocation_sizes[i] = val;
	// 		num_integers += val / sizeof(int);
	// 	}
			
	// 	d_allocation_sizes.copyToDevice(allocation_sizes, num_allocations);

	// 	int blockSize{256};
	// 	int gridSize = Utils::divup(num_allocations, blockSize);

	// 	// Start measurement
	// 	timing.startMeasurement();
		
	// 	#ifdef TEST_BASELINE

	// 	int* requirements{nullptr};
	// 	CHECK_ERROR(cudaMalloc(&requirements, (num_allocations + 1) * sizeof(int)));

	// 	d_baseline_requirements <<< gridSize, blockSize >>> (d_allocation_sizes.get(), num_allocations, requirements);

	// 	// Exclusive sum
	// 	Helper::thrustExclusiveSum(requirements, num_allocations + 1);

	// 	// Copy back num elements
	// 	int num_items{0};
	// 	CHECK_ERROR(cudaMemcpy(&num_items, requirements + num_allocations, sizeof(int), cudaMemcpyDeviceToHost));

	// 	// Allocate memory
	// 	int* storage_area{nullptr};
	// 	CHECK_ERROR(cudaMalloc(&storage_area, num_items * sizeof(int)));

	// 	// Write ptrs to verification ptrs
	// 	d_baseline_write <<< gridSize, blockSize >>>(requirements, num_allocations, storage_area, d_verification_ptrs.get());

	// 	#else
	// 	d_memorymanager_alloc <<< gridSize, blockSize >>>(memory_manager, d_allocation_sizes.get(), num_allocations, d_verification_ptrs.get());
	// 	#endif

	// 	// **********
	// 	// **********
	// 	// Now we could write to this area
	// 	// **********
	// 	// **********

	// 	#ifdef TEST_BASELINE
	// 	CHECK_ERROR(cudaFree(requirements));
	// 	CHECK_ERROR(cudaFree(storage_area));
	// 	#else
	// 	d_memorymanager_free <<< gridSize, blockSize >>>(memory_manager, d_allocation_sizes.get(), num_allocations, d_verification_ptrs.get());
	// 	#endif

	// 	// Stop Measurement
	// 	timing.stopMeasurement();
	// 	std::cout << "#" << std::flush;
	// }
	// std::cout << std::endl;

	PerfMeasure timing;

	for(auto i = 0; i < num_iterations; ++i)
	{
		std::mt19937 gen(i); //Standard mersenne_twister_engine seeded with iteration
    	std::uniform_real_distribution<> dis(0.0, 1.0);
		// Generate sizes
		srand(i);
		int num_integers{0};
		for(auto i = 0; i < num_allocations; ++i)
		{
			int val = Utils::alignment(offset + dis(gen) * range, sizeof(int));
			allocation_sizes[i] = val;
			num_integers += val / sizeof(int);
		}
			
		d_allocation_sizes.copyToDevice(allocation_sizes, num_allocations);

		int blockSize{256};
		int gridSize = Utils::divup(num_allocations, blockSize);

		int* requirements{nullptr};
		CHECK_ERROR(cudaMalloc(&requirements, (num_allocations + 1) * sizeof(int)));

		d_baseline_requirements <<< gridSize, blockSize >>> (d_allocation_sizes.get(), num_allocations, requirements);

		// Exclusive sum
		Helper::thrustExclusiveSum(requirements, num_allocations + 1);
		
		#ifdef TEST_BASELINE

		// Copy back num elements
		int num_items{0};
		CHECK_ERROR(cudaMemcpy(&num_items, requirements + num_allocations, sizeof(int), cudaMemcpyDeviceToHost));

		// Allocate memory
		int* storage_area{nullptr};
		CHECK_ERROR(cudaMalloc(&storage_area, num_items * sizeof(int)));

		// Write ptrs to verification ptrs
		d_baseline_write <<< gridSize, blockSize >>>(requirements, num_allocations, storage_area, d_verification_ptrs.get());

		#else
		d_memorymanager_alloc <<< gridSize, blockSize >>>(memory_manager, d_allocation_sizes.get(), num_allocations, d_verification_ptrs.get());
		#endif

		// **********
		// **********
		// Now we could write to this area
		// **********
		// **********

		CudaUniquePointer<int> d_assignment(num_integers);
		CudaUniquePointer<int> d_pos(num_integers);
		d_write_assignment<<<gridSize, blockSize>>>(d_assignment.get(), d_pos.get(), num_allocations, requirements);

		timing.startMeasurement();
		gridSize = Utils::divup(num_integers, blockSize);
		for(auto j = 0; j < 100; ++j)
			d_write<<<gridSize, blockSize>>>(d_assignment.get(), d_pos.get(), requirements, d_verification_ptrs.get(), num_integers);
		
		timing.stopMeasurement();

		#ifdef TEST_BASELINE
		CHECK_ERROR(cudaFree(requirements));
		CHECK_ERROR(cudaFree(storage_area));
		#else
		d_memorymanager_free <<< gridSize, blockSize >>>(memory_manager, d_allocation_sizes.get(), num_allocations, d_verification_ptrs.get());
		#endif

		std::cout << "#" << std::flush;
	}
	std::cout << std::endl;

	auto result = timing.generateResult();
	results << result.mean_ << "," << result.std_dev_ << "," << result.min_ << "," << result.max_ << "," << result.median_;
	std::cout << "Timing: " << result.mean_ << " ms" << std::endl;

	return 0;
}