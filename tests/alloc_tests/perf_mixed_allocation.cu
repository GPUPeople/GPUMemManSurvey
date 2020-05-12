#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>

#include "UtilityFunctions.cuh"
#include "PerformanceMeasure.cuh"
#include "DevicePerformanceMeasure.cuh"

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
__global__ void d_testAllocation(MemoryManagerType mm, int** verification_ptr, int num_allocations, unsigned int* allocation_size)
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

	verification_ptr[tid] = reinterpret_cast<int*>(mm.malloc(allocation_size[tid]));
}

template <typename MemoryManagerType>
__global__ void d_testAllocation(MemoryManagerType mm, int** verification_ptr, int num_allocations, unsigned int* allocation_size, DevicePerfMeasure::Type* timing)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	DevicePerf perf_measure;
	auto alloc_size = allocation_size[tid];
	
	// Start Measure
	perf_measure.startThreadMeasure();

	auto ptr = reinterpret_cast<int*>(mm.malloc(alloc_size));
	
	// Stop Measure
	timing[tid] = perf_measure.stopThreadMeasure();

	verification_ptr[tid] = ptr;
}

__global__ void d_testWriteToMemory(int** verification_ptr, int num_allocations, unsigned int* allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	auto ptr = verification_ptr[tid];
	auto alloc_size = allocation_size[tid];
	for(auto i = 0; i < (alloc_size / sizeof(int)); ++i)
	{
		ptr[i] = tid;
	}
}

__global__ void d_testReadFromMemory(int** verification_ptr, int num_allocations, unsigned int* allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	auto ptr = verification_ptr[tid];
	auto alloc_size = allocation_size[tid];
	for(auto i = 0; i < (alloc_size / sizeof(int)); ++i)
	{
		if(ptr[i] != tid)
		{
			printf("%d | We got a wrong value here! %d vs %d\n", tid, ptr[i], tid);
			__trap();
		}
	}
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

template <typename MemoryManagerType>
__global__ void d_testFree(MemoryManagerType mm, int** verification_ptr, int num_allocations, DevicePerfMeasure::Type* timing)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	DevicePerf perf_measure;

	// Start Measure
	perf_measure.startThreadMeasure();

	mm.free(verification_ptr[tid]);

	// Stop Measure
	timing[tid] = perf_measure.stopThreadMeasure();
}

int main(int argc, char* argv[])
{
	unsigned int num_allocations{10000};
	unsigned int allocation_size_byte_low{4};
	unsigned int allocation_size_byte_high{8192};
	int num_iterations {100};
	bool warp_based{false};
	bool print_output{true};
	bool onDeviceMeasure{false};
	bool generate_output{false};
	bool free_memory{true};
	std::string alloc_csv_path{"../results/tmp/"};
	std::string free_csv_path{"../results/tmp/"};
	if(argc >= 11)
	{
		num_allocations = atoi(argv[1]);
		allocation_size_byte_low = atoi(argv[2]);
		allocation_size_byte_high = atoi(argv[3]);
		num_iterations = atoi(argv[4]);
		onDeviceMeasure = static_cast<bool>(atoi(argv[5]));
		warp_based = static_cast<bool>(atoi(argv[6]));
		generate_output = static_cast<bool>(atoi(argv[7]));
		free_memory = static_cast<bool>(atoi(argv[8]));
		alloc_csv_path = std::string(argv[9]);
		free_csv_path = std::string(argv[10]);
	}
	else
	{
		std::cout << "Invalid configuration!\n";
		std::cout << "Call as ./mixed_allocation <num_alloc> <min_size_range> <max_size_range> ";
		std::cout << "<num_iter> <device_measure> <warp_based> <output> <free_mem> <alloc_csv> <free_csv>\n";
		exit(-1);
	}
			
	std::cout << "--- " << mem_name << "---\n";
	MemoryManager memory_manager;

	std::vector<unsigned int> allocation_sizes(num_allocations);
	unsigned int* d_allocation_sizes{nullptr};
	int** d_memory{nullptr};
	CHECK_ERROR(cudaMalloc(&d_memory, sizeof(int*) * num_allocations));
	CHECK_ERROR(cudaMalloc(&d_allocation_sizes, sizeof(unsigned int) * num_allocations));

	std::ofstream results_alloc, results_free;
	if(generate_output)
	{
		results_alloc.open(alloc_csv_path.c_str(), std::ios_base::app);
		results_free.open(free_csv_path.c_str(), std::ios_base::app);
	}

	int blockSize {256};
	int gridSize {Utils::divup<int>(num_allocations, blockSize)};
	if(warp_based)
		gridSize *= 32;

	PerfMeasure timing_allocation;
	PerfMeasure timing_free;

	DevicePerfMeasure per_thread_timing_allocation(num_allocations, num_iterations);
	DevicePerfMeasure per_thread_timing_free(num_allocations, num_iterations);


	auto range = allocation_size_byte_high - allocation_size_byte_low;
	auto offset = allocation_size_byte_low;
	for(auto i = 0; i < num_iterations; ++i)
	{
		std::mt19937 gen(i); //Standard mersenne_twister_engine seeded with iteration
    	std::uniform_real_distribution<> dis(0.0, 1.0);
		// Generate sizes
		srand(i);
		for(auto i = 0; i < num_allocations; ++i)
			allocation_sizes[i] = Utils::alignment(offset + dis(gen) * range, sizeof(int));
		CHECK_ERROR(cudaMemcpy(d_allocation_sizes, allocation_sizes.data(), sizeof(unsigned int) * num_allocations, cudaMemcpyHostToDevice));

		// TODO:: Continue
		if(onDeviceMeasure)
		{
			d_testAllocation <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, d_allocation_sizes, per_thread_timing_allocation.getDevicePtr());
			CHECK_ERROR(cudaDeviceSynchronize());
			per_thread_timing_allocation.acceptResultsFromDevice();
		}
		else
		{
			timing_allocation.startMeasurement();
			if(warp_based)
				d_testAllocation <decltype(memory_manager), true> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, d_allocation_sizes);
			else
				d_testAllocation <decltype(memory_manager), false> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, d_allocation_sizes);
			timing_allocation.stopMeasurement();
			CHECK_ERROR(cudaDeviceSynchronize());
		}

		d_testWriteToMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, d_allocation_sizes);

		CHECK_ERROR(cudaDeviceSynchronize());

		d_testReadFromMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, d_allocation_sizes);

		CHECK_ERROR(cudaDeviceSynchronize());

		if(free_memory)
		{
			if(onDeviceMeasure)
			{
				d_testFree <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, per_thread_timing_free.getDevicePtr());
				CHECK_ERROR(cudaDeviceSynchronize());
				per_thread_timing_free.acceptResultsFromDevice();
			}
			else
			{
				timing_free.startMeasurement();
				if(warp_based)
					d_testFree <decltype(memory_manager), true> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations);
				else
					d_testFree <decltype(memory_manager), false> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations);
				timing_free.stopMeasurement();
				CHECK_ERROR(cudaDeviceSynchronize());
			}
		}
	}

	if(onDeviceMeasure)
	{
		auto alloc_result = per_thread_timing_allocation.generateResult();
		auto free_result = per_thread_timing_free.generateResult();

		if(print_output)
		{
			std::cout << "Timing Allocation: Mean:" << alloc_result.mean_ << "cycles | Median: " << alloc_result.median_ << " ms" << std::endl;
			std::cout << "Timing       Free: Mean:" << free_result.mean_ << "cycles | Median: " << free_result.median_ << " ms" << std::endl;
		}
		if(generate_output)
		{
			results_alloc << alloc_result.mean_ << "," << alloc_result.std_dev_ << "," << alloc_result.min_ << "," << alloc_result.max_ << "," << alloc_result.median_;
			results_free << free_result.mean_ << "," << free_result.std_dev_ << "," << free_result.min_ << "," << free_result.max_ << "," << free_result.median_;
		}
	}
	else
	{
		auto alloc_result = timing_allocation.generateResult();
		auto free_result = timing_free.generateResult();
		if(print_output)
		{
			std::cout << "Timing Allocation: Mean:" << alloc_result.mean_ << "ms" << std::endl;// " | Median: " << alloc_result.median_ << " ms" << std::endl;
			std::cout << "Timing       Free: Mean:" << free_result.mean_ << "ms" << std::endl;// "  | Median: " << free_result.median_ << " ms" << std::endl;
		}
		if(generate_output)
		{
			results_alloc << alloc_result.mean_ << "," << alloc_result.std_dev_ << "," << alloc_result.min_ << "," << alloc_result.max_ << "," << alloc_result.median_;
			results_free << free_result.mean_ << "," << free_result.std_dev_ << "," << alloc_result.min_ << "," << alloc_result.max_ << "," << free_result.median_;
		}
	}

	CHECK_ERROR(cudaFree(d_allocation_sizes));
	CHECK_ERROR(cudaFree(d_memory));

    return 0;
}