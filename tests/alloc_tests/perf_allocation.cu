#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "UtilityFunctions.cuh"
#include "PerformanceMeasure.cuh"
#include "DevicePerformanceMeasure.cuh"

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

template <typename MemoryManagerType>
__global__ void d_testAllocation(MemoryManagerType mm, int** verification_ptr, int num_allocations, int allocation_size, DevicePerfMeasure::Type* timing)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	DevicePerf perf_measure;
	
	// Start Measure
	perf_measure.startThreadMeasure();

	auto ptr = reinterpret_cast<int*>(mm.malloc(allocation_size));
	
	// Stop Measure
	timing[tid] = perf_measure.stopThreadMeasure();

	verification_ptr[tid] = ptr;
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
	// Usage: <num_allocations> <size_of_allocation_in_byte> <num_iterations> <onDeviceMeasure> <warp-based> <generateoutput> <free_memory> <initial_path>
	unsigned int num_allocations{10000};
	unsigned int allocation_size_byte{8192};
	int num_iterations {100};
	bool warp_based{false};
	bool onDeviceMeasure{false};
	bool print_output{true};
	bool generate_output{false};
	bool write_header{false};
	bool free_memory{true};
	std::string initial_path{"../results/tmp/"};
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
					onDeviceMeasure = static_cast<bool>(atoi(argv[4]));
					if(argc >= 6)
					{
						warp_based = static_cast<bool>(atoi(argv[5]));
						if(onDeviceMeasure && warp_based)
						{
							std::cout << "OnDeviceMeasure and warp-based not possible at the same!" << std::endl;
							exit(-1);
						}
						if(argc >= 7)
						{
							generate_output = static_cast<bool>(atoi(argv[6]));
							if(argc >= 8)
							{
								write_header = static_cast<bool>(atoi(argv[7]));
								if(argc >= 9)
								{
									free_memory = static_cast<bool>(atoi(argv[8]));
									if(argc >= 9)
									{
										initial_path = std::string(argv[9]);
									}
								}
							}
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

	int** d_memory{nullptr};
	CHECK_ERROR(cudaMalloc(&d_memory, sizeof(int*) * num_allocations));

	std::ofstream results_alloc, results_free;
	if(generate_output)
	{
		results_alloc.open((initial_path + std::string("alloc_") + prop.name  + "_" + mem_name + "_" + std::to_string(num_allocations) + ".csv").c_str(), std::ios_base::app);
		results_free.open((initial_path + std::string("free_") + prop.name + "_" + mem_name + "_" + std::to_string(num_allocations) + ".csv").c_str(), std::ios_base::app);
		if(write_header)
		{
			results_alloc << "AllocationSize (in Byte), mean, std-dev, min, max, median";
			results_free << "AllocationSize (in Byte), mean, std-dev, min, max, median";
		}
		results_alloc << "\n" << allocation_size_byte << ",";
		results_free << "\n" << allocation_size_byte << ",";
	}

	int blockSize {256};
	int gridSize {Utils::divup<int>(num_allocations, blockSize)};

	PerfMeasure timing_allocation;
	PerfMeasure timing_free;

	DevicePerfMeasure per_thread_timing_allocation(num_allocations, num_iterations);
	DevicePerfMeasure per_thread_timing_free(num_allocations, num_iterations);

	for(auto i = 0; i < num_iterations; ++i)
	{
		if(onDeviceMeasure)
		{
			d_testAllocation <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, allocation_size_byte, per_thread_timing_allocation.getDevicePtr());
			CHECK_ERROR(cudaDeviceSynchronize());
			per_thread_timing_allocation.acceptResultsFromDevice();
		}
		else
		{
			timing_allocation.startMeasurement();
			if(warp_based)
				d_testAllocation <decltype(memory_manager), true> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, allocation_size_byte);
			else
				d_testAllocation <decltype(memory_manager), false> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, allocation_size_byte);
			timing_allocation.stopMeasurement();
			CHECK_ERROR(cudaDeviceSynchronize());
		}

		d_testWriteToMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);

		CHECK_ERROR(cudaDeviceSynchronize());

		d_testReadFromMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);

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
	
	return 0;
}