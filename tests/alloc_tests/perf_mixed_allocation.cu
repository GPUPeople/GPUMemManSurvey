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
#elif TEST_HALLOC
#include "halloc/Instance.cuh"
#elif TEST_XMALLOC
#include "xmalloc/Instance.cuh"
#elif TEST_SCATTERALLOC
#include "scatteralloc/Instance.cuh"
#elif TEST_OUROBOROS
#include "ouroboros/Instance.cuh"
#elif TEST_FDG
#include "fdg/Instance.cuh"
#elif TEST_REGEFF
#include "regeff/Instance.cuh"
#endif

int main(int argc, char* argv[])
{
	unsigned int num_allocations{10000};
	unsigned int allocation_size_byte_low{4};
	unsigned int allocation_size_byte_high{8192};
	int num_iterations {100};
	bool warp_based{false};
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
			

#ifdef TEST_CUDA
	std::cout << "--- CUDA ---\n";
	MemoryManagerCUDA memory_manager;
	std::string mem_name("CUDA");
#elif TEST_XMALLOC
	std::cout << "--- XMalloc ---\n";
	MemoryManagerXMalloc memory_manager;
	std::string mem_name("XMalloc");
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

	std::vector<unsigned int> allocation_sizes(num_allocations);
	unsigned int* d_allocation_sizes{nullptr};
	int* d_memory{nullptr};
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
		std::mt19937 gen(i); //Standard mersenne_twister_engine seeded with rd()
    	std::uniform_real_distribution<> dis(0.0, 1.0);
		// Generate sizes
		srand(i);
		for(auto i = 0; i < num_allocations; ++i)
			allocation_sizes[i] = offset + dis(gen) * range;
		CHECK_ERROR(cudaMemcpy(d_allocation_sizes, allocation_sizes.data(), sizeof(unsigned int) * num_allocations, cudaMemcpyHostToDevice));

		// TODO:: Continue
	}

	CHECK_ERROR(cudaFree(d_allocation_sizes));
	CHECK_ERROR(cudaFree(d_memory));

    return 0;
}