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

__global__ void mallocKernel(MemoryManager mm, int** __restrict verification_ptr)
{
	verification_ptr[threadIdx.x + blockIdx.x * blockDim.x] = reinterpret_cast<int*>(mm.malloc(32));
}

__global__ void freeKernel(MemoryManager mm, int** __restrict verification_ptr)
{
	mm.free(verification_ptr[threadIdx.x + blockIdx.x * blockDim.x]);
}

int main(int argc, char* argv[])
{
	std::string csv_path{"../results/tmp/"};
	if(argc >= 2)
	{
		csv_path = std::string(argv[1]);
	}
	std::ofstream results;
	results.open(csv_path.c_str(), std::ios_base::app);

	struct cudaFuncAttributes funcAttribMalloc;
	CHECK_ERROR(cudaFuncGetAttributes(&funcAttribMalloc, mallocKernel));
	printf("%s numRegs = %d\n","Malloc-Kernel",funcAttribMalloc.numRegs);
	results << funcAttribMalloc.numRegs << ", ";

	struct cudaFuncAttributes funcAttribFree;
	CHECK_ERROR(cudaFuncGetAttributes(&funcAttribFree, freeKernel));
	printf("%s numRegs = %d\n","Free-Kernel",funcAttribFree.numRegs);
	results << funcAttribFree.numRegs;

	return 0;
}