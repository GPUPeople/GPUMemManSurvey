#pragma once

#include "TestInstance.cuh"
#include "gpualloc_impl.cuh"

enum class RegEffVariants
{
	CudaMalloc,		// - mallocCudaMalloc, freeCudaMalloc
	AtomicMalloc,	// - mallocAtomicMalloc
	AWMalloc,		// - mallocAtomicWrapMalloc
	CMalloc, 		// - mallocCircularMalloc, freeCircularMalloc
	CFMalloc,		// - mallocCircularFusedMalloc, freeCircularFusedMalloc
	CMMalloc,		// - mallocCircularMultiMalloc, freeCircularMultiMalloc
	CFMMalloc		// - mallocCircularFusedMultiMalloc, freeCircularFusedMultiMalloc
};

template <RegEffVariants variant=RegEffVariants::CudaMalloc>
struct MemoryManagerRegEff : public MemoryManagerBase
{
	explicit MemoryManagerRegEff(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : MemoryManagerBase(instantiation_size) {}
	~MemoryManagerRegEff(){}

	virtual void init() override
	{
		// Allocate the memory for the heap
		if(variant == RegEffVariants::CudaMalloc)
		{
			cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
		}
		else
		{
			
			
		}

		// Prepare the memory for the heap
		if (variant == RegEffVariants::AtomicMalloc || variant == RegEffVariants::AWMalloc)
		{
			void* m_mallocData{nullptr};
			cudaMalloc(&m_mallocData, size);
			cudaMemcpyToSymbol(g_heapBase, reinterpret_cast<char**>(&m_mallocData), sizeof(char*));
			unsigned int _g_heapOffset{0};
			cudaMemcpyToSymbol(g_heapOffset, &_g_heapOffset, sizeof(unsigned int));
		}
		else if (variant == RegEffVariants::CMalloc || variant == RegEffVariants::CFMalloc ||
			variant == RegEffVariants::CMMalloc || variant == RegEffVariants::CFMMalloc)
		{

		}
	}

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		if(variant == RegEffVariants::CudaMalloc)
		{
			return mallocCudaMalloc(size);
		}
		else if (variant == RegEffVariants::AtomicMalloc)
		{
			return mallocAtomicMalloc(size);
		}
		else if (variant == RegEffVariants::AWMalloc)
		{
			return mallocAtomicWrapMalloc(size);
		}
		else if (variant == RegEffVariants::CMalloc)
		{
			return mallocCircularMalloc(size);
		}
		else if (variant == RegEffVariants::CFMalloc)
		{
			return mallocCircularFusedMalloc(size);
		}
		else if (variant == RegEffVariants::CMMalloc)
		{
			return mallocCircularMultiMalloc(size);
		}
		else if (variant == RegEffVariants::CFMMalloc)
		{
			return mallocCircularFusedMultiMalloc(size);
		}
		else
		{
			printf("Variant not implemented!\n");
			return nullptr;
		}
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		if(variant == RegEffVariants::CudaMalloc)
		{
			freeCudaMalloc(ptr);
		}
		else if (variant == RegEffVariants::CMalloc)
		{
			freeCircularMalloc(ptr);
		}
		else if (variant == RegEffVariants::CFMalloc)
		{
			freeCircularFusedMalloc(ptr);
		}
		else if (variant == RegEffVariants::CMMalloc)
		{
			freeCircularMultiMalloc(ptr);
		}
		else if (variant == RegEffVariants::CFMMalloc)
		{
			freeCircularFusedMultiMalloc(ptr);
		}
		else
		{
			// variant == RegEffVariants::AtomicMalloc || variant == RegEffVariants::AWMalloc
			// Those variants have no deallocations
		}
	}

	__host__ std::string getDescriptor()
	{
		if(variant == RegEffVariants::CudaMalloc)
		{
			return std::string("RegEff - CudaMalloc");
		}
		else if (variant == RegEffVariants::AtomicMalloc)
		{
			return std::string("RegEff - AtomicMalloc");
		}
		else if (variant == RegEffVariants::AWMalloc)
		{
			return std::string("RegEff - AWMalloc");
		}
		else if (variant == RegEffVariants::CMalloc)
		{
			return std::string("RegEff - CMalloc");
		}
		else if (variant == RegEffVariants::CFMalloc)
		{
			return std::string("RegEff - CFMalloc");
		}
		else if (variant == RegEffVariants::CMMalloc)
		{
			return std::string("RegEff - CMMalloc");
		}
		else if (variant == RegEffVariants::CFMMalloc)
		{
			return std::string("RegEff - CFMMalloc");
		}
	}

	
};
