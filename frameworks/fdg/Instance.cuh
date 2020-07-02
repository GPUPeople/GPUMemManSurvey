#pragma once

#include "TestInstance.cuh"
#include "FDGMalloc_def.cuh"
#include "FDGMalloc_impl.cuh"

struct MemoryManagerFDG : public MemoryManagerBase
{
	explicit MemoryManagerFDG(size_t instantiation_size) : MemoryManagerBase(instantiation_size) 
	{
		if(initialized)
			return;
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
		initialized = true;
	}
	~MemoryManagerFDG(){}

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		FDG::Warp* warp = FDG::Warp::start();
		return warp->alloc(size);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		//warp->end();
		//warp->tidyUp();
	}

	static bool initialized;
};

bool MemoryManagerFDG::initialized = false;