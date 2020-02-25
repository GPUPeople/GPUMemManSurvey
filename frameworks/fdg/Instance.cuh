#pragma once

#include "TestInstance.cuh"
#include "FDGMalloc_def.cuh"
#include "FDGMalloc_impl.cuh"

struct MemoryManagerFDG : public MemoryManagerBase
{
	explicit MemoryManagerFDG(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : MemoryManagerBase(instantiation_size) {}
	~MemoryManagerFDG(){}

	virtual void init() override
	{
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
	}

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		FDG::Warp* warp = FDG::Warp::start();
		return warp->alloc(size);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		FDG::Warp* warp = FDG::Warp::start();
		// warp->end();
		warp->tidyUp();
	}
};
