#pragma once

#include "TestInstance.cuh"
#include "gpualloc_impl.cuh"

struct MemoryManagerRegEff : public MemoryManagerBase
{
	explicit MemoryManagerRegEff(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : MemoryManagerBase(instantiation_size) {}
	~MemoryManagerRegEff(){}

	virtual void init() override
	{
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
	}

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		return nullptr;
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
	}
};
