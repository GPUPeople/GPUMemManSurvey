#pragma once
#include "TestInstance.cuh"

struct MemoryManagerCUDA : public MemoryManagerBase
{
	explicit MemoryManagerCUDA(size_t instantiation_size) : MemoryManagerBase(instantiation_size)
	{
		if(initialized)
			return;
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
		initialized = true;
	}
	~MemoryManagerCUDA(){};

	static constexpr size_t alignment{16ULL};

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		return ::malloc(size);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		::free(ptr);
	};

	static bool initialized;
};

bool MemoryManagerCUDA::initialized = false;