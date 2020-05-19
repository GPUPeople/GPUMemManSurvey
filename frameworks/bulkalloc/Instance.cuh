#pragma once
#include "TestInstance.cuh"

struct MemoryManagerBulkAlloc : public MemoryManagerBase
{
	explicit MemoryManagerBulkAlloc(size_t instantiation_size) : MemoryManagerBase(instantiation_size)
	{
		if(initialized)
			return;
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
		initialized = true;
	}
	~MemoryManagerBulkAlloc(){};

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

bool MemoryManagerBulkAlloc::initialized = false;