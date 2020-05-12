#pragma once
#include "TestInstance.cuh"

#include "Xmalloc.cuh"

struct MemoryManagerXMalloc : public MemoryManagerBase
{
	explicit MemoryManagerXMalloc(size_t instantiation_size) : MemoryManagerBase(instantiation_size)
	{
		if(initialized)
			return;
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
		if(xmcInit(size) == 0)
		{
			printf("Error on Init for XMalloc!\n");
			exit(-1);
		}
		initialized = true;
	}
	~MemoryManagerXMalloc(){};

	static constexpr size_t alignment{16ULL};

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		return xmcMalloc(size);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		xmcFree(ptr);
	};

	static bool initialized;
};

bool MemoryManagerXMalloc::initialized = false;