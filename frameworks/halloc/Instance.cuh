#pragma once

#include "TestInstance.cuh"
#include "repository/src/halloc.cuh"

struct MemoryManagerHalloc : public MemoryManagerBase
{
	explicit MemoryManagerHalloc(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : MemoryManagerBase(instantiation_size){}
	~MemoryManagerHalloc()
	{
		ha_shutdown();
	}

	virtual void init() override
	{
		ha_init(halloc_opts_t(size));
	}

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		return hamalloc(size);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		hafree(ptr);
	}
};