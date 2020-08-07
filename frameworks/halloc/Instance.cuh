#pragma once

#include "TestInstance.cuh"
#include "repository/src/halloc.cu"

struct MemoryManagerHalloc : public MemoryManagerBase
{
	explicit MemoryManagerHalloc(size_t instantiation_size) : MemoryManagerBase(instantiation_size)
	{
		ha_init(halloc_opts_t(instantiation_size));
	}

	MemoryManagerHalloc(const MemoryManagerHalloc& src) : IAMACOPY{true} {}

	~MemoryManagerHalloc()
	{
		if(!IAMACOPY)
			ha_shutdown();
	}

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		return hamalloc(size);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		hafree(ptr);
	}

	bool IAMACOPY{false}; // TODO: That is an ugly hack so we don't get a double free when making a copy for the device
};