#pragma once

#include "TestInstance.cuh"
#include "FDGMalloc_def.cuh"

struct MemoryManagerFDG : public MemoryManagerBase
{
	explicit MemoryManagerFDG(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : MemoryManagerBase(instantiation_size) {}
	~MemoryManagerFDG(){}

	virtual void init() override
	{
	}

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		return nullptr;
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		return;
	}

};