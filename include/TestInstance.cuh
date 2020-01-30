#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

struct MemoryManagerBase
{
	explicit MemoryManagerBase(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : size{instantiation_size}{}
	virtual ~MemoryManagerBase(){};

	virtual void init() = 0;

	virtual __device__ __forceinline__ void* malloc(size_t size) = 0;
	virtual __device__ __forceinline__ void free(void* ptr) = 0;

	size_t size;
};