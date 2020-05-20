#pragma once

template <unsigned int CHUNK_SIZE>
class UAlloc
{
public:
	__device__ __forceinline__ void* malloc(size_t size)
	{
		printf("UAlloc malloc!\n");
		return ::malloc(size);
	}

	__device__ __forceinline__ void free(void* ptr)
	{
		printf("UAlloc free!\n");
		::free(ptr);
	}

private:

};