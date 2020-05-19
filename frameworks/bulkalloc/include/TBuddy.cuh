#pragma once

template <unsigned int CHUNK_SIZE>
class TBuddy
{
public:
	__device__ __forceinline__ void* malloc(size_t size)
	{
		printf("TBuddy malloc!\n");
		return nullptr;
	}

	__device__ __forceinline__ void free(void* ptr)
	{
		printf("TBuddy free!\n");
	}

private:

};