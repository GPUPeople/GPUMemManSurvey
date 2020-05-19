#pragma once

#include "TBuddy.cuh"
#include "UAlloc.cuh"
#include "Utility.cuh"
#include "BulkSemaphore.cuh"

template <unsigned int CHUNK_SIZE>
class BulkAllocator
{
public:
	static constexpr unsigned int ChunkSize{ CHUNK_SIZE }; // Largest size serviceable by UAlloc, leaf node size of TBuddy

	__device__ __forceinline__ void* malloc(size_t size)
	{
		if(size > (CHUNK_SIZE))
			return tbuddy.malloc(size);
		else
			return ualloc.malloc(size);
	}

	__device__ __forceinline__ void free(void* ptr)
	{
		if(BUtils::template isAlignedToSize<ChunkSize>(base, ptr))
			tbuddy.free(ptr);
		else
			ualloc.free(ptr);
	}

private:
	// Members
	TBuddy<ChunkSize> tbuddy;
	UAlloc<ChunkSize> ualloc;
	char* base;
};