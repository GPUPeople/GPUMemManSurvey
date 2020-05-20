#pragma once

#include "TBuddy.cuh"
#include "UAlloc.cuh"
#include "Utility.cuh"
#include "BulkSemaphore.cuh"

template <unsigned long long ALLOCATION_SIZE, unsigned int CHUNK_SIZE>
class BulkAllocator
{
public:
	static constexpr unsigned long long AllocationSize{ ALLOCATION_SIZE }; // 
	static constexpr unsigned int ChunkSize{ CHUNK_SIZE }; // Largest size serviceable by UAlloc, leaf node size of TBuddy

	BulkAllocator(size_t instantiation_size)
	{
		CHECK_ERROR(cudaMalloc(&base, instantiation_size));
		tbuddy.print();
	}

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
	TBuddy<AllocationSize, ChunkSize> tbuddy;
	UAlloc<ChunkSize> ualloc;
	char* base;
};