#pragma once

#include "TBuddy_impl.cuh"
#include "UAlloc_impl.cuh"
#include "Utility.cuh"
#include "BulkSemaphore.cuh"

template <unsigned long long ALLOCATION_SIZE, unsigned int NUM_SMs>
class BulkAllocator
{
public:
	static constexpr unsigned long long AllocationSize{ ALLOCATION_SIZE }; // 
	static constexpr unsigned int ChunkSize{ 512U * 1024U }; // 
	static constexpr unsigned int BinSize{ 4U * 1024U }; // Largest size serviceable by UAlloc is (BinSize / 2), leaf node size of TBuddy
	static constexpr unsigned int BinsPerChunk{ ChunkSize / BinSize };
	static constexpr unsigned int NumSMs{NUM_SMs};

	BulkAllocator(size_t instantiation_size)
	{
		CHECK_ERROR(cudaMalloc(&base, instantiation_size));
		tbuddy.print();
	}

	__device__ __forceinline__ void* malloc(size_t size)
	{
		// First align to next power of 2
		auto aligned_size = BUtils::getNextPow2(size);
		if(aligned_size > (BinSize >> 1))
			return tbuddy.malloc(aligned_size);
		else
			return ualloc.malloc(aligned_size);
	}

	__device__ __forceinline__ void free(void* ptr)
	{
		// Only if it is not aligned to the bin size, then it has to be in UAlloc
		if(BUtils::template isAlignedToSize<BinSize>(base, ptr))
			tbuddy.free(ptr, base);
		else
			ualloc.free(ptr);
	}

private:
	// Members
	TBuddy<AllocationSize, BinSize> tbuddy;
	UAlloc<ChunkSize, BinSize, NumSMs> ualloc;
	memory_t* base;
};