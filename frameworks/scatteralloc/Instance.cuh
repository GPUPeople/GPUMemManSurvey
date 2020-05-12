#pragma once

#include "TestInstance.cuh"

// #define HEAPARGS SCATTER_ALLOC_PAGESIZE, SCATTER_ALLOC_ACCESSBLOCKS, SCATTER_ALLOC_REGIONSIZE, SCATTER_ALLOC_WASTEFACTOR, SCATTER_ALLOC_COALESCING, SCATTER_ALLOC_RESETPAGES
// #include "heap_impl.cuh"
// #include "utils.h"

// template __global__ void GPUTools::initHeap<HEAPARGS>(DeviceHeap<HEAPARGS>* heap, void* heapmem, uint memsize);

// struct MemoryManagerScatterAlloc : public MemoryManagerBase
// {
// 	explicit MemoryManagerScatterAlloc(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : 
// 		MemoryManagerBase(instantiation_size)
// 	{
// 		cudaMalloc(&heapmem, instantiation_size);
// 		GPUTools::initHeap<HEAPARGS>(&theHeap, heapmem, instantiation_size);
// 	}
// 	~MemoryManagerScatterAlloc()
// 	{
// 		cudaFree(heapmem);
// 	}
	
// 	virtual __device__ __forceinline__ void* malloc(size_t size) override
// 	{
// 		return theHeap.alloc(size);
// 	}

// 	virtual __device__ __forceinline__ void free(void* ptr) override
// 	{
// 		theHeap.dealloc(ptr);
// 	}
// 	void* heapmem{nullptr};
// };



#include "mallocMC.hpp"
#include "alignmentPolicies/Shrink.hpp"

namespace MC = mallocMC;

using ScatterAllocator = MC::Allocator<
  MC::CreationPolicies::Scatter<>,
  MC::DistributionPolicies::XMallocSIMD<>,
  MC::OOMPolicies::ReturnNull,
  MC::ReservePoolPolicies::SimpleCudaMalloc,
  MC::AlignmentPolicies::Shrink<MC::AlignmentPolicies::ShrinkConfig::DefaultShrinkConfig>
  >;

struct MemoryManagerScatterAlloc : public MemoryManagerBase
{
	explicit MemoryManagerScatterAlloc(size_t instantiation_size) : 
		MemoryManagerBase(instantiation_size), 
		sa{ new ScatterAllocator(instantiation_size)}, sah{sa->getAllocatorHandle()}{}

	~MemoryManagerScatterAlloc(){if(!IAMACOPY) {delete sa;}}
	MemoryManagerScatterAlloc(const MemoryManagerScatterAlloc& src) : sa{src.sa}, sah{src.sah}, IAMACOPY{true} {}

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		return sah.malloc(size);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		sah.free(ptr);
	}

	ScatterAllocator* sa;
	ScatterAllocator::AllocatorHandle sah;
	bool IAMACOPY{false}; // TODO: That is an ugly hack so we don't get a double free when making a copy for the device
};