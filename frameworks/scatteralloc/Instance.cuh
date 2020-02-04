#pragma once

#include "TestInstance.cuh"
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
	explicit MemoryManagerScatterAlloc(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : 
		MemoryManagerBase(instantiation_size), 
		sa{ new ScatterAllocator(instantiation_size)}, sah{sa->getAllocatorHandle()}{}

	~MemoryManagerScatterAlloc(){if(!IAMACOPY) {delete sa;}}
	MemoryManagerScatterAlloc(const MemoryManagerScatterAlloc& src) : sa{src.sa}, sah{src.sah}, IAMACOPY{true} {}

	virtual void init() override
	{
	}

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