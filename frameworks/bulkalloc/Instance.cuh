#pragma once
#include "TestInstance.cuh"
#include "include/BulkAllocator.cuh"

struct MemoryManagerBulkAlloc : public MemoryManagerBase
{
	static constexpr unsigned int ChunkSize{8192U};
	explicit MemoryManagerBulkAlloc(size_t instantiation_size) : MemoryManagerBase(instantiation_size), allocator(new BulkAllocator<ChunkSize>())
	{
		if(initialized)
			return;
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
		initialized = true;
	}
	~MemoryManagerBulkAlloc(){if(!IAMACOPY) {delete allocator;}}
	MemoryManagerBulkAlloc(const MemoryManagerBulkAlloc& src) : allocator{src.allocator}, IAMACOPY{true} {}

	static constexpr size_t alignment{16ULL};

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		return allocator->malloc(size);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		allocator->free(ptr);
	};

	static bool initialized;
	BulkAllocator<ChunkSize>* allocator{nullptr};
	bool IAMACOPY{false}; // TODO: That is an ugly hack so we don't get a double free when making a copy for the device
};

bool MemoryManagerBulkAlloc::initialized = false;