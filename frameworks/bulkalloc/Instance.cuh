#pragma once
#include "TestInstance.cuh"
#include "include/BulkAllocator.cuh"

struct MemoryManagerBulkAlloc : public MemoryManagerBase
{
	static constexpr unsigned long long AllocationSize{8192ULL * 1024ULL * 1024ULL};
	static constexpr unsigned int ChunkSize{512 * 1024}; // 512 kib
	explicit MemoryManagerBulkAlloc(size_t instantiation_size) : MemoryManagerBase(instantiation_size), allocator(new BulkAllocator<AllocationSize, ChunkSize>(instantiation_size))
	{
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024ULL * 1024ULL * 1024ULL);
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

	BulkAllocator<AllocationSize, ChunkSize>* allocator{nullptr};
	bool IAMACOPY{false}; // TODO: That is an ugly hack so we don't get a double free when making a copy for the device
};
