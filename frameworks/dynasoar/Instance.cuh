#pragma once
#include "TestInstance.cuh"
#include "dynasoar.h"

// Pre-Declare all classes
class AllocationElement;

// Declare allocator type. First argument is max. number of objects that can be created.
using AllocatorT = SoaAllocator<512*1024*1024, AllocationElement>;

// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;


class AllocationElement : public AllocatorT::Base 
{
public:
	// Pre-declare types of all fields.
	declare_field_types(AllocationElement, int)

	 // Declare fields.
	 SoaField<AllocationElement, 0> field1_;  // int

	 __device__ AllocationElement(int f1) : field1_(f1) {}
};

struct MemoryManagerDynaSOAr : public MemoryManagerBase
{
	explicit MemoryManagerDynaSOAr(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : MemoryManagerBase(instantiation_size)
	{
		// Create new allocator.
		allocator_handle = new AllocatorHandle<AllocatorT>();
		AllocatorT* dev_ptr = allocator_handle->device_pointer();
		cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
						   cudaMemcpyHostToDevice);
	}
	~MemoryManagerDynaSOAr(){};

	static constexpr size_t alignment{16ULL};

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		return reinterpret_cast<void*>(new(device_allocator) AllocationElement(0));
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		destroy(device_allocator, reinterpret_cast<AllocationElement*>(ptr));
	};

};
