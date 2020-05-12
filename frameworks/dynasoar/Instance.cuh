#pragma once
#include "TestInstance.cuh"
#include "UtilityFunctions.cuh"

// No Debug-messages with this flag
#define NDEBUG
#include "dynasoar.h"

// Pre-Declare all classes
class AllocationElement;

// Declare allocator type. First argument is max. number of objects that can be created.
using AllocatorT = SoaAllocator<512*1024*1024, AllocationElement>;

// Allocator handles.
__device__ AllocatorT* device_allocator;

class AllocationElement : public AllocatorT::Base 
{
public:
	// Pre-declare types of all fields.
	declare_field_types(AllocationElement, int)

	 // Declare fields.
	 SoaField<AllocationElement, 0> field1_;  // int

	 __device__ AllocationElement(int f1) : field1_(f1) {}
	 __device__ AllocationElement() : field1_(0) {}
};

struct MemoryManagerDynaSOAr : public MemoryManagerBase
{
	explicit MemoryManagerDynaSOAr(size_t instantiation_size) : MemoryManagerBase(instantiation_size)
	{
		// Create new allocator.
		allocator_handle = new AllocatorHandle<AllocatorT>();
		AllocatorT* dev_ptr = allocator_handle->device_pointer();
		cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
						   cudaMemcpyHostToDevice);
	}
	~MemoryManagerDynaSOAr(){if(!IAMACOPY) {delete allocator_handle;}};
	MemoryManagerDynaSOAr(const MemoryManagerDynaSOAr& src) : allocator_handle{src.allocator_handle}, IAMACOPY{true} {}

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		AllocationElement* elements = new (device_allocator) AllocationElement[Utils::divup(size, sizeof(int))];
		for(auto i = 0; i < Utils::divup(size, sizeof(int)); ++i)
			printf("%d - sizeof: %llu - Ptr: %p - diff to prev: %llu\n", threadIdx.x, sizeof(AllocationElement), &elements[i], &elements[i] - &elements[0]);
		return reinterpret_cast<void*>(elements);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		destroy(device_allocator, reinterpret_cast<AllocationElement*>(ptr));
	};

	AllocatorHandle<AllocatorT>* allocator_handle;
	bool IAMACOPY{false}; // TODO: That is an ugly hack so we don't get a double free when making a copy for the device
};
