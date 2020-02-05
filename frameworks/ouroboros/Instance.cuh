#pragma once

#include "TestInstance.cuh"

#include "repository/include/device/Ouroboros_impl.cuh"
#include "repository/include/device/MemoryInitialization.cuh"
#include "repository/include/InstanceDefinitions.cuh"

struct MemoryManagerOuroboros : public MemoryManagerBase
{
	explicit MemoryManagerOuroboros(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : MemoryManagerBase(instantiation_size), memory_manager{new OuroPQ()} {}
	~MemoryManagerOuroboros(){if(!IAMACOPY) {delete memory_manager;}}
	MemoryManagerOuroboros(const MemoryManagerOuroboros& src) : memory_manager{src.memory_manager}, IAMACOPY{true} {}

	virtual void init() override
	{
		memory_manager->initialize();
		d_memory_manager = reinterpret_cast<OuroPQ*>(memory_manager->memory.d_memory);
	}

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		MemoryIndex new_index;
		return d_memory_manager->template allocPage<int>(size / sizeof(int), new_index);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
	}

	OuroPQ* memory_manager;
	OuroPQ* d_memory_manager{nullptr};
	bool IAMACOPY{false}; // TODO: That is an ugly hack so we don't get a double free when making a copy for the device
};