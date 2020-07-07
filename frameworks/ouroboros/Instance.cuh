#pragma once

#include "TestInstance.cuh"

#include "repository/include/device/Ouroboros_impl.cuh"
#include "repository/include/device/MemoryInitialization.cuh"
#include "repository/include/InstanceDefinitions.cuh"

template <typename OuroborosType=OuroPQ>
struct MemoryManagerOuroboros : public MemoryManagerBase
{
	explicit MemoryManagerOuroboros(size_t instantiation_size) : MemoryManagerBase(instantiation_size), memory_manager{new OuroborosType()}
	{
		memory_manager->initialize(instantiation_size);
		d_memory_manager = memory_manager->getDeviceMemoryManager();
	}
	~MemoryManagerOuroboros(){if(!IAMACOPY) {delete memory_manager;}}
	MemoryManagerOuroboros(const MemoryManagerOuroboros& src) : memory_manager{src.memory_manager}, d_memory_manager{src.d_memory_manager}, IAMACOPY{true} {}

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		return d_memory_manager->malloc(size);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		d_memory_manager->free(ptr);
	}

	OuroborosType* memory_manager{nullptr};
	OuroborosType* d_memory_manager{nullptr};
	bool IAMACOPY{false}; // TODO: That is an ugly hack so we don't get a double free when making a copy for the device
};