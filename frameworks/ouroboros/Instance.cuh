#pragma once

#include "TestInstance.cuh"

#include "repository/include/device/Ouroboros_impl.cuh"
#include "repository/include/device/MemoryInitialization.cuh"
#include "repository/include/InstanceDefinitions.cuh"

struct MemoryManagerOuroboros : public MemoryManagerBase
{
	using OuroborosType = OuroPQ;

	explicit MemoryManagerOuroboros(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : MemoryManagerBase(instantiation_size), memory_manager{new OuroborosType()} {}
	~MemoryManagerOuroboros(){if(!IAMACOPY) {delete memory_manager;}}
	MemoryManagerOuroboros(const MemoryManagerOuroboros& src) : memory_manager{src.memory_manager}, d_memory_manager{src.d_memory_manager}, IAMACOPY{true} {}

	virtual void init() override
	{
		memory_manager->initialize();
		d_memory_manager = memory_manager->getDeviceMemoryManager();
	}

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