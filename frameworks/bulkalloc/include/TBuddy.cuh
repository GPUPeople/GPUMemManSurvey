#pragma once

#include "Utility.cuh"
#include "BulkSemaphore_impl.cuh"
#include "Mutex.cuh"

enum class NodeStatus : int
{
	Busy,
	Partial,
	Available
};

template <unsigned int NUMLEVELS>
struct StaticBinaryTree
{
	static constexpr unsigned int NumLevels{NUMLEVELS};
	static constexpr unsigned int NumLeaves{1 << (NumLevels - 1)};
	static constexpr unsigned int NumNodes{(2 * NumLeaves) - 1};

	__device__ __forceinline__ void initialize();

	__device__ __forceinline__ int accessOrderOffset(int order) { return (1 << order) - 1; }

	__device__ __forceinline__ void* allocate(int order);

	// ###########################################################################################################
	//
	// Members
	//
	BulkSemaphore per_order_semaphore[NumLevels];
	NodeStatus nodes[NumNodes];
	Mutex per_node_locks[NumNodes];

	// ###########################################################################################################
	//
	void print()
	{
		std::cout
		<< NumLeaves << " | "
		<< NumNodes << " | " 
		<< std::endl;
	}
};

template <unsigned long long ALLOCATION_SIZE, unsigned int NODE_SIZE>
struct TBuddy
{
	static constexpr unsigned long long AllocationSize{ALLOCATION_SIZE};
	static constexpr unsigned int AllocationSizePow{BUtils::static_getNextPow2Pow_l<AllocationSize>()};
	static constexpr unsigned int NodeSize{NODE_SIZE};
	static constexpr unsigned int NodeSizePow{BUtils::static_getNextPow2Pow<NodeSize>()};
	static constexpr unsigned int NumLevels{AllocationSizePow - NodeSizePow + 1};

	void print()
	{
		std::cout
		<< AllocationSize << " | "
		<< AllocationSizePow << " | "
		<< NodeSize << " | "
		<< NodeSizePow << " | "
		<< NumLevels << " | " 
		<< std::endl;
		availability_nodes.print();
	}

	__device__ __forceinline__ void* malloc(size_t size);

	__device__ __forceinline__ void free(void* ptr, memory_t* base);

	// ###########################################################################################################
	//
	// Members
	//
	StaticBinaryTree<NumLevels> availability_nodes;
};
