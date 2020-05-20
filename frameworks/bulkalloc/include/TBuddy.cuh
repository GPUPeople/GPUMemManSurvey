#pragma once

#include "Utility.cuh"
#include "BulkSemaphore.cuh"

enum class Node{
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

	void print()
	{
		std::cout
		<< NumLeaves << " | "
		<< NumNodes << std::endl;
	}

	BulkSemaphore per_order_semaphore[NumLevels];
	Node nodes[NumNodes];
};

template <unsigned long long ALLOCATION_SIZE, unsigned int CHUNK_SIZE>
struct TBuddy
{
	static constexpr unsigned long long AllocationSize{ALLOCATION_SIZE};
	static constexpr unsigned int AllocationSizePow{BUtils::static_getNextPow2Pow_l<AllocationSize>()};
	static constexpr unsigned int ChunkSize{CHUNK_SIZE};
	static constexpr unsigned int ChunkSizePow{BUtils::static_getNextPow2Pow<ChunkSize>()};
	static constexpr unsigned int NumLevels{AllocationSizePow - ChunkSizePow + 1};

	void print()
	{
		std::cout
		<< AllocationSize << " | "
		<< AllocationSizePow << " | "
		<< ChunkSize << " | "
		<< ChunkSizePow << " | "
		<< NumLevels << " | " << std::endl;
		availability_nodes.print();
	}

	__device__ __forceinline__ void* malloc(size_t size)
	{
		printf("TBuddy malloc!\n");
		return nullptr;
	}

	__device__ __forceinline__ void free(void* ptr)
	{
		printf("TBuddy free!\n");
	}

	StaticBinaryTree<NumLevels> availability_nodes;
};
