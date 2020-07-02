#pragma once

# include "TBuddy.cuh"

// ###########################################################################################################
//
template <unsigned int NUMLEVELS>
__device__ __forceinline__ void StaticBinaryTree<NUMLEVELS>::initialize()
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NumNodes; i += blockDim.x * gridDim.x)
	{
		if(i == 0)
		{
			nodes[0] = NodeStatus::Available;
			per_order_semaphore[0].signal(1);
		}
		else
		{
			nodes[i] = NodeStatus::Busy;
		}
	}
}


// ###########################################################################################################
//
template <unsigned int NUMLEVELS>
__device__ __forceinline__ void* StaticBinaryTree<NUMLEVELS>::allocate(int order)
{
	// Should return
	per_order_semaphore[order].wait(1, 1, [&]()
	{

	});

	// There currently is a page free of this size

	return nullptr;
}

// ###########################################################################################################
//
template <unsigned long long ALLOCATION_SIZE, unsigned int NODE_SIZE>
__device__ __forceinline__ void* TBuddy<ALLOCATION_SIZE, NODE_SIZE>::malloc(size_t size)
{
	// The smallest size we get in here is NodeSize
	int order = size >> NodeSizePow;
	auto ret_ptr = availability_nodes.allocate(order);
	// return ret_ptr;
	return ::malloc(size);
}

// ###########################################################################################################
//
template <unsigned long long ALLOCATION_SIZE, unsigned int NODE_SIZE>
__device__ __forceinline__ void TBuddy<ALLOCATION_SIZE, NODE_SIZE>::free(void* ptr, memory_t* base)
{
	printf("TBuddy free!\n");
	::free(ptr);
}
