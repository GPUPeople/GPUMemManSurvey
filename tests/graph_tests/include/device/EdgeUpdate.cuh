#pragma once
#include "Definitions.h"
#include "EdgeUpdate.h"
#include "Utility.cuh"
#include "MemoryLayout.h"

#include <type_traits>

// ##############################################################################################################################################
// Counts the number of updates per src index
template <typename UpdateDataType>
__global__ void d_updateInstanceCounter(const UpdateDataType* __restrict edge_update_data,
	index_t* edge_src_counter,
	const int number_vertices,
	const int batch_size)
{
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= batch_size)
		return;

	const auto edge_src = edge_update_data[tid].source;

	if (edge_src >= number_vertices)
		return;

	atomicAdd(&(edge_src_counter[edge_src]), 1);
}

// ##############################################################################################################################################
// Counts the number of updates per src index
template <typename SomeType>
__global__ void d_bringInOrder(const size_t size, const vertex_t* __restrict pos, const vertex_t* __restrict weights, vertex_t* helper_weights)
{
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= size)
		return;
	
	helper_weights[tid] = weights[pos[tid]];
}

inline void EdgeDataUpdateDevice::bringDataInOrder(){}

inline void EdgeDataWeightUpdateDevice::bringDataInOrder()
{
	CudaUniquePtr<vertex_t> d_helper_weights(size);
	auto block_size = 256;
	auto grid_size = divup(size , block_size);
	d_bringInOrder<int><<<grid_size, block_size>>>(size, pos.get(), d_weights.get(), d_helper_weights.get());
	d_weights = std::move(d_helper_weights);
}

struct EdgeUpdatePreProcessing
{
	template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
	void process(EdgeUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>& update_batch, vertex_t number_vertices)
	{
		using UpdateType = typename TypeResolution<VertexDataType, EdgeDataType>::EdgeUpdateType;
		const int batch_size = update_batch.edge_update.size();
		auto block_size = 256;
		int grid_size = divup(batch_size, block_size);

		// Allocate update helper
		d_update_src_helper.allocate((number_vertices + 1) * 2);
		d_update_src_helper.memSet(0, number_vertices + 1);

		d_updateInstanceCounter<UpdateType> << <grid_size, block_size >> > (
			update_batch.d_edge_update.get(),
			d_update_src_helper.get(),
			number_vertices,
			batch_size);

		Helper::thrustExclusiveSum(d_update_src_helper.get(), number_vertices + 1, d_update_src_helper.get() + (number_vertices + 1));
	}

	CudaUniquePtr<index_t> d_update_src_helper; // Holds updates per vertex in first half and offsets based on this in second half
};

template <typename VertexDataType, typename EdgeDataType, typename DeviceUpdateType>
inline void sortUpdates(DeviceUpdateType& edge_update_cub, DeviceUpdateType& d_edge_update_cub, int max_bits)
{
	Helper::cubSortPairs(d_edge_update_cub.d_src_dest.get(), d_edge_update_cub.d_pos.get(), edge_update_cub.size, 0, max_bits);
}

template <>
inline void sortUpdates<VertexData, EdgeData, EdgeDataUpdateDevice>(EdgeDataUpdateDevice& edge_update_cub, EdgeDataUpdateDevice& d_edge_update_cub, int max_bits)
{
	Helper::cubSort(d_edge_update_cub.d_src_dest.get(), edge_update_cub.size, 0, max_bits);
}


template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void EdgeUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>::prepareEdgeUpdates(bool sort)
{
    using UpdateType = typename TypeResolution<VertexDataType, EdgeDataType>::EdgeUpdateType;

	if(THRUST_SORT)
	{
		// THRUST Sort
		d_edge_update.allocate(edge_update.size());
		d_edge_update.copyToDevice(edge_update.data(), edge_update.size());
		if(sort)
			Helper::thrustSort(d_edge_update.get(), edge_update.size());
	}
	else
	{
		// CUB Sort
		d_edge_update_cub.deviceAllocate(edge_update_cub.size);
		d_edge_update_cub.copyData(edge_update_cub);

		if(sort)
			sortUpdates<VertexDataType, EdgeDataType, DeviceUpdateType>(edge_update_cub, d_edge_update_cub, max_bits);
		
		d_edge_update_cub.bringDataInOrder();
	}
}


template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void EdgeUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>::generateEdgeUpdates(size_t number_vertices, size_t batch_size, unsigned int seed, unsigned int range, unsigned int offset)
{
    if(THRUST_SORT)
    {
        edge_update.resize(batch_size);
    }
    else
    {
        edge_update_cub.resize(batch_size);
    }
    
    // Generate random edge updates
    srand(seed + 1);
    
    for(decltype(batch_size) i = 0; i < batch_size; ++i)
    {
        vertex_t src{0};
        vertex_t dest{0};
        vertex_t intermediate = rand() % ((range && (range < number_vertices)) ? range : number_vertices);
        if(offset + intermediate < number_vertices)
            src = offset + intermediate;
        else
            src = intermediate;
        dest = rand() % number_vertices;

        if(THRUST_SORT)
        {
            edge_update[i].source = src;
            edge_update[i].update.destination = dest;
            //printf("Update %u | %u\n", src, dest);
        }
        else
        {
            edge_update_cub.setValue(i, (src << (32 - cntlz(number_vertices))) + dest);
        }        
    }
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void EdgeUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>::generateEdgeUpdates(DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>& dynGraph, size_t batch_size, unsigned int seed, unsigned int range, unsigned int offset)
{
    if(THRUST_SORT)
    {
        edge_update.resize(batch_size);
    }
    else
    {
        edge_update_cub.resize(batch_size);
    }

    // Get current graph
    CSR<float> current_graph;
    dynGraph.dynGraphToCSR(current_graph);

    srand(seed + 1);
    for(decltype(batch_size) i = 0; i < batch_size; ++i)
    {
        vertex_t src{0};
        vertex_t dest{0};
        vertex_t intermediate = rand() % ((range && (range < current_graph.rows)) ? range : current_graph.rows);
        if(offset + intermediate < current_graph.rows)
            src = offset + intermediate;
        else
            src = intermediate;

        auto offset = current_graph.row_offsets[src];
        auto neighbours = current_graph.row_offsets[src] - offset;
        dest = current_graph.col_ids[offset + (rand() % neighbours)];

        if(THRUST_SORT)
        {
            edge_update[i].source = src;
            edge_update[i].update.destination = dest;
        }
        else
        {
            edge_update_cub.setValue(i, (src << (32 - cntlz(current_graph.rows))) + dest);
        }   
    }
}