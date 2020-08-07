#pragma once
#include "device/EdgeUpdate.cuh"
#include "device/dynamicGraph_impl.cuh"

//------------------------------------------------------------------------------
//
template <typename UpdateDataType>
__forceinline__ __device__ bool d_binarySearchDeletion(UpdateDataType* edge_update_data, index_t search_element,
	index_t start_index, index_t number_updates)
{
	int lower_bound = start_index;
	int upper_bound = start_index + (number_updates - 1);
	index_t search_index;
	while (lower_bound <= upper_bound)
	{
		search_index = lower_bound + ((upper_bound - lower_bound) / 2);
		index_t update = edge_update_data[search_index].update.destination;

		// First check if we get a hit
		if (update == search_element)
		{
			// We have a duplicate
			return true;
		}
		if (update < search_element)
		{
			lower_bound = search_index + 1;
		}
		else
		{
			upper_bound = search_index - 1;
		}
	}
	return false;
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, bool VECTORIZED_COPY>
__global__ void d_edgeDeletionVertexCentric(VertexDataType* vertices, 
                                             unsigned int number_vertices,
                                             MemoryManagerType memory_manager,
                                             const typename TypeResolution<VertexDataType, EdgeDataType>::EdgeUpdateType* __restrict edge_update_data,
                                             const int batch_size,
                                             const index_t* __restrict update_src_offsets)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= number_vertices)
        return;
    
    // Early-Out for no updates for this vertexs
	const auto number_updates = update_src_offsets[tid];
	if (number_updates == 0)
		return;
	
    VertexDataType vertex = vertices[tid];
    EdgeDataType* end_iterator{vertex.adjacency + (vertex.meta_data.neighbours)}; // Point to one behind end
    const auto index_offset = update_src_offsets[(number_vertices + 1) + tid];
	auto actual_updates{ 0U };
	
	auto adjacency{vertex.adjacency};
    while(actual_updates != number_updates && end_iterator != adjacency)
	{
		// Get current destination
		auto dest = adjacency->destination;

		// Try to locate edge in updates
		if (d_binarySearchDeletion(edge_update_data, dest, index_offset, number_updates))
		{
			// Move compaction iterator forward
			--end_iterator;

			// This element can been deleted
			++actual_updates;

			// Do Compaction // TODO: If the end needs to be cleared, use this instead
			// vertex.adjacency->destination = end_iterator->destination;
			// end_iterator->destination = DELETIONMARKER;
			adjacency->destination = end_iterator != adjacency ? end_iterator->destination : DELETIONMARKER;
		}
		else
			++(adjacency);
	}
    
    const auto page_size = Helper::AllocationHelper::template getPageSize<unsigned int, minPageSize>(vertex.meta_data.neighbours * sizeof(EdgeDataType));
    vertex.meta_data.neighbours -= actual_updates;
    const auto new_page_size = Helper::AllocationHelper::template getPageSize<unsigned int, minPageSize>(vertex.meta_data.neighbours * sizeof(EdgeDataType));
	
	if(new_page_size != page_size)
    {
        auto adjacency = reinterpret_cast<EdgeDataType*>(memory_manager.malloc(new_page_size));
        if(adjacency == nullptr)
		{
			printf("Could not allocate Page for Vertex %u for Size %u!\n", tid, new_page_size);
			return;
		}
		
		if(VECTORIZED_COPY)
		{
			// Copy over data vectorized
			auto iterations = Utils::divup(new_page_size, sizeof(uint4));
			for (auto i = 0U; i < iterations; ++i)
			{
				reinterpret_cast<uint4*>(adjacency)[i] = reinterpret_cast<uint4*>(vertex.adjacency)[i];
			}
		}
		else
		{
			for (auto i = 0U; i < vertex.meta_data.neighbours; ++i)
			{
				adjacency[i] = vertex.adjacency[i];
			}
		}

        // Free old page and set new pointer and index
		memory_manager.free(vertex.adjacency);
		vertices[tid].adjacency = adjacency;
    }
	// Update neighbours
	vertices[tid].meta_data.neighbours = vertex.meta_data.neighbours;
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>::edgeDeletion(EdgeUpdateBatch<VertexDataType, EdgeDataType>& update_batch)
{
	using UpdateType = typename TypeResolution<VertexDataType, EdgeDataType>::EdgeUpdateType;
    
    const int batch_size = update_batch.edge_update.size();
	auto block_size = 256;
	int grid_size = Utils::divup(number_vertices, block_size);

	// Copy update data to device and sort
	update_batch.prepareEdgeUpdates(true);
	
    // #######################################################################################
	// Preprocessing
	EdgeUpdatePreProcessing pre_processing;
	pre_processing.process<VertexDataType, EdgeDataType>(update_batch, number_vertices);
	
    // #######################################################################################
	// Deletion
	delete_performance.startMeasurement();
	d_edgeDeletionVertexCentric<VertexDataType, EdgeDataType, MemoryManagerType, true> << <grid_size, block_size >> > (
		d_vertices,
        number_vertices,
        memory_manager,
		update_batch.d_edge_update.get(),
		batch_size,
		pre_processing.d_update_src_helper.get());
		delete_performance.stopMeasurement();
		CHECK_ERROR(cudaDeviceSynchronize());
}