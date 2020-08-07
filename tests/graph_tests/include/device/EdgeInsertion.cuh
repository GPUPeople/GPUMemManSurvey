#pragma once

#include "device/EdgeUpdate.cuh"
#include "device/dynamicGraph_impl.cuh"

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, bool VECTORIZED_COPY>
__global__ void d_edgeInsertionVertexCentric(VertexDataType* vertices, 
                                             unsigned int number_vertices,
                                             MemoryManagerType memory_manager,
                                             const typename TypeResolution<VertexDataType, EdgeDataType>::EdgeUpdateType* __restrict edge_update_data,
                                             const int batch_size,
                                             const index_t* __restrict update_src_offsets)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= number_vertices)
		return;

	const auto number_updates = update_src_offsets[tid];
	if (number_updates == 0)
		return;

	const auto index_offset = update_src_offsets[(number_vertices + 1) + tid];
	const auto range_updates = update_src_offsets[(number_vertices + 1) + tid + 1] - index_offset;

	// Do insertion here
	VertexDataType vertex = vertices[tid];
	const auto page_size = Helper::AllocationHelper::template getPageSize<unsigned int, minPageSize>(vertex.meta_data.neighbours * sizeof(EdgeDataType));
	const auto new_page_size = Helper::AllocationHelper::template getPageSize<unsigned int, minPageSize>((vertex.meta_data.neighbours + number_updates) * sizeof(EdgeDataType));
	if (page_size != new_page_size)
	{
		// Have to reallocate here
        auto adjacency = reinterpret_cast<EdgeDataType*>(memory_manager.malloc(new_page_size));
		if(adjacency == nullptr)
		{
			printf("Could not allocate Page for Vertex %u for Size %u!\n", tid, new_page_size);
			return;
		}

		if(VECTORIZED_COPY)
		{
			// Copy over data vectorized
			auto iterations = Utils::divup(vertex.meta_data.neighbours * sizeof(EdgeDataType), sizeof(uint4));
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
		

		// Copy over data vectorized
		auto iterations = Utils::divup(vertex.meta_data.neighbours * sizeof(EdgeDataType), sizeof(unsigned int));
		

		// Do insertion now
		for (auto i = 0U, j = vertex.meta_data.neighbours; i < range_updates; ++i)
		{
			if (edge_update_data[index_offset + i].update.destination != DELETIONMARKER)
			{
				adjacency[j++].destination = edge_update_data[index_offset + i].update.destination;
			}
		}

		// Free old page and set new pointer and index
        memory_manager.free(vertex.adjacency);
		vertices[tid].adjacency = adjacency;
	}
	else
	{
		// Do insertion now
		for (auto i = 0U, j = vertex.meta_data.neighbours; i < range_updates; ++i)
		{
			if (edge_update_data[index_offset + i].update.destination != DELETIONMARKER)
			{
				vertex.adjacency[j++].destination = edge_update_data[index_offset + i].update.destination;
			}
		}
	}

	// Update number of neighbours
	vertices[tid].meta_data.neighbours += number_updates;
}

// ##############################################################################################################################################
// Duplicate checking in Graph
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
__global__ void d_duplicateCheckingInSortedBatch2Graph(VertexDataType* vertices,
                                                       UpdateDataType* edge_update_data,
                                                       int batch_size,
                                                       index_t* edge_src_counter)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= batch_size)
		return;

	auto edge_update = edge_update_data[tid];
	auto vertex = vertices[edge_update.source];
	for (auto i = 0; i < vertex.meta_data.neighbours; ++i)
	{
		if (vertex.adjacency[i].destination == edge_update.update.destination)
		{
			if(updateValues)
			{
				// Update with new values
				vertex.adjacency[i] = edge_update.update;
				edge_update_data[tid].update.destination = DELETIONMARKER;
			}
			else
			{
				edge_update_data[tid].update.destination = DELETIONMARKER;
			}
			
			atomicSub(&edge_src_counter[edge_update.source], 1);
			return;
		}
	}
}

// ##############################################################################################################################################
// Check duplicated updates in sorted batch
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
__global__ void d_duplicateCheckingInSortedBatch(UpdateDataType* edge_update_data,
                                                 int batch_size,
                                                 index_t* edge_src_counter)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= batch_size)
		return;

	UpdateDataType edge_update = edge_update_data[tid];
	const auto number_updates = edge_src_counter[edge_update.source];
	auto firstElementPerVertex = tid == 0;
	if(!firstElementPerVertex && edge_update.source != edge_update_data[tid - 1].source)
		firstElementPerVertex = true;

	if(firstElementPerVertex)
	{
		for(auto i = 0; i < number_updates - 1; ++i)
		{
			if(edge_update.update.destination == edge_update_data[++tid].update.destination)
			{
				edge_update_data[tid].update.destination = DELETIONMARKER;
				--edge_src_counter[edge_update.source];
			}
			else
			{
				// Look at the next update
				edge_update = edge_update_data[tid];
			}
			
		}
	}
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>::edgeInsertion(EdgeUpdateBatch<VertexDataType, EdgeDataType>& update_batch)
{
    using UpdateType = typename TypeResolution<VertexDataType, EdgeDataType>::EdgeUpdateType;
    
    int batch_size = update_batch.edge_update.size();

    // Copy update data to device and sort
	update_batch.prepareEdgeUpdates(true);
	
    // #######################################################################################
	// Preprocessing
	EdgeUpdatePreProcessing pre_processing;
	pre_processing.template process<VertexDataType, EdgeDataType>(update_batch, number_vertices);

    // #######################################################################################
	// Duplicate checking
	auto block_size = 256;
    auto grid_size = Utils::divup(batch_size, block_size);
    
    // #######################################################################################
	// Duplicate checking in batch
	d_duplicateCheckingInSortedBatch<VertexDataType, EdgeDataType, UpdateType> << < grid_size, block_size >> >(
		update_batch.d_edge_update.get(),
		batch_size,
		pre_processing.d_update_src_helper.get());
	
	CHECK_ERROR(cudaDeviceSynchronize());
    
    // #######################################################################################
	// Duplicate checking in Graph
	d_duplicateCheckingInSortedBatch2Graph<VertexDataType, EdgeDataType, UpdateType> << < grid_size, block_size >> > (
		d_vertices,
		update_batch.d_edge_update.get(),
		batch_size,
		pre_processing.d_update_src_helper.get());
		
	CHECK_ERROR(cudaDeviceSynchronize());
    
    // #######################################################################################
	// Insertion
	grid_size = Utils::divup(number_vertices, block_size);
	insert_performance.startMeasurement();
	d_edgeInsertionVertexCentric<VertexDataType, EdgeDataType, MemoryManagerType, true> << <grid_size, block_size >> >(
        d_vertices,
        number_vertices,
        memory_manager,
		update_batch.d_edge_update.get(),
		batch_size,
		pre_processing.d_update_src_helper.get());
	insert_performance.stopMeasurement();
}