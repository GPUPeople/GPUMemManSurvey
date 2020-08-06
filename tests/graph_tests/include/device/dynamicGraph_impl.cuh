#pragma once

#include "UtilityFunctions.cuh"
#include "dCSR.h"
#include "device/dynamicGraph.cuh"

// ##############################################################################################################################################
//
//
// Initialization
//
//
// ##############################################################################################################################################


template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void setupGraph(MemoryManagerType mm, VertexDataType* vertices, int num_vertices,
    const unsigned int* __restrict row_offsets,	const unsigned int* __restrict col_ids)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= num_vertices)
        return;

    auto offset = row_offsets[tid];
    VertexDataType vertex;
    vertex.meta_data.neighbours = row_offsets[tid + 1] - offset;
    vertex.adjacency = reinterpret_cast<EdgeDataType*>(mm.malloc(Helper::AllocationHelper::template getPageSize<unsigned int, minPageSize>(vertex.meta_data.neighbours * sizeof(EdgeDataType))));
    for(auto i = 0; i < vertex.meta_data.neighbours; ++i)
    {
        vertex.adjacency[i].destination = col_ids[offset + i];
    }
    vertices[tid] = vertex;
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
template <typename DataType>
void DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>::init(CSR<DataType>& input_graph)
{
    if(initialized)
        return;
    initialized = true;
    // CSR on device
	dCSR<DataType> d_csr_graph;
    convert(d_csr_graph, input_graph, 0);

    number_vertices = input_graph.rows;
    
	CHECK_ERROR(cudaMalloc(&d_vertices, sizeof(VertexDataType) * number_vertices));
	
    int blockSize {256};
    int gridSize {Utils::divup<int>(number_vertices, blockSize)};
    init_performance.startMeasurement();
    setupGraph<VertexDataType, EdgeDataType, MemoryManagerType> <<<gridSize, blockSize>>>(
        memory_manager, 
        d_vertices, 
        number_vertices,
        d_csr_graph.row_offsets,
        d_csr_graph.col_ids);
	init_performance.stopMeasurement();
	CHECK_ERROR(cudaDeviceSynchronize());
}

// ##############################################################################################################################################
//
//
// DynGraph -> CSR
//
//
// ##############################################################################################################################################
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_getOffsets(VertexDataType* vertices,
                             unsigned int* __restrict offset,
                             unsigned int number_vertices)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= number_vertices)
		return;

    offset[tid] = vertices[tid].meta_data.neighbours;
}

template <typename VertexDataType, typename EdgeDataType>
__global__ void d_dynGraph_To_CSR(VertexDataType* vertices,
                                  unsigned int* __restrict offset,
                                  unsigned int* __restrict adjacency,
                                  unsigned int number_vertices)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= number_vertices)
		return;

	auto adj_offset = offset[tid];
	auto neighbours = vertices[tid].meta_data.neighbours;
	auto adj = vertices[tid].adjacency;

	for(auto i = 0; i < neighbours; ++i)
	{
		adjacency[adj_offset + i] = adj[i].destination;
	}
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
template <typename DataType>
void DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>::dynGraphToCSR(CSR<DataType>& output_graph)
{
    auto block_size = 256;
    int grid_size = (number_vertices / block_size) + 1;

    // Allocate output graph
	dCSR<DataType> d_output_graph;
	d_output_graph.rows = number_vertices;
    d_output_graph.cols = number_vertices;
    CHECK_ERROR(cudaMalloc(&(d_output_graph.row_offsets), sizeof(unsigned int) * (number_vertices + 1)));
    CHECK_ERROR(cudaMemset(d_output_graph.row_offsets, 0, number_vertices + 1));

    d_getOffsets<VertexDataType, EdgeDataType> << <grid_size, block_size >> > (d_vertices, d_output_graph.row_offsets, number_vertices);
    CHECK_ERROR(cudaDeviceSynchronize());

    // Sum up offsets correctly
    Helper::thrustExclusiveSum(d_output_graph.row_offsets, number_vertices + 1);

    auto num_edges{ 0U };
    CHECK_ERROR(cudaMemcpy(&num_edges, d_output_graph.row_offsets + number_vertices, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // Allocate rest
	d_output_graph.nnz = num_edges;
	CHECK_ERROR(cudaMalloc(&(d_output_graph.col_ids), sizeof(unsigned int) * num_edges));
    CHECK_ERROR(cudaMalloc(&(d_output_graph.data), sizeof(DataType) * num_edges));
    
    // Write Adjacency back
	d_dynGraph_To_CSR<VertexDataType, EdgeDataType> << <grid_size, block_size >> > (
        d_vertices, d_output_graph.row_offsets, d_output_graph.col_ids, number_vertices);
    
    CHECK_ERROR(cudaDeviceSynchronize());
    
    convert(output_graph, d_output_graph);
}

// ##############################################################################################################################################
//
//
// Destructor
//
//
// ##############################################################################################################################################
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void freeAdjacencies(MemoryManagerType mm, VertexDataType* vertices, int num_vertices)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= num_vertices)
        return;
    
    const auto adjacency = vertices[tid].adjacency;
    if(adjacency)
        mm.free(adjacency);
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>::cleanup()
{
    freeAdjacencies<VertexDataType, EdgeDataType, MemoryManagerType> <<<Utils::divup(number_vertices, 256), 256>>>(memory_manager, d_vertices, number_vertices);
    CHECK_ERROR(cudaDeviceSynchronize());

    // Free vertices
    if(d_vertices)
    {
        CHECK_ERROR(cudaFree(d_vertices));
        d_vertices = nullptr;
    }

    number_vertices = 0;
    initialized = false;
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>::~DynGraph()
{
    if(initialized)
        cleanup();
}