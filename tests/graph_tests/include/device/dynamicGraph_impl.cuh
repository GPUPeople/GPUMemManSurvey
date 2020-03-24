#pragma once

#include "UtilityFunctions.cuh"
#include "dCSR.h"
#include "device/dynamicGraph.cuh"

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void setupGraph(MemoryManagerType mm, VertexDataType* vertices, int num_vertices,
    const unsigned int* __restrict row_offsets,	const unsigned int* __restrict col_ids)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= num_vertices)
        return;

    auto offset = row_offsets[tid];
    auto neighbours = row_offsets[tid + 1] - offset;
    VertexDataType vertex;
    vertex.meta_data.neighbours = neighbours;
    vertex.adjacency = reinterpret_cast<EdgeDataType*>(mm.malloc(sizeof(EdgeDataType) * neighbours));
    for(auto i = 0; i < neighbours; ++i)
    {
        vertex.adjacency[i].destination = col_ids[offset + i];
    }
    vertices[tid] = vertex;
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
template <typename DataType>
void DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>::init(CSR<DataType>& input_graph)
{
    // CSR on device
	dCSR<DataType> d_csr_graph;
    convert(d_csr_graph, input_graph, 0);
    
    memory_manager.init();

    CHECK_ERROR(cudaMalloc(&d_vertices, sizeof(VertexDataType) * d_csr_graph.rows));

    int blockSize {256};
    int gridSize {Utils::divup<int>(d_csr_graph.rows, blockSize)};
    init_performance.startMeasurement();
    setupGraph<VertexDataType, EdgeDataType, MemoryManagerType> <<<gridSize, blockSize>>>(
        memory_manager, 
        d_vertices, 
        d_csr_graph.rows,
        d_csr_graph.row_offsets,
        d_csr_graph.col_ids);
    init_performance.stopMeasurement();
}