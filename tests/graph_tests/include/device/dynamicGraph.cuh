#pragma once

#include "MemoryLayout.h"
#include "PerformanceMeasure.cuh"

// Forward Declaration
template<typename T>
struct CSR;
template <typename VertexDataType, typename EdgeDataType>
struct EdgeUpdateBatch;

template <typename VertexDataType, typename EdgeDataType, class MemoryManagerType>
struct DynGraph
{
    DynGraph() : memory_manager{2ULL * 1024ULL * 1024ULL * 1024ULL}{}
    DynGraph(size_t allocationSize) : memory_manager{allocationSize}{}
    ~DynGraph();

    // Members
    MemoryManagerType memory_manager;

    VertexDataType* d_vertices{nullptr};
    unsigned int number_vertices;
    bool initialized{false};
    
    // Performance    
    PerfMeasure init_performance;
    PerfMeasure insert_performance;
    PerfMeasure delete_performance;

    // Methods
    template <typename DataType>
    void init(CSR<DataType>& input_graph);

    template <typename DataType>
	void dynGraphToCSR(CSR<DataType>& output_graph);
    
    void edgeInsertion(EdgeUpdateBatch<VertexDataType, EdgeDataType>& update_batch);

    void edgeDeletion(EdgeUpdateBatch<VertexDataType, EdgeDataType>& update_batch);

    void cleanup();
};