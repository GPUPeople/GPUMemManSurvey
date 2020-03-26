#pragma once

#include <vector>

#include "device/dynamicGraph.cuh"

/*! \class EdgeUpdateBatch
\brief Templatised class to hold a batch of edge updates
*/
template <typename VertexDataType, typename EdgeDataType>
struct EdgeUpdateBatch
{
	EdgeUpdateBatch(vertex_t num_vertices) : max_bits{2 * (32 - Utils::cntlz(num_vertices)) }{}

    using UpdateType = typename TypeResolution<VertexDataType, EdgeDataType>::EdgeUpdateType;
    using DeviceUpdateType = typename TypeResolution<VertexDataType, EdgeDataType>::DeviceEdgeUpdateType;
    // Host side
    std::vector<UpdateType> edge_update;
    UpdateType* raw_edge_update{nullptr};
    DeviceUpdateType edge_update_cub;
	int max_bits{0};

    // Device side
    CudaUniquePtr<UpdateType> d_edge_update;
    DeviceUpdateType d_edge_update_cub;

    void prepareEdgeUpdates(bool sort = true);

    // Methods
    void generateEdgeUpdates(size_t number_vertices, size_t batch_size, unsigned int seed, unsigned int range = 0, unsigned int offset = 0);
    template <typename MemoryManagerType>
    void generateEdgeUpdates(DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>& dynGraph, size_t batch_size, unsigned int seed, unsigned int range = 0, unsigned int offset = 0);
};