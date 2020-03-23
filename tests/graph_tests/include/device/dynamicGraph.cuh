#pragma once

#include "MemoryLayout.h"

// Forward Declaration
template<typename T>
struct CSR;
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
struct EdgeUpdateBatch;

template <typename VertexDataType, typename EdgeDataType, class MemoryManagerType>
struct DynGraph
{
    // Members
    MemoryManagerType memory_manager;
    cudaEvent_t ce_start, ce_stop;
};