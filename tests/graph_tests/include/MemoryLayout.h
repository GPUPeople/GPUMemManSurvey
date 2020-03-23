#pragma once

#include "Definitions.h"
#include "device/CudaUniquePtr.cuh"

//------------------------------------------------------------------------------
// EdgeData Variants for simple, with weight or semantic graphs AOS
//------------------------------------------------------------------------------
//

struct EdgeDataUpdate;

struct EdgeData
{
  vertex_t destination;
  friend __host__ __device__ bool operator<(const EdgeData &lhs, const EdgeData &rhs) { return (lhs.destination < rhs.destination); }
  static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t); }
};

struct EdgeDataWeight
{
  vertex_t destination;
  vertex_t weight;
  friend __host__ __device__ bool operator<(const EdgeDataWeight &lhs, const EdgeDataWeight &rhs) { return (lhs.destination < rhs.destination); };
  static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t) + sizeof(vertex_t); }
};


//------------------------------------------------------------------------------
// EdgeData Variants for simple, with weight or semantic graphs SOA
//------------------------------------------------------------------------------
//

struct EdgeDataSOA
{
	vertex_t destination;
	static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t); }
};

struct EdgeDataWeightSOA
{
	vertex_t destination;
	static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t) + sizeof(vertex_t); }
};

//------------------------------------------------------------------------------
// EdgeUpdate Variants for simple, with weight or semantic graphs
//------------------------------------------------------------------------------
//

struct EdgeDataUpdate
{
	vertex_t source;
	EdgeData update;
  friend __host__ __device__ bool operator<(const EdgeDataUpdate &lhs, const EdgeDataUpdate &rhs) 
  { 
    return ((lhs.source < rhs.source) || (lhs.source == rhs.source && (lhs.update < rhs.update))); 
  }
};

struct EdgeDataUpdateDevice
{
	std::unique_ptr<uint64_t[]> src_dest;
	CudaUniquePtr<uint64_t> d_src_dest;
	int size{0};

	template <typename CountType>
	void resize(CountType batch_size)
	{
		size = batch_size;
		src_dest = std::make_unique<uint64_t[]>(size);
	}

	void setValue(int index, uint64_t new_src_dest)
	{
		src_dest[index] = new_src_dest;
	}

	template <typename CountType>
	void deviceAllocate(CountType batch_size)
	{
		size = batch_size;
		d_src_dest.allocate(size);
	}

	void copyData(EdgeDataUpdateDevice& edge_update)
	{
		d_src_dest.copyToDevice(edge_update.src_dest.get(), size);
	}

	void bringDataInOrder();
};


struct EdgeDataWeightUpdate
{
	vertex_t source;
	EdgeDataWeight update;
  friend __host__ __device__ bool operator<(const EdgeDataWeightUpdate &lhs, const EdgeDataWeightUpdate &rhs) 
  { 
    return ((lhs.source < rhs.source) || (lhs.source == rhs.source && (lhs.update < rhs.update))); 
  };
};

struct EdgeDataWeightUpdateDevice
{
	std::unique_ptr<uint64_t[]> src_dest;
	std::unique_ptr<uint32_t[]> pos;
	std::unique_ptr<vertex_t[]> weights;
	CudaUniquePtr<uint64_t> d_src_dest;
	CudaUniquePtr<uint32_t> d_pos;
	CudaUniquePtr<vertex_t> d_weights;
	int size{0};

	template <typename CountType>
	void resize(CountType batch_size)
	{
		size = batch_size;
		src_dest = std::make_unique<uint64_t[]>(size);
		pos = std::make_unique<uint32_t[]>(size);
		weights = std::make_unique<vertex_t[]>(size);
	}

	void setValue(int index, uint64_t new_src_dest)
	{
		src_dest[index] = new_src_dest;
		pos[index] = index;
	}

	template <typename CountType>
	void deviceAllocate(CountType batch_size)
	{
		size = batch_size;
		d_src_dest.allocate(size);
		d_pos.allocate(size);
		d_weights.allocate(size);
	}

	void copyData(EdgeDataWeightUpdateDevice& edge_update)
	{
		d_src_dest.copyToDevice(edge_update.src_dest.get(), size);
		d_pos.copyToDevice(edge_update.pos.get(), size);
		d_weights.copyToDevice(edge_update.weights.get(), size);
	}

	void bringDataInOrder();
};


//------------------------------------------------------------------------------
// VertexData Variants for simple, with weight or semantic graphs
//------------------------------------------------------------------------------
//

struct alignas(16) VertexMetaData
{
	unsigned int locking;
	unsigned int neighbours;
	unsigned int host_identifier;
};

struct alignas(16) VertexMetaDataWeight : uint4
{
	unsigned int locking;
	unsigned int neighbours;
	unsigned int host_identifier;
	unsigned int weight;
};

typedef struct VertexData
{
	VertexMetaData meta_data;
	EdgeData* adjacency;
}VertexData;

typedef struct VertexDataWeight
{
	VertexMetaDataWeight meta_data;
	EdgeDataWeight* adjacency;
}VertexDataWeight;

//------------------------------------------------------------------------------
// VertexUpdate Variants for simple, with weight or semantic graphs
//------------------------------------------------------------------------------
//

typedef struct VertexUpdate
{
  index_t identifier;
}VertexUpdate;

__forceinline__ __host__ __device__ bool operator<(const VertexUpdate &lhs, const VertexUpdate &rhs) { return (lhs.identifier < rhs.identifier); };

typedef struct VertexUpdateWeight
{
  index_t identifier;
  vertex_t weight;
}VertexUpdateWeight;

__forceinline__ __host__ __device__ bool operator<(const VertexUpdateWeight &lhs, const VertexUpdateWeight &rhs) { return (lhs.identifier < rhs.identifier); };

typedef struct VertexUpdateSemantic
{
  index_t identifier;
  vertex_t weight;
  vertex_t type;
}VertexUpdateSemantic;

__forceinline__ __host__ __device__ bool operator<(const VertexUpdateSemantic &lhs, const VertexUpdateSemantic &rhs) { return (lhs.identifier < rhs.identifier); };


//------------------------------------------------------------------------------
// CSR Structure for matrices
//------------------------------------------------------------------------------
//
class CSRMatrix
{
public:
  OffsetList_t offset;
  AdjacencyList_t adjacency;
  MatrixList_t matrix_values;
};


template <typename VertexDataType, typename EdgeDataType>
struct TypeResolution;

template <>
struct TypeResolution<VertexData, EdgeData>
{
	using VertexUpdateType = VertexUpdate;
	using EdgeUpdateType = EdgeDataUpdate;
	using DeviceEdgeUpdateType = EdgeDataUpdateDevice;

	template<int page_size>
	__forceinline__ __device__ static constexpr vertex_t getEdgesPerBlock()
	{
		return (page_size - sizeof(index_t)) / sizeof(EdgeData);
	}
};

template <>
struct TypeResolution<VertexData, EdgeDataSOA>
{
	using VertexUpdateType = VertexUpdate;
	using EdgeUpdateType = EdgeDataUpdate;
	using DeviceEdgeUpdateType = EdgeDataUpdateDevice;

	template<int page_size>
	__forceinline__ __device__ static constexpr vertex_t getEdgesPerBlock()
	{
		return (page_size - sizeof(index_t)) / sizeof(EdgeData);
	}
};

template <>
struct TypeResolution<VertexDataWeight, EdgeDataWeight>
{
	using VertexUpdateType = VertexUpdateWeight;
	using EdgeUpdateType = EdgeDataWeightUpdate;
	using DeviceEdgeUpdateType = EdgeDataWeightUpdateDevice;
	static constexpr int number_of_indices{ 1 };

	
	template<int page_size>
	__forceinline__ __device__ static constexpr vertex_t getEdgesPerBlock()
	{
		return (page_size - sizeof(index_t)) / sizeof(EdgeDataWeight);
	}
};

template <>
struct TypeResolution<VertexDataWeight, EdgeDataWeightSOA>
{
	using VertexUpdateType = VertexUpdateWeight;
	using EdgeUpdateType = EdgeDataWeightUpdate;
	using DeviceEdgeUpdateType = EdgeDataWeightUpdateDevice;

	template<int page_size>
	__forceinline__ __device__ static constexpr vertex_t getEdgesPerBlock()
	{
		return (page_size - sizeof(index_t)) / sizeof(EdgeDataWeight);
	}
};
