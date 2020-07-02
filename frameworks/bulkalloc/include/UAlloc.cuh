#pragma once

#include "Mutex.cuh"

template <unsigned int CHUNK_SIZE, unsigned int BIN_SIZE>
struct SMArena
{
	static constexpr unsigned int ChunkSize{CHUNK_SIZE};
	static constexpr unsigned int BinSize{BIN_SIZE};


};

template <unsigned int CHUNK_SIZE>
struct Chunk
{
	static constexpr unsigned int ChunkSize{CHUNK_SIZE};
};

template <unsigned int CHUNK_SIZE>
struct ChunkList
{
	static constexpr unsigned int ChunkSize{CHUNK_SIZE};

	Mutex mutex;
	Chunk<ChunkSize>* chunk_list;
};

template <unsigned int CHUNK_SIZE, unsigned int BIN_SIZE, unsigned int NUM_SMS>
struct UAlloc
{
	static constexpr unsigned int ChunkSize{CHUNK_SIZE};
	static constexpr unsigned int BinSize{BIN_SIZE};
	static constexpr unsigned int ChunkOrder{(ChunkSize / BinSize) - 1};
	static constexpr unsigned int NumSMs{NUM_SMS};

	__device__ __forceinline__ void* malloc(size_t size)
	{
		printf("UAlloc malloc!\n");
		return ::malloc(size);
	}

	__device__ __forceinline__ void free(void* ptr)
	{
		printf("UAlloc free!\n");
		::free(ptr);
	}

	SMArena<ChunkSize, BinSize> arenas[NumSMs];
};