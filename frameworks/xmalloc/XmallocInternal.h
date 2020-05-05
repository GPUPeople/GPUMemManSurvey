/* Copyright (c) 2010 The Board of Trustees of the University of Illinois.
 * All rights reserved.
 */

#include <stdint.h>

///////////////////////////////////////////////////////////////////////////////
// Data types

// A pointer to a heap address with at least 8-byte alignment,
// packed into 32 bits.
typedef uint32_t PackedHeapPtr;

#define PackedHeapNULL ((uint32_t)0)
#define PackedHeapFAIL (~(uint32_t)0)

///////////////////////////////////////////////////////////////////////////////
// Constants

// Log2 of the size of the smallest basic block in the system
#define START_LEVEL_SHIFT 9

// Number of heap object size classes
#define MemoryLevels 5

// Value indicating that a memory block has no size class
#define SIZE_CLASS_NOTHING (~0U)

// Number of BasicBlocks per SuperBlock
#define BasicBlockNumber 16

// Initial capacity of each free list
#define LOG2_FREELIST_SIZE 8

// Number of free lists (1, 2, or 4)
#define FREELIST_COUNT 4

// Pick a free list
#define WHICHFL \
  ((threadIdx.x * 829 + blockIdx.x + clock()) & (FREELIST_COUNT - 1))

// The maximum memory request size that will be coalesced
#define MAX_COALESCE_SIZE 256

// if the thread cannot get resource, the maximum loop it can wait for
#define MAXI_LOOP  10000000	/* 10 million */

// A memory block type.  Must be in the range [0..3].
enum BlockType {
  BlockType_INVALID    = 0,
  BlockType_THREAD     = 1,	// A component of a coalesced memory request
  BlockType_BASIC      = 2,	// A basic block
  BlockType_SUPERALLOC = 3	// A super-allocator block
};

// 64-bit compare and swap.
// Return true if CAS was successful, false otherwise.
static inline __device__ bool
CAS_AND_CHECK(uint64_t *ptr, uint64_t oldval, uint64_t newval) {
  return atomicCAS((unsigned long long *)ptr,
		   (unsigned long long)oldval,
		   (unsigned long long)newval) == oldval;
}

// Performance counter acces
#ifdef XMC_PERFCOUNT
# define XMC_DECLARE_COUNTER(counter) \
  __device__ unsigned long long counter
# define XMC_COUNTER_INIT(counter) ((counter) = 0)
# define XMC_COUNTER_INC(counter) (atomicAdd(&(counter), 1))
# define XMC_COUNTER_READ(counter) (xmcCounterRead(#counter))
#else
# define XMC_DECLARE_COUNTER(counter)
# define XMC_COUNTER_INIT(counter)
# define XMC_COUNTER_INC(counter)
# define XMC_COUNTER_READ(counter)
#endif

#include "LockFreeFIFO.h"
#include "System.h"
#include "Coalescing.h"
#include "Superblock.h"
