/* Copyright (c) 2010 The Board of Trustees of the University of Illinois.
 * All rights reserved.
 */

#ifndef _SYSTEM_
#define _SYSTEM_

//////////////////////////////////////////////////////////////////////////
// ConstantSystemHeader  :      the data structure for memory alocation header, but save it in constant memory
// SystemHeader* MemoryManagementStart;      start point for memory management
// FIFOQueue* BBfifo[MemoryLevels];        Free lists.
// unsigned int StartLevelShift;        the smallest memory block in our system
// unsigned int size;              size of buffer
//////////////////////////////////////////////////////////////////////////
struct alignas(8) ConstantSystemHeader
{
  // Free lists.  These hold pointers to BasicBlock payloads, not headers.
  // If XMC_DISABLE_SUPERBLOCKS, then these hold pointers to
  //  SuperallocBlock payloads.
  FIFOQueuePtr BBfifo[FREELIST_COUNT][MemoryLevels];

  FIFOQueuePtr SBfifo[MemoryLevels]; // Lists of partly-free superblocks. 
                                // These hold pointers to the superblocks.
                                // If XMC_DISABLE_SUPERBLOCKS, then unused.
  void *heapOrigin;		// An address at or before the heap origin
  size_t heapSize;		// Heap size
};

// All constants used by XMalloc on the compute device
__constant__ ConstantSystemHeader xmcConstants;

//////////////////////////////////////////////////////////////////////////
// MemoryFlagReader	: use to read the memory type
// _pad			: ignored
// flag			: read the memory flag, a BlockType value
//////////////////////////////////////////////////////////////////////////
struct MemoryFlagReader
{
	unsigned _pad	: 30;   // useless 			
	unsigned flag	: 2;	// read the memory flag
};

union _MemoryFlagReader
{
  MemoryFlagReader value;
  uint32_t atomic;
};

//////////////////////////////////////////////////////////////////////////
// SuperallocBlock	: a block produced by the superallocator
// offeset		: offset of this block relative to parent
// sizeClass            : superblock's size class, or SIZE_CLASS_NOTHING
// flag			: always BlockType_SUPERALLOC
//////////////////////////////////////////////////////////////////////////
struct SuperallocBlock
{
  unsigned _pad   	: 19;
  signed sizeClass      : 7;
  unsigned offset  	: 4;
  unsigned flag		: 2;
};

union SuperallocBlockAtomic
{
  SuperallocBlock value;
  uint32_t atomic;
};

///////////////////////////////////////////////////////////////////////////////
// Performance counters

// Success/failure at putting a freed block into a free list
XMC_DECLARE_COUNTER(xmcCounter_FreeListEnqueueSuccess);
XMC_DECLARE_COUNTER(xmcCounter_FreeListEnqueueFail);

// Refilling a free list
XMC_DECLARE_COUNTER(xmcCounter_FreeListRefill);
XMC_DECLARE_COUNTER(xmcCounter_FreeListRefillOverflow);

// Success/failure at taking a block from a free list
XMC_DECLARE_COUNTER(xmcCounter_FreeListDequeueFail);
XMC_DECLARE_COUNTER(xmcCounter_FreeListDequeueSuccess);

// Success/failure at taking a block from a recycle list
XMC_DECLARE_COUNTER(xmcCounter_RecycleListEnqueue);
XMC_DECLARE_COUNTER(xmcCounter_RecycleListDequeueFail);
XMC_DECLARE_COUNTER(xmcCounter_RecycleListDequeueSuccess);

///////////////////////////////////////////////////////////////////////////////

__device__ int getSizeClass(unsigned int size);
__device__ int sizeClassSize(unsigned int size_class);
__device__ int getSBfifoSize(size_t heap_size,
			     unsigned int size_class);
__device__ void *SuperallocMalloc(unsigned int size, unsigned int size_class);
__device__ void SuperallocFree(void *ptr);

__device__ void *SuperallocMallocWithFreelist(unsigned int size,
					      unsigned int size_class);
__device__ void SuperallocFreeWithFreelist(void *ptr, unsigned int size_class);

// Given a pointer to the heap that is at least 8-byte-aligned,
// pack it into 32 bits.  Only heap pointers can be packed this way.
static inline __device__ PackedHeapPtr
packHeapPointer(void *hptr)
{
  uintptr_t offset =(char *)hptr - (char *)xmcConstants.heapOrigin;
  //if (offset & 7) printf("xmalloc: unaligned pointer %p\n", hptr); // DEBUG
  return offset >> 3;
}

static inline __device__ void *
unpackHeapPointer(PackedHeapPtr pptr)
{
  return (char *)xmcConstants.heapOrigin + ((uintptr_t)pptr << 3);
}


#endif
