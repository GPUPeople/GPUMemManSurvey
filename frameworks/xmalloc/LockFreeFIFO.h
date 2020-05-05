/* Copyright (c) 2010 The Board of Trustees of the University of Illinois.
 * All rights reserved.
 */

#ifndef _LOCKFREEFIFO_
#define _LOCKFREEFIFO_

///////////////////////////////////////////////////////////////////////////////
// An entry in a FIFO queue.
// Must be exactly 64 bits because it is accessed atomically.
struct __align__(8) QueueNode
{
  PackedHeapPtr buf;		// buffer for last element
  uint32_t position;		// current Tailer position
};

// use to convert QueueNode to long long
union __align__(8)  _QueueNode
{
  QueueNode node;
  uint64_t llvalue;
};

//////////////////////////////////////////////////////////////////////////
// A FIFO data structure.  FIFOs must be allocated on the heap.
struct __align__(8) FIFOQueue
{
  uint32_t log2_size;		// base-2 log of number of FIFO queue nodes
  uint32_t header;		// Index of the FIFO head
  QueueNode    tailer;		// The FIFO tail
};

typedef  FIFOQueue* FIFOQueuePtr;

__device__ void
initFIFO(FIFOQueuePtr myqueue, const uint32_t log2_size);

__device__ FIFOQueuePtr
newFIFO(const uint32_t log2_size);

__device__ bool
isFull(const uint32_t header, 
       const QueueNode tailer,
       const uint32_t size);

__device__ bool
isEmpty(const uint32_t header,
	const QueueNode tailer);

__device__ unsigned int
FIFOOccupancy(FIFOQueuePtr myqueue);

__device__ bool
Enqueue(FIFOQueuePtr myqueue, PackedHeapPtr element);

__device__ PackedHeapPtr
Dequeue(FIFOQueuePtr myqueue);

static inline __host__ __device__ unsigned int
getFIFOsize(const uint32_t log2_size)
{
  return sizeof(FIFOQueue) + (1 << log2_size) * sizeof(QueueNode);
}

#endif
