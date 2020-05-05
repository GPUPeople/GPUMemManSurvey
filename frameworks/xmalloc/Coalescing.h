/* Copyright (c) 2010 The Board of Trustees of the University of Illinois.
 * All rights reserved.
 */

#ifndef _COALESCING_
#define _COALESCING_

//////////////////////////////////////////////////////////////////////////
// ThreadHeader: The data structure header for malloc for a warp
// full   :  True if this block is in use, False if it's been freed
// offset : the offset from warp header to address of this object
// flag   : the block type, always BlockType_THREAD
//////////////////////////////////////////////////////////////////////////
struct ThreadHeader
{
  unsigned inUse  : 1;		// True if this block hasn't been freed
  unsigned offset : 29;		// Offset relative to warp header
  unsigned flag  : 2;		// Block type; always BlockType_THREAD
};

union ThreadHeaderAtomic
{
  ThreadHeader value;
  uint32_t atomic;
};

//////////////////////////////////////////////////////////////////////////
// _ThreadHeader	: the data structure for atomic operation
// atomic			: atomic part
// uivalue			: value for atomic operataion in CUDA
//////////////////////////////////////////////////////////////////////////
union _ThreadHeader
{
	ThreadHeader atomic;
	unsigned int uivalue;
};

__device__ unsigned int
getThreadBlockSize(unsigned int payload_size);

__device__ void
initThreadHeader(ThreadHeaderAtomic *hdr, unsigned int position);

__device__ void
freeThreadHeader(ThreadHeader *hdr);

#endif
