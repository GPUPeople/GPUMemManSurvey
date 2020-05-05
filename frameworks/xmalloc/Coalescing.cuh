/* Copyright (c) 2010 The Board of Trustees of the University of Illinois.
 * All rights reserved.
 */

// Get the size of a thread coalescing block, including its header.
// The block size is always a multiple of 8 bytes and includes a 4-byte header.
__device__ unsigned int
getThreadBlockSize(unsigned int payload_size)
{
  // Add 4 bytes, then round up to a multiple of 8
  return (payload_size + 4 + 7) & ~7;
}

// Initialize the header of one thread's block within a coalesced memory
// request.  This does not need a memory fence since the header is private
// to the current thread.
__device__ void
initThreadHeader(ThreadHeaderAtomic *hdr, unsigned int position)
{
  ThreadHeaderAtomic myheader;
  myheader.value.inUse = true;
  myheader.value.offset = position;
  myheader.value.flag = BlockType_THREAD;
  hdr->atomic = myheader.atomic;
}
