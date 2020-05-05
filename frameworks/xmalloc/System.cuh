/* Copyright (c) 2010 The Board of Trustees of the University of Illinois.
 * All rights reserved.
 */

// Determine the size class of a memory request of size "size".
// The size class determines which free lists are used to get the memory.
__device__ int getSizeClass(unsigned int size)
{
  for (int i = 0; i < MemoryLevels; ++i)
  {
    if (size <= (1 << (START_LEVEL_SHIFT + i)) - sizeof(BasicBlock))
    {
      return i;
    }
  }

  return -1;
}

// Compute the maximum size of a memory request of size "size_class".
__device__ int sizeClassSize(unsigned int size_class)
{
  return (1 << (START_LEVEL_SHIFT + size_class)) - sizeof(BasicBlock);
}

// Compute a superblock buffer size that is large enough to hold the maximum
// possible number of superblocks.
//
// The number of superblocks can be computed from the heap size.
// When the buffer holds all N superblocks, the memory used is
// N * (size of superblock + size of buffer's queue node).
__device__ int getSBfifoSize(size_t heap_size, unsigned int size_class)
{
  unsigned int superblock_space =
    SuperBlockSize(size_class) + sizeof(QueueNode);
  unsigned int log2_queue_size;
  for (log2_queue_size = LOG2_FREELIST_SIZE;
       (heap_size / superblock_space) >= (1 << log2_queue_size);
       log2_queue_size++);

  return log2_queue_size;
}

//////////////////////////////////////////////////////////////////////////
// SuperBlock memory allocation -- normal
//////////////////////////////////////////////////////////////////////////

// Allocate a block from the superallocator.  Return a pointer to its
// payload.
//
// The size_class parameter should be ~0 if this superblock does not
// fall into a size class.  This will always be the case in normal
// operation.  The size class may be something else when
// XMC_DISABLE_SUPERBLOCKS is defined.
__device__ void *SuperallocMalloc(unsigned int size,
				  unsigned int size_class)
{
  // Allocate from the superallocator.  Add 8 bytes for the header.
  // If allocating in a size class, allocate for the size class instead
  // of the actual request size.
  unsigned int superalloc_size =
    8 + (size_class != SIZE_CLASS_NOTHING ? sizeClassSize(size_class) : size);
  void *ptr = malloc(superalloc_size);
  void *payload = (char *)ptr + 8;

  if (ptr == NULL) {
    printf("xmcMalloc: Heap exhausted\n");
    return NULL;
  }

  // Write a header
  SuperallocBlockAtomic *memflag =
    (SuperallocBlockAtomic *)((char *)payload - sizeof(SuperallocBlock));
  SuperallocBlockAtomic header_value;
  header_value.value.offset = 8 - sizeof(SuperallocBlock);
  header_value.value.sizeClass = size_class;
  header_value.value.flag = BlockType_SUPERALLOC;
  memflag->atomic = header_value.atomic;
  return payload;
}

// Given a pointer to a superallocator block's payload, free the block.
__device__ void SuperallocFree(void *ptr)
{
  // Get the offset of the actual block
  void *block_ptr = (char *)ptr - 8;

  free(block_ptr);
}

//////////////////////////////////////////////////////////////////////////
// SuperBlock memory allocation -- with free lists
//////////////////////////////////////////////////////////////////////////

__device__ void *SuperallocMallocWithFreelist(unsigned int size,
					      unsigned int size_class)
{
#ifndef XMC_DISABLE_SUPERBLOCKS
  printf("SuperallocMallocWithFreelist: This feature is disabled!\n");
  return NULL;
#else
  // Pick a freelist to use
  int whichfl = WHICHFL;

  PackedHeapPtr ptr;

  // Try to take from free list
  if (size_class != SIZE_CLASS_NOTHING) {
    do ptr = Dequeue(xmcConstants.BBfifo[whichfl][size_class]);
    while (ptr == PackedHeapFAIL);
  }
  else ptr = PackedHeapNULL;

  // If free list is empty, allocate a new block
  if (ptr == PackedHeapNULL) return SuperallocMalloc(size, size_class);

  // Otherwise, return the allocated memory
  return unpackHeapPointer(ptr);
#endif
}
__device__ void SuperallocFreeWithFreelist(void *ptr, unsigned int size_class)
{
#ifndef XMC_DISABLE_SUPERBLOCKS
  printf("SuperallocFreeWithFreelist: This feature is disabled!\n");
#else
  // Pick a freelist to use
  int whichfl = WHICHFL;

  PackedHeapPtr packed_addr = packHeapPointer(ptr);

  // Try to put into free list
  bool enqueued;
  if (size_class != SIZE_CLASS_NOTHING) {
    enqueued = Enqueue(xmcConstants.BBfifo[whichfl][size_class], packed_addr);
  }
  else enqueued = false;

  // If free list is full or block doesn't belong in free list,
  // deallocate the block
  if (!enqueued)
    SuperallocFree(ptr);
#endif
}

