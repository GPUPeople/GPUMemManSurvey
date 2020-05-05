/* Copyright (c) 2010 The Board of Trustees of the University of Illinois.
 * All rights reserved.
 */

// Define this to not use the superblock queues.
// The allocator will leak memory when free lists are full.
// #define NO_REUSE_SUPERBLOCKS

// Get the address of a basic block's parent
static inline __device__ SuperBlock *
BasicBlockParent(const BasicBlock *bb_pointer, const BasicBlock &bb)
{
  unsigned int byte_offset = bb.offset * 8;
  return (SuperBlock *)((char *)bb_pointer - byte_offset);
}

// Get the size in bytes of a superblock in the given size class
__device__ unsigned int
SuperBlockSize(const unsigned int size_class)
{
  // Size of basicblock in bytes (including header and payload)
  unsigned int basicblockSize = 1 << (START_LEVEL_SHIFT + size_class);

  // Size of superblock in bytes (including header and payload)
  unsigned int superblockSize =
    sizeof(SuperBlock) + basicblockSize * BasicBlockNumber;
  
  return superblockSize;
}

// Get the address of a basic block inside a SuperBlock
static __device__ BasicBlock *
SuperBlockChild(const SuperBlock *sb,
		unsigned int bb_size,
		unsigned int index)
{
  return (BasicBlock *)((char *)sb + sizeof(SuperBlock) + bb_size * index);
}

// Get the address of a basic block's payload inside a SuperBlock
static __device__ void *
SuperBlockChildPayload(const SuperBlock *sb,
		       unsigned int bb_size,
		       unsigned int index)
{
  BasicBlock *bb = SuperBlockChild(sb, bb_size, index);
  return (void *)((char *)bb + sizeof(BasicBlock));
}

//////////////////////////////////////////////////////////////////////////
// Allocate a superblock from global memory pool.
// size_class: the size class to allocate memory for
//////////////////////////////////////////////////////////////////////////
__device__ SuperBlock*
SuperBlockMalloc(const unsigned int size_class)
{
  // no access to shared data here and no thread will use this super block
  // so we do not deal with atomic 

  // Size of basicblock in bytes (including header and payload)
  unsigned int basicblockSize = 1 << (START_LEVEL_SHIFT + size_class);
  // Size of superblock in bytes (including header and payload)
  unsigned int superblockSize = SuperBlockSize(size_class);

  // Allocate memory
  SuperBlock* superblockPointer = (SuperBlock *)malloc(superblockSize);
  if (superblockPointer == NULL) {
    printf("xmalloc: Heap exhausted\n");
    return NULL;
  }

  // init the header
  _SuperBlockAtomic init_action;
  init_action.atomic.available = ~0;
  init_action.atomic.count = BasicBlockNumber;
  init_action.atomic.total = BasicBlockNumber;
  SuperBlockConstant init_constant;
  init_constant.value.size = basicblockSize;
  init_constant.value.sizeClass = size_class;

  superblockPointer->action.llvalue = init_action.llvalue;
  superblockPointer->constant.atomic = init_constant.atomic;

  // initialize the basic blocks
  for (unsigned int i = 0; i < BasicBlockNumber; ++i)
  {
    BasicBlockAtomic* bbPointer = (BasicBlockAtomic *)
      SuperBlockChild(superblockPointer, basicblockSize, i);

    BasicBlockAtomic myBasicBlock;
    myBasicBlock.value.offset = ((char *)bbPointer - (char *)superblockPointer) / 8;
    myBasicBlock.value.bits.count = 0;
    myBasicBlock.value.bits.index = i;
    myBasicBlock.value.bits.tag = 0;
    myBasicBlock.value.bits.flag = BlockType_BASIC;

    bbPointer->atomic = myBasicBlock.atomic;
  }

  __threadfence();		// Ensure all initialization has completed
  return superblockPointer;
}

__device__ void
SuperBlockFree(SuperBlock *sb)
{
  // printf("Free superblock\n");
  free(sb);
}

//////////////////////////////////////////////////////////////////////////
// Put all the superblock's free BasicBlocks into the freelist.
//
// The superblock is owned by the current thread but it may contain
// BasicBlocks that are owned by other threads.
//
// if return false, the superblock is still not empty
//////////////////////////////////////////////////////////////////////////
__device__ bool EnqueueBasicBlockFromSuperBlock(SuperBlock* sb)
{
  _SuperBlockAtomic oldAtomic, newAtomic;
  SuperBlockConstant sb_constant;
  sb_constant.atomic = XMC_VLD_U32(sb->constant.atomic);

  XMC_COUNTER_INC(xmcCounter_FreeListRefill);

  // Pick a freelist to use
  int whichfl = WHICHFL;

  // Do until sb contains no free BasicBlocks,
  // or until a freelist enqueue fails
  do {
    unsigned int bb_index;	// Index of current basic block

    // Get index of one free BasicBlock in bb_index
    do {
      oldAtomic.llvalue = newAtomic.llvalue = XMC_VLD_U64(sb->action.llvalue);

      // If no free BasicBlocks, then end
      if (oldAtomic.atomic.count == 0) return true;
      
      // Find the index of the first available block
      bb_index = __ffs(newAtomic.atomic.available) - 1;
      unsigned int shift = 1 << bb_index;
      newAtomic.atomic.available ^= shift;
      newAtomic.atomic.count--;
      newAtomic.atomic.tag++;
    } while (!CAS_AND_CHECK((uint64_t*)sb,
			    oldAtomic.llvalue,
			    newAtomic.llvalue));

    // Get address of basic block payload
    BasicBlock *bb = SuperBlockChild(sb, sb_constant.value.size, bb_index);
    PackedHeapPtr reservation =
      packHeapPointer(SuperBlockChildPayload(sb,
					     sb_constant.value.size,
					     bb_index));
    
    // Put the BasicBlock in the free list
    if ( Enqueue(xmcConstants.BBfifo[whichfl][sb_constant.value.sizeClass],
		 reservation) == false)
    {
      unsigned int shift = 1 << bb_index;

      // Failed to put back.  Return the BasicBlock to the superblock.
      do {
	oldAtomic.llvalue = newAtomic.llvalue =
	  XMC_VLD_U64(sb->action.llvalue);
	newAtomic.atomic.available ^= shift;
	newAtomic.atomic.count++;
	newAtomic.atomic.tag++;
      } while (!CAS_AND_CHECK((uint64_t*)sb,
			      oldAtomic.llvalue,
			      newAtomic.llvalue));

      // printf("xmcMalloc: enqueue failed\n");
      return false;
    }
  }while(newAtomic.atomic.count > 0);

  return true;
};

// Return a basic block to its parent superblock after freeing it.
// The basic block is not in use.
// The parameters are a pointer to the parent superblock, and the header of
// the BasicBlock.
// There is no race condition wrt the BasicBlock because only constant
// fields are read.
__device__ bool RestoreSuperblock(SuperBlock* sb, const BasicBlock &bb)
{
  _SuperBlockAtomic oldAtomic, newAtomic;

  // Mark the block as available
  do 
  {
    oldAtomic.llvalue = newAtomic.llvalue = XMC_VLD_U64(sb->action.llvalue);
    newAtomic.atomic.count++;
    newAtomic.atomic.tag++;
    newAtomic.atomic.available |= (1 << bb.bits.index);
  } while (!CAS_AND_CHECK((uint64_t*)sb,
			  oldAtomic.llvalue,
			  newAtomic.llvalue));
  
  // If the superblock transitioned from completely used to partly used,
  // then put it into the superblock FIFO
  if (oldAtomic.atomic.count == 0)
  {
    //cuPrintf("RecycleListEnqueue: %d\n", sb);
    SuperBlockConstant sb_constant;
    sb_constant.atomic = XMC_VLD_U32(sb->constant.atomic);
    unsigned int size_class = sb_constant.value.sizeClass;
    RecycleListEnqueue(sb, size_class);
  }

  // If all BasicBlocks are unused, then deallocate it
  else if (newAtomic.atomic.count == newAtomic.atomic.total)
  {
#if 0
    // NOTE: Can't free the superblock because it's in the queue
    // Should remove from queue and free
    SuperBlockFree(sb);
#endif
  }

  return true; 
}

//////////////////////////////////////////////////////////////////////////
// recycle functions
//////////////////////////////////////////////////////////////////////////

// Attempt to enqueue a superblock.
// size_class must be the superblock's actual size class.
// Return true if successful, false otherwise.
__device__ bool RecycleListEnqueue(SuperBlock* sb,
				   const int size_class)
{
#ifndef NO_REUSE_SUPERBLOCKS
  XMC_COUNTER_INC(xmcCounter_RecycleListEnqueue);
  return Enqueue(xmcConstants.SBfifo[size_class], packHeapPointer(sb));
#else
  return true;
#endif
}

// Get a superblock from the recycle list, or allocate a new superblock.
// Return the superblock, a NULL pointer if dequeue failed, or
// a flag if out of memory.
__device__ RecycleListDequeueResult RecycleListDequeue(const int size_class)
{
#ifndef NO_REUSE_SUPERBLOCKS
  PackedHeapPtr packedsb = Dequeue(xmcConstants.SBfifo[size_class]);

  RecycleListDequeueResult res;
  if (packedsb == PackedHeapFAIL)
  {
    res.sb = NULL;
    res.outOfMemory = false;
  }
  else if (packedsb == PackedHeapNULL) 
  {
    // Queue is empty, so allocate a new superblock.
    // If superblock is NULL, the heap is exhausted.
    XMC_COUNTER_INC(xmcCounter_RecycleListDequeueFail);

    res.sb = SuperBlockMalloc(size_class);
    res.outOfMemory = res.sb == NULL;   
  }
  else
  {
    XMC_COUNTER_INC(xmcCounter_RecycleListDequeueSuccess);
    res.sb = (SuperBlock *)unpackHeapPointer(packedsb);
    res.outOfMemory = false;   
  }
#else
  // Always allocate a new superblock.
  // If superblock is NULL, the heap is exhausted.
  RecycleListDequeueResult res;
  res.sb = SuperBlockMalloc(size_class);
  res.outOfMemory = res.sb == NULL;   
#endif
  return res;
}

//////////////////////////////////////////////////////////////////////////
// BasicBlock memory allocation
//////////////////////////////////////////////////////////////////////////

__device__ bool FreeBasicBlock(void* addr)
{
#ifdef XMC_DISABLE_SUPERBLOCKS
  printf("FreeBasicBlock: This feature was disabled!\n");
  return false;
#else
  // Pick a freelist to use
  int whichfl = WHICHFL;

  PackedHeapPtr packed_addr = packHeapPointer(addr);
  BasicBlockAtomic* myBasicBlock = (BasicBlockAtomic*)addr - 1;
  BasicBlockAtomic myBasicBlockValue;
  myBasicBlockValue.atomic = XMC_VLD_U64(myBasicBlock->atomic);

  SuperBlock* sb = BasicBlockParent(&myBasicBlock->value,
				    myBasicBlockValue.value);

  SuperBlockConstant sb_constant;
  sb_constant.atomic = XMC_VLD_U32(sb->constant.atomic);
  unsigned int size_class = sb_constant.value.sizeClass;

  // The 'enqueue' operation performs a memory barrier.
  // Any prior writes to the payload will be completed by that point.
  if (Enqueue(xmcConstants.BBfifo[whichfl][size_class], packed_addr) == false)
  {
#ifndef NO_REUSE_SUPERBLOCKS
    //cuPrintf("index: %d,    header:%d,   tailer:%d\n",sb->index, constantManagement.BBfifo[sb->index]->header,constantManagement.BBfifo[sb->index]->tailer.position );

    RestoreSuperblock(sb, myBasicBlockValue.value);
#endif
  }
  return true;
#endif
};

// Allocate some memory through a free list.  Returns a pointer to
// the payload of a BasicBlock.
//
// This function contains a lock-based critical section.
__device__ void* BasicBlockMalloc(const unsigned int size_class)
{
#ifdef XMC_DISABLE_SUPERBLOCKS
  printf("BasicBlockMalloc: This feature was disabled!\n");
  return NULL;
#else
  // Pick a freelist to use
  int whichfl = WHICHFL;

  // An owned superblock.
  // If a superblock is owned, then it must be put into the freelist or
  // into the superblock queue.
  //
  // This thread must increment the refill counter before it acquires a
  // superblock, and decrement it after it gets rid of it.
  SuperBlock *sb = NULL;

  for(int retry = 0; retry < MAXI_LOOP; ++retry) {
    // 1. If we own a superblock, get rid of ownership.  This will add to
    //    the free list if it isn't full.
    if (sb != NULL) {
      if (EnqueueBasicBlockFromSuperBlock(sb) == false) {
	// Oops, we tried to refill the free list, but it's full
	XMC_COUNTER_INC(xmcCounter_FreeListRefillOverflow);
	bool success = RecycleListEnqueue(sb, size_class);
	if (!success) continue;
      }
      sb = NULL;
    }

    // 2. Get a pointer from the free list
    PackedHeapPtr paddr = Dequeue(xmcConstants.BBfifo[whichfl][size_class]);

#if FREELIST_COUNT == 1
    // Nothing
#elif FREELIST_COUNT == 2
    if (paddr == PackedHeapNULL)
      paddr = Dequeue(xmcConstants.BBfifo [whichfl ^ 1][size_class]);
#elif FREELIST_COUNT == 4
    if (paddr == PackedHeapNULL) {
      paddr = Dequeue(xmcConstants.BBfifo [whichfl ^ 1][size_class]);
      if (paddr == PackedHeapNULL) {
	paddr = Dequeue(xmcConstants.BBfifo [whichfl ^ 2][size_class]);
	if (paddr == PackedHeapNULL)
	  paddr = Dequeue(xmcConstants.BBfifo [whichfl ^ 3][size_class]);
      }
    }
#else
# error "Unsupported value of FREELIST_COUNT"
#endif

    if (paddr == PackedHeapNULL)
    {
      // If free list is empty, get a superblock.  It will be used to
      // replenish the free list.

      XMC_COUNTER_INC(xmcCounter_FreeListDequeueFail);

      RecycleListDequeueResult result = RecycleListDequeue(size_class);
      if (result.outOfMemory) break;
      sb = result.sb;
      continue;
    }
    else
    {
      // Got a pointer
      XMC_COUNTER_INC(xmcCounter_FreeListDequeueSuccess);

      return unpackHeapPointer(paddr);
    }
  }

  printf("xmcMalloc: Allocation failed\n");
  return NULL;
#endif
}
