/* Copyright (c) 2010 The Board of Trustees of the University of Illinois.
 * All rights reserved.
 */

#ifndef _SUPERBLOCK_
#define _SUPERBLOCK_

//////////////////////////////////////////////////////////////////////////
// SuperBlockAtomic  : the data structure for Superblock atomic operation
// count : 6 ;      how many basicblocks available
// total : 6;      total number of basicblocks
// tag   : 20;      tag to avoid ABA problem
// available : 32;    which block is avaible
//////////////////////////////////////////////////////////////////////////

struct alignas(8) SuperBlockAtomic
{
  unsigned count : 6 ;    // how many basicblocks available
  unsigned total : 6;      // total number of basicblocks
  unsigned tag   : 20;    // tag to avoid ABA problem
  unsigned available : 32;  // which block is avaible
};

union _SuperBlockAtomic
{
  SuperBlockAtomic atomic;
  uint64_t llvalue;
};

//////////////////////////////////////////////////////////////////////////
// BasicBlock  : the data structure for basicblock header
// offset  : 32;    the basic block's byte offset relative to its parent
//                  superblock, divided by eight
// count  : 5;    how many thread blocks inside this block
// index  : 5;    which basic block within superblock
// tag  : 20;      useless, to avoid ABA problem
// flag  : 2;      memory type
//////////////////////////////////////////////////////////////////////////
struct BasicBlockBits {
  unsigned count  	: 5;    // how many thread block
  unsigned index  	: 5;    // which basic block within superblock
  unsigned tag  	: 20;   // useless, to avoid ABA problem (volatile)
  unsigned flag  	: 2;    // memory type
};

union _BasicBlockBits {
  BasicBlockBits value;
  uint32_t atomic;
};

struct alignas(8) BasicBlock
{
  uint32_t offset;		// the block's offset relative to its parent
  BasicBlockBits bits;
};

union BasicBlockAtomic
{
  BasicBlock value;
  uint64_t atomic;
  struct {
    uint32_t _offset;
    uint32_t atomic_bits;
  };
};

//////////////////////////////////////////////////////////////////////////
// SuperBlock  : the data structure for SuperBlock header
// SuperBlockAtomic action;        atomic part
// unsigned int   size   : 25;      the size of basic block + size of basicblock header. Max is 2^25 = 32M
// unsigned int   sizeClass  : 7 ;      which queue it belongs to. Max is 128 level
// BasicBlock     _pad;        useless, but need to reserve for the first basicblock 
//////////////////////////////////////////////////////////////////////////
union SuperBlockConstant
{
  struct {
    unsigned int   size   : 25;    // the size of basic block including its header. Max is 2^25 = 32M
    unsigned int   sizeClass : 7 ; // which queue it belongs to. Max is 128 level
  } value;
  uint32_t atomic;
};
    
struct alignas(8) SuperBlock
{
  _SuperBlockAtomic action;      // atomic part
  SuperBlockConstant constant;	// constant part
  uint32_t       _pad;		// Ensure that BasicBlocks are 8-byte-aligned
};

struct RecycleListDequeueResult {
  SuperBlock *sb;
  bool outOfMemory;
};

__device__ unsigned int SuperBlockSize(const unsigned int size_class);

// malloc a super block
__device__ SuperBlock *SuperBlockMalloc(const unsigned int index, const unsigned int BBnumber );
__device__ bool EnqueueBasicBlockFromSuperBlock(SuperBlock* sb);
__device__ bool RecycleListEnqueue(SuperBlock* sb, const int size_class);
__device__ RecycleListDequeueResult RecycleListDequeue(const int size_class);
__device__ bool RestoreSuperblock(BasicBlock* bb);
__device__ bool FreeBasicBlock(void* addr);
__device__ void *BasicBlockMalloc(const unsigned int index);

#endif
