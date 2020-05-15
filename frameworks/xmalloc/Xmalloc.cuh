/* Copyright (c) 2010 The Board of Trustees of the University of Illinois.
 * All rights reserved.
 */

#include "Xmalloc.h"
#include "XmallocInternal.h"

///////////////////////////////////////////////////////////////////////////////
// Globals used by XMalloc

// An address at or before the origin of the heap
void *xmcHeapOrigin;

// Constants used by XMalloc.  Defined in System.h.
// __constant__ struct ConstantSystemHeader xmcConstants;

///////////////////////////////////////////////////////////////////////////////
// Other components of XMalloc

#include "LockFreeFIFO.cuh"
#include "System.cuh"
#include "Coalescing.cuh"
#include "Superblock.cuh"

///////////////////////////////////////////////////////////////////////////////

struct AllocatedFreeLists
{
  void *freelist_area;		// Space reserved for free lists
  FIFOQueuePtr SBfifo[MemoryLevels]; // Pointers to superblock FIFOs
};

// Alloate and initialize free lists in the GPU heap.
//
// This kernel is run by one thread.  This kernel runs before xmcConstants
// is initialized.
static __global__ void
AllocateFreeLists(AllocatedFreeLists *fl_pointer,
		  size_t heap_size,
		  unsigned int freelists_size)
{
  // Allocate memory for free lists 
  fl_pointer->freelist_area = malloc(freelists_size);

  // Allocate and initialize memory for superblock FIFOs
  int i;
  for (i = 0; i < MemoryLevels; i++) {
    fl_pointer->SBfifo[i] = newFIFO(getSBfifoSize(heap_size, i));
  }
}

// Initialize GPU heap data structures.
//
// Each thread initializes one free list and one superblock FIFO.
static __global__ void
InitializeGPUHeap(void)
{
  int index = threadIdx.x;
  int i;

  // Initialize a free list
  for (i = 0; i < FREELIST_COUNT; i++) {
    initFIFO(xmcConstants.BBfifo[i][index], LOG2_FREELIST_SIZE);
  }

  // Initialize the performance counters
  if (threadIdx.x == 0) {
    XMC_COUNTER_INIT(xmcCounter_FreeListEnqueueSuccess);
    XMC_COUNTER_INIT(xmcCounter_FreeListEnqueueFail);
    XMC_COUNTER_INIT(xmcCounter_FreeListRefill);
    XMC_COUNTER_INIT(xmcCounter_FreeListRefillOverflow);
    XMC_COUNTER_INIT(xmcCounter_FreeListDequeueFail);
    XMC_COUNTER_INIT(xmcCounter_FreeListDequeueSuccess);
    XMC_COUNTER_INIT(xmcCounter_RecycleListEnqueue);
    XMC_COUNTER_INIT(xmcCounter_RecycleListDequeueFail);
    XMC_COUNTER_INIT(xmcCounter_RecycleListDequeueSuccess);
  }
}

// Initialize the heap.
//
// Allocate (don't initialize) global heap data and free lists.
// Initialize constant data in xmcConstants.
int
xmcInit(size_t heap_size)
{
  ConstantSystemHeader constants;

  // How big is the heap?
  // size_t heap_size;
	// cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
  constants.heapSize = heap_size;

  // Allocate free lists on the heap.  Also, use the memory address to
  // estimate the location of the heap origin.
  unsigned int freelist_size = getFIFOsize(LOG2_FREELIST_SIZE);
  unsigned int all_freelists_size =
    freelist_size * FREELIST_COUNT * MemoryLevels;
  AllocatedFreeLists freelist_pointers;

  {
    AllocatedFreeLists *d_freelist_pointers;

    // Call the CUDA SDK malloc() to allocate free lists
    cudaMalloc(&d_freelist_pointers, sizeof(AllocatedFreeLists));
    AllocateFreeLists<<<1,1>>> (d_freelist_pointers,
				heap_size,
				all_freelists_size);
    cudaMemcpy(&freelist_pointers,
	       d_freelist_pointers,
	       sizeof(AllocatedFreeLists),
	       cudaMemcpyDeviceToHost);
    cudaFree(d_freelist_pointers);
  }
  constants.heapOrigin = xmcHeapOrigin = 
    (void *)((char *)freelist_pointers.freelist_area - heap_size);

  // Set up pointers to FIFOs.
  // The freelist area is partitioned into free lists.
  for (int i = 0; i < MemoryLevels; ++i)
  {
    for (int j = 0; j < FREELIST_COUNT; ++j) {
      constants.BBfifo[j][i] =
	(FIFOQueuePtr)((char *)freelist_pointers.freelist_area +
		       (i * FREELIST_COUNT + j) * freelist_size);
    }
    constants.SBfifo[i] = freelist_pointers.SBfifo[i];
  }

  cudaMemcpyToSymbol(xmcConstants, &constants, sizeof(ConstantSystemHeader));

  // Initialize CUDA memory structures
  InitializeGPUHeap<<<1, MemoryLevels>>>();

  return 1;
}

void xmcPrintStatistics(void)
{
#ifdef XMC_PERFCOUNT		// Only do something if compiled in
# define PRINTCOUNTER(c) printf("%36s: %llu\n", #c, XMC_COUNTER_READ(c));
  PRINTCOUNTER(xmcCounter_FreeListEnqueueSuccess)
  PRINTCOUNTER(xmcCounter_FreeListEnqueueFail)
  PRINTCOUNTER(xmcCounter_FreeListRefill)
  PRINTCOUNTER(xmcCounter_FreeListRefillOverflow)
  PRINTCOUNTER(xmcCounter_FreeListDequeueFail)
  PRINTCOUNTER(xmcCounter_FreeListDequeueSuccess)
  PRINTCOUNTER(xmcCounter_RecycleListEnqueue)
  PRINTCOUNTER(xmcCounter_RecycleListDequeueFail)
  PRINTCOUNTER(xmcCounter_RecycleListDequeueSuccess)
# undef PRINTCOUNTER
#endif
}

#ifdef XMC_PERFCOUNT
unsigned long long xmcCounterRead(const char *counter_name)
{
  unsigned long long host_value;
  cudaMemcpyFromSymbol(&host_value, counter_name, sizeof(unsigned long long));
  return host_value;
}
#endif


// Perform a coalesced memory allocation.  This is usually
// run by one thread in a warp, but can run on several threads simultaneously.
// The returned memory will be aligned to a multiple of 8 bytes.
//
// Allocates a BasicBlock or superallocator block and returns its payload.
static inline __device__ void*
SubCudaMalloc(unsigned int size)
{
  if (size == 0) return NULL;

  int index = getSizeClass(size);

#ifdef _NOBUFFER_
  return SuperallocMalloc (size, SIZE_CLASS_NOTHING);
#else

  if (index == -1)
    return SuperallocMalloc(size, SIZE_CLASS_NOTHING);
  else {
#ifdef XMC_DISABLE_SUPERBLOCKS
    return SuperallocMallocWithFreelist(sizeClassSize(index), index);
#else
    return BasicBlockMalloc(index);
#endif // XMC_DISABLE_SUPERBLOCKS
  }

#endif // _NOBUFFER_
}

// Allocate memory.
//
// Allocation works one of two ways, depending on whether thread coalescing
// occurs.  The code is written to call SubCudaMalloc() in exactly one place
// regardless of which allocation strategy is used.
__device__ void *
xmcMalloc(unsigned int size)
{
  // The number of threads in the current warp that will allocate
  unsigned int threadCount = 0;

  // sizeCount[i]    holds the size of warp i's memory request
  // temporaryPtr[i] holds thread i's returned memory request
  __shared__ unsigned int sizeCount[32];
  __shared__ char *temporaryPtr[32];

  unsigned int warpId = threadIdx.x >> 5;
  sizeCount[warpId]   = sizeof(uint32_t); // Reserve space for number of sub-blocks

  // Will be set to True if this thread will participate in coalescing
  bool use_coalescing = false; 

  // Position of this thread's data in the coalesced block;
  // ignored if thread doesn't participates in coalescing
  unsigned int position = 0;

  /*
  ** Calculate how much memory to allocate
  */

#ifndef XMC_DISABLE_COALESCING
  // Permit coalescing for this range of sizes
  bool coalescible = size != 0 && size < MAX_COALESCE_SIZE;

  // Count number of threads that will allocate a coalescible number of bytes
  #if (__CUDA_ARCH__ >= 700)
  threadCount = __popc(__ballot_sync(__activemask(), coalescible));
  #else
  threadCount = __popc(__ballot(coalescible));
  #endif

  if (coalescible && threadCount != 1) {
    unsigned int thread_request_size = getThreadBlockSize(size);

    // Coalesce memory requests from half-warps (16 threads)
    position = atomicAdd(&sizeCount[warpId], thread_request_size);
    use_coalescing = true;

    //printf("sz  %d, pos %d, sz+ %d, td+ %d\n",
    // size, position, sizeCount[warpId],threadCount[warpId]);
  }
#endif

  unsigned int request_size;	// Size of memory that will be malloc'd, or 0
  if (use_coalescing)
    // One of the coalesced threads will reserve the first position within
    // the coalesced block.  Only this thread will perform allocation.
    request_size = (position == sizeof(uint32_t)) ? sizeCount[warpId] : 0;
  else
    // Threads that don't participate in coalescing will request their memory
    request_size = size;

  /*
  ** Allocate memory
  */
  char *allocated_pointer = (char *)SubCudaMalloc(request_size);

  /*
  ** Initialize and return the new memory
  */
  if (request_size && use_coalescing) {
    // If this thread is allocating coalesced memory,
    // pass the data to other threads and initialize the block.
    // Store the allocated pointer
    temporaryPtr[warpId] = allocated_pointer;

    // Store the number of sub-blocks
    if (allocated_pointer != NULL)
      *(uint32_t *)allocated_pointer = threadCount;
  }

  // Ensure that the allocated memory is visible to the entire block
  __threadfence_block();

  void *returned_pointer;

  if (use_coalescing) {
    char *coalesced_pointer = temporaryPtr[warpId];
    if (coalesced_pointer == NULL) return NULL;

    // Initialize this thread's block within the new memory area
    char *thread_block = coalesced_pointer + position;
    initThreadHeader((ThreadHeaderAtomic*)thread_block, position);

    // Return the payload
    //printf("a: %p\n", thread_block + 4);
    returned_pointer = (void*)(thread_block + sizeof(uint32_t));
  }
  else {
    returned_pointer = allocated_pointer;
  }
  return returned_pointer;
}

// Deallocate memory
__device__ void
xmcFree(void *ptr)
{
  if (ptr == NULL) return;

  _MemoryFlagReader* myflag = (_MemoryFlagReader*)ptr - 1;
  _MemoryFlagReader myflag_val;
  myflag_val.atomic = XMC_VLD_U32(myflag->atomic);

  switch(myflag_val.value.flag) 
  {
  case BlockType_BASIC:
    FreeBasicBlock(ptr);
    break;
  case BlockType_SUPERALLOC:
#ifdef XMC_DISABLE_SUPERBLOCKS
    {
      SuperallocBlockAtomic sb_flag;
      sb_flag.atomic = myflag_val.atomic;
      unsigned int size_class = sb_flag.value.sizeClass;
      SuperallocFreeWithFreelist(ptr, size_class);
    }
#else
    SuperallocFree(ptr);
#endif
    break;
  case BlockType_THREAD:
    {
      ThreadHeaderAtomic* header = (ThreadHeaderAtomic*)ptr - 1;

      // Ensure that all uses of this block have completed before freeing it
      __threadfence();

      // Mark the header as invalid
      ThreadHeaderAtomic header_value;
      header_value.atomic = XMC_VLD_U32(header->atomic);
      header_value.value.inUse = false;
      header_value.value.flag = BlockType_INVALID;
      header->atomic = header_value.atomic;

      // Find the parent block's payload area
      uint32_t* parent_block =
	(uint32_t*)((char*)header - header_value.value.offset);

      // Update the counter.  If parent is empty, deallocate it.
      if (atomicSub(parent_block, 1) == 1)
	{
	  _MemoryFlagReader* parent_flag =
	    (_MemoryFlagReader*)parent_block - 1;
	  myflag_val.atomic = XMC_VLD_U32(parent_flag->atomic);

	  switch(myflag_val.value.flag) {
	  case BlockType_BASIC:
	    FreeBasicBlock(parent_block);
	    break;
	  case BlockType_SUPERALLOC:
#ifdef XMC_DISABLE_SUPERBLOCKS
	    {
	      SuperallocBlockAtomic sb_flag;
        sb_flag.atomic = myflag_val.atomic;
	      unsigned int size_class = sb_flag.value.sizeClass;
	      SuperallocFreeWithFreelist(parent_block, size_class);
	    }
#else
	    SuperallocFree(parent_block);
#endif
	    break;
	  default:
	    printf("xmcFree: Invalid header\n");
	    //printf("xmcFree: Invalid header: B %p %p\n", parent_block, *((void **)parent_block - 1));
	  }
	}
      break;
    }
  default:
    printf("xmcFree: Invalid header\n");
      //printf("xmcFree: Invalid header: A %p %p\n", ptr, *((void **)ptr - 1));
  }
}
