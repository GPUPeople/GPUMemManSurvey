/* Copyright (c) 2010 The Board of Trustees of the University of Illinois.
 * All rights reserved.
 */

// This file is included in Xmalloc.cu

//////////////////////////////////////////////////////////////////////////
// initlize the FIFO queue
// myqueue : FIFO memory buffer.
// log2_size   : the size of the fifo queue is 1<<log2_size
//////////////////////////////////////////////////////////////////////////
__device__ void initFIFO(FIFOQueuePtr myqueue,
			 const uint32_t log2_size)
{
  unsigned int size 		= 1 << log2_size;

  myqueue->log2_size        	= log2_size;
  myqueue->header        	= 0;
  myqueue->tailer.buf      	= NULL;
  myqueue->tailer.position  	= ~(unsigned int)(0);

  QueueNode* buffer = (QueueNode*)(myqueue + 1);

  for ( unsigned int i = 0; i < size; ++i )
  {
    buffer[i].position = i - size;
  }
}

__device__ FIFOQueuePtr
newFIFO(const uint32_t log2_size)
{
  FIFOQueuePtr queue = (FIFOQueuePtr)malloc(getFIFOsize(log2_size));
  if (queue == NULL) return NULL;
  initFIFO(queue, log2_size);
  return queue;
}


//////////////////////////////////////////////////////////////////////////
// check whether the FIFO is full
// myqueue : FIFO memory buffer.
//////////////////////////////////////////////////////////////////////////

// check whether the FIFO is full, the input is from registers directly
__device__ bool 
isFull(const unsigned int header,
	    const QueueNode tailer,
	    const unsigned int size)
{
  return (tailer.position - header + 1) == size;
}

// check whether the FIFO is empty, the input is from registers directly
__device__ bool 
isEmpty(const unsigned int header,
	     const QueueNode tailer)
{
  return (tailer.position - header + 1) == 0;
}

// Get the number of elements in the FIFO.  This is not threadsafe.
__device__ unsigned int
FIFOOccupancy(FIFOQueuePtr myqueue)
{
  unsigned int header = XMC_VLD_U32(myqueue->header);
  _QueueNode tailer;
  tailer.llvalue = XMC_VLD_U64(myqueue->tailer);
  return tailer.node.position - header + 1;
}

// push a data to the FIFO
__device__ bool Enqueue( FIFOQueue* myqueue, PackedHeapPtr element)
{
  unsigned int oldHeader;
  _QueueNode newTailer, oldTailer;
  unsigned int size = 1 << myqueue->log2_size; // Nonvolatile load OK
  unsigned int index;
  QueueNode* buffer = (QueueNode*)(myqueue + 1);

  __threadfence(); // Prior writes must complete before enqueueing
  do
  {
    oldHeader = XMC_VLD_U32(myqueue->header);
    newTailer.llvalue = oldTailer.llvalue = XMC_VLD_U64(myqueue->tailer);

    if (isFull(oldHeader, oldTailer.node, size))
    {
      return false;
    }

    _QueueNode oldNode;
    index = oldTailer.node.position % size;

    oldNode.llvalue = XMC_VLD_U64(buffer[index]);

    if ( oldNode.node.position + size == oldTailer.node.position )
    {
      atomicCAS((unsigned long long*)&(buffer[index]),oldNode.llvalue, oldTailer.llvalue);
    }

    newTailer.node.buf = element;
    newTailer.node.position++;
  } while (atomicCAS((unsigned long long*)&(myqueue->tailer), oldTailer.llvalue, newTailer.llvalue) != oldTailer.llvalue);

  return true;
}

// pop a data from them
__device__ PackedHeapPtr Dequeue( FIFOQueue* myqueue)
{
  unsigned int newHeader, oldHeader;  // use for compare
  _QueueNode oldTailer;      // current FIFO tailer

  PackedHeapPtr returnElement;
  unsigned int size = 1 << myqueue->log2_size; // Nonvolatile load OK
  QueueNode* buffer = (QueueNode*)(myqueue + 1);

  do 
  {
    newHeader = oldHeader = XMC_VLD_U32(myqueue->header);
    oldTailer.llvalue = XMC_VLD_U64(myqueue->tailer);

    if (isEmpty(oldHeader, oldTailer.node) == true)
    {
      return PackedHeapNULL;
    }

    if ( oldHeader == oldTailer.node.position )
    {
      returnElement = oldTailer.node.buf;
    }
    else
    {
      returnElement = XMC_VLD_U32(buffer[oldHeader % size].buf);
    }
    newHeader++;

  } while (atomicCAS(&(myqueue->header), oldHeader, newHeader) != oldHeader);

  __threadfence(); // Dequeue must complete before accessing data
  return returnElement;
}

