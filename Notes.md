# Some Notes on the Allocators

## Ouroboros
---

## FDGMalloc
---
* Only allows for allocations on a warp-basis and cannot really free memory it seems

## CUDA
---
* Cuda Allocator cannot be resized once its size has been set once and allocations happened!
  * Probably needs a context reset for it to work
  * This means also that you can only have one instance of it running
    * Not sure if this is a major use case, but nonetheless
* It seems there is also no difference between different iterations
  * So no difference between the first round and subsequent iterations

## ScatterAlloc
---
* Have two variants of that now, the one from GitHub in mallocMC (currently in use) and the base version found in the RegEff Code
  * Probably should get the Original some time xD
* Currently it only works correctly in sync-mode

## Halloc
---
* The masks are quite hard to translate to modern CUDA
  * Currently it works with Sync Build

## Reg-Eff
---
### Atomic
* Simply increments an offset for each new allocation `return d_heap_base + offset`
  * Can be used together with coalescing on a warp basis
* Has no de-allocation and no re-use
* Can only increase its offset, hence will over time simply run out of memory as soon as it reaches the end of the allocated memory
### Atomic Wrap
* Works the same as the basis *atomic* allocator, the only difference is what happens once it reaches the end of the allocated memory
  * In this case it will try to *wrap around* to the beginning using successive atomicCAS operations
    * So it will simply start overwriting data from the front
* Has no de-allocation and no re-use
### Circular
* Not yet working in simple testcase
### Circular Fused
Text
### Circular Multi
* Not yet working in simple testcase
### Circular Fused
Text

## XMalloc
---
* Only works in sync-mode
* Allocates from the cudaHeap, hence cannot reallocate unfortunately
* Problem for `mixed_allocation` testcase for `100.000` allocations with range `512-8192`
  * `10.000` works without a problem

## DynaSOAr
---
* Can only allocate objects implemented in their specific format
  * Even with a hack it will not work for a general purpose memory allocator