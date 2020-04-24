# Some Notes on the Allocators

## Ouroboros

## FDGMalloc
* Only allows for allocations on a warp-basis and cannot really free memory it seems

## CUDA
* Cuda Allocator cannot be resized once its size has been set once and allocations happened!
  * Probably needs a context reset for it to work
  * This means also that you can only have one instance of it running
    * Not sure if this is a major use case, but nonetheless

## ScatterAlloc
* Have two variants of that now, the one from GitHub in mallocMC (currently in use) and the base version found in the RegEff Code
  * Probably should get the Original some time xD

## Halloc
* Currently only works in warp-based manner

## Reg-Eff