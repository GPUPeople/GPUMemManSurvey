# Some Notes on the Allocators

## Ouroboros

## FDGMalloc
* Only allows for allocations on a warp-basis and cannot really free memory it seems

## CUDA
* Cuda Allocator cannot be resized once its size has been set once and allocations happened!
  * Probably needs a context reset for it to work

## ScatterAlloc

## Halloc
* Currently only works in warp-based manner

## Reg-Eff