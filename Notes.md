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


These methods get slower over time during allocation if no free happens

### Circular

### Circular Fused
Text
### Circular Multi

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

## Notes Performance
* `Performance`
  * `10.000`
    * `Oro - C - VA` fail with increasing likelihood for the larger allocation sizes
    * `Oro - C - VL` seems to work, but is quite slow
  * `100.000`
    * Reg-Eff-CF failed at `8192`
    * Reg-Eff-CFM fails a few times after `7376`
    * Reg-Eff-CM fails a few times after `6768`

* `Mixed Performance`
  * `10.000`
    * `Reg-Eff-C` fails in between for sizes `32,64,256`
    * `Oro - P - VL` fails after `32`
  * `100.000`
    * `Reg-Eff-C` fails after `16`
    * `Reg-Eff-CM` fails after `1024`
    * `Reg-Eff-CFM` fails after `4096`
    * `Oro - C - VA` fails after `2048` -> got manual results with less iterations
    * `Oro - P - VL` fails after `32`

## Notes Scaling
* `Oro - C - S` fails for the largest two sizes `4096` and `8192` the largest two thread counts `500.000` and `1.000.000`

## Notes Mixed

## Notes Fragmentation
* `Fragmentation`
  * Missing still for `Reg-Eff-CF`, `Reg-Eff-CM` and `Reg-Eff-CFM`
* `OOM`
  * `Oro - C - VA` and `Oro - C - VL` become really slow after a few hundred iterations, probably not moving the front correctly.
  * `Reg-Eff-A*` also align to 16 Bytes internally, hence they don't get to maximum in the beginning
  * `Reg-Eff-C*` are painfully slow, hence typically are reigned in by the timeout
    * Also get slower with every passing iteration

## Notes Dynamic Graph
* Graph Stats captured :heavy_check_mark:
* `Init`
  * `Oro - P - V*` not everything works, `VA` dies sometimes with `died in freePage`
    * `VA` is missing `333SP`, `hugetric` and `adaptive`
    * `VL` is missing `caidaRouterLevel`, `delaunay_n20`
* `Update`
  * `Reg-Eff` does not return 16-byte aligned memory, hence copying data over vectorized does not work

## Notes Synthetic
* `Workload`
  * `Oro - P - VL` fails after 1024
  * `Reg-Eff-C` fails after `8192`
* Could also test how write performance to that memory region is, not only the allocation speed