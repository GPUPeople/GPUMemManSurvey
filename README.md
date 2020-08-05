# GPUMemManSurvey
Evaluating different memory managers for dynamic GPU memory

# Instructions
* `git clone https://github.com/GPUPeople/GPUMemManSurvey.git <chosen_directory>`
* `cd <chosen_directory>`
* `git submodule init`
* `git submodule update`
* `mkdir build && cd build`
* `ccmake ..` -> set correct CC (some only build in sync mode)
* `make`
* Similar procedure for all tests, just navigate to, e.g. `tests/alloc_tests` and create build folder the same as before

| Framework | Status | Link to Paper | Code |
|:---:|:---:|:---:| :---:|
| CUDA Device Allocator | :heavy_check_mark: 	| - | - |
| XMalloc (2010)				| 	:heavy_check_mark: 	| [Webpage](http://hdl.handle.net/2142/16137) | - |
| ScatterAlloc (2012) 			| :heavy_check_mark: 	| [Webpage](https://ieeexplore.ieee.org/document/6339604) | [GitHub - Repository](https://github.com/ax3l/scatteralloc) |
| FDGMalloc (2013) 			    |  :question: 	| [Webpage](https://www.gcc.tu-darmstadt.de/media/gcc/papers/Widmer_2013_FDM.pdf) | [Webpage](https://www.gcc.tu-darmstadt.de/home/proj/fdgmalloc/index.en.jsp) |
| Register Efficient (2014)	    | :heavy_check_mark:	| [Webpage](https://diglib.eg.org/bitstream/handle/10.2312/hpg.20141090.019-027/019-027.pdf?sequence=1&isAllowed=y) | [Webpage](http://decibel.fi.muni.cz/~xvinkl/CMalloc/) |
| Halloc (2014)				    |  :heavy_check_mark: 	| [Presentation](http://on-demand.gputechconf.com/gtc/2014/presentations/S4271-halloc-high-throughput-dynamic-memory-allocator.pdf) | [GitHub - Repository](https://github.com/canonizer/halloc) |
| DynaSOAr (2019)               |   Not usable   | [Webpage](https://drops.dagstuhl.de/opus/volltexte/2019/10809/pdf/LIPIcs-ECOOP-2019-17.pdf) | [GitHub - Repository](https://github.com/prg-titech/dynasoar)|
| Bulk-Sempaphore (2019)		| 	:watch: 	| [Webpage](https://research.nvidia.com/publication/2019-02_Throughput-oriented-GPU-memory) | - |
| Orooboros (2020)			    | :heavy_check_mark:	| [Paper](https://dl.acm.org/doi/pdf/10.1145/3392717.3392742) | [GitHub - Repository](https://github.com/GPUPeople/Orooboros) |

# Notes to individual approaches
## FDGMalloc
* Currently still buggy and fails early in multiple cases
* Also, has several limitations, only can do warp-based allocation, cannot really free memory, only tidy up full warp, etc.

## Register-Efficient
* Has the option to `coalesce` within a warp -> adds up all allocations and then only does one large allocation
  * Does not work with free
    * Not sure if this is not implemented correctly, but this immediately hangs, as then possible threads are trying to delete some memory that is only part of a larger allocation and cannot be individually freed
  * Even without free, not all sizes sime to work properly
* Currently, `coalesceWarp` is turned off so that it works

# Test table TITAN V

| | Build |Init|Reg.| Perf. 10K | Perf. 100K | Warp 10K | Warp 100K | Mix 10K | Mix 100K | Scale | Frag. 1|OOM|Graph Init.|Graph Up.|Graph Range|Synth.|
|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**CUDA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**XMalloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:boom:|:heavy_check_mark:|-|:heavy_check_mark:|:boom:|:boom:|:boom:|:boom:|:question:|:boom:|:boom:|:heavy_check_mark:|
|**ScatterAlloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Halloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - A**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:boom:|:boom:|:boom:|:heavy_check_mark:|
|**Reg-Eff - AW**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:boom:|:boom:|:boom:|:heavy_check_mark:|
|**Reg-Eff - C**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:watch:|-|:boom:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:question:|:boom:|:boom:|:boom:|
|**Reg-Eff - CF**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|-|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:question:|:boom:|:boom:|:heavy_check_mark:|
|**Reg-Eff - CM**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|-|-|:heavy_check_mark:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:question:|:boom:|:boom:|:heavy_check_mark:|
|**Reg-Eff - CFM**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|-|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:question:|:boom:|:boom:|:heavy_check_mark:|
|**Oro - P - S**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - P - VA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - P - VL**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - C - S**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - C - VA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:question:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - C - VL**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|


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

## Notes Synthetic
* `Workload`
  * `Oro - P - VL` fails after 1024
  * `Reg-Eff-C` fails after `8192`
* Could also test how write performance to that memory region is, not only the allocation speed


