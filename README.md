# GPUMemManSurvey
Evaluating different memory managers for dynamic GPU memory

# Instructions
* `git clone --recursive https://github.com/GPUPeople/GPUMemManSurvey.git <chosen_directory>`
* `cd <chosen_directory>`
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
## XMalloc
`XMalloc` is currently not publicly available, but an existing implementation is included in this repository which was obtained a few years ago, the license file is included `<frameworks/xmalloc/LICENSE>`

## ScatterAlloc
`ScatterAlloc` is cloned from a [public GitHub - Repository](https://github.com/ax3l/scatteralloc)

## FDGMalloc
`FDGMalloc` is included as well, the original source code can be downloaded from their respecive [webpage](https://www.gcc.tu-darmstadt.de/home/proj/fdgmalloc/index.en.jsp).
Unfortunately, it does not seem to work even for simple cases and as other limitations do not fit the description of a general purpose memory manager, it is ommited in all tests

## Register-Efficient
`Register-Efficient` is downloaded from their [webpage](http://decibel.fi.muni.cz/~xvinkl/CMalloc/)
* Has the option to `coalesce` within a warp -> adds up all allocations and then only does one large allocation
  * Does not work with free
    * Not sure if this is not implemented correctly, but this immediately hangs, as then possible threads are trying to delete some memory that is only part of a larger allocation and cannot be individually freed
  * Even without free, not all sizes sime to work properly
* Currently, `coalesceWarp` is turned off so that it works

## Halloc
`Halloc` is cloned from a [public GitHub - Repository](https://github.com/canonizer/halloc)

## DynaSOAr
`DynaSOAr` cannot be used as a general purpose memory allocator, as it can only manage pre-defined objects instantiated in their specific format

## Bulk-Semaphore
The two memory managers described in the [paper](https://research.nvidia.com/publication/2019-02_Throughput-oriented-GPU-memory) are unfortunately not publicly available

## Ouroboros
All variants of `Ouroboros` can be cloned from their [public GitHub - Repository](https://github.com/GPUPeople/Orooboros)


# Test table TITAN V

| | Build |Init|Reg.| Perf. 10K | Perf. 100K | Warp 10K | Warp 100K | Mix 10K | Mix 100K | Scale | Frag. 1|OOM|Graph Init.|Graph Up.|Graph Range|Synth.4-64|Synth.4-4096|
|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**CUDA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**XMalloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:boom:|:boom:|:boom:|:boom:|:question:|:boom:|:boom:|:heavy_check_mark:|:heavy_check_mark:|
|**ScatterAlloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Halloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - A**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - AW**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - C**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|-|:boom:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:boom:|:boom:|:boom:|:boom:|
|**Reg-Eff - CF**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - CM**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - CFM**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - P - S**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - P - VA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - P - VL**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - C - S**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - C - VA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - C - VL**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:watch:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|

