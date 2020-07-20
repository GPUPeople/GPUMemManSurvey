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

# Work in Progress!

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
| Ouroboros (2020)			    | :heavy_check_mark:	| [Paper](https://dl.acm.org/doi/pdf/10.1145/3392717.3392742) | [GitHub - Repository](https://github.com/GPUPeople/Ouroboros) |

# Notes to individual approaches
## FDGMalloc
* Currently still buggy and fails early in multiple cases
* Also, has several limitations, only can do warp-based allocation, cannot really free memory, only tidy up full warp, etc.

# Test table TITAN V

| | Sync :a: - Async :b: |Init| Perf. 10K | Perf. 100K | Mixed 10K | Mixed 100K | Scaling 2¹ - 2²⁰| Frag. 1|Frag. 2|Graph Init.|Graph Updates|Synthetic|
|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**CUDA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|-|-|-|
|**ScatterAlloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|-|-|-|
|**Halloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:question:|:heavy_check_mark:|-|-|-|-|
|**XMalloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:boom:|:heavy_check_mark:|:boom:|:boom:|-|-|-|-|-|
|**Our - P - S**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|-|-|:heavy_check_mark:|-|-|-|-|
|**Our - P - VA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|-|-|:heavy_check_mark:|-|-|-|-|
|**Our - P - VL**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|-|-|:heavy_check_mark:|-|-|-|-|
|**Our - C - S**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|-|-|:heavy_check_mark:|-|-|-|-|
|**Our - C - VA**|:ab:|:heavy_check_mark:|-| -|-|-|-|-|-|-|-|-|
|**Our - C - VL**|:ab:|:heavy_check_mark:|-| -|-|-|-|-|-|-|-|-|
|**Reg-Eff - A**|:a:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|-|-|-|:watch:|-|-|-|-|
|**Reg-Eff - AW**|:a:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|-|-|-|:watch:|-|-|-|-|
|**Reg-Eff - C**|:a:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|-|-|-|:watch:|-|-|-|-|
|**Reg-Eff - CF**|:a:|:heavy_check_mark:|:heavy_check_mark:| :watch:|-|-|-|:watch:|-|-|-|-|
|**Reg-Eff - CM**|:a:|:heavy_check_mark:|:heavy_check_mark:| :watch:|-|-|-|:watch:|-|-|-|-|
|**Reg-Eff - CFM**|:a:|:heavy_check_mark:|:heavy_check_mark:| :watch:|-|-|-|:watch:|-|-|-|-|
|**FDGMalloc**|:a:|-| -|-|-|-|-|-|-|-|-|-|
|**BulkAlloc**|:b:|-| -|-|-|-|-|-|-|-|-|-|


## Notes Performance
* `100.000`
  * Reg-Eff-CF failed at `8192`
  * Reg-Eff-CFM fails a few times after `7376`
  * Reg-Eff-CM fails a few times after `6768`

## Notes Scaling

## Notes Mixed

## Notes Fragmentation
* Stopped for `Reg-Eff-C` at `4088`

## Notes Dynamic Graph

## Notes Synthetic
* Could also test how write performance to that memory region is, not only the allocation speed


