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
| FDGMalloc (2013) 			    |  :watch: 	| [Webpage](https://www.gcc.tu-darmstadt.de/media/gcc/papers/Widmer_2013_FDM.pdf) | [Webpage](https://www.gcc.tu-darmstadt.de/home/proj/fdgmalloc/index.en.jsp) |
| Register Efficient (2014)	    | :watch:	| [Webpage](https://diglib.eg.org/bitstream/handle/10.2312/hpg.20141090.019-027/019-027.pdf?sequence=1&isAllowed=y) | [Webpage](http://decibel.fi.muni.cz/~xvinkl/CMalloc/) |
| Halloc (2014)				    |  :heavy_check_mark: 	| [Presentation](http://on-demand.gputechconf.com/gtc/2014/presentations/S4271-halloc-high-throughput-dynamic-memory-allocator.pdf) | [GitHub - Repository](https://github.com/canonizer/halloc) |
| DynaSOAr (2019)               |   Not usable   | [Webpage](https://drops.dagstuhl.de/opus/volltexte/2019/10809/pdf/LIPIcs-ECOOP-2019-17.pdf) | [GitHub - Repository](https://github.com/prg-titech/dynasoar)|
| Bulk-Sempaphore (2019)		| 	:watch: 	| [Webpage](https://research.nvidia.com/publication/2019-02_Throughput-oriented-GPU-memory) | - |
| Ouroboros (2020)			    | :heavy_check_mark:	| [Paper](https://dl.acm.org/doi/pdf/10.1145/3392717.3392742) | [GitHub - Repository](https://github.com/GPUPeople/Ouroboros) |

# Test table

| | CUDA | ScatterAlloc | Halloc | XMalloc | Ouroboros | Reg-Eff | FDGMalloc | BulkAlloc|
|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|
| **Build** (Sync :a: - Async :b:) | :ab: | :a: | :a: | :a: | :ab: | :a:| :a: | :b:|
|Performance 10K|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|-|-|-|
|Performance 100K|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_multiplication_x:|:watch:|-|-|-|
|Mixed 10K|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|-|-|-|
|Mixed 100K|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_multiplication_x:|-|-|-|-|
|Scaling 2¹ - 2²⁰|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_multiplication_x:|-|-|-|-|
|Fragmentation 1|-|-|-|-|-|-|-|-|
|Fragmentation 2|-|-|-|-|-|-|-|-|
|Graph Initialization|-|-|-|-|-|-|-|-|
|Graph Updates|-|-|-|-|-|-|-|-|

## Notes Performance
* `Ouroboros` has issues for page-based standard and VA for 1 million allocations?
* `XMalloc` fails after about `256` Bytes for `100.000` allocations

## Notes Scaling
* Since `Halloc` delegates the larger allocations to `CUDA`, it fails a little bit early, since it has to split its memory space, standard is `75% Halloc / 25% CUDA`
* `XMalloc` start failing right around `256` Bytes

## Notes Mixed
* `XMalloc`
  *  for `100.000` allocations with range `512-8192`
  * `10.000` works without a problem

## Notes Fragmentation
* Text

## Notes Dynamic Graph
* Text