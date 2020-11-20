# GPUMemManSurvey
Evaluating different memory managers for dynamic GPU memory

# Requirements
Tested on Windows 10, Arch Linux <5.9.9> as well as Manjaro <5.4>
* CUDA Toolkit
  * Tested on `10.1`, `10.2`, `11.0` and `11.1`
* C++ Compiler
  * Tested on
    * `gcc 9.0` and `gcc 10.2`
    * `VS 2019`
* boost (required for ScatterAlloc)
  * Tested with boost `1.66`
* CMake
  * Version `>= 3.16`, tested up until `3.18`
* Python
  * Tested with `Python 3.8`
  * Requires packages
    * `argparse` (`python pip install argparse`)
    * `numpy` (`python pip install numpy`)

# Setup Instructions
* `git clone --recursive https://github.com/GPUPeople/GPUMemManSurvey.git <chosen_directory>`
* `cd <chosen_directory>`
* `python init.py`
* Two options
  * If you want to build everything, call `python setupAll.py --cc XX` set correct CC (tested with 61, 70 and 75)
  * You can build each testcase, there is a `setup.py` in each tests folder, you can call each individually
    * `python setup.py --cc XX` set correct CC tested with 61, 70, 75
* On `Windows`
  * Use the `Developer PowerShell for VS 20XX` (`msbuild` is needed) to call the scripts
* To clean the build folders, simply call `python clean.py`

| Framework | Status | Link to Paper | Code |
|:---:|:---:|:---:| :---:|
| CUDA Device Allocator 		| :heavy_check_mark: | - | - |
| XMalloc (2010)				| :heavy_check_mark: | [Webpage](http://hdl.handle.net/2142/16137) | - |
| ScatterAlloc (2012) 			| :heavy_check_mark: | [Webpage](https://ieeexplore.ieee.org/document/6339604) | [GitHub - Repository](https://github.com/ax3l/scatteralloc) |
| FDGMalloc (2013) 			    | :question: 		 | [Webpage](https://www.gcc.tu-darmstadt.de/media/gcc/papers/Widmer_2013_FDM.pdf) | [Webpage](https://www.gcc.tu-darmstadt.de/home/proj/fdgmalloc/index.en.jsp) |
| Register Efficient (2014)	    | :heavy_check_mark: | [Webpage](https://diglib.eg.org/bitstream/handle/10.2312/hpg.20141090.019-027/019-027.pdf?sequence=1&isAllowed=y) | [Webpage](http://decibel.fi.muni.cz/~xvinkl/CMalloc/) |
| Halloc (2014)				    | :heavy_check_mark: | [Presentation](http://on-demand.gputechconf.com/gtc/2014/presentations/S4271-halloc-high-throughput-dynamic-memory-allocator.pdf) | [GitHub - Repository](https://github.com/canonizer/halloc) |
| DynaSOAr (2019)               | :x: 	 | [Webpage](https://drops.dagstuhl.de/opus/volltexte/2019/10809/pdf/LIPIcs-ECOOP-2019-17.pdf) | [GitHub - Repository](https://github.com/prg-titech/dynasoar)|
| Bulk-Sempaphore (2019)		| :watch: 			 | [Webpage](https://research.nvidia.com/publication/2019-02_Throughput-oriented-GPU-memory) | - |
| Ouroboros (2020)			    | :heavy_check_mark: | [Paper](https://dl.acm.org/doi/pdf/10.1145/3392717.3392742) | [GitHub - Repository](https://github.com/GPUPeople/Ouroboros) |

# Testcases
Each testcase is controlled and executed via python scripts, a commonality of all scripts is that to run the testcase, one has to pass `-runtest` to the script, to gather all results into one file one can pass `-genres`
## Allocation Testcases
### Single Threaded Allocation Performance
To test single threaded performance, navigate to `tests/alloc_tests` and call the script `test_allocation.py`
* `python test_allocation.py`

## Fragmentation Testcases

## Dynamic Graph Testcases

## Synthetic Testcases

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

# Test table 2080Ti
| | Build |Init|Reg.| Perf. 10K | Perf. 100K | Warp 10K | Warp 100K | Mix 10K | Mix 100K | Scale | Frag. 1|OOM|Graph Init.|Graph Up.|Graph Range|Synth.4-64|Synth.4-4096|
|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**CUDA**|:ab:|||||||||||||||||
|**XMalloc**|:a:|||||||||||||||||
|**ScatterAlloc**|:a:|||||||||||||||||
|**Halloc**|:a:|||||||||||||||||
|**Reg-Eff - A**|:a:|||||||||||||||||
|**Reg-Eff - AW**|:a:|||||||||||||||||
|**Reg-Eff - C**|:a:|||||||||||||||||
|**Reg-Eff - CF**|:a:|||||||||||||||||
|**Reg-Eff - CM**|:a:|||||||||||||||||
|**Reg-Eff - CFM**|:a:|||||||||||||||||
|**Oro - P - S**|:ab:|||||||||||||||||
|**Oro - P - VA**|:ab:|||||||||||||||||
|**Oro - P - VL**|:ab:|||||||||||||||||
|**Oro - C - S**|:ab:|||||||||||||||||
|**Oro - C - VA**|:ab:|||||||||||||||||
|**Oro - C - VL**|:ab:|||||||||||||||||
