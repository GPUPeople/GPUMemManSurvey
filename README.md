# GPUMemManSurvey
Evaluating different memory managers for dynamic GPU memory allocation.

# Requirements
The framework was tested on Windows 10, Arch Linux <5.9.9> as well as Manjaro <5.4>
* **CUDA Toolkit**
  * Tested on `10.1`, `10.2`, `11.0` and `11.1`
    * Windows [Download](https://developer.nvidia.com/cuda-downloads)
    * Arch Linux (`pacman -S cuda`)
* **C++ Compiler**
  * Tested on
    * `gcc 9.0` and `gcc 10.2`
      * Arch Linux (`pacman -S gcc`)
    * `VS 2019`
      * [Download](https://visualstudio.microsoft.com/vs/community/)
* **boost** (required for ScatterAlloc)
  * Tested with boost `1.66` and `1.74`
    * Windows [Download](https://www.boost.org/users/download/)
      * Set the installed location in `BaseCMake.cmake`
    * Arch Linux (`pacman -S boost`)
* **CMake**
  * Version `>= 3.16`, tested with `3.18`
    * Windows [Download](https://cmake.org/download/)
    * Arch Linux (`pacman -S cmake`)

* **Python**
  * Tested with `Python 3.8` and `Python 3.9`
    * Windows [Download](https://www.python.org/downloads/release/python-390/) or download via Windows Store
    * Arch Linux (`pacman -S python`)
  * Requires packages
    * `argparse` (`python pip -m install argparse`)

# Setup Instructions
* Make sure all requirements are installed and configured correctly
  * On `Windows` also set the correct boost path in `BaseCMake.cmake`
* **Setup**
  * **Option A:** Setup from Archive
    * Extract archive
    * In top-level directory, call
      * `git submodule init`
      * `git submodule update`
  * **Option B:** Setup from GitHub
    * `git clone --recursive -b AEsubmission https://github.com/GPUPeople/GPUMemManSurvey.git`
* `python init.py`
* **Install**
  * On `Windows` use the `Developer PowerShell for VS 20XX` (`msbuild` is needed) to call the scripts
  * **Option A:**
    * If you want to build everything, call `python setupAll.py --cc XX`, set correct CC (tested with 61, 70 and 75)
  * **OptionB:**
    * You can build each testcase separately, there is a `setup.py` in each tests folder
      * `python setup.py --cc XX`, set correct CC tested with 61, 70, 75
* To **clean/reset** the build folders, simply call `python cleanAll.py`
  * Once again, there is a separate `clean.py` in every test subfolder

# Testcase Instructions
To run a representative testsuite, simply call
* `python testAll.py -mem_size 8 -device 0 -runtest -genres`
  * The memory size is in GB
  * The device ID of the device to use (has to match with the CC passed in build stage)
  
These runtime measures were measured for the limited testcase as setup in `testAll.py` on a TITAN V and an Intel Core i7-8700X on Windows 10 and Manjaro respectively.

| Task | Time (min:sec) - Linux| Time (min:sec) - Windows|
|:---:|:---:|:---:|
|Overall|28 min 28 sec|1 h 3 min 47 sec|
|Build | 9 min 45 sec | 28 min 15 sec |
|Test All |18 min 43 sec| 35 min 32 sec|
|-|-|-|
| Allocation|1 min 48 sec| 2 min 39 sec |
| Mixed Allocation | 0 min 35 sec | 2 min 46 sec |
| Scaling | 2 min 52 sec | 4 min 03 sec |
| Fragmentation | 1 min 47 sec | 2 min 36 sec |
| Out-of-Memory | 7 min 15 sec | 8 min 21 sec |
|Graph Init| 0 min 11 sec | 1 min 44 sec |
| Graph Update | 0 min 11 sec | 1 min 44 sec |
| Graph Update Range | 0 min 11 sec | 1 min 44 sec|
| Register Footprint | 0 min 03 sec | 0 min 05 sec| 
| Initialization | 0 min 05 sec | 0 min 12 sec |
| Synthetic Workload | 1 min 58 sec | 4 min 50 sec |
| Synthetic Workload Write | 1 min 57 sec | 4 min 48 sec |

The framework **does not perform many sanity checks**, please read the documentation first if something is not working as expected if some parameter was not configured correctly for example.

# Folder Structure
* `frameworks` -> includes code for all frameworks
* `externals` -> not all CUDA versions have CUB yet
* `include` / `src` / `scripts` -> framework code
* `tests` -> all test implementations
  * `alloc_test` -> all allocation tests
    * `test_allocation.py`
    * `test_mixed_allocation.py`
    * `test_scaling.py`
  * `frag_test` -> all memory/fragmentation tests
    * `test_fragmenation.py`
    * `test_oom.py`
  * `graph_test` -> all graph tests
    * `test_graph_init.py`
    * `test_graph_update.py`
  * `synth_test` -> all synthetic tests
    * `test_registers.py`
    * `test_synth_init.py`
    * `test_synth_workload.py`

# Frameworks
| Framework | Status | Paper | Code |
|:---:|:---:|:---:| :---:|
| CUDA Device Allocator 		| :heavy_check_mark: | - | - |
| XMalloc (2010)				| :heavy_check_mark: | [Webpage](http://hdl.handle.net/2142/16137) | - |
| ScatterAlloc (2012) 			| :heavy_check_mark: | [Webpage](https://ieeexplore.ieee.org/document/6339604) | [GitHub - Repository](https://github.com/ax3l/scatteralloc) |
| FDGMalloc (2013) 			    | :question: 		 | [Webpage](https://www.gcc.tu-darmstadt.de/media/gcc/papers/Widmer_2013_FDM.pdf) | [Webpage](https://www.gcc.tu-darmstadt.de/home/proj/fdgmalloc/index.en.jsp) |
| Register Efficient (2014)	    | :heavy_check_mark: | [Webpage](https://diglib.eg.org/bitstream/handle/10.2312/hpg.20141090.019-027/019-027.pdf?sequence=1&isAllowed=y) | [Webpage](http://decibel.fi.muni.cz/~xvinkl/CMalloc/) |
| Halloc (2014)				    | :heavy_check_mark: | [Presentation](http://on-demand.gputechconf.com/gtc/2014/presentations/S4271-halloc-high-throughput-dynamic-memory-allocator.pdf) | [GitHub - Repository](https://github.com/canonizer/halloc) |
| DynaSOAr (2019)               | :x: 	 | [Webpage](https://drops.dagstuhl.de/opus/volltexte/2019/10809/pdf/LIPIcs-ECOOP-2019-17.pdf) | [GitHub - Repository](https://github.com/prg-titech/dynasoar)|
| Bulk-Sempaphore (2019)		| :x: 			 | [Webpage](https://research.nvidia.com/publication/2019-02_Throughput-oriented-GPU-memory) | - |
| Ouroboros (2020)			    | :heavy_check_mark: | [Paper](https://dl.acm.org/doi/pdf/10.1145/3392717.3392742) | [GitHub - Repository](https://github.com/GPUPeople/Ouroboros) |

# Testcases
Each testcase is controlled and executed via python scripts, a commonality of all scripts is that to run the testcase, one has to pass `-runtest` to the script, to gather all results into one file one can pass `-genres`.
Pass `-h` to print a help screen with all parameters.
All testcases get a `-device` parameter to control which device should execute the GPU code (e.g. `0`) and how much memory on this device should be reserved for the memory manager, specified via `-allocsize` (size on GB).

## Data to Plot - Map
This table shows which test file can be used to generate which plot used in the paper.

| Figure/Section | Script | Command |
|:---:|:---:|:---:|
|Sec. `4.1`|`test_registers.py`|`python test_registers.py -t o+s+h+c+r+x -runtest -genres -allocsize 8 -device 0`|
|Sec. `4.1`|`test_synth_init.py`|`python test_synth_init.py -t o+s+h+c+r+x -runtest -genres -allocsize 8 -device 0`|
|Fig. `9.a`|`test_allocation.py`|`python test_allocation.py -t o+s+h+c+r+x -num 100000 -range 4-8192 -iter 100 -runtest -genres -timeout 120 -allocsize 8 -device 0`|
|Fig. `9.b`|`test_allocation.py`|`python test_allocation.py -t o+s+h+c+r+x -num 100000 -range 4-8192 -iter 100 -runtest -genres -timeout 120 -allocsize 8 -device 0`|
|Fig. `9.c`|`test_allocation.py`|`python test_allocation.py -t o+s+h+c+r+x -num 10000 -range 4-8192 -iter 100 -runtest -genres -warp -timeout 120 -allocsize 8 -device 0`|
|Fig. `9.d`|`test_mixed_allocation.py`|`python test_mixed_allocation.py -t o+s+h+c+r+x -num 10000 -range 4-8192 -iter 100 -runtest -genres -timeout 120 -allocsize 8 -device 0`|
|Fig. `10.x`|`test_scaling.py`|`python test_scaling.py -t o+s+h+c+r+x -byterange 4-8192 -threadrange 0-20 -iter 100 -runtest -genres -timeout 300 -allocsize 8 -device 0`|
|Fig. `11.a`|`test_fragmentation.py`|`python test_fragmentation.py -t o+s+h+c+r+x -num 100000 -range 4-8192 -iter 100 -runtest -genres -timeout 60 -allocsize 8 -device 0`|
|Fig. `11.b`|`test_oom.py`|`python test_oom.py -t o+s+h+c+r+x -num 100000 -range 4-8192 -runtest -genres -timeout 3600 -allocsize 2 -device 0`|
|Fig. `11.c`|`test_synth_workload.py`|`python test_synth_workload.py -t o+s+h+c+r+x -threadrange 0-20 -range 4-64 -iter 100 -runtest -genres -timeout 300 -allocsize 8 -device 0`|
|Fig. `11.d`|`test_synth_workload.py`|`python test_synth_workload.py -t o+s+h+c+r+x -threadrange 0-20 -range 4-4096 -iter 100 -runtest -genres -timeout 300 -allocsize 8 -device 0`|
|Fig. `11.e`|`test_synth_workload.py`|`python test_synth_workload.py -t o+s+h+c+r+x -threadrange 0-20 -range 4-64 -iter 100 -runtest -genres -timeout 300 -allocsize 8 -device 0 -testwrite`|
|Fig. `11.f`|`test_graph_init.py`|`python test_graph_init.py -t o+s+h+c+r+x -configfile config_init.json -runtest -genres -timeout 600 -allocsize 8 -device 0`|
|Fig. `11.g`|`test_graph_update.py`|`python test_graph_update.py -t o+s+h+c+r+x -configfile config_update_range.json -runtest -genres -timeout 600 -allocsize 8 -device 0`|


## Allocation Testcases
### Single Threaded / Single Warp Allocation Performance
To test single threaded or single warp performance, navigate to `tests/alloc_tests` and call the script `test_allocation.py`
* `python test_allocation.py -t o+s+h+c+r+x -num 10000 -range 4-64 -iter 50 -runtest -timeout 60 -allocsize 8 -device 0`
  * This will start `10000` threads, each of them will start by allocating `4` Bytes and then increase linearly up to `64` Bytes

This will generate one csv file for each approach with `mean`, `min`, `max`, `median` performance averaged over the number of iterations.
To generate one file with all approaches already executed, pass option `-genres` instead or additional to `-runtest`.

| Option | Parameter-Example | Description |
|:---:|:---:|:---:|
|`-t`|`o+s+h+c+f+r+x`|Specify which frameworks to test, first letter of approach separated by `+`, e.g. `c` : cuda or `s` : scatteralloc|
|`-num`|`10000`| How many threads/warps to start, e.g. `10000`|
|`-range`|`4-64`|Which allocation range to test, e.g. `4-64` Bytes|
|`-iter`|`50`|How often to run test and average over runs, e.g. `50`|
|`-runtest`||Pass this flag to execute the testcase and run the approaches|
|`-genres`||Pass this flag to gather all results from existing csv files into one|
|`-warp`||Pass this flag to start 1 warp instead of 1 warp per allocation|
|`-timeout`|`120`|Timeout in seconds, each individual testcase run will be canceled after this timeout, **default** is `600`|
|`-allocsize`|`8`|How large the manageable memory ares per memory manager should be in `GB`|
|`-device`|`0`|Which GPU device to use|

### Mixed Range Allocation Performance
To test allocation performance when threads are allocating with different sizes (constrained by a maximum/minimum allocation size), navigate to `tests/alloc_tests` and call the script `test_mixed_allocation.py`
* `python test_mixed_allocation.py -t o+s+h+c+r+x -num 10000 -range 4-64 -iter 50 -runtest -timeout 60 -allocsize 8 -device 0`
	* This will start `10000` threads, each of them will allocate in the range of `4-64` Bytes

This will generate one csv file for each approach with `mean`, `min`, `max`, `median` performance averaged over the number of iterations.
To generate one file with all approaches already executed, pass option `-genres` instead or additional to `-runtest`.

| Option | Parameter-Example | Description |
|:---:|:---:|:---:|
|`-t`|`o+s+h+c+f+r+x`|Specify which frameworks to test, first letter of approach separated by `+`, e.g. `c` : cuda or `s` : scatteralloc|
|`-num`|`10000`| How many threads/warps to start, e.g. `10000`|
|`-range`|`4-64`|Which allocation range to test, e.g. `4-64` Bytes|
|`-iter`|`50`|How often to run test and average over runs, e.g. `50`|
|`-runtest`||Pass this flag to execute the testcase and run the approaches|
|`-genres`||Pass this flag to gather all results from existing csv files into one|
|`-warp`||Pass this flag to start 1 warp instead of 1 warp per allocation|
|`-timeout`|`120`|Timeout in seconds, each individual testcase run will be canceled after this timeout, **default** is `600`|
|`-allocsize`|`8`|How large the manageable memory ares per memory manager should be in `GB`|
|`-device`|`0`|Which GPU device to use|

### Performance Scaling
To test performance scaling over a changing number of threads, navigate to `tests/alloc_tests` and call the script `test_scaling.py`
* `python test_scaling.py -t o+s+h+c+r+x -byterange 4-64 -threadrange 0-10 -iter 50 -runtest -timeout 60 -allocsize 8 -device 0`
  * This will start with `2⁰` threads up to `2¹⁰` threads, testing all powers of 2 in-between, and for each number of threads test the range `4-64` Bytes

This will generate one csv file for each approach and for each number of threads with `mean`, `min`, `max`, `median` performance averaged over the number of iterations.
To generate one file with all approaches already executed, pass option `-genres` instead or additional to `-runtest`.
Can also be started with one warp per allocation by passing `-warp`.

| Option | Parameter-Example | Description |
|:---:|:---:|:---:|
|`-t`|`o+s+h+c+f+r+x`|Specify which frameworks to test, first letter of approach separated by `+`, e.g. `c` : cuda or `s` : scatteralloc|
|`-threadrange`|`0-10`| The range of threads to test, given as a power of 2, e.g. `0-10` would test `2⁰`, `2¹`, ..., `2¹⁰` threads for the given `-byterange`|
|`-byterange`|`4-64`|Which allocation range to test, e.g. `4-64` Bytes|
|`-iter`|`50`|How often to run test and average over runs, e.g. `50`|
|`-runtest`||Pass this flag to execute the testcase and run the approaches|
|`-genres`||Pass this flag to gather all results from existing csv files into one|
|`-warp`||Pass this flag to start 1 warp instead of 1 warp per allocation|
|`-timeout`|`120`|Timeout in seconds, each individual testcase run will be canceled after this timeout, **default** is `600`|
|`-allocsize`|`8`|How large the manageable memory ares per memory manager should be in `GB`|
|`-device`|`0`|Which GPU device to use|

## Fragmentation Testcases
### Memory Fragmentation Testcase
This testcase tests the fragmentation of the returned addresses of a given allocation by reporting the maximum address range returned by each allocating thread. It also tracks the static maximum over a number of iterations.
It continues to allocate and free a number of allocations for the number of `-iter` and returns those ranges.
* `python test_fragmentation.py -t o+s+h+c+r+x -num 10000 -range 4-64 -iter 50 -runtest -timeout 60 -allocsize 8 -device 0`
  * This will start `10000` threads, each of them will start by allocating `4` Bytes and then increase linearly up to `64` Bytes, reporting the current range and static maximum range

This will generate one csv file for each approach with `min address range`, `max address range`, `min address range (static)` and `max address range (max)`.
To generate one file with all approaches already executed, pass option `-genres` instead or additional to `-runtest`.

| Option | Parameter-Example | Description |
|:---:|:---:|:---:|
|`-t`|`o+s+h+c+f+r+x`|Specify which frameworks to test, first letter of approach separated by `+`, e.g. `c` : cuda or `s` : scatteralloc|
|`-num`|`10000`| Starts `10000` threads|
|`-range`|`4-64`|Which allocation range to test, e.g. `4-64` Bytes|
|`-iter`|`50`|How often to run test and average over runs, e.g. `50`|
|`-runtest`||Pass this flag to execute the testcase and run the approaches|
|`-genres`||Pass this flag to gather all results from existing csv files into one|
|`-timeout`|`120`|Timeout in seconds, each individual testcase run will be canceled after this timeout, **default** is `600`|
|`-allocsize`|`8`|How large the manageable memory ares per memory manager should be in `GB`|
|`-device`|`0`|Which GPU device to use|

### Out-of-Memory Testcase
Tests out-of-memory behavior for a range of allocation sizes, hence how efficient the memory is utilized. The range will be sampled for each power of 2 in-between the given `-range`
* `python test_oom.py -t o+s+h+c+r+x -num 10000 -range 4-64 -runtest -timeout 60 -allocsize 8 -device 0`
  * This starts `10000` allocating threads, tests powers of 2 in the range `4-64` and continues to allocate until out-of-memory is reported, recording the number of iterations in the csv file

This will generate one csv file for each approach and records the number of successful iterations. 
To generate one file with all approaches already executed, pass option `-genres` instead or additional to `-runtest`.

| Option | Parameter-Example | Description |
|:---:|:---:|:---:|
|`-t`|`o+s+h+c+f+r+x`|Specify which frameworks to test, first letter of approach separated by `+`, e.g. `c` : cuda or `s` : scatteralloc|
|`-num`|`10000`| Starts `10000` threads|
|`-range`|`4-64`|Which allocation range to test, e.g. `4-64` Bytes|
|`-runtest`||Pass this flag to execute the testcase and run the approaches|
|`-genres`||Pass this flag to gather all results from existing csv files into one|
|`-timeout`|`120`|Timeout in seconds, each individual testcase run will be canceled after this timeout, **default** is `600`|
|`-allocsize`|`8`|How large the manageable memory ares per memory manager should be in `GB`|
|`-device`|`0`|Which GPU device to use|

## Dynamic Graph Testcases
Graph testcases require a `config.json` file, which has the following parameters
| Parameter | Value-Example | Description |
|:---:|:---:|:---:|
|`iterations`|`10`| How many iterations to do, in which the graph is initialized new, e.g. `10`|
|`update_iterations`|`10`| How many edge update iterations to perform|
|`batch_size`|`10000`| How many edges to insert each iteration|
|`range`|`0`|If `range` is `0`, the edge sources are randomly distributed amongst the available vertices, if `> 0`, then updates will be focused on this smaller range, which is shifted over the graph `update_iterations` times|
|`test_init`|`true`|If this is set to true, only initialization will be measured.|
|`verify`|`false`|If this is set to true, then each operation will be verified against a host dynamic graph `-> takes quite a long time`|
|`realistic_deletion`|`false`|If this is set to `false`, the deletion operation will delete exactly the same edges that where introduced during the insertion opertion. Otherwise, random edges will be selected from the graph.|

The testcase can handle `.mtx` (Matrix Market Format) files, which can be downloaded from the [SuiteSparse Collection](https://sparse.tamu.edu/), and will automatically convert each file into a more efficient binary format, which greatly improves load times for multiple runs.

### Graph Initialization
This testcase will test dynamic graph initialization. One has to pass a configfile as described above, the list of graphs to test is given at the top of `test_graph_init.py`. 
* `python test_graph_init.py -t o+s+h+c+r+x -configfile config_init.json -runtest -timeout 120 -allocsize 8 -device 0`
  * Tests initialization performance for all graphs noted in `test_graph_init.py`, configured according to `config_init.json`

| Option | Parameter-Example | Description |
|:---:|:---:|:---:|
|`-t`|`o+s+h+c+f+r+x`|Specify which frameworks to test, first letter of approach separated by `+`, e.g. `c` : cuda or `s` : scatteralloc|
|`-configfile`|`config_init.json`|All the configuration details for this testcase, as described above|
|`-graphstats`||Writes out graph statistics, does **not** run the actual testcase afterwards|
|`-runtest`||Pass this flag to execute the testcase and run the approaches|
|`-genres`||Pass this flag to gather all results from existing csv files into one|
|`-timeout`|`120`|Timeout in seconds, each individual testcase run will be canceled after this timeout, **default** is `600`|
|`-allocsize`|`8`|How large the manageable memory ares per memory manager should be in `GB`|
|`-device`|`0`|Which GPU device to use|

### Graph Edge Updates
This testcase will test dynamic graph updates. One has to pass a configfile as described above, the list of graphs to test is given at the top of `test_graph_update.py`. 
* `python test_graph_update.py -t o+s+h+c+r+x -configfile config_update.json -runtest -timeout 120 -allocsize 8 -device 0`
  * Tests edge update performance for all graphs noted in `test_graph_update.py`, configured according to `config_update.json`, this will test random edge updates
* `python test_graph_update.py -t o+s+h+c+r+x -configfile config_update_range.json -runtest -timeout 120 -allocsize 8 -device 0`
  * Tests edge update performance for all graphs noted in `test_graph_update.py`, configured according to `config_update_range.json`, this will test pressured edge updates with a given range of source vertices shifted over the graph

| Option | Parameter-Example | Description |
|:---:|:---:|:---:|
|`-t`|`o+s+h+c+f+r+x`|Specify which frameworks to test, first letter of approach separated by `+`, e.g. `c` : cuda or `s` : scatteralloc|
|`-configfile`|`config_update.json`|All the configuration details for this testcase, as described above|
|`-graphstats`||Writes out graph statistics, does **not** run the actual testcase afterwards|
|`-runtest`||Pass this flag to execute the testcase and run the approaches|
|`-genres`||Pass this flag to gather all results from existing csv files into one|
|`-timeout`|`120`|Timeout in seconds, each individual testcase run will be canceled after this timeout, **default** is `600`|
|`-allocsize`|`8`|How large the manageable memory ares per memory manager should be in `GB`|
|`-device`|`0`|Which GPU device to use|

## Synthetic Testcases
### Register Requirements
This testcase will report the number of registers required for a respective call to `malloc` or `free`.
* `python test_registers.py -t o+s+h+c+r+x -runtest -allocsize 8 -device 0`

| Option | Parameter-Example | Description |
|:---:|:---:|:---:|
|`-t`|`o+s+h+c+f+r+x`|Specify which frameworks to test, first letter of approach separated by `+`, e.g. `c` : cuda or `s` : scatteralloc|
|`-runtest`||Pass this flag to execute the testcase and run the approaches|
|`-genres`||Pass this flag to gather all results from existing csv files into one|
|`-allocsize`|`8`|How large the manageable memory ares per memory manager should be in `GB`|
|`-device`|`0`|Which GPU device to use|

### Memory Manager Initialization
This testcase will test how long it takes to initialize each memory manager.
* `python test_synth_init.py -t o+s+h+c+r+x -runtest -allocsize 8 -device 0`

| Option | Parameter-Example | Description |
|:---:|:---:|:---:|
|`-t`|`o+s+h+c+f+r+x`|Specify which frameworks to test, first letter of approach separated by `+`, e.g. `c` : cuda or `s` : scatteralloc|
|`-runtest`||Pass this flag to execute the testcase and run the approaches|
|`-genres`||Pass this flag to gather all results from existing csv files into one|
|`-allocsize`|`8`|How large the manageable memory ares per memory manager should be in `GB`|
|`-device`|`0`|Which GPU device to use|

### Workload Testcase
This testcase will test the classic case of a number of threads producing varying numbers of output elements and compares it to a baseline implemented with an `CUB::ExclusiveSum`.
* `python test_synth_workload.py -t o+s+h+c+r+x -threadrange 0-10 -range 4-64 -iter 50 -runtest -timeout 60 -allocsize 8 -device 0`
  * This will start with `2⁰` threads up to `2¹⁰` threads, testing all powers of 2 in-between, and for each number of threads test the range `4-64` Bytes
  * The option `-testwrite` will test write performance to this memory area

| Option | Parameter-Example | Description |
|:---:|:---:|:---:|
|`-t`|`o+s+h+c+f+r+x+b`|Specify which frameworks to test, first letter of approach separated by `+`, e.g. `b` : baseline (CUB exclusive sum) or `c` : cuda or `s` : scatteralloc|
|`-threadrange`|`0-10`| The range of threads to test, given as a power of 2, e.g. `0-10` would test `2⁰`, `2¹`, ..., `2¹⁰` threads for the given `-byterange`|
|`-range`|`4-64`|Which allocation range to test, e.g. `4-64` Bytes|
|`-iter`|`50`|How often to run test and average over runs, e.g. `50`|
|`-runtest`||Pass this flag to execute the testcase and run the approaches|
|`-genres`||Pass this flag to gather all results from existing csv files into one|
|`-allocsize`|`8`|How large the manageable memory ares per memory manager should be in `GB`|
|`-device`|`0`|Which GPU device to use|
|`-testwrite`||If parameter is passed, not the allocation performance is measured but the write performance to these allocations|

# Test table TITAN V
| | Build |Init|Reg.| Perf. 10K | Perf. 100K | Warp 10K | Warp 100K | Mix 10K | Mix 100K | Scale | Frag. 1|OOM|Graph Init.|Graph Up.|Graph Range|Synth.4-64|Synth.4-4096|Synth. Write|
|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**CUDA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**XMalloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:boom:|:boom:|:boom:|:boom:|:question:|:boom:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**ScatterAlloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Halloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - AW**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - C**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:boom:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:boom:|:boom:|:boom:|:boom:|:boom:|
|**Reg-Eff - CF**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - CM**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - CFM**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - P - S**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - P - VA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - P - VL**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - C - S**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - C - VA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - C - VL**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|

# Test table 2080Ti
| | Build |Init|Reg.| Perf. 10K | Perf. 100K | Warp 10K | Warp 100K | Mix 10K | Mix 100K | Scale | Frag. 1|OOM|Graph Init.|Graph Up.|Graph Range|Synth.4-64|Synth.4-4096|Synth. Write|
|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**CUDA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**XMalloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|||:heavy_check_mark:|:boom:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**ScatterAlloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Halloc**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - AW**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - C**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|||:heavy_check_mark:|:boom:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - CF**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|||:heavy_check_mark:|:boom:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - CM**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|||:heavy_check_mark:|:boom:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Reg-Eff - CFM**|:a:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|||:heavy_check_mark:|:boom:|:boom:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - P - S**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - P - VA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - P - VL**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - C - S**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|**Oro - C - VA**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:boom:|
|**Oro - C - VL**|:ab:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
