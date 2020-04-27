# GPUMemManSurvey
Evaluating different memory managers for dynamic GPU memory

# Instructions
To use the automated setup script, please install `gitpython` (Arch Linux: `pacman -S python-gitpython`, Windows: `pip install gitpython`).
Then execute the automated script!

# Work in Progress!

| Framework | Status | Link to Paper | Code |
|:---:|:---:|:---:| :---:|
| CUDA Device Allocator | Done 	| - | - |
| XMalloc (2010)				| 	X 	| [Webpage](http://hdl.handle.net/2142/16137) | - |
| ScatterAlloc (2012) 			| Done 	| [Webpage](https://ieeexplore.ieee.org/document/6339604) | [GitHub - Repository](https://github.com/ax3l/scatteralloc) |
| FDGMalloc (2013) 			    |  Bug 	| [Webpage](https://www.gcc.tu-darmstadt.de/media/gcc/papers/Widmer_2013_FDM.pdf) | [Webpage](https://www.gcc.tu-darmstadt.de/home/proj/fdgmalloc/index.en.jsp) |
| Register Efficient (2014)	    | Done	| [Webpage](https://diglib.eg.org/bitstream/handle/10.2312/hpg.20141090.019-027/019-027.pdf?sequence=1&isAllowed=y) | [Webpage](http://decibel.fi.muni.cz/~xvinkl/CMalloc/) |
| Halloc (2014)				    |  Bug 	| [Webpage](http://on-demand.gputechconf.com/gtc/2014/presentations/S4271-halloc-high-throughput-dynamic-memory-allocator.pdf) | [GitHub - Repository](https://github.com/canonizer/halloc) |
| DynaSOAr (2019)               |   WIP   | [Webpage](https://drops.dagstuhl.de/opus/volltexte/2019/10809/pdf/LIPIcs-ECOOP-2019-17.pdf) | [GitHub - Repository](https://github.com/prg-titech/dynasoar)|
| Bulk-Sempaphore (2019)		| 	X 	| [Webpage](https://research.nvidia.com/publication/2019-02_Throughput-oriented-GPU-memory) | - |
| Ouroboros (2020)			    | Done	| - | [GitHub - Repository](https://github.com/GPUPeople/Ouroboros) |
