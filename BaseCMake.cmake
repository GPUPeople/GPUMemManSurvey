##########################################################################
# Choose for which CC to build and if to enable Debug
option(CC50 "Build with compute capability 5.0 support" FALSE)
option(CC52 "Build with compute capability 5.2 support" FALSE)
option(CC61 "Build with compute capability 6.1 support" FALSE)
option(CC70_SYNC "Build with compute capability 7.0 support - SYNC" FALSE)
option(CC70_ASYNC "Build with compute capability 7.0 support - ASYNC" FALSE)
option(CC75_SYNC "Build with compute capability 7.5 support - SYNC" FALSE)
option(CC75_ASYNC "Build with compute capability 7.5 support - ASYNC" FALSE)
option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" TRUE)
option(CUDA_BUILD_DEBUG "Build with kernel debug" FALSE)

##########################################################################
# CUDA Flags
if (CC50)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50")
endif ()
if (CC52)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_52,code=sm_52")
endif ()
if (CC61)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61")
endif ()
if (CC70_SYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_70")
endif ()
if (CC70_ASYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70")
endif ()
if (CC75_SYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_75")
endif ()
if (CC75_ASYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_75,code=sm_75")
endif ()

string(APPEND CMAKE_CUDA_FLAGS "  -Xcompiler -Wall -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")

if (CUDA_BUILD_INFO)
	string(APPEND CMAKE_CUDA_FLAGS " -keep --ptxas-options=-v -lineinfo")
endif ()

if (CUDA_BUILD_DEBUG)
	string(APPEND CMAKE_CUDA_FLAGS " -G")
endif ()

##########################################################################
# CXX Flags
if(WIN32)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
if(MSVC)
  string(APPEND CMAKE_CXX_FLAGS "/wd4464 /wd4514 /wd4820 /wd4668 /wd4574 /wd4571 /wd4324 /wd4710 /wd4711 /wd4365 /wd4515 /wd4201 /wd4267 /wd5027 /wd4626")
endif()
else()
set(CUDA_PROPAGATE_HOST_FLAGS ON)
SET(GCC_COVERAGE_LINK_FLAGS  "-lstdc++fs")
string(APPEND CMAKE_CXX_FLAGS "-std=c++14 -mlzcnt -Wno-unknown-pragmas")
endif()