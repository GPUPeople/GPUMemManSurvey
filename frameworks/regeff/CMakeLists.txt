cmake_minimum_required(VERSION 3.16)
project(memorymanagerregeff LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_LOCATION "../..")
include(${BASE_LOCATION}/BaseCMake.cmake)

##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${BASE_LOCATION}/include)
include_directories(${BASE_PATH}/externals/cub)

add_executable(memorymanagerrefeff_example
	Instance.cuh
	main.cu
	warp_common.cuh
	warp_common_impl.cuh
	gpualloc_impl.cuh
	gpualloc.cuh
)