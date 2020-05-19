cmake_minimum_required(VERSION 3.16)
project(memorymanagerfdg LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_LOCATION "../..")
include(${BASE_LOCATION}/BaseCMake.cmake)

##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${BASE_LOCATION}/include)
include_directories(${BASE_LOCATION}/externals/cub)
include_directories(header)

add_executable(memorymanagerfdg_example
	Instance.cuh
	main.cu
)