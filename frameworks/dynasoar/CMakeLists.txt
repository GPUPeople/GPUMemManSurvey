cmake_minimum_required(VERSION 3.8)
project(memorymanagerdynasoar LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_LOCATION "../..")
include(${BASE_LOCATION}/BaseCMake.cmake)

##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${BASE_LOCATION}/include)
include_directories(${BASE_LOCATION}/externals/cub)
include_directories(repository)

add_executable(memorymanagerdynasoar_example
	main.cu
)