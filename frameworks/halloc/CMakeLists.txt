cmake_minimum_required(VERSION 3.16)
project(memorymanagerhalloc LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_LOCATION "../..")
set(HALLOC_LOCATION "repository/src")
include(${BASE_LOCATION}/BaseCMake.cmake)

##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${BASE_LOCATION}/include)
include_directories(${HALLOC_LOCATION})

add_executable(memorymanagerhalloc_example
	${HALLOC_LOCATION}/globals.cuh
	${HALLOC_LOCATION}/grid.cuh
	${HALLOC_LOCATION}/grid.h
	${HALLOC_LOCATION}/halloc.cuh
	${HALLOC_LOCATION}/halloc.h
	${HALLOC_LOCATION}/sbset.cuh
	${HALLOC_LOCATION}/sbset.h
	${HALLOC_LOCATION}/size-info.cuh
	${HALLOC_LOCATION}/size-info.h
	${HALLOC_LOCATION}/slab.cuh
	${HALLOC_LOCATION}/slab.h
	${HALLOC_LOCATION}/statistics.cuh
	${HALLOC_LOCATION}/utils.cuh
	${HALLOC_LOCATION}/utils.h
	Instance.cuh
	main.cu
)
