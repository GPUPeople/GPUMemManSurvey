cmake_minimum_required(VERSION 3.16)
project(memorymanagerscatteralloc LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_LOCATION "../..")
set(SCATTERALLOC_LOCATION "repository/src/include/mallocMC")
include(${BASE_LOCATION}/BaseCMake.cmake)

##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${BASE_LOCATION}/include)
include_directories(${SCATTERALLOC_LOCATION})

# Library
# add_library(memorymanagerhalloc STATIC
# 	${HALLOC_LOCATION}/globals.cuh
# 	${HALLOC_LOCATION}/grid.cuh
# 	${HALLOC_LOCATION}/grid.h
# 	${HALLOC_LOCATION}/halloc.cuh
# 	${HALLOC_LOCATION}/halloc.h
# 	${HALLOC_LOCATION}/sbset.cuh
# 	${HALLOC_LOCATION}/sbset.h
# 	${HALLOC_LOCATION}/size-info.cuh
# 	${HALLOC_LOCATION}/size-info.h
# 	${HALLOC_LOCATION}/slab.cuh
# 	${HALLOC_LOCATION}/slab.h
# 	${HALLOC_LOCATION}/statistics.cuh
# 	${HALLOC_LOCATION}/utils.cu
# 	${HALLOC_LOCATION}/utils.h
# 	Instance.cuh
# )
# # Optional
# target_compile_definitions(memorymanagerhalloc PUBLIC LINEAR_MEMORY_LAYOUT)

add_executable(memorymanagerscatteralloc
	main.cu
)

# target_link_libraries(memorymanagerhalloc_example memorymanagerhalloc)