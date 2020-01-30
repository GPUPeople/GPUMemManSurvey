cmake_minimum_required(VERSION 3.8)
project(memorymanagercuda LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_LOCATION "../..")
include(${BASE_LOCATION}/BaseCMake.cmake)

##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${BASE_LOCATION}/include)

# add_library(memorymanagercuda
# 	Instance.cuh
# )

add_executable(memorymanagercuda_example
	main.cu
)

# target_link_libraries(memorymanagercuda_example memorymanagercuda)