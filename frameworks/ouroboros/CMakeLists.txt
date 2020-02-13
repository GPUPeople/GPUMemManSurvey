cmake_minimum_required(VERSION 3.16)
project(memorymanagerouroboros LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_LOCATION "../..")
set(OUROBOROS_LOCATION "repository")
include(${BASE_LOCATION}/BaseCMake.cmake)

##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${BASE_LOCATION}/include)
include_directories(${OUROBOROS_LOCATION}/include)

add_executable(memorymanagerouroboros_example
	Instance.cuh
	main.cu
)

# target_link_libraries(memorymanagerhalloc_example memorymanagerhalloc)