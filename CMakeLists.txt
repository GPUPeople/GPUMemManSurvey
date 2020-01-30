cmake_minimum_required(VERSION 3.8)
project(GPUMemoryManagers LANGUAGES CXX CUDA VERSION 1.0)

include(BaseCMake.cmake)

##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)
include_directories(frameworks)

add_executable(main
	src/main.cu
)
target_compile_definitions(main PUBLIC TEST_CUDA)