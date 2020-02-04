cmake_minimum_required(VERSION 3.16)
project(GPUMemoryManagers LANGUAGES CXX CUDA VERSION 1.0)

include(BaseCMake.cmake)

option(TEST_CUDA "Test CUDA" True)
option(TEST_HALLOC "Test Halloc" True)
option(TEST_SCATTERALLOC "Test ScatterAlloc" True)

if(TEST_HALLOC)
	add_subdirectory(frameworks/halloc)
endif ()

##########################################################################
##########################################################################
##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)
include_directories(frameworks)

##########################################################################
# CUDA
if(TEST_CUDA)
	add_executable(c_main
		src/main.cu
	)
	target_compile_definitions(c_main PUBLIC TEST_CUDA)
endif()

##########################################################################
# Halloc
if(TEST_HALLOC)
	add_executable(h_main
		src/main.cu
	)
	target_compile_definitions(h_main PUBLIC TEST_HALLOC)
endif ()

##########################################################################
##########################################################################
##########################################################################
# Linking
if(TEST_HALLOC)
	target_link_libraries(h_main memorymanagerhalloc)
endif ()
