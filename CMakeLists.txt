cmake_minimum_required(VERSION 3.16)
project(GPUMemoryManagers LANGUAGES CXX CUDA VERSION 1.0)

include(BaseCMake.cmake)

# Which frameworks to test
option(TEST_CUDA "Test CUDA" True)
option(TEST_HALLOC "Test Halloc" True)
option(TEST_SCATTERALLOC "Test ScatterAlloc" True)
option(TEST_OUROBOROS "Test Ouroboros" True)

##########################################################################
##########################################################################
##########################################################################
# Add Libraries
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
if(TEST_SCATTERALLOC)
	include_directories(frameworks/scatteralloc/repository/src/include/mallocMC)
endif ()
if(TEST_OUROBOROS)
	include_directories(frameworks/ouroboros/repository/include)
endif ()

##########################################################################
##########################################################################
##########################################################################
# Add Executables

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
# ScatterAlloc
if(TEST_SCATTERALLOC)
	add_executable(s_main
		src/main.cu
	)
	target_compile_definitions(s_main PUBLIC TEST_SCATTERALLOC)
endif ()

##########################################################################
# Ouroboros
if(TEST_OUROBOROS)
	add_executable(o_main
		src/main.cu
	)
	target_compile_definitions(o_main PUBLIC TEST_OUROBOROS)
endif ()

##########################################################################
##########################################################################
##########################################################################
# Linking
if(TEST_HALLOC)
	target_link_libraries(h_main memorymanagerhalloc)
endif ()
