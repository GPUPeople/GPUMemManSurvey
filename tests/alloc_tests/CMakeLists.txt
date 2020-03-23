cmake_minimum_required(VERSION 3.16)
project(GPUMemoryManagers LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_PATH "../../")

include(${BASE_PATH}BaseCMake.cmake)

# Which frameworks to test
option(TEST_CUDA "Test CUDA" True)
option(TEST_HALLOC "Test Halloc" True)
option(TEST_SCATTERALLOC "Test ScatterAlloc" True)
option(TEST_OUROBOROS "Test Ouroboros" True)
option(TEST_FDG "Test FDGMalloc" True)
option(TEST_REG "Test Register-Efficient" True)

##########################################################################
##########################################################################
##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${BASE_PATH}include)
include_directories(${BASE_PATH}frameworks)

##########################################################################
##########################################################################
##########################################################################
# CUDA
##########################################################################
##########################################################################
##########################################################################
if(TEST_CUDA)

	add_executable(c_alloc_test
		perf_allocation.cu
	)
	target_compile_definitions(c_alloc_test PUBLIC TEST_CUDA)

endif()

##########################################################################
##########################################################################
##########################################################################
# Halloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_HALLOC)
	
	add_executable(h_alloc_test
		perf_allocation.cu
	)
	target_compile_definitions(h_alloc_test PUBLIC TEST_HALLOC)
	target_include_directories(h_alloc_test PUBLIC ${BASE_PATH}frameworks/halloc/repository/src)

endif ()

##########################################################################
##########################################################################
##########################################################################
# ScatterAlloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_SCATTERALLOC)

	add_executable(s_alloc_test
		perf_allocation.cu
	)
	target_compile_definitions(s_alloc_test PUBLIC TEST_SCATTERALLOC)
	target_include_directories(s_alloc_test PUBLIC ${BASE_PATH}frameworks/scatteralloc/repository/src/include/mallocMC)

endif ()

##########################################################################
##########################################################################
##########################################################################
# Ouroboros
##########################################################################
##########################################################################
##########################################################################
if(TEST_OUROBOROS)

	add_executable(o_alloc_test_p
	perf_allocation.cu
	)
	target_compile_definitions(o_alloc_test_p PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_alloc_test_p PUBLIC TEST_PAGES)
	target_include_directories(o_alloc_test_p PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_alloc_test_c
	perf_allocation.cu
	)
	target_compile_definitions(o_alloc_test_c PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_alloc_test_c PUBLIC TEST_CHUNKS)
	target_include_directories(o_alloc_test_c PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_alloc_test_vap
	perf_allocation.cu
	)
	target_compile_definitions(o_alloc_test_vap PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_alloc_test_vap PUBLIC TEST_PAGES)
	target_compile_definitions(o_alloc_test_vap PUBLIC TEST_VIRTUALIZED_ARRAY)
	target_include_directories(o_alloc_test_vap PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_alloc_test_vac
	perf_allocation.cu
	)
	target_compile_definitions(o_alloc_test_vac PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_alloc_test_vac PUBLIC TEST_CHUNKS)
	target_compile_definitions(o_alloc_test_vap PUBLIC TEST_VIRTUALIZED_ARRAY)
	target_include_directories(o_alloc_test_vac PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_alloc_test_vlp
	perf_allocation.cu
	)
	target_compile_definitions(o_alloc_test_vlp PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_alloc_test_vlp PUBLIC TEST_PAGES)
	target_compile_definitions(o_alloc_test_vlp PUBLIC TEST_VIRTUALIZED_LIST)
	target_include_directories(o_alloc_test_vlp PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_alloc_test_vlc
	perf_allocation.cu
	)
	target_compile_definitions(o_alloc_test_vlc PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_alloc_test_vlc PUBLIC TEST_CHUNKS)
	target_compile_definitions(o_alloc_test_vlc PUBLIC TEST_VIRTUALIZED_LIST)
	target_include_directories(o_alloc_test_vlc PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

endif ()

##########################################################################
##########################################################################
##########################################################################
# FDGMalloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_FDG)

	add_executable(f_alloc_test
	perf_allocation.cu
	)
	target_compile_definitions(f_alloc_test PUBLIC TEST_FDG)

endif()

##########################################################################
##########################################################################
##########################################################################
# Register-Efficient Allocation
##########################################################################
##########################################################################
##########################################################################
if(TEST_REG)

	add_executable(r_alloc_test_a
		perf_allocation.cu
	)
	target_compile_definitions(r_alloc_test_a PUBLIC TEST_REGEFF)
	target_compile_definitions(r_alloc_test_a PUBLIC TEST_ATOMIC)

	add_executable(r_alloc_test_aw
		perf_allocation.cu
	)
	target_compile_definitions(r_alloc_test_aw PUBLIC TEST_REGEFF)
	target_compile_definitions(r_alloc_test_aw PUBLIC TEST_ATOMIC_WRAP)

	add_executable(r_alloc_test_c
		perf_allocation.cu
	)
	target_compile_definitions(r_alloc_test_c PUBLIC TEST_REGEFF)
	target_compile_definitions(r_alloc_test_c PUBLIC TEST_CIRCULAR)

	add_executable(r_alloc_test_cf
		perf_allocation.cu
	)
	target_compile_definitions(r_alloc_test_cf PUBLIC TEST_REGEFF)
	target_compile_definitions(r_alloc_test_cf PUBLIC TEST_CIRCULAR_FUSED)

	add_executable(r_alloc_test_cm
		perf_allocation.cu
	)
	target_compile_definitions(r_alloc_test_cm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_alloc_test_cm PUBLIC TEST_CIRCULAR_MULTI)

	add_executable(r_alloc_test_cfm
		perf_allocation.cu
	)
	target_compile_definitions(r_alloc_test_cfm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_alloc_test_cfm PUBLIC TEST_CIRCULAR_FUSED_MULTI)

endif()