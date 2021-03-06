cmake_minimum_required(VERSION 3.16)
project(GPUMemoryManagers LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_PATH "../../")

include(${BASE_PATH}BaseCMake.cmake)

# Which frameworks to test
option(TEST_CUDA "Test CUDA" True)
option(TEST_HALLOC "Test Halloc" True)
option(TEST_SCATTERALLOC "Test ScatterAlloc" True)
option(TEST_OUROBOROS "Test Ouroboros" True)
option(TEST_FDG "Test FDGMalloc" False)
option(TEST_REG "Test Register-Efficient" True)
option(TEST_XMALLOC "Test XMalloc" True)

if(SYNC_BUILD)
set(TEST_CUDA OFF)
set(TEST_OUROBOROS OFF)
else()
set(TEST_HALLOC OFF)
set(TEST_SCATTERALLOC OFF)
set(TEST_FDG OFF)
set(TEST_REG OFF)
set(TEST_XMALLOC OFF)
endif()

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

	add_executable(c_frag_test
		frag_memory_overhead.cu
	)
	target_compile_definitions(c_frag_test PUBLIC TEST_CUDA)
	set_property(TARGET c_frag_test PROPERTY CUDA_ARCHITECTURES OFF)

endif()

##########################################################################
##########################################################################
##########################################################################
# Halloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_HALLOC AND NOT CUDA_BUILD_CC70_ASYNC)
	
	add_executable(h_frag_test
		frag_memory_overhead.cu
	)
	target_compile_definitions(h_frag_test PUBLIC TEST_HALLOC)
	target_include_directories(h_frag_test PUBLIC ${BASE_PATH}frameworks/halloc/repository/src)
	set_property(TARGET h_frag_test PROPERTY CUDA_ARCHITECTURES OFF)

endif ()

##########################################################################
##########################################################################
##########################################################################
# XMalloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_XMALLOC)
	
	add_executable(x_frag_test
		frag_memory_overhead.cu
	)
	target_compile_definitions(x_frag_test PUBLIC TEST_XMALLOC)
	target_include_directories(x_frag_test PUBLIC ${BASE_PATH}frameworks/xmalloc)
	set_property(TARGET x_frag_test PROPERTY CUDA_ARCHITECTURES OFF)

endif ()

##########################################################################
##########################################################################
##########################################################################
# ScatterAlloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_SCATTERALLOC)

	add_executable(s_frag_test
		frag_memory_overhead.cu
	)
	target_compile_definitions(s_frag_test PUBLIC TEST_SCATTERALLOC)
	target_include_directories(s_frag_test PUBLIC ${BASE_PATH}frameworks/scatteralloc/repository/src/include/mallocMC)
	set_property(TARGET s_frag_test PROPERTY CUDA_ARCHITECTURES OFF)

endif ()

##########################################################################
##########################################################################
##########################################################################
# Ouroboros
##########################################################################
##########################################################################
##########################################################################
if(TEST_OUROBOROS)

	add_executable(o_frag_test_p
	frag_memory_overhead.cu
	)
	target_compile_definitions(o_frag_test_p PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_frag_test_p PUBLIC TEST_PAGES)
	target_include_directories(o_frag_test_p PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_frag_test_p PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_frag_test_c
	frag_memory_overhead.cu
	)
	target_compile_definitions(o_frag_test_c PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_frag_test_c PUBLIC TEST_CHUNKS)
	target_include_directories(o_frag_test_c PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_frag_test_c PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_frag_test_vap
	frag_memory_overhead.cu
	)
	target_compile_definitions(o_frag_test_vap PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_frag_test_vap PUBLIC TEST_PAGES)
	target_compile_definitions(o_frag_test_vap PUBLIC TEST_VIRTUALIZED_ARRAY)
	target_include_directories(o_frag_test_vap PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_frag_test_vap PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_frag_test_vac
	frag_memory_overhead.cu
	)
	target_compile_definitions(o_frag_test_vac PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_frag_test_vac PUBLIC TEST_CHUNKS)
	target_compile_definitions(o_frag_test_vac PUBLIC TEST_VIRTUALIZED_ARRAY)
	target_include_directories(o_frag_test_vac PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_frag_test_vac PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_frag_test_vlp
	frag_memory_overhead.cu
	)
	target_compile_definitions(o_frag_test_vlp PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_frag_test_vlp PUBLIC TEST_PAGES)
	target_compile_definitions(o_frag_test_vlp PUBLIC TEST_VIRTUALIZED_LIST)
	target_include_directories(o_frag_test_vlp PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_frag_test_vlp PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_frag_test_vlc
	frag_memory_overhead.cu
	)
	target_compile_definitions(o_frag_test_vlc PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_frag_test_vlc PUBLIC TEST_CHUNKS)
	target_compile_definitions(o_frag_test_vlc PUBLIC TEST_VIRTUALIZED_LIST)
	target_include_directories(o_frag_test_vlc PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_frag_test_vlc PROPERTY CUDA_ARCHITECTURES OFF)

endif ()

##########################################################################
##########################################################################
##########################################################################
# FDGMalloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_FDG)

	add_executable(f_frag_test
	frag_memory_overhead.cu
	)
	target_compile_definitions(f_frag_test PUBLIC TEST_FDG)
	set_property(TARGET f_frag_test PROPERTY CUDA_ARCHITECTURES OFF)

endif()

##########################################################################
##########################################################################
##########################################################################
# Register-Efficient Allocation
##########################################################################
##########################################################################
##########################################################################
if(TEST_REG)

	add_executable(r_frag_test_a
	frag_memory_overhead.cu
	)
	target_compile_definitions(r_frag_test_a PUBLIC TEST_REGEFF)
	target_compile_definitions(r_frag_test_a PUBLIC TEST_ATOMIC)
	set_property(TARGET r_frag_test_a PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_frag_test_aw
	frag_memory_overhead.cu
	)
	target_compile_definitions(r_frag_test_aw PUBLIC TEST_REGEFF)
	target_compile_definitions(r_frag_test_aw PUBLIC TEST_ATOMIC_WRAP)
	set_property(TARGET r_frag_test_aw PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_frag_test_c
	frag_memory_overhead.cu
	)
	target_compile_definitions(r_frag_test_c PUBLIC TEST_REGEFF)
	target_compile_definitions(r_frag_test_c PUBLIC TEST_CIRCULAR)
	set_property(TARGET r_frag_test_c PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_frag_test_cf
	frag_memory_overhead.cu
	)
	target_compile_definitions(r_frag_test_cf PUBLIC TEST_REGEFF)
	target_compile_definitions(r_frag_test_cf PUBLIC TEST_CIRCULAR_FUSED)
	set_property(TARGET r_frag_test_cf PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_frag_test_cm
	frag_memory_overhead.cu
	)
	target_compile_definitions(r_frag_test_cm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_frag_test_cm PUBLIC TEST_CIRCULAR_MULTI)
	set_property(TARGET r_frag_test_cm PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_frag_test_cfm
	frag_memory_overhead.cu
	)
	target_compile_definitions(r_frag_test_cfm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_frag_test_cfm PUBLIC TEST_CIRCULAR_FUSED_MULTI)
	set_property(TARGET r_frag_test_cfm PROPERTY CUDA_ARCHITECTURES OFF)

endif()