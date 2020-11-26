cmake_minimum_required(VERSION 3.16)
project(GPUMemoryManagers LANGUAGES CXX CUDA VERSION 1.0)

include(BaseCMake.cmake)

set(BASE_PATH "")

# Which frameworks to test
option(TEST_CUDA "Test CUDA" ON)
option(TEST_HALLOC "Test Halloc" ON)
option(TEST_SCATTERALLOC "Test ScatterAlloc" ON)
option(TEST_OUROBOROS "Test Ouroboros" ON)
option(TEST_FDG "Test FDGMalloc" OFF)
option(TEST_REG "Test Register-Efficient" ON)
option(TEST_XMALLOC "Test XMalloc" ON)

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
include_directories(${BASE_PATH}externals/cub)

##########################################################################
##########################################################################
##########################################################################
# CUDA
##########################################################################
##########################################################################
##########################################################################
if(TEST_CUDA)

	add_executable(c_main
		src/main.cu
	)
	target_compile_definitions(c_main PUBLIC TEST_CUDA)
	set_property(TARGET c_main PROPERTY CUDA_ARCHITECTURES OFF)

endif()

##########################################################################
##########################################################################
##########################################################################
# XMalloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_XMALLOC)

	add_executable(x_main
		src/main.cu
	)
	target_compile_definitions(x_main PUBLIC TEST_XMALLOC)
	target_include_directories(x_main PUBLIC ${BASE_PATH}frameworks/xmalloc)
	set_property(TARGET x_main PROPERTY CUDA_ARCHITECTURES OFF)

endif ()

##########################################################################
##########################################################################
##########################################################################
# Halloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_HALLOC AND NOT CUDA_BUILD_CC70_ASYNC)
	
	add_executable(h_main
		src/main.cu
	)
	target_compile_definitions(h_main PUBLIC TEST_HALLOC)
	target_include_directories(h_main PUBLIC ${BASE_PATH}frameworks/halloc/repository/src)
	set_property(TARGET h_main PROPERTY CUDA_ARCHITECTURES OFF)

endif ()

##########################################################################
##########################################################################
##########################################################################
# ScatterAlloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_SCATTERALLOC)

	add_executable(s_main
		src/main.cu
	)
	target_compile_definitions(s_main PUBLIC TEST_SCATTERALLOC)
	target_include_directories(s_main PUBLIC ${BASE_PATH}frameworks/scatteralloc/repository/src/include/mallocMC)
	set_property(TARGET s_main PROPERTY CUDA_ARCHITECTURES OFF)

endif ()

##########################################################################
##########################################################################
##########################################################################
# Ouroboros
##########################################################################
##########################################################################
##########################################################################
if(TEST_OUROBOROS)

	add_executable(o_main_p
	src/main.cu
	)
	target_compile_definitions(o_main_p PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_main_p PUBLIC TEST_PAGES)
	target_include_directories(o_main_p PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_main_p PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_main_c
	src/main.cu
	)
	target_compile_definitions(o_main_c PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_main_c PUBLIC TEST_CHUNKS)
	target_include_directories(o_main_c PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_main_c PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_main_vap
	src/main.cu
	)
	target_compile_definitions(o_main_vap PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_main_vap PUBLIC TEST_PAGES)
	target_compile_definitions(o_main_vap PUBLIC TEST_VIRTUALIZED_ARRAY)
	target_include_directories(o_main_vap PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_main_vap PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_main_vac
	src/main.cu
	)
	target_compile_definitions(o_main_vac PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_main_vac PUBLIC TEST_CHUNKS)
	target_compile_definitions(o_main_vac PUBLIC TEST_VIRTUALIZED_ARRAY)
	target_include_directories(o_main_vac PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_main_vac PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_main_vlp
	src/main.cu
	)
	target_compile_definitions(o_main_vlp PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_main_vlp PUBLIC TEST_PAGES)
	target_compile_definitions(o_main_vlp PUBLIC TEST_VIRTUALIZED_LIST)
	target_include_directories(o_main_vlp PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_main_vlp PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_main_vlc
	src/main.cu
	)
	target_compile_definitions(o_main_vlc PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_main_vlc PUBLIC TEST_CHUNKS)
	target_compile_definitions(o_main_vlc PUBLIC TEST_VIRTUALIZED_LIST)
	target_include_directories(o_main_vlc PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_main_vlc PROPERTY CUDA_ARCHITECTURES OFF)

endif ()

##########################################################################
##########################################################################
##########################################################################
# FDGMalloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_FDG)

	add_executable(f_main
	src/main.cu
	)
	target_compile_definitions(f_main PUBLIC TEST_FDG)
	set_property(TARGET f_main PROPERTY CUDA_ARCHITECTURES OFF)

endif()

##########################################################################
##########################################################################
##########################################################################
# Register-Efficient Allocation
##########################################################################
##########################################################################
##########################################################################
if(TEST_REG)

	add_executable(r_main_a
	src/main.cu
	)
	target_compile_definitions(r_main_a PUBLIC TEST_REGEFF)
	target_compile_definitions(r_main_a PUBLIC TEST_ATOMIC)
	set_property(TARGET r_main_a PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_main_aw
	src/main.cu
	)
	target_compile_definitions(r_main_aw PUBLIC TEST_REGEFF)
	target_compile_definitions(r_main_aw PUBLIC TEST_ATOMIC_WRAP)
	set_property(TARGET r_main_aw PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_main_c
	src/main.cu
	)
	target_compile_definitions(r_main_c PUBLIC TEST_REGEFF)
	target_compile_definitions(r_main_c PUBLIC TEST_CIRCULAR)
	set_property(TARGET r_main_c PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_main_cf
	src/main.cu
	)
	target_compile_definitions(r_main_cf PUBLIC TEST_REGEFF)
	target_compile_definitions(r_main_cf PUBLIC TEST_CIRCULAR_FUSED)
	set_property(TARGET r_main_cf PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_main_cm
	src/main.cu
	)
	target_compile_definitions(r_main_cm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_main_cm PUBLIC TEST_CIRCULAR_MULTI)
	set_property(TARGET r_main_cm PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_main_cfm
	src/main.cu
	)
	target_compile_definitions(r_main_cfm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_main_cfm PUBLIC TEST_CIRCULAR_FUSED_MULTI)
	set_property(TARGET r_main_cfm PROPERTY CUDA_ARCHITECTURES OFF)

endif()