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
include_directories(include)

add_library(helper
			include/helper/COO.h 
			include/helper/CSR.h 
			include/helper/dCSR.h 
			include/helper/Vector.h
			${BASE_PATH}include/json.h

			src/helper/COO.cpp
			src/helper/CSR.cpp
			src/helper/dCSR.cpp
			src/helper/Vector.cpp
)
target_include_directories(helper PUBLIC include/helper)

##########################################################################
##########################################################################
##########################################################################
# CUDA
##########################################################################
##########################################################################
##########################################################################
if(TEST_CUDA)

	add_executable(c_graph_test
		graph_test.cu
	)
	target_compile_definitions(c_graph_test PUBLIC TEST_CUDA)
	set_property(TARGET c_graph_test PROPERTY CUDA_ARCHITECTURES OFF)
	target_link_libraries(c_graph_test helper)

endif()

##########################################################################
##########################################################################
##########################################################################
# Halloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_HALLOC AND NOT CUDA_BUILD_CC70_ASYNC)
	
	add_executable(h_graph_test
		graph_test.cu
	)
	target_compile_definitions(h_graph_test PUBLIC TEST_HALLOC)
	target_include_directories(h_graph_test PUBLIC ${BASE_PATH}frameworks/halloc/repository/src)
	set_property(TARGET h_graph_test PROPERTY CUDA_ARCHITECTURES OFF)
	target_link_libraries(h_graph_test helper)

endif ()

##########################################################################
##########################################################################
##########################################################################
# XMalloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_XMALLOC)
	
	add_executable(x_graph_test
		graph_test.cu
	)
	target_compile_definitions(x_graph_test PUBLIC TEST_XMALLOC)
	target_include_directories(x_graph_test PUBLIC ${BASE_PATH}frameworks/xmalloc)
	set_property(TARGET x_graph_test PROPERTY CUDA_ARCHITECTURES OFF)
	target_link_libraries(x_graph_test helper)

endif ()

##########################################################################
##########################################################################
##########################################################################
# ScatterAlloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_SCATTERALLOC)

	add_executable(s_graph_test
		graph_test.cu
	)
	target_compile_definitions(s_graph_test PUBLIC TEST_SCATTERALLOC)
	target_include_directories(s_graph_test PUBLIC ${BASE_PATH}frameworks/scatteralloc/repository/src/include/mallocMC)
	set_property(TARGET s_graph_test PROPERTY CUDA_ARCHITECTURES OFF)
	target_link_libraries(s_graph_test helper)

endif ()

##########################################################################
##########################################################################
##########################################################################
# Ouroboros
##########################################################################
##########################################################################
##########################################################################
if(TEST_OUROBOROS)

	add_executable(o_graph_test_p
		graph_test.cu
	)
	target_compile_definitions(o_graph_test_p PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_graph_test_p PUBLIC TEST_PAGES)
	target_include_directories(o_graph_test_p PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_graph_test_p PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_graph_test_c
		graph_test.cu
	)
	target_compile_definitions(o_graph_test_c PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_graph_test_c PUBLIC TEST_CHUNKS)
	target_include_directories(o_graph_test_c PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_graph_test_c PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_graph_test_vap
		graph_test.cu
	)
	target_compile_definitions(o_graph_test_vap PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_graph_test_vap PUBLIC TEST_PAGES)
	target_compile_definitions(o_graph_test_vap PUBLIC TEST_VIRTUALIZED_ARRAY)
	target_include_directories(o_graph_test_vap PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_graph_test_vap PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_graph_test_vac
		graph_test.cu
	)
	target_compile_definitions(o_graph_test_vac PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_graph_test_vac PUBLIC TEST_CHUNKS)
	target_compile_definitions(o_graph_test_vac PUBLIC TEST_VIRTUALIZED_ARRAY)
	target_include_directories(o_graph_test_vac PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_graph_test_vac PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_graph_test_vlp
		graph_test.cu
	)
	target_compile_definitions(o_graph_test_vlp PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_graph_test_vlp PUBLIC TEST_PAGES)
	target_compile_definitions(o_graph_test_vlp PUBLIC TEST_VIRTUALIZED_LIST)
	target_include_directories(o_graph_test_vlp PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_graph_test_vlp PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(o_graph_test_vlc
		graph_test.cu
	)
	target_compile_definitions(o_graph_test_vlc PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_graph_test_vlc PUBLIC TEST_CHUNKS)
	target_compile_definitions(o_graph_test_vlc PUBLIC TEST_VIRTUALIZED_LIST)
	target_include_directories(o_graph_test_vlc PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
	set_property(TARGET o_graph_test_vlc PROPERTY CUDA_ARCHITECTURES OFF)

	target_link_libraries(o_graph_test_p helper)
	target_link_libraries(o_graph_test_c helper)
	target_link_libraries(o_graph_test_vap helper)
	target_link_libraries(o_graph_test_vac helper)
	target_link_libraries(o_graph_test_vlp helper)
	target_link_libraries(o_graph_test_vlc helper)

endif ()

##########################################################################
##########################################################################
##########################################################################
# FDGMalloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_FDG)

	add_executable(f_graph_test
		graph_test.cu
	)
	target_compile_definitions(f_graph_test PUBLIC TEST_FDG)
	set_property(TARGET f_graph_test PROPERTY CUDA_ARCHITECTURES OFF)
	target_link_libraries(f_graph_test helper)

endif()

##########################################################################
##########################################################################
##########################################################################
# Register-Efficient Allocation
##########################################################################
##########################################################################
##########################################################################
if(TEST_REG)

	add_executable(r_graph_test_a
		graph_test.cu
	)
	target_compile_definitions(r_graph_test_a PUBLIC TEST_REGEFF)
	target_compile_definitions(r_graph_test_a PUBLIC TEST_ATOMIC)
	set_property(TARGET r_graph_test_a PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_graph_test_aw
		graph_test.cu
	)
	target_compile_definitions(r_graph_test_aw PUBLIC TEST_REGEFF)
	target_compile_definitions(r_graph_test_aw PUBLIC TEST_ATOMIC_WRAP)
	set_property(TARGET r_graph_test_aw PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_graph_test_c
		graph_test.cu
	)
	target_compile_definitions(r_graph_test_c PUBLIC TEST_REGEFF)
	target_compile_definitions(r_graph_test_c PUBLIC TEST_CIRCULAR)
	set_property(TARGET r_graph_test_c PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_graph_test_cf
		graph_test.cu
	)
	target_compile_definitions(r_graph_test_cf PUBLIC TEST_REGEFF)
	target_compile_definitions(r_graph_test_cf PUBLIC TEST_CIRCULAR_FUSED)
	set_property(TARGET r_graph_test_cf PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_graph_test_cm
		graph_test.cu
	)
	target_compile_definitions(r_graph_test_cm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_graph_test_cm PUBLIC TEST_CIRCULAR_MULTI)
	set_property(TARGET r_graph_test_cm PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_graph_test_cfm
		graph_test.cu
	)
	target_compile_definitions(r_graph_test_cfm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_graph_test_cfm PUBLIC TEST_CIRCULAR_FUSED_MULTI)
	set_property(TARGET r_graph_test_cfm PROPERTY CUDA_ARCHITECTURES OFF)

	target_link_libraries(r_graph_test_a helper)
	target_link_libraries(r_graph_test_aw helper)
	target_link_libraries(r_graph_test_c helper)
	target_link_libraries(r_graph_test_cf helper)
	target_link_libraries(r_graph_test_cm helper)
	target_link_libraries(r_graph_test_cfm helper)

endif()
