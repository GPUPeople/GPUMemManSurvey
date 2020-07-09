cmake_minimum_required(VERSION 3.16)
project(synth_tests LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_PATH "../../")

include(${BASE_PATH}BaseCMake.cmake)

# Which frameworks to test
option(TEST_CUDA "Test CUDA" True)
option(TEST_HALLOC "Test Halloc" True)
option(TEST_SCATTERALLOC "Test ScatterAlloc" True)
option(TEST_OUROBOROS "Test Ouroboros" True)
option(TEST_FDG "Test FDGMalloc" True)
option(TEST_REG "Test Register-Efficient" True)
option(TEST_XMALLOC "Test XMalloc" True)

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
# Baseline
##########################################################################
##########################################################################
##########################################################################
add_executable(b_synth_test
		synth_workgeneration.cu
)
target_compile_definitions(b_synth_test PUBLIC TEST_BASELINE)

##########################################################################
##########################################################################
##########################################################################
# CUDA
##########################################################################
##########################################################################
##########################################################################
if(TEST_CUDA)

	add_executable(c_synth_test
		synth_workgeneration.cu
	)
	target_compile_definitions(c_synth_test PUBLIC TEST_CUDA)

	add_executable(c_init_test
		synth_init.cu
	)
	target_compile_definitions(c_init_test PUBLIC TEST_CUDA)

endif()

##########################################################################
##########################################################################
##########################################################################
# Halloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_HALLOC AND NOT CUDA_BUILD_CC70_ASYNC)
	
	add_executable(h_synth_test
		synth_workgeneration.cu
	)
	target_compile_definitions(h_synth_test PUBLIC TEST_HALLOC)
	target_include_directories(h_synth_test PUBLIC ${BASE_PATH}frameworks/halloc/repository/src)

	add_executable(h_init_test
	synth_init.cu
	)
	target_compile_definitions(h_init_test PUBLIC TEST_HALLOC)
	target_include_directories(h_init_test PUBLIC ${BASE_PATH}frameworks/halloc/repository/src)

endif ()

##########################################################################
##########################################################################
##########################################################################
# XMalloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_XMALLOC)
	
	add_executable(x_synth_test
		synth_workgeneration.cu
	)
	target_compile_definitions(x_synth_test PUBLIC TEST_XMALLOC)
	target_include_directories(x_synth_test PUBLIC ${BASE_PATH}frameworks/xmalloc)

	add_executable(x_init_test
	synth_init.cu
	)
	target_compile_definitions(x_init_test PUBLIC TEST_XMALLOC)
	target_include_directories(x_init_test PUBLIC ${BASE_PATH}frameworks/xmalloc)

endif ()

##########################################################################
##########################################################################
##########################################################################
# ScatterAlloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_SCATTERALLOC)

	add_executable(s_synth_test
		synth_workgeneration.cu
	)
	target_compile_definitions(s_synth_test PUBLIC TEST_SCATTERALLOC)
	target_include_directories(s_synth_test PUBLIC ${BASE_PATH}frameworks/scatteralloc/repository/src/include/mallocMC)

	add_executable(s_init_test
	synth_init.cu
	)
	target_compile_definitions(s_init_test PUBLIC TEST_SCATTERALLOC)
	target_include_directories(s_init_test PUBLIC ${BASE_PATH}frameworks/scatteralloc/repository/src/include/mallocMC)

endif ()

##########################################################################
##########################################################################
##########################################################################
# Ouroboros
##########################################################################
##########################################################################
##########################################################################
if(TEST_OUROBOROS)

	add_executable(o_synth_test_p
	synth_workgeneration.cu
	)
	target_compile_definitions(o_synth_test_p PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_synth_test_p PUBLIC TEST_PAGES)
	target_include_directories(o_synth_test_p PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_synth_test_c
	synth_workgeneration.cu
	)
	target_compile_definitions(o_synth_test_c PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_synth_test_c PUBLIC TEST_CHUNKS)
	target_include_directories(o_synth_test_c PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_synth_test_vap
	synth_workgeneration.cu
	)
	target_compile_definitions(o_synth_test_vap PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_synth_test_vap PUBLIC TEST_PAGES)
	target_compile_definitions(o_synth_test_vap PUBLIC TEST_VIRTUALIZED_ARRAY)
	target_include_directories(o_synth_test_vap PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_synth_test_vac
	synth_workgeneration.cu
	)
	target_compile_definitions(o_synth_test_vac PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_synth_test_vac PUBLIC TEST_CHUNKS)
	target_compile_definitions(o_synth_test_vac PUBLIC TEST_VIRTUALIZED_ARRAY)
	target_include_directories(o_synth_test_vac PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_synth_test_vlp
	synth_workgeneration.cu
	)
	target_compile_definitions(o_synth_test_vlp PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_synth_test_vlp PUBLIC TEST_PAGES)
	target_compile_definitions(o_synth_test_vlp PUBLIC TEST_VIRTUALIZED_LIST)
	target_include_directories(o_synth_test_vlp PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_synth_test_vlc
	synth_workgeneration.cu
	)
	target_compile_definitions(o_synth_test_vlc PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_synth_test_vlc PUBLIC TEST_CHUNKS)
	target_compile_definitions(o_synth_test_vlc PUBLIC TEST_VIRTUALIZED_LIST)
	target_include_directories(o_synth_test_vlc PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

##########################################################################

	add_executable(o_init_test_p
	synth_init.cu
	)
	target_compile_definitions(o_init_test_p PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_init_test_p PUBLIC TEST_PAGES)
	target_include_directories(o_init_test_p PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_init_test_c
	synth_init.cu
	)
	target_compile_definitions(o_init_test_c PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_init_test_c PUBLIC TEST_CHUNKS)
	target_include_directories(o_init_test_c PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_init_test_vap
	synth_init.cu
	)
	target_compile_definitions(o_init_test_vap PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_init_test_vap PUBLIC TEST_PAGES)
	target_compile_definitions(o_init_test_vap PUBLIC TEST_VIRTUALIZED_ARRAY)
	target_include_directories(o_init_test_vap PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_init_test_vac
	synth_init.cu
	)
	target_compile_definitions(o_init_test_vac PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_init_test_vac PUBLIC TEST_CHUNKS)
	target_compile_definitions(o_init_test_vac PUBLIC TEST_VIRTUALIZED_ARRAY)
	target_include_directories(o_init_test_vac PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_init_test_vlp
	synth_init.cu
	)
	target_compile_definitions(o_init_test_vlp PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_init_test_vlp PUBLIC TEST_PAGES)
	target_compile_definitions(o_init_test_vlp PUBLIC TEST_VIRTUALIZED_LIST)
	target_include_directories(o_init_test_vlp PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

	add_executable(o_init_test_vlc
	synth_init.cu
	)
	target_compile_definitions(o_init_test_vlc PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_init_test_vlc PUBLIC TEST_CHUNKS)
	target_compile_definitions(o_init_test_vlc PUBLIC TEST_VIRTUALIZED_LIST)
	target_include_directories(o_init_test_vlc PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)

endif ()

##########################################################################
##########################################################################
##########################################################################
# FDGMalloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_FDG)

	add_executable(f_synth_test
	synth_workgeneration.cu
	)
	target_compile_definitions(f_synth_test PUBLIC TEST_FDG)

	add_executable(f_init_test
	synth_init.cu
	)
	target_compile_definitions(f_init_test PUBLIC TEST_FDG)

endif()

##########################################################################
##########################################################################
##########################################################################
# Register-Efficient Allocation
##########################################################################
##########################################################################
##########################################################################
if(TEST_REG)

	add_executable(r_synth_test_a
	synth_workgeneration.cu
	)
	target_compile_definitions(r_synth_test_a PUBLIC TEST_REGEFF)
	target_compile_definitions(r_synth_test_a PUBLIC TEST_ATOMIC)

	add_executable(r_synth_test_aw
	synth_workgeneration.cu
	)
	target_compile_definitions(r_synth_test_aw PUBLIC TEST_REGEFF)
	target_compile_definitions(r_synth_test_aw PUBLIC TEST_ATOMIC_WRAP)

	add_executable(r_synth_test_c
	synth_workgeneration.cu
	)
	target_compile_definitions(r_synth_test_c PUBLIC TEST_REGEFF)
	target_compile_definitions(r_synth_test_c PUBLIC TEST_CIRCULAR)

	add_executable(r_synth_test_cf
	synth_workgeneration.cu
	)
	target_compile_definitions(r_synth_test_cf PUBLIC TEST_REGEFF)
	target_compile_definitions(r_synth_test_cf PUBLIC TEST_CIRCULAR_FUSED)

	add_executable(r_synth_test_cm
	synth_workgeneration.cu
	)
	target_compile_definitions(r_synth_test_cm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_synth_test_cm PUBLIC TEST_CIRCULAR_MULTI)

	add_executable(r_synth_test_cfm
	synth_workgeneration.cu
	)
	target_compile_definitions(r_synth_test_cfm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_synth_test_cfm PUBLIC TEST_CIRCULAR_FUSED_MULTI)

	##########################################################################

	add_executable(r_init_test_a
	synth_init.cu
	)
	target_compile_definitions(r_init_test_a PUBLIC TEST_REGEFF)
	target_compile_definitions(r_init_test_a PUBLIC TEST_ATOMIC)

	add_executable(r_init_test_aw
	synth_init.cu
	)
	target_compile_definitions(r_init_test_aw PUBLIC TEST_REGEFF)
	target_compile_definitions(r_init_test_aw PUBLIC TEST_ATOMIC_WRAP)

	add_executable(r_init_test_c
	synth_init.cu
	)
	target_compile_definitions(r_init_test_c PUBLIC TEST_REGEFF)
	target_compile_definitions(r_init_test_c PUBLIC TEST_CIRCULAR)

	add_executable(r_init_test_cf
	synth_init.cu
	)
	target_compile_definitions(r_init_test_cf PUBLIC TEST_REGEFF)
	target_compile_definitions(r_init_test_cf PUBLIC TEST_CIRCULAR_FUSED)

	add_executable(r_init_test_cm
	synth_init.cu
	)
	target_compile_definitions(r_init_test_cm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_init_test_cm PUBLIC TEST_CIRCULAR_MULTI)

	add_executable(r_init_test_cfm
	synth_init.cu
	)
	target_compile_definitions(r_init_test_cfm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_init_test_cfm PUBLIC TEST_CIRCULAR_FUSED_MULTI)

endif()