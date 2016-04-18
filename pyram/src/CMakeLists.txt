
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP)

if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -mpopcnt -ffast-math -fPIC")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -mpopcnt -ffast-math -fPIC")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mpopcnt -ffast-math -fPIC")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mpopcnt -ffast-math -fPIC")

if (USE_CUDA)
	find_package(CUDA)
endif(USE_CUDA)

if (CUDA_FOUND)
    set(CUDA_SEPARABLE_COMPILATION ON)
    
	set (CUDA_NVCC_FLAGS, ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
	set (CUDA_NVCC_FLAGS, ${CUDA_NVCC_FLAGS};-use_fast_math;--compiler-options)
	
	if (USE_MIXED_OMP_CUDA)
		add_definitions(-DTRAIN_ON_CPU)
		cuda_add_library (vgram SHARED 	vgram_base.cpp vgram_train.cpp vgram_base.cu vgram_test.cu 
									vgram_data.cpp vgram_synapse.cpp vgram_output.cpp vgram_error.cpp vgram_utils.cpp)
	else()
		cuda_add_library (vgram SHARED 	vgram_base.cu vgram_train.cu vgram_test.cu 
									vgram_data.cpp vgram_synapse.cpp vgram_output.cpp vgram_error.cpp vgram_utils.cpp)
	endif(USE_MIXED_OMP_CUDA)
	
	cuda_add_executable(vgram-test vgram_main.cpp vgram_files.cpp)
	target_link_libraries(vgram-test vgram m ${CUDA_LIBRARIES})
else ()
	add_library (vgram SHARED 	vgram_base.cpp vgram_train.cpp vgram_test.cpp vgram_test_dtw.cpp
								vgram_data.cpp vgram_synapse.cpp vgram_output.cpp vgram_error.cpp vgram_utils.cpp)
	add_executable(vgram-test vgram_main.cpp vgram_files.cpp)
	target_link_libraries(vgram-test vgram m)
endif (CUDA_FOUND)

set_target_properties(vgram PROPERTIES PREFIX "" )
set_target_properties(vgram PROPERTIES SUFFIX .so)

ADD_CUSTOM_COMMAND(TARGET vgram POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/vgram.so ${PROJECT_SOURCE_DIR}/mae/vgram)

