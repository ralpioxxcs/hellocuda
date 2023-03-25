cmake_minimum_required(VERSION 3.17)

project(CudaCross C CXX CUDA)

if(${CMAKE_CROSSCOMPILING})
  message(STATUS "cross compile ON")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(CudaCross main.cu)

set_property(TARGET CudaCross PROPERTY CUDA_ARCHITECTURES 50)
set_property(TARGET CudaCross PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(CudaCross PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(CudaCross PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_50>)
