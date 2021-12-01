cmake_minimum_required(VERSION 3.12...3.18)
project(brainpylib LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

# CPU op library
pybind11_add_module(cpu_ops ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_ops.cc)
install(TARGETS cpu_ops DESTINATION brainpylib)

# GPU op library
if (BRAINPY_CUDA)
  enable_language(CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  pybind11_add_module(
    gpu_ops
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels.cc.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_ops.cc)
  install(TARGETS gpu_ops DESTINATION brainpylib)
else()
  message(STATUS "Building without CUDA")
endif()