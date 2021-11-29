cmake_minimum_required(VERSION 3.12...3.18)
project(kepler_jax LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

# CPU op library
pybind11_add_module(cpu_ops ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_ops.cc)
install(TARGETS cpu_ops DESTINATION brainpy_extension)
