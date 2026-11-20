# Standalone build for the HexecutorBackend ExecuTorch delegate.
#
# This project does not edit the repository's root or hexecutor CMake files. It
# compiles the existing hexecutor host sources by path and builds the minimal
# ExecuTorch runtime from a source tree whose root directory is named exactly
# `executorch` (ExecuTorch's CMake enforces that name).
#
#   cmake -S hexecutor/executorch -B hexecutor/.work/executorch-build \
#         -DHEXECUTOR_ET_SOURCE_DIR=<path to .../executorch> -DCMAKE_BUILD_TYPE=Release
#
# Targets:
#   hexecutor_blob_test        blob parser tests (no ExecuTorch needed)
#   hexecutor_executorch_backend  static lib with the BackendInterface
#   hexecutor_et_run           loads a .pte and runs it (needs the ET runtime)
#   hexecutor_et_selftest      drives the backend on an HXFG image (no .pte needed)
#   hexecutor_et               (HEXECUTOR_ET_SHARED=ON) shared library: the backend, the hexecutor host
#                              sources and the ExecuTorch runtime, for programs that link hexecutor
#                              dynamically (LGPL-3.0-or-later; see hexecutor/LICENSE.md)
cmake_minimum_required(VERSION 3.24)
project(hexecutor_executorch LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

set(HEXECUTOR_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
get_filename_component(HEXECUTOR_REPO_ROOT "${HEXECUTOR_REPO_ROOT}" ABSOLUTE)
set(HEXECUTOR_ET_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.work/et-src/executorch"
    CACHE PATH "ExecuTorch v1.0.0 source tree (its directory must be named executorch)")

find_package(Threads REQUIRED)

option(HEXECUTOR_ET_SHARED "Build libhexecutor_et, a shared library holding the backend and the ExecuTorch runtime" OFF)
if(HEXECUTOR_ET_SHARED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON) # every archive below is folded into the shared library
endif()

# Blob parser: no ExecuTorch dependency.
add_library(hexecutor_blob STATIC blob.cpp)
target_include_directories(hexecutor_blob PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(hexecutor_blob PRIVATE -Wall -Wextra -Wpedantic)

add_executable(hexecutor_blob_test test_blob.cpp)
target_link_libraries(hexecutor_blob_test PRIVATE hexecutor_blob)

# Existing hexecutor host sources, referenced by path. This list mirrors the
# hexecutor_host target in hexecutor/CMakeLists.txt; keep the two in sync.
set(HEXECUTOR_HOST_SOURCES
    ${HEXECUTOR_REPO_ROOT}/hexecutor/host/cpp/executor.cpp
    ${HEXECUTOR_REPO_ROOT}/hexecutor/host/cpp/workload_memory.cpp
    ${HEXECUTOR_REPO_ROOT}/hexecutor/host/cpp/frame_transport.cpp
    ${HEXECUTOR_REPO_ROOT}/hexecutor/host/cpp/broker.cpp
    ${HEXECUTOR_REPO_ROOT}/hexecutor/host/cpp/xrservice_featurenn.cpp
    ${HEXECUTOR_REPO_ROOT}/hexecutor/native/tensor/graph.c
    ${HEXECUTOR_REPO_ROOT}/hexecutor/native/tensor/session.c
    ${HEXECUTOR_REPO_ROOT}/hexecutor/native/qnn16/program.c
    ${HEXECUTOR_REPO_ROOT}/hexecutor/native/program/nprg2_runtime.c)
add_library(hexecutor_host_sources STATIC ${HEXECUTOR_HOST_SOURCES})
target_include_directories(hexecutor_host_sources PUBLIC ${HEXECUTOR_REPO_ROOT}/hexecutor/include)
target_include_directories(hexecutor_host_sources PRIVATE
    ${HEXECUTOR_REPO_ROOT}/hexecutor/native/program ${HEXECUTOR_REPO_ROOT}/hexecutor/native)
target_link_libraries(hexecutor_host_sources PUBLIC Threads::Threads)
target_compile_options(hexecutor_host_sources PRIVATE -Wall -Wextra -Wpedantic)

# ExecuTorch: minimal runtime plus portable kernels. Everything optional is off.
if(NOT EXISTS "${HEXECUTOR_ET_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "ExecuTorch source not found at ${HEXECUTOR_ET_SOURCE_DIR}. "
        "Clone v1.0.0 and point HEXECUTOR_ET_SOURCE_DIR at its executorch directory.")
endif()
set(EXECUTORCH_BUILD_PORTABLE_OPS ON)
set(EXECUTORCH_BUILD_PTHREADPOOL OFF)
set(EXECUTORCH_BUILD_CPUINFO OFF)
set(EXECUTORCH_BUILD_PYBIND OFF)
set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF)
set(EXECUTORCH_BUILD_TESTS OFF)
set(EXECUTORCH_BUILD_XNNPACK OFF)
set(EXECUTORCH_BUILD_EXTENSION_MODULE OFF)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR OFF)
set(BUILD_TESTING OFF)
add_subdirectory("${HEXECUTOR_ET_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/executorch")

add_library(hexecutor_executorch_backend STATIC
    hexecutor_backend.cpp
    ${HEXECUTOR_REPO_ROOT}/hexecutor/host/cpp/tensor_reference.cpp)
target_include_directories(hexecutor_executorch_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(hexecutor_executorch_backend PRIVATE
    ${HEXECUTOR_REPO_ROOT}/hexecutor/include ${HEXECUTOR_REPO_ROOT}/hexecutor/native)
target_link_libraries(hexecutor_executorch_backend
    PUBLIC hexecutor_blob hexecutor_host_sources executorch
    PRIVATE Threads::Threads)
target_compile_options(hexecutor_executorch_backend PRIVATE -Wall -Wextra -Wpedantic)

add_executable(hexecutor_et_run hexecutor_et_run.cpp)
target_link_libraries(hexecutor_et_run PRIVATE
    hexecutor_executorch_backend extension_data_loader extension_tensor)
# Portable CPU kernels for ops left outside the delegate (whole archive so the
# static kernel registrations are kept). Their codegen needs Python torchgen on
# the build host, which the Frame does not have, so this is off by default:
# without it a .pte must be fully delegated.
option(HEXECUTOR_ET_PORTABLE_OPS "Link ExecuTorch portable CPU kernels into hexecutor_et_run" OFF)
if(HEXECUTOR_ET_PORTABLE_OPS)
    target_link_libraries(hexecutor_et_run PRIVATE
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,portable_ops_lib>" portable_kernels)
endif()
target_compile_options(hexecutor_et_run PRIVATE -Wall -Wextra)

add_executable(hexecutor_et_selftest hexecutor_et_selftest.cpp)
target_link_libraries(hexecutor_et_selftest PRIVATE
    hexecutor_executorch_backend hexecutor_blob extension_data_loader extension_tensor)
target_compile_options(hexecutor_et_selftest PRIVATE -Wall -Wextra)

# libhexecutor_et: everything an ExecuTorch program needs to run HexecutorBackend delegates, as one
# replaceable shared library. The ExecuTorch runtime is folded in (whole archives) rather than linked
# beside it: the backend registry is a static object of executorch_core, so a process must hold exactly
# one copy, the one the backend registers into. Consumers link only this library and get the ExecuTorch
# and backend headers through its interface; they must not link the static ExecuTorch archives as well.
if(HEXECUTOR_ET_SHARED)
    add_library(hexecutor_et SHARED hexecutor_et_shared.cpp)
    set_target_properties(hexecutor_et PROPERTIES OUTPUT_NAME hexecutor_et SOVERSION 1)
    target_link_libraries(hexecutor_et PRIVATE
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,hexecutor_executorch_backend,hexecutor_blob,hexecutor_host_sources>"
        # executorch (the prim ops) arrives whole through its own interface; listing it again loads it twice.
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,executorch_core,extension_data_loader,extension_tensor>"
        executorch
        Threads::Threads)
    target_include_directories(hexecutor_et INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<TARGET_PROPERTY:executorch_core,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:extension_tensor,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:extension_data_loader,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(hexecutor_et INTERFACE
        $<TARGET_PROPERTY:executorch_core,INTERFACE_COMPILE_DEFINITIONS>)
    target_compile_features(hexecutor_et INTERFACE cxx_std_17)
    if(UNIX AND NOT APPLE)
        target_link_options(hexecutor_et PRIVATE "-Wl,-z,defs")
    endif()
endif()
