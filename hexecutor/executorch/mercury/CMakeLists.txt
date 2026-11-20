# fhm_nn with ExecuTorch and HexecutorBackend, for Steam-Frame-Hand-Tracking's mercury/. Its CMakeLists links the
# fhm_nn target from this project when given -DFHM_NN_CMAKE_DIR=<this directory> (build.sh: FHT_CMAKE_ARGS), and
# FHM_NN_SOURCE_DIR must be its mercury/ directory (fhm_nn.cpp/.h). Also builds fhm_nn_check. HexecutorBackend (LGPL-3.0-or-later) is linked dynamically: it and the
# ExecuTorch runtime are the separate shared library libhexecutor_et.so, which this project's MIT code only
# calls, so it stays MIT and the library can be replaced (THIRD_PARTY.md). Also builds standalone (no Monado), e.g. for a reference-mode check:
#
#   cmake -S hexecutor/executorch/mercury -B build/fhm-nn -DCMAKE_BUILD_TYPE=Release \
#         -DFHM_NN_SOURCE_DIR=<Steam-Frame-Hand-Tracking>/mercury -DFHM_HEXECUTOR_ROOT=<this checkout> -DHEXECUTOR_ET_SOURCE_DIR=<.../executorch>
#
# HexecutorBackend and its hexecutor host sources come from FHM_HEXECUTOR_ROOT (hexecutor/executorch); the
# ExecuTorch v1.0.0 source tree from HEXECUTOR_ET_SOURCE_DIR. Neither has a default.
cmake_minimum_required(VERSION 3.24)
project(fhm_nn LANGUAGES C CXX)

set(FHM_NN_SOURCE_DIR "" CACHE PATH "Steam-Frame-Hand-Tracking mercury/ directory holding fhm_nn.cpp")
if(NOT EXISTS "${FHM_NN_SOURCE_DIR}/fhm_nn.cpp")
    message(FATAL_ERROR "FHM_NN_SOURCE_DIR must name Steam-Frame-Hand-Tracking/mercury")
endif()
set(FHM_HEXECUTOR_ROOT "" CACHE PATH "frame-spatial-runtime checkout providing hexecutor/executorch")
if(NOT EXISTS "${FHM_HEXECUTOR_ROOT}/hexecutor/executorch/CMakeLists.txt")
    message(FATAL_ERROR "FHM_HEXECUTOR_ROOT must name a frame-spatial-runtime checkout (got '${FHM_HEXECUTOR_ROOT}')")
endif()
if(NOT EXISTS "${HEXECUTOR_ET_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HEXECUTOR_ET_SOURCE_DIR must name an ExecuTorch v1.0.0 source tree (got '${HEXECUTOR_ET_SOURCE_DIR}')")
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # linked into libframehand_mercury
set(HEXECUTOR_ET_SHARED ON CACHE BOOL "" FORCE)
add_subdirectory("${FHM_HEXECUTOR_ROOT}/hexecutor/executorch" "${CMAKE_CURRENT_BINARY_DIR}/hexecutor-executorch"
                 EXCLUDE_FROM_ALL)

add_library(fhm_nn STATIC ${FHM_NN_SOURCE_DIR}/fhm_nn.cpp)
set_target_properties(fhm_nn PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_definitions(fhm_nn PRIVATE FHM_EXECUTORCH=1)
target_include_directories(fhm_nn PUBLIC "${FHM_NN_SOURCE_DIR}")
target_link_libraries(fhm_nn PUBLIC hexecutor_et) # the shared library only, never the static archives
target_compile_options(fhm_nn PRIVATE -Wall -Wextra)

add_executable(fhm_nn_check fhm_nn_check.cpp)
set_target_properties(fhm_nn_check PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_link_libraries(fhm_nn_check PRIVATE fhm_nn)
set_target_properties(fhm_nn_check PROPERTIES BUILD_RPATH_USE_ORIGIN ON)
target_compile_options(fhm_nn_check PRIVATE -Wall -Wextra)
