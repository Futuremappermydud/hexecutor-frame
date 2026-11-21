# SPDX-License-Identifier: LGPL-3.0-or-later
cmake_minimum_required(VERSION 3.18)
project(hexecutor_sharp LANGUAGES C CXX)

if(NOT TARGET hexecutor_host)
  add_subdirectory(../../hexecutor hexecutor)
endif()

add_executable(sharp_embed sharp_embed.cpp)
target_compile_features(sharp_embed PRIVATE cxx_std_17)
target_link_libraries(sharp_embed PRIVATE hexecutor_host)

add_executable(sharp_blocks sharp_blocks.cpp)
target_compile_features(sharp_blocks PRIVATE cxx_std_17)
target_link_libraries(sharp_blocks PRIVATE hexecutor_host)

add_executable(sharp_merge sharp_merge.cpp)
target_compile_features(sharp_merge PRIVATE cxx_std_17)

add_executable(sharp_fusion sharp_fusion.cpp)
target_compile_features(sharp_fusion PRIVATE cxx_std_17)
target_link_libraries(sharp_fusion PRIVATE hexecutor_host)

add_executable(sharp_disparity sharp_disparity.cpp)
target_compile_features(sharp_disparity PRIVATE cxx_std_17)
target_link_libraries(sharp_disparity PRIVATE hexecutor_host)

add_executable(sharp_ply sharp_ply.cpp)
target_compile_features(sharp_ply PRIVATE cxx_std_17)

add_executable(sharp_compose sharp_compose.cpp)
target_compile_features(sharp_compose PRIVATE cxx_std_17)

add_executable(sharp_groupnorm sharp_groupnorm.cpp)
target_compile_features(sharp_groupnorm PRIVATE cxx_std_17)

if(UNIX)
  find_package(Threads REQUIRED)
  add_executable(sharp_run sharp_run.cpp)
  target_compile_features(sharp_run PRIVATE cxx_std_17)
  target_link_libraries(sharp_run PRIVATE Threads::Threads)
endif()
