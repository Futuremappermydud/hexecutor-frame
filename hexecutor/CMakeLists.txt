cmake_minimum_required(VERSION 3.16)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(hexecutor LANGUAGES C CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
option(FRAME_BUILD_TESTS "Build portable host/protocol tests" ON)
find_package(Threads REQUIRED)

add_library(hexecutor_host
    host/cpp/executor.cpp
    host/cpp/workload_memory.cpp
    host/cpp/frame_transport.cpp
    host/cpp/broker.cpp
    host/cpp/xrservice_featurenn.cpp
    native/tensor/graph.c
    native/tensor/session.c
    native/qnn16/program.c
    native/program/nprg2_runtime.c)
target_include_directories(hexecutor_host PUBLIC include)
target_include_directories(hexecutor_host PRIVATE native/program native)
target_link_libraries(hexecutor_host PUBLIC Threads::Threads)
target_compile_options(hexecutor_host PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(hexecutor_host PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(hexecutor_chunk_probe host/cpp/chunk_probe_run.cpp)
target_include_directories(hexecutor_chunk_probe PRIVATE host/cpp)
target_link_libraries(hexecutor_chunk_probe PRIVATE hexecutor_host)
set_target_properties(hexecutor_chunk_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_library(xrservice_broker_shim SHARED host/cpp/xrservice_broker_shim.cpp
    host/cpp/xrservice_inline_hook.cpp)
target_include_directories(xrservice_broker_shim PUBLIC include)
target_link_libraries(xrservice_broker_shim PRIVATE hexecutor_host ${CMAKE_DL_LIBS})
set_target_properties(xrservice_broker_shim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_executable(hexecutord host/cpp/hexecutord.cpp)
add_executable(hexecutor_broker_bench host/cpp/broker_bench.cpp)
target_link_libraries(hexecutor_broker_bench PRIVATE hexecutor_host)
set_target_properties(hexecutor_broker_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(hexecutord PRIVATE hexecutor_host)
set_target_properties(hexecutord PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
install(TARGETS hexecutord xrservice_broker_shim
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
install(FILES include/hexecutor/xrservice_broker_shim.h DESTINATION include)
add_executable(hexecutor_qnn16_run host/cpp/qnn16_run.cpp)
target_link_libraries(hexecutor_qnn16_run PRIVATE hexecutor_host)
set_target_properties(hexecutor_qnn16_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_executable(hexecutor_tensor_run host/cpp/tensor_run.cpp host/cpp/tensor_reference.cpp)
target_include_directories(hexecutor_tensor_run PRIVATE native)
target_link_libraries(hexecutor_tensor_run PRIVATE hexecutor_host)
set_target_properties(hexecutor_tensor_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_executable(hexecutor_jtlb_probe host/cpp/jtlb_probe_run.cpp)
target_include_directories(hexecutor_jtlb_probe PRIVATE host/cpp)
target_link_libraries(hexecutor_jtlb_probe PRIVATE hexecutor_host)
set_target_properties(hexecutor_jtlb_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_executable(hexecutor_lfm_run host/cpp/lfm_run.cpp host/cpp/tensor_reference.cpp)
target_include_directories(hexecutor_lfm_run PRIVATE native)
target_link_libraries(hexecutor_lfm_run PRIVATE hexecutor_host)
set_target_properties(hexecutor_lfm_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_executable(hexecutor_decode_run host/cpp/decode_run.cpp host/cpp/tensor_reference.cpp)
target_include_directories(hexecutor_decode_run PRIVATE native)
target_link_libraries(hexecutor_decode_run PRIVATE hexecutor_host)
set_target_properties(hexecutor_decode_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(FRAME_BUILD_TESTS)
    enable_testing()
    add_executable(test_qnn16_hvx tests/test_qnn16_hvx.c
        native/qnn16/qnn16_hvx.c)
    target_include_directories(test_qnn16_hvx PRIVATE native/qnn16)
    target_link_libraries(test_qnn16_hvx PRIVATE m)
    add_test(NAME qnn16_hvx COMMAND test_qnn16_hvx)
    add_executable(test_qnn16_transfer tests/test_qnn16_transfer.c
        native/qnn16/qnn16_transfer.c)
    target_include_directories(test_qnn16_transfer PRIVATE native/qnn16)
    add_test(NAME qnn16_transfer COMMAND test_qnn16_transfer)
    add_executable(test_broker tests/test_broker.cpp)
    target_link_libraries(test_broker PRIVATE hexecutor_host)
    add_test(NAME broker COMMAND test_broker)
    set_tests_properties(broker PROPERTIES TIMEOUT 60)
    add_executable(test_xrservice_featurenn tests/test_xrservice_featurenn.cpp)
    target_link_libraries(test_xrservice_featurenn PRIVATE hexecutor_host)
    add_test(NAME xrservice_featurenn COMMAND test_xrservice_featurenn)
    add_executable(test_xrservice_broker_shim tests/test_xrservice_broker_shim.cpp)
    target_link_libraries(test_xrservice_broker_shim PRIVATE xrservice_broker_shim hexecutor_host)
    add_test(NAME xrservice_broker_shim COMMAND test_xrservice_broker_shim)
    add_executable(test_dynamic_cdsp_mem tests/test_dynamic_cdsp_mem.cpp)
    target_link_libraries(test_dynamic_cdsp_mem PRIVATE hexecutor_host Threads::Threads)
    add_test(NAME dynamic_cdsp_mem COMMAND test_dynamic_cdsp_mem)
    add_executable(test_cdsp_mem_core tests/test_cdsp_mem_core.cpp device/kmod/cdsp_mem_core.c)
    target_include_directories(test_cdsp_mem_core PRIVATE device/kmod include)
    target_link_libraries(test_cdsp_mem_core PRIVATE Threads::Threads)
    add_test(NAME cdsp_mem_core COMMAND test_cdsp_mem_core)
    add_executable(test_graph_entry tests/test_graph_entry.c)
    add_test(NAME graph_entry COMMAND test_graph_entry)
    add_executable(test_tensor_graph tests/test_tensor_graph.cpp)
    target_include_directories(test_tensor_graph PRIVATE native)
    target_link_libraries(test_tensor_graph PRIVATE hexecutor_host)
    add_test(NAME tensor_graph COMMAND test_tensor_graph)
    add_executable(test_tensor_pack tests/test_tensor_pack.cpp)
    target_include_directories(test_tensor_pack PRIVATE native)
    add_test(NAME tensor_pack COMMAND test_tensor_pack)

    # Preserve the build-root test paths used by the broker gate and host scripts.
    set_target_properties(
        test_qnn16_hvx test_qnn16_transfer test_broker test_xrservice_featurenn
        test_xrservice_broker_shim test_dynamic_cdsp_mem test_cdsp_mem_core
        test_graph_entry test_tensor_graph test_tensor_pack
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
