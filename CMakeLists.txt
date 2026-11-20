cmake_minimum_required(VERSION 3.20)
project(frame-spatial-runtime C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(hexecutor/host/cpp)

add_executable(hexecutord hexecutor/host/cpp/hexecutord.cpp)
target_include_directories(hexecutord PRIVATE hexecutor/host/cpp)
target_link_libraries(hexecutord PRIVATE hexecutor_host)
set_target_properties(hexecutord PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(hexecutor_chunk_probe hexecutor/host/cpp/chunk_probe_run.cpp)
target_include_directories(hexecutor_chunk_probe PRIVATE hexecutor/host/cpp)
target_link_libraries(hexecutor_chunk_probe PRIVATE hexecutor_host)
set_target_properties(hexecutor_chunk_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_executable(hexecutor_lfm_run hexecutor/host/cpp/lfm_run.cpp hexecutor/host/cpp/tensor_reference.cpp)
target_include_directories(hexecutor_lfm_run PRIVATE hexecutor/native)
target_link_libraries(hexecutor_lfm_run PRIVATE hexecutor_host)
set_target_properties(hexecutor_lfm_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_executable(hexecutor_jtlb_probe hexecutor/host/cpp/jtlb_probe_run.cpp)
target_include_directories(hexecutor_jtlb_probe PRIVATE hexecutor/host/cpp)
target_link_libraries(hexecutor_jtlb_probe PRIVATE hexecutor_host)
set_target_properties(hexecutor_jtlb_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_library(depth_model_host depth-model/deployment/cpp/package.cpp)
target_include_directories(depth_model_host PUBLIC depth-model/include)
target_link_libraries(depth_model_host PUBLIC hexecutor_host)
target_compile_options(depth_model_host PRIVATE -Wall -Wextra -Wpedantic)

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs highgui)

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(fusion_core runtime/src/fusion_core.cpp)
target_include_directories(fusion_core PUBLIC runtime/include)
target_link_libraries(fusion_core PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(fusion_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(depth_host runtime/src/main.cpp)
target_link_libraries(depth_host PRIVATE fusion_core)
target_compile_options(depth_host PRIVATE -Wall -Wextra -Wpedantic)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found: enabling viewer")
    add_executable(depth_viewer runtime/src/viewer.cpp)
    target_link_libraries(depth_viewer PRIVATE fusion_core ${OpenCV_LIBS})
    target_compile_options(depth_viewer PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(depth_worker runtime/src/depth_worker.cpp)
target_include_directories(depth_worker PRIVATE depth-model/include runtime/include)
target_link_libraries(depth_worker PRIVATE depth_model_host fusion_core)
target_compile_options(depth_worker PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(test_fusion_core runtime/tests/test_fusion_core.cpp)
target_link_libraries(test_fusion_core PRIVATE fusion_core)
target_compile_options(test_fusion_core PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME test_fusion_core COMMAND test_fusion_core)

add_executable(test_broker hexecutor/tests/test_broker.cpp)
target_include_directories(test_broker PRIVATE hexecutor/include)
target_link_libraries(test_broker PRIVATE hexecutor_host)
add_test(NAME test_broker COMMAND test_broker)

add_executable(test_cdsp_mem_core hexecutor/tests/test_cdsp_mem_core.cpp hexecutor/device/kmod/cdsp_mem_core.c)
target_include_directories(test_cdsp_mem_core PRIVATE hexecutor/device/kmod)
add_test(NAME test_cdsp_mem_core COMMAND test_cdsp_mem_core)

add_library(xrservice_broker_shim SHARED hexecutor/host/cpp/xrservice_broker_shim.cpp)
target_include_directories(xrservice_broker_shim PRIVATE hexecutor/include)
target_link_libraries(xrservice_broker_shim PRIVATE hexecutor_host)
target_compile_options(xrservice_broker_shim PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_xrservice_broker_shim hexecutor/tests/test_xrservice_broker_shim.cpp)
target_include_directories(test_xrservice_broker_shim PRIVATE hexecutor/include)
target_link_libraries(test_xrservice_broker_shim PRIVATE hexecutor_host xrservice_broker_shim)
add_test(NAME test_xrservice_broker_shim COMMAND test_xrservice_broker_shim)

add_executable(test_tensor_graph hexecutor/tests/test_tensor_graph.cpp hexecutor/native/tensor/graph.c hexecutor/native/tensor/session.c hexecutor/native/tensor/engine.c hexecutor/native/tensor/fp16.c)
target_include_directories(test_tensor_graph PRIVATE hexecutor/native/tensor hexecutor/native)
target_compile_options(test_tensor_graph PRIVATE -Wall -Wextra)
add_test(NAME test_tensor_graph COMMAND test_tensor_graph)

add_executable(hexecutor_tensor_run hexecutor/host/cpp/tensor_run.cpp hexecutor/host/cpp/tensor_reference.cpp)
target_include_directories(hexecutor_tensor_run PRIVATE hexecutor/native)
target_link_libraries(hexecutor_tensor_run PRIVATE hexecutor_host)
set_target_properties(hexecutor_tensor_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_executable(hexecutor_decode_run hexecutor/host/cpp/decode_run.cpp hexecutor/host/cpp/tensor_reference.cpp)
target_include_directories(hexecutor_decode_run PRIVATE hexecutor/native)
target_link_libraries(hexecutor_decode_run PRIVATE hexecutor_host)
set_target_properties(hexecutor_decode_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_library(hand_tracking_host hand-tracking/src/tracker.cpp)
target_include_directories(hand_tracking_host PUBLIC hand-tracking/include)
target_link_libraries(hand_tracking_host PUBLIC hexecutor_host)
target_compile_options(hand_tracking_host PRIVATE -Wall -Wextra -Wpedantic)
add_executable(frame_hand_tracker hand-tracking/tools/hand_tracker.cpp)
target_include_directories(frame_hand_tracker PRIVATE hand-tracking/include)
target_link_libraries(frame_hand_tracker PRIVATE hand_tracking_host hexecutor_host)
target_compile_options(frame_hand_tracker PRIVATE -Wall -Wextra -Wpedantic)
