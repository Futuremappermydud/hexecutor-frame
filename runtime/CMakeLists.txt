cmake_minimum_required(VERSION 3.16)
project(FrameSpatialRuntime C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# Include paths
set(OPENXR_INCLUDE "$ENV{OPENXR_INCLUDE}" CACHE PATH "Path to OpenXR include directory")
if(NOT OPENXR_INCLUDE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../hand-tracking/.work/openxr-sdk/include/openxr/openxr.h")
        set(OPENXR_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../hand-tracking/.work/openxr-sdk/include")
    endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /opt/steamvr/tools/hellovr_vulkan_linux/src/openvr/headers
)
if(OPENXR_INCLUDE)
    include_directories(${OPENXR_INCLUDE})
endif()

# Library paths
link_directories(
    /opt/steamvr/bin/linuxarm64
)

# Core Runtime Executable
add_executable(frame_spatial_runtime
    src/main.cpp
    src/dma_capture.cpp
    src/overlay_renderer.cpp
    src/vk_detile.cpp
    src/depth_pipeline.cpp
    src/persistent_depth.cpp
    src/gpu_depth_pipeline.cpp
    src/vulkan_stereo_pipeline.cpp
)

add_executable(frame_openxr_point_cloud
    src/openxr_point_cloud.cpp
    src/vulkan_guarded_stereo_pipeline.cpp
    src/cloud_worker.cpp
    src/model_depth_pipeline.cpp src/model_temporal.cpp
    src/dma_capture.cpp
    src/vk_detile.cpp
    src/depth_pipeline.cpp
    src/persistent_depth.cpp
    src/vulkan_stereo_pipeline.cpp
)

set(STEREO_SHADER_NAMES stereo_census stereo_match stereo_filter stereo_support
    stereo_guarded_seed stereo_guarded_anchors stereo_guarded_masks stereo_guarded_arms stereo_guarded_match)
set(STEREO_SHADER_OUTPUTS)
foreach(SHADER_NAME IN LISTS STEREO_SHADER_NAMES)
    set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}.comp")
    set(SHADER_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT "${SHADER_OUTPUT}"
        COMMAND glslc -O "${SHADER_SOURCE}" -o "${SHADER_OUTPUT}"
        DEPENDS "${SHADER_SOURCE}"
        VERBATIM
    )
    list(APPEND STEREO_SHADER_OUTPUTS "${SHADER_OUTPUT}")
endforeach()
add_custom_target(stereo_shaders ALL DEPENDS ${STEREO_SHADER_OUTPUTS})
add_dependencies(frame_spatial_runtime stereo_shaders)

set(POINT_SHADER_OUTPUTS)
foreach(SHADER_FILE point_cloud.vert point_cloud.frag occlusion_demo.frag)
    set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_FILE}")
    set(SHADER_OUTPUT "${SHADER_SOURCE}.spv")
    add_custom_command(
        OUTPUT "${SHADER_OUTPUT}"
        COMMAND glslc -O "${SHADER_SOURCE}" -o "${SHADER_OUTPUT}"
        DEPENDS "${SHADER_SOURCE}"
        VERBATIM
    )
    list(APPEND POINT_SHADER_OUTPUTS "${SHADER_OUTPUT}")
endforeach()
add_custom_target(point_cloud_shaders ALL DEPENDS ${POINT_SHADER_OUTPUTS})
add_dependencies(frame_openxr_point_cloud stereo_shaders point_cloud_shaders)

target_link_libraries(frame_spatial_runtime
    PRIVATE
    OpenMP::OpenMP_CXX
    openvr_api
    pthread
    m
    vulkan
)

target_link_libraries(frame_openxr_point_cloud
    PRIVATE
    OpenMP::OpenMP_CXX
    openxr_loader
    pthread
    m
    vulkan
)

target_compile_options(frame_spatial_runtime PRIVATE -O3 -Wall -Wextra)
target_compile_options(frame_openxr_point_cloud PRIVATE -O3 -Wall -Wextra)
set_target_properties(frame_spatial_runtime PROPERTIES
    BUILD_RPATH "/opt/steamvr/bin/linuxarm64"
    INSTALL_RPATH "/opt/steamvr/bin/linuxarm64"
)
set_target_properties(frame_openxr_point_cloud PROPERTIES
    BUILD_RPATH "/opt/steamvr/bin/linuxarm64"
    INSTALL_RPATH "/opt/steamvr/bin/linuxarm64"
)

set(MNN_PREFIX "$ENV{HOME}/.local/frame-mnn" CACHE PATH "MNN Vulkan install prefix")
if(EXISTS "${MNN_PREFIX}/include/MNN/Interpreter.hpp")
    target_compile_definitions(frame_spatial_runtime PRIVATE FRAME_WITH_MNN)
    target_include_directories(frame_spatial_runtime PRIVATE "${MNN_PREFIX}/include")
    target_link_directories(frame_spatial_runtime PRIVATE "${MNN_PREFIX}/lib")
    target_link_libraries(frame_spatial_runtime PRIVATE MNN)
    set_property(TARGET frame_spatial_runtime APPEND PROPERTY BUILD_RPATH "${MNN_PREFIX}/lib")
    set_property(TARGET frame_spatial_runtime APPEND PROPERTY INSTALL_RPATH "${MNN_PREFIX}/lib")
endif()

add_library(frame_openxr_occlusion SHARED
    src/openxr_occlusion_layer.cpp src/occlusion_renderer.cpp src/cloud_worker.cpp src/model_depth_pipeline.cpp src/model_temporal.cpp src/vulkan_guarded_stereo_pipeline.cpp
    src/dma_capture.cpp src/vk_detile.cpp src/depth_pipeline.cpp src/persistent_depth.cpp src/vulkan_stereo_pipeline.cpp)

add_executable(test_persistent_depth
    tools/test_persistent_depth.cpp
    src/persistent_depth.cpp)
target_include_directories(test_persistent_depth PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(test_persistent_depth PRIVATE -O3 -Wall -Wextra)
target_link_libraries(frame_openxr_occlusion PRIVATE OpenMP::OpenMP_CXX pthread m vulkan)
target_compile_options(frame_openxr_occlusion PRIVATE -O3 -Wall -Wextra -fvisibility=hidden)
set(OCCLUSION_SHADER_OUTPUTS)
foreach(SHADER_FILE occlusion.vert occlusion.frag)
    set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_FILE}")
    set(SHADER_OUTPUT "${SHADER_SOURCE}.spv")
    add_custom_command(OUTPUT "${SHADER_OUTPUT}"
        COMMAND glslc -O "${SHADER_SOURCE}" -o "${SHADER_OUTPUT}"
        DEPENDS "${SHADER_SOURCE}" VERBATIM)
    list(APPEND OCCLUSION_SHADER_OUTPUTS "${SHADER_OUTPUT}")
endforeach()
add_custom_target(occlusion_shaders DEPENDS ${OCCLUSION_SHADER_OUTPUTS})
add_dependencies(frame_openxr_occlusion stereo_shaders occlusion_shaders)

set_target_properties(frame_openxr_occlusion PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

add_executable(test_model_depth_geometry tools/test_model_depth_geometry.cpp)
target_compile_options(test_model_depth_geometry PRIVATE -O3 -Wall -Wextra)
add_executable(test_model_depth_pipeline tools/test_model_depth_pipeline.cpp src/model_depth_pipeline.cpp src/model_temporal.cpp src/vulkan_stereo_pipeline.cpp)
target_compile_options(test_model_depth_pipeline PRIVATE -O3 -Wall -Wextra)
target_link_libraries(test_model_depth_pipeline PRIVATE pthread vulkan)
add_dependencies(test_model_depth_pipeline stereo_shaders)

# The standalone runtime build also needs the native learned-depth worker.
if(NOT TARGET frame_depth_worker)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_BINARY_DIR}/native-host")
endif()
add_dependencies(frame_openxr_point_cloud frame_depth_worker)
add_dependencies(frame_spatial_runtime frame_depth_worker)

set_target_properties(frame_depth_worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(frame_hand_viewer tools/frame_hand_viewer.cpp src/hand_viewer_xr.cpp src/dma_capture.cpp src/vk_detile.cpp src/depth_pipeline.cpp)
target_include_directories(frame_hand_viewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../hand-tracking/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../hand-tracking/mercury/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../hexecutor/include
)
target_link_libraries(frame_hand_viewer PRIVATE hand_tracking_host mercury_executor_networks OpenMP::OpenMP_CXX openxr_loader pthread vulkan)
target_compile_options(frame_hand_viewer PRIVATE -O3 -Wall -Wextra)
set_target_properties(frame_hand_viewer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_property(TARGET frame_hand_viewer PROPERTY BUILD_RPATH "/opt/steamvr/bin/linuxarm64")

find_program(GLSLC glslc)
set(HAND_SHADER_OUTPUTS)
foreach(SHADER_FILE hand_joints.vert hand_joints.frag)
    set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_FILE}")
    set(SHADER_OUTPUT "${SHADER_SOURCE}.spv")
    if(GLSLC)
        add_custom_command(
            OUTPUT "${SHADER_OUTPUT}"
            COMMAND "${GLSLC}" -O "${SHADER_SOURCE}" -o "${SHADER_OUTPUT}"
            DEPENDS "${SHADER_SOURCE}"
            VERBATIM
        )
    endif()
    list(APPEND HAND_SHADER_OUTPUTS "${SHADER_OUTPUT}")
endforeach()
if(GLSLC)
    add_custom_target(hand_viewer_shaders ALL DEPENDS ${HAND_SHADER_OUTPUTS})
    add_dependencies(frame_hand_viewer hand_viewer_shaders)
endif()

set(NIR_ORT "$ENV{NIR_ORT}" CACHE PATH "Path to ONNX Runtime directory for NIR viewer")
if(NOT NIR_ORT)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../hand-tracking/.work/ort/onnxruntime-linux-aarch64-1.24.1")
        set(NIR_ORT "${CMAKE_CURRENT_SOURCE_DIR}/../hand-tracking/.work/ort/onnxruntime-linux-aarch64-1.24.1")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../hand-tracking/.work/ort/onnxruntime-osx-arm64-1.24.1")
        set(NIR_ORT "${CMAKE_CURRENT_SOURCE_DIR}/../hand-tracking/.work/ort/onnxruntime-osx-arm64-1.24.1")
    endif()
endif()
if(NIR_ORT AND EXISTS "${NIR_ORT}/include/onnxruntime_cxx_api.h")
    add_executable(frame_nir_hand_viewer
        tools/frame_nir_hand_viewer.cpp
        src/hand_viewer_xr.cpp
        ../hand-tracking/nirhands/src/pipeline.cpp
        ../hand-tracking/nirhands/src/skeleton.cpp
        ../hand-tracking/nirhands/src/geometry.cpp
        ../hand-tracking/nirhands/src/geometry_ops.cpp
    )
    target_include_directories(frame_nir_hand_viewer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../hand-tracking/nirhands/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../hexecutor/include
        ${NIR_ORT}/include
    )
    target_link_directories(frame_nir_hand_viewer PRIVATE ${NIR_ORT}/lib)
    target_link_libraries(frame_nir_hand_viewer PRIVATE
        hexecutor_host
        onnxruntime
        openxr_loader
        pthread
        vulkan
    )
    target_compile_options(frame_nir_hand_viewer PRIVATE -O3 -Wall -Wextra)
    set_target_properties(frame_nir_hand_viewer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    set_property(TARGET frame_nir_hand_viewer PROPERTY BUILD_RPATH "/opt/steamvr/bin/linuxarm64;${NIR_ORT}/lib")
    if(TARGET hand_viewer_shaders)
        add_dependencies(frame_nir_hand_viewer hand_viewer_shaders)
    endif()
endif()
