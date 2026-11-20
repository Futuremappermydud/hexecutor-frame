CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -fopenmp -MMD -MP -Iinclude -I/opt/steamvr/tools/hellovr_vulkan_linux/src/openvr/headers
ifneq ($(OPENXR_INCLUDE),)
CXXFLAGS += -I$(OPENXR_INCLUDE)
else ifneq ($(wildcard ../samples/hand-tracking/.work/openxr-sdk/include/openxr/openxr.h),)
CXXFLAGS += -I../samples/hand-tracking/.work/openxr-sdk/include
endif
LDFLAGS = -L/opt/steamvr/bin/linuxarm64 -lopenvr_api -Wl,-rpath=/opt/steamvr/bin/linuxarm64 -lpthread -lm -lvulkan

MNN_PREFIX ?= $(HOME)/.local/frame-mnn
ifneq ($(wildcard $(MNN_PREFIX)/include/MNN/Interpreter.hpp),)
CXXFLAGS += -DFRAME_WITH_MNN -I$(MNN_PREFIX)/include
LDFLAGS += -L$(MNN_PREFIX)/lib -lMNN -Wl,-rpath,$(MNN_PREFIX)/lib
endif

OPENXR_INC ?= $(HOME)/frame-spatial-components/third_party/openxr/include
ifneq ($(wildcard $(OPENXR_INC)/openxr/openxr.h),)
CXXFLAGS += -I$(OPENXR_INC)
endif

SRC = src/main.cpp src/dma_capture.cpp src/timed_stereo_capture.cpp src/v4l2_capture_clock.cpp src/overlay_renderer.cpp src/vk_detile.cpp src/depth_pipeline.cpp src/gpu_depth_pipeline.cpp src/vulkan_stereo_pipeline.cpp
OBJ = $(SRC:.cpp=.o)
DEP = $(OBJ:.o=.d)
TARGET = bin/frame_spatial_runtime
SPV = shaders/stereo_census.spv shaders/stereo_match.spv shaders/stereo_filter.spv shaders/stereo_support.spv
OPENXR_TARGET = bin/frame_openxr_point_cloud
MESH_TARGET = bin/frame_openxr_mesh_viewer
ROOM_REPLAY_TARGET = bin/replay_room_tsdf
OPENXR_SRC = src/openxr_point_cloud.cpp src/cloud_worker.cpp src/tsdf_volume.cpp src/model_depth_pipeline.cpp src/model_temporal.cpp src/ir_pair_capture.cpp src/ir_dot_bridge.cpp src/ir_dot_refine.cpp src/persistent_depth.cpp src/vulkan_guarded_stereo_pipeline.cpp src/dma_capture.cpp src/timed_stereo_capture.cpp src/v4l2_capture_clock.cpp src/vk_detile.cpp src/depth_pipeline.cpp src/vulkan_stereo_pipeline.cpp
OPENXR_OBJ = $(OPENXR_SRC:.cpp=.o)
POINT_SPV = shaders/point_cloud.vert.spv shaders/point_cloud.frag.spv shaders/occlusion_demo.frag.spv
MESH_SPV = shaders/mesh.vert.spv shaders/mesh.frag.spv

all: bin/frame_depth_worker dirs $(SPV) $(POINT_SPV) $(MESH_SPV) $(TARGET) $(OPENXR_TARGET) $(MESH_TARGET)

dirs:
	mkdir -p bin

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OPENXR_TARGET): $(OPENXR_OBJ) | dirs $(SPV) $(POINT_SPV) $(MESH_SPV)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L/opt/steamvr/bin/linuxarm64 \
		-lopenxr_loader -Wl,-rpath=/opt/steamvr/bin/linuxarm64 -lpthread -lm -lvulkan

$(MESH_TARGET): $(OPENXR_TARGET) | dirs
	ln -sf frame_openxr_point_cloud $(MESH_TARGET)

$(ROOM_REPLAY_TARGET): tools/replay_room_tsdf.cpp src/tsdf_volume.cpp | dirs
	$(CXX) $(CXXFLAGS) -o $@ $^ -lpthread -lm

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

shaders/%.spv: shaders/%.comp
	glslc -O $< -o $@

shaders/point_cloud.vert.spv: shaders/point_cloud.vert
	glslc -O $< -o $@

shaders/point_cloud.frag.spv: shaders/point_cloud.frag
	glslc -O $< -o $@

shaders/mesh.vert.spv: shaders/mesh.vert
	glslc -O $< -o $@

shaders/mesh.frag.spv: shaders/mesh.frag
	glslc -O $< -o $@

clean:
	rm -f $(TARGET) $(OPENXR_TARGET) $(MESH_TARGET) $(OBJ) $(OPENXR_OBJ) $(DEP) $(SPV) $(POINT_SPV) $(MESH_SPV) $(OCCLUSION_TARGET) $(OCCLUSION_OBJ) $(OCCLUSION_OBJ:.o=.d) $(OCCLUSION_SPV) $(DIAGNOSTIC_SPV) $(GUARDED_SPV) $(GUARDED_DIAGNOSTIC_SPV)

-include $(DEP)

.PHONY: all clean dirs

# Standalone live camera/hand inspector. Build from depth-runtime/; private model and
# matching DSP paths are supplied at launch, never copied into the executable.
HAND_NATIVE_BUILD ?= ../build
.PHONY: hand-viewer-host hand-viewer
hand-viewer-host:
	cmake -S .. -B $(HAND_NATIVE_BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(HAND_NATIVE_BUILD) --parallel --target hand_tracking_host mercury_executor_networks mercury_hand_tracking hexecutor_host

HAND_SPV = shaders/hand_joints.vert.spv shaders/hand_joints.frag.spv
hand-viewer: bin/frame_hand_viewer $(HAND_SPV)

NIR_ORT ?= ../samples/hand-tracking/.work/ort/onnxruntime-linux-aarch64-1.24.1
.PHONY: nir-hand-viewer nir-hand-viewer-host
nir-hand-viewer: bin/frame_nir_hand_viewer $(HAND_SPV)
nir-hand-viewer-host:
	cmake -S .. -B $(HAND_NATIVE_BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(HAND_NATIVE_BUILD) --parallel --target hexecutor_host

bin/frame_nir_hand_viewer: tools/frame_nir_hand_viewer.cpp $(wildcard ../samples/hand-tracking/nirhands/include/nirhands/*.hpp) src/hand_viewer_xr.o ../samples/hand-tracking/nirhands/src/pipeline.cpp ../samples/hand-tracking/nirhands/src/skeleton.cpp ../samples/hand-tracking/nirhands/src/geometry.cpp ../samples/hand-tracking/nirhands/src/geometry_ops.cpp nir-hand-viewer-host | dirs
	$(CXX) $(CXXFLAGS) -I../samples/hand-tracking/nirhands/include -I../hexecutor/include -I$(NIR_ORT)/include \
		tools/frame_nir_hand_viewer.cpp src/hand_viewer_xr.o ../samples/hand-tracking/nirhands/src/*.cpp \
		$(HAND_NATIVE_BUILD)/libhexecutor_host.a \
		-L$(NIR_ORT)/lib -lonnxruntime -Wl,-rpath,$(abspath $(NIR_ORT))/lib \
		-L/opt/steamvr/bin/linuxarm64 -lopenxr_loader -Wl,-rpath,/opt/steamvr/bin/linuxarm64 -lvulkan -lpthread -o $@

GLSLC ?= glslc

shaders/hand_joints.vert.spv: shaders/hand_joints.vert
	@if command -v $(GLSLC) >/dev/null 2>&1; then $(GLSLC) -O $< -o $@; elif [ -f $@ ]; then echo "glslc not found; using existing $@"; else echo "Error: glslc not found and $@ missing" >&2; exit 1; fi

shaders/hand_joints.frag.spv: shaders/hand_joints.frag
	@if command -v $(GLSLC) >/dev/null 2>&1; then $(GLSLC) -O $< -o $@; elif [ -f $@ ]; then echo "glslc not found; using existing $@"; else echo "Error: glslc not found and $@ missing" >&2; exit 1; fi

bin/frame_hand_viewer: tools/frame_hand_viewer.cpp src/hand_viewer_xr.o ../samples/hand-tracking/include/hand_tracking/viewer_geometry.hpp src/dma_capture.o src/timed_stereo_capture.o src/v4l2_capture_clock.o src/vk_detile.o src/depth_pipeline.o hand-viewer-host | dirs
	$(CXX) $(CXXFLAGS) -I../samples/hand-tracking/include -I../samples/hand-tracking/mercury/include -I../hexecutor/include \
		tools/frame_hand_viewer.cpp src/hand_viewer_xr.o src/dma_capture.o src/timed_stereo_capture.o src/v4l2_capture_clock.o src/vk_detile.o src/depth_pipeline.o \
		$(HAND_NATIVE_BUILD)/libhand_tracking_host.a $(HAND_NATIVE_BUILD)/libmercury_executor_networks.a \
		$(HAND_NATIVE_BUILD)/libmercury_hand_tracking.a $(HAND_NATIVE_BUILD)/libhexecutor_host.a \
		-L/opt/steamvr/bin/linuxarm64 -lopenxr_loader -Wl,-rpath,/opt/steamvr/bin/linuxarm64 -lvulkan -lpthread -o $@

-include src/hand_viewer_xr.d

OCCLUSION_TARGET = bin/libframe_openxr_occlusion.so
OCCLUSION_SRC = src/openxr_occlusion_layer.cpp src/occlusion_renderer.cpp src/cloud_worker.cpp src/tsdf_volume.cpp src/model_depth_pipeline.cpp src/model_temporal.cpp src/ir_pair_capture.cpp src/ir_dot_bridge.cpp src/ir_dot_refine.cpp src/persistent_depth.cpp src/vulkan_guarded_stereo_pipeline.cpp src/dma_capture.cpp src/timed_stereo_capture.cpp src/v4l2_capture_clock.cpp src/vk_detile.cpp src/depth_pipeline.cpp src/vulkan_stereo_pipeline.cpp
OCCLUSION_OBJ = $(OCCLUSION_SRC:.cpp=.pic.o)
OCCLUSION_SPV = shaders/occlusion.vert.spv shaders/occlusion.frag.spv

occlusion: $(OCCLUSION_TARGET)

$(OCCLUSION_TARGET): $(OCCLUSION_OBJ) | dirs $(SPV) $(OCCLUSION_SPV)
	$(CXX) -shared -o $@ $^ -fopenmp -lpthread -lm -lvulkan

%.pic.o: %.cpp
	$(CXX) $(CXXFLAGS) -fPIC -fvisibility=hidden -c $< -o $@

shaders/occlusion.vert.spv: shaders/occlusion.vert
	glslc -O $< -o $@

shaders/occlusion.frag.spv: shaders/occlusion.frag
	glslc -O $< -o $@

-include $(OPENXR_OBJ:.o=.d) $(OCCLUSION_OBJ:.o=.d)
.PHONY: occlusion

bin/test_tsdf_volume: tools/test_tsdf_volume.cpp src/tsdf_volume.cpp include/tsdf_volume.hpp | dirs
	$(CXX) -O3 -std=c++17 -Wall -Wextra -fopenmp -Iinclude tools/test_tsdf_volume.cpp src/tsdf_volume.cpp -o $@

bin/test_tsdf_incremental: tools/test_tsdf_incremental.cpp src/tsdf_volume.cpp include/tsdf_volume.hpp | dirs
	$(CXX) -O3 -std=c++17 -Wall -Wextra -fopenmp -Iinclude tools/test_tsdf_incremental.cpp src/tsdf_volume.cpp -o $@

bin/replay_model_mesh: tools/replay_model_mesh.cpp src/tsdf_volume.cpp include/tsdf_volume.hpp include/model_depth_geometry.hpp include/depth_speckle_filter.hpp | dirs
	$(CXX) -O3 -std=c++17 -Wall -Wextra -fopenmp -Iinclude tools/replay_model_mesh.cpp src/tsdf_volume.cpp -o $@

bin/test_plane_patches: tools/test_plane_patches.cpp src/plane_patches.cpp include/plane_patches.hpp | dirs
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Iinclude tools/test_plane_patches.cpp src/plane_patches.cpp -o $@

bin/replay_plane_patches: tools/replay_plane_patches.cpp src/plane_patches.cpp include/plane_patches.hpp | dirs
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Iinclude tools/replay_plane_patches.cpp src/plane_patches.cpp -o $@

bin/test_depth_postprocess: tools/test_depth_postprocess.cpp include/model_depth_geometry.hpp include/depth_speckle_filter.hpp | dirs
	$(CXX) $(CXXFLAGS) -UNDEBUG $< -o $@

bin/test_meshing_pipeline: tools/test_meshing_pipeline.cpp include/meshing_pipeline.hpp | dirs
	$(CXX) $(CXXFLAGS) -UNDEBUG $< -o $@

bin/benchmark_depth_processing: tools/benchmark_depth_processing.cpp src/tsdf_volume.cpp include/meshing_pipeline.hpp include/model_depth_geometry.hpp include/depth_speckle_filter.hpp include/tsdf_volume.hpp | dirs
	$(CXX) $(CXXFLAGS) tools/benchmark_depth_processing.cpp src/tsdf_volume.cpp -o $@

bin/test_tsdf_e07: tools/test_tsdf_e07.cpp src/tsdf_volume.cpp include/tsdf_volume.hpp | dirs
	$(CXX) -O3 -std=c++17 -Wall -Wextra -fopenmp -Iinclude tools/test_tsdf_e07.cpp src/tsdf_volume.cpp -o $@

IR_DOT_TEST_SRC = src/ir_dot_refine.cpp src/ir_dot_bridge.cpp src/ir_pair_capture.cpp src/v4l2_capture_clock.cpp

bin/test_ir_dot_plane: tools/test_ir_dot_plane.cpp $(IR_DOT_TEST_SRC) include/ir_dot_refine.hpp include/ir_dot_bridge.hpp | dirs
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Iinclude tools/test_ir_dot_plane.cpp $(IR_DOT_TEST_SRC) -lpthread -o $@

bin/test_ir_dot_bridge: tools/test_ir_dot_bridge.cpp src/ir_dot_bridge.cpp include/ir_dot_bridge.hpp | dirs
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Iinclude tools/test_ir_dot_bridge.cpp src/ir_dot_bridge.cpp -lpthread -o $@

bin/probe_ir_pair: tools/probe_ir_pair.cpp src/ir_pair_capture.cpp src/v4l2_capture_clock.cpp include/ir_pair_capture.hpp | dirs
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Iinclude tools/probe_ir_pair.cpp src/ir_pair_capture.cpp src/v4l2_capture_clock.cpp -lpthread -o $@

bin/record_ir_pairs: tools/record_ir_pairs.cpp src/ir_pair_capture.cpp src/v4l2_capture_clock.cpp include/ir_pair_capture.hpp | dirs
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Iinclude tools/record_ir_pairs.cpp src/ir_pair_capture.cpp src/v4l2_capture_clock.cpp -lpthread -o $@

bin/record_head_poses: tools/record_head_poses.cpp | dirs
	$(CXX) -O2 -std=c++17 -Wall -Wextra -I/opt/steamvr/tools/hellovr_vulkan_linux/src/openvr/headers $< -L/opt/steamvr/bin/linuxarm64 -lopenvr_api -Wl,-rpath=/opt/steamvr/bin/linuxarm64 -o $@

bin/mark_surfaces: tools/mark_surfaces.cpp | dirs
	$(CXX) -O2 -std=c++17 -Wall -Wextra -I/opt/steamvr/tools/hellovr_vulkan_linux/src/openvr/headers $< -L/opt/steamvr/bin/linuxarm64 -lopenvr_api -Wl,-rpath=/opt/steamvr/bin/linuxarm64 -o $@

bin/model_batch_to_v2: tools/model_batch_to_v2.cpp include/model_depth_geometry.hpp | dirs
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Iinclude $< -o $@

bin/test_persistent_depth: tools/test_persistent_depth.cpp src/persistent_depth.cpp include/persistent_depth.hpp | dirs
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Iinclude tools/test_persistent_depth.cpp src/persistent_depth.cpp -o $@

bin/test_occlusion_completion: tools/test_occlusion_completion.cpp include/occlusion_completion.hpp include/occlusion_surface.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

bin/evaluate_occlusion_completion: tools/evaluate_occlusion_completion.cpp src/vulkan_stereo_pipeline.o include/occlusion_completion.hpp include/occlusion_surface.hpp | dirs $(SPV)
	$(CXX) $(CXXFLAGS) tools/evaluate_occlusion_completion.cpp src/vulkan_stereo_pipeline.o -lvulkan -o $@

bin/test_vulkan_stereo: tools/test_vulkan_stereo.cpp src/vulkan_stereo_pipeline.o | dirs $(SPV)
	$(CXX) $(CXXFLAGS) tools/test_vulkan_stereo.cpp src/vulkan_stereo_pipeline.o -lvulkan -o $@

bin/capture_rectified_pair: tools/capture_rectified_pair.cpp src/dma_capture.o src/timed_stereo_capture.o src/v4l2_capture_clock.o src/vk_detile.o src/depth_pipeline.o | dirs
	$(CXX) $(CXXFLAGS) tools/capture_rectified_pair.cpp src/dma_capture.o src/timed_stereo_capture.o src/v4l2_capture_clock.o src/vk_detile.o src/depth_pipeline.o -lvulkan -lpthread -o $@

shaders/occlusion_demo.frag.spv: shaders/occlusion_demo.frag
	glslc -O $< -o $@

bin/analyze_occlusion_benchmark: tools/analyze_occlusion_benchmark.cpp include/occlusion_benchmark.hpp include/occlusion_surface.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -lpthread -o $@

bin/test_occlusion_benchmark: tools/test_occlusion_benchmark.cpp include/occlusion_benchmark.hpp include/occlusion_surface.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -lpthread -o $@

# Diagnostic variants have extra storage outputs; production shaders stay unchanged.
DIAGNOSTIC_SPV = shaders/stereo_match_diagnostic.spv shaders/stereo_filter_diagnostic.spv shaders/stereo_support_diagnostic.spv
shaders/%_diagnostic.spv: shaders/%.comp
	glslc -O -DFRAME_STEREO_DIAGNOSTICS $< -o $@

bin/test_vulkan_stereo bin/evaluate_occlusion_completion: | $(DIAGNOSTIC_SPV)

bin/replay_occlusion_stages: tools/replay_occlusion_stages.cpp src/vulkan_stereo_pipeline.o include/occlusion_benchmark.hpp include/occlusion_completion.hpp | dirs $(SPV) $(DIAGNOSTIC_SPV)
	$(CXX) $(CXXFLAGS) tools/replay_occlusion_stages.cpp src/vulkan_stereo_pipeline.o -lvulkan -lpthread -o $@

bin/test_occlusion_surface: tools/test_occlusion_surface.cpp include/occlusion_surface.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

bin/test_occlusion_expiry: tools/test_occlusion_expiry.cpp include/occlusion_surface.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

bin/test_occlusion_budget: tools/test_occlusion_budget.cpp include/occlusion_benchmark.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -lpthread -o $@

bin/analyze_occlusion_mesh: tools/analyze_occlusion_mesh.cpp include/occlusion_benchmark.hpp include/occlusion_surface.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -lpthread -o $@

# The layer tests include its implementation to exercise intercepted calls.
OCCLUSION_TEST_SUPPORT = src/vulkan_guarded_stereo_pipeline.o src/occlusion_renderer.o src/cloud_worker.o src/dma_capture.o src/timed_stereo_capture.o src/v4l2_capture_clock.o src/vk_detile.o src/depth_pipeline.o src/vulkan_stereo_pipeline.o
bin/test_occlusion_renderer: tools/test_occlusion_renderer.cpp src/occlusion_renderer.o | dirs $(OCCLUSION_SPV)
	$(CXX) $(CXXFLAGS) $^ -lvulkan -lpthread -o $@

bin/test_occlusion_layer: tools/test_occlusion_layer.cpp src/openxr_occlusion_layer.cpp $(OCCLUSION_TEST_SUPPORT) | dirs $(OCCLUSION_SPV) $(SPV)
	$(CXX) $(CXXFLAGS) tools/test_occlusion_layer.cpp $(OCCLUSION_TEST_SUPPORT) -lvulkan -lpthread -o $@

bin/test_occlusion_expiry_gpu: tools/test_occlusion_expiry_gpu.cpp tools/test_occlusion_renderer.cpp src/openxr_occlusion_layer.cpp $(OCCLUSION_TEST_SUPPORT) | dirs $(OCCLUSION_SPV) $(SPV)
	$(CXX) $(CXXFLAGS) tools/test_occlusion_expiry_gpu.cpp $(OCCLUSION_TEST_SUPPORT) -lvulkan -lpthread -o $@

bin/replay_occlusion_stages: include/occlusion_stage_diagnostics.hpp include/vulkan_stereo_pipeline.hpp include/occlusion_surface.hpp include/depth_speckle_filter.hpp
bin/evaluate_occlusion_completion: include/occlusion_stage_diagnostics.hpp include/vulkan_stereo_pipeline.hpp include/depth_speckle_filter.hpp

bin/test_occlusion_history_events: tools/test_occlusion_history_events.cpp include/occlusion_surface.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

bin/test_occlusion_capture_generation: tools/test_occlusion_capture_generation.cpp src/vulkan_guarded_stereo_pipeline.o src/cloud_worker.o src/dma_capture.o src/timed_stereo_capture.o src/v4l2_capture_clock.o src/vk_detile.o src/depth_pipeline.o src/vulkan_stereo_pipeline.o | dirs $(SPV)
	$(CXX) $(CXXFLAGS) $^ -lvulkan -lpthread -o $@

-include $(OCCLUSION_TEST_SUPPORT:.o=.d) $(wildcard bin/*.d)

# Matcher research tools are opt-in and are not linked into the live runtime.
bin/compare_stereo_matchers: tools/compare_stereo_matchers.cpp include/stereo_matcher_experiment.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

bin/test_stereo_matcher_experiment: tools/test_stereo_matcher_experiment.cpp include/stereo_matcher_experiment.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

bin/test_stereo_matcher_experiment_gpu: tools/test_stereo_matcher_experiment.cpp include/stereo_matcher_experiment.hpp src/vulkan_stereo_pipeline.o | dirs $(SPV) $(DIAGNOSTIC_SPV)
	$(CXX) $(CXXFLAGS) -DFRAME_STEREO_EXPERIMENT_VULKAN tools/test_stereo_matcher_experiment.cpp src/vulkan_stereo_pipeline.o -lvulkan -o $@

bin/export_stereo_matcher_inputs: tools/export_stereo_matcher_inputs.cpp src/vulkan_stereo_pipeline.o include/occlusion_benchmark.hpp include/occlusion_stage_diagnostics.hpp include/depth_speckle_filter.hpp | dirs $(SPV) $(DIAGNOSTIC_SPV)
	$(CXX) $(CXXFLAGS) tools/export_stereo_matcher_inputs.cpp src/vulkan_stereo_pipeline.o -lvulkan -lpthread -o $@

bin/test_stereo_adaptive: tools/test_stereo_adaptive.cpp include/stereo_matcher_experiment.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

bin/test_stereo_guarded: tools/test_stereo_guarded.cpp include/stereo_matcher_experiment.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

GUARDED_SPV = shaders/stereo_guarded_arms.spv shaders/stereo_guarded_seed.spv shaders/stereo_guarded_anchors.spv shaders/stereo_guarded_masks.spv shaders/stereo_guarded_match.spv
GUARDED_DIAGNOSTIC_SPV = shaders/stereo_guarded_match_diagnostic.spv

bin/preview_stereo_guarded: tools/preview_stereo_guarded.cpp src/overlay_renderer.o src/dma_capture.o src/timed_stereo_capture.o src/v4l2_capture_clock.o src/vk_detile.o src/depth_pipeline.o src/vulkan_stereo_pipeline.o src/vulkan_guarded_stereo_pipeline.o include/stereo_matcher_experiment.hpp include/depth_speckle_filter.hpp | dirs $(SPV) $(GUARDED_SPV)
	$(CXX) $(CXXFLAGS) tools/preview_stereo_guarded.cpp src/overlay_renderer.o src/dma_capture.o src/timed_stereo_capture.o src/v4l2_capture_clock.o src/vk_detile.o src/depth_pipeline.o src/vulkan_stereo_pipeline.o src/vulkan_guarded_stereo_pipeline.o -L/opt/steamvr/bin/linuxarm64 -lopenvr_api -Wl,-rpath,/opt/steamvr/bin/linuxarm64 -lpthread -lm -lvulkan -o $@

bin/test_vulkan_guarded: tools/test_vulkan_guarded.cpp src/vulkan_guarded_stereo_pipeline.o include/stereo_matcher_experiment.hpp include/vulkan_guarded_stereo_pipeline.hpp | dirs $(SPV) $(DIAGNOSTIC_SPV) $(GUARDED_SPV) $(GUARDED_DIAGNOSTIC_SPV)
	$(CXX) $(CXXFLAGS) tools/test_vulkan_guarded.cpp src/vulkan_guarded_stereo_pipeline.o -lvulkan -o $@

-include src/vulkan_guarded_stereo_pipeline.d

$(OPENXR_TARGET): | $(GUARDED_SPV)

$(OCCLUSION_TARGET): | $(GUARDED_SPV)

bin/benchmark_stereo_guarded: tools/benchmark_stereo_guarded.cpp src/vulkan_guarded_stereo_pipeline.o | dirs $(SPV) $(GUARDED_SPV)
	$(CXX) $(CXXFLAGS) $^ -lvulkan -o $@


bin/test_model_depth_geometry: tools/test_model_depth_geometry.cpp include/model_depth_geometry.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

bin/test_depth_features: tools/test_depth_features.cpp src/depth_features.cpp include/depth_features.hpp include/depth_observation.hpp | dirs
	$(CXX) $(CXXFLAGS) tools/test_depth_features.cpp src/depth_features.cpp -o $@

bin/test_model_depth_pipeline: tools/test_model_depth_pipeline.cpp src/model_depth_pipeline.o src/model_temporal.o src/vulkan_stereo_pipeline.o | dirs $(SPV)
	$(CXX) $(CXXFLAGS) tools/test_model_depth_pipeline.cpp src/model_depth_pipeline.o src/model_temporal.o src/vulkan_stereo_pipeline.o -lvulkan -lpthread -o $@

bin/test_model_temporal: tools/test_model_temporal.cpp src/model_temporal.cpp include/model_temporal.hpp | dirs
	$(CXX) $(CXXFLAGS) tools/test_model_temporal.cpp src/model_temporal.cpp -o $@

bin/probe_timed_pairs: tools/probe_timed_pairs.cpp src/dma_capture.o src/timed_stereo_capture.o src/v4l2_capture_clock.o src/vk_detile.o | dirs
	$(CXX) $(CXXFLAGS) $^ -lvulkan -lpthread -o $@

bin/capture_model_inputs: tools/capture_model_inputs.cpp src/dma_capture.o src/timed_stereo_capture.o src/v4l2_capture_clock.o src/vk_detile.o src/depth_pipeline.o | dirs
	$(CXX) $(CXXFLAGS) $^ -lvulkan -lpthread -o $@

bin/rectify_raw_pair: tools/rectify_raw_pair.cpp src/depth_pipeline.o | dirs
	$(CXX) $(CXXFLAGS) $^ -lpthread -o $@

bin/probe_capture_timing: tools/probe_capture_timing.cpp src/dma_capture.o src/timed_stereo_capture.o src/v4l2_capture_clock.o src/vk_detile.o | dirs
	$(CXX) $(CXXFLAGS) $^ -lvulkan -lpthread -o $@

# Native learned-depth worker; all host/model dependencies are built from the monorepo.
bin/frame_depth_worker: ../CMakeLists.txt ../hexecutor/CMakeLists.txt \
    ../hexecutor/host/cpp/executor.cpp ../hexecutor/host/cpp/frame_transport.cpp \
    ../hexecutor/host/cpp/detail.hpp ../hexecutor/include/hexecutor/executor.hpp \
    ../hexecutor/native/program/nprg2_runtime.c tools/frame_depth_worker.cpp \
    include/depth_worker_protocol.hpp src/depth_worker_protocol.cpp \
    ../samples/depth-model/include/depth_model/deployment.hpp \
    ../samples/depth-model/deployment/cpp/package.cpp | dirs
	$(MAKE) -C .. build-worker
	cp ../build/bin/frame_depth_worker $@
$(OPENXR_TARGET) $(MESH_TARGET): | bin/frame_depth_worker
