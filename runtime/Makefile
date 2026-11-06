CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -fopenmp -MMD -MP -Iinclude -I/opt/steamvr/tools/hellovr_vulkan_linux/src/openvr/headers
LDFLAGS = -L/opt/steamvr/bin/linuxarm64 -lopenvr_api -Wl,-rpath=/opt/steamvr/bin/linuxarm64 -lpthread -lm -lvulkan

MNN_PREFIX ?= $(HOME)/.local/frame-mnn
ifneq ($(wildcard $(MNN_PREFIX)/include/MNN/Interpreter.hpp),)
CXXFLAGS += -DFRAME_WITH_MNN -I$(MNN_PREFIX)/include
LDFLAGS += -L$(MNN_PREFIX)/lib -lMNN -Wl,-rpath,$(MNN_PREFIX)/lib
endif

SRC = src/main.cpp src/dma_capture.cpp src/overlay_renderer.cpp src/vk_detile.cpp src/depth_pipeline.cpp src/gpu_depth_pipeline.cpp src/vulkan_stereo_pipeline.cpp
OBJ = $(SRC:.cpp=.o)
DEP = $(OBJ:.o=.d)
TARGET = bin/frame_spatial_runtime
SPV = shaders/stereo_census.spv shaders/stereo_match.spv shaders/stereo_filter.spv shaders/stereo_support.spv
OPENXR_TARGET = bin/frame_openxr_point_cloud
MESH_TARGET = bin/frame_openxr_mesh_viewer
OPENXR_SRC = src/openxr_point_cloud.cpp src/cloud_worker.cpp src/tsdf_volume.cpp src/model_depth_pipeline.cpp src/persistent_depth.cpp src/vulkan_guarded_stereo_pipeline.cpp src/dma_capture.cpp src/vk_detile.cpp src/depth_pipeline.cpp src/vulkan_stereo_pipeline.cpp
OPENXR_OBJ = $(OPENXR_SRC:.cpp=.o)
POINT_SPV = shaders/point_cloud.vert.spv shaders/point_cloud.frag.spv shaders/occlusion_demo.frag.spv
MESH_SPV = shaders/mesh.vert.spv shaders/mesh.frag.spv

all: dirs $(SPV) $(POINT_SPV) $(MESH_SPV) $(TARGET) $(OPENXR_TARGET) $(MESH_TARGET)

dirs:
	mkdir -p bin

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OPENXR_TARGET): $(OPENXR_OBJ) | dirs $(SPV) $(POINT_SPV) $(MESH_SPV)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L/opt/steamvr/bin/linuxarm64 \
		-lopenxr_loader -Wl,-rpath=/opt/steamvr/bin/linuxarm64 -lpthread -lm -lvulkan

$(MESH_TARGET): $(OPENXR_TARGET) | dirs
	ln -sf frame_openxr_point_cloud $(MESH_TARGET)

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

OCCLUSION_TARGET = bin/libframe_openxr_occlusion.so
OCCLUSION_SRC = src/openxr_occlusion_layer.cpp src/occlusion_renderer.cpp src/cloud_worker.cpp src/tsdf_volume.cpp src/model_depth_pipeline.cpp src/persistent_depth.cpp src/vulkan_guarded_stereo_pipeline.cpp src/dma_capture.cpp src/vk_detile.cpp src/depth_pipeline.cpp src/vulkan_stereo_pipeline.cpp
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

bin/test_persistent_depth: tools/test_persistent_depth.cpp src/persistent_depth.cpp include/persistent_depth.hpp | dirs
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Iinclude tools/test_persistent_depth.cpp src/persistent_depth.cpp -o $@

bin/test_occlusion_completion: tools/test_occlusion_completion.cpp include/occlusion_completion.hpp include/occlusion_surface.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

bin/evaluate_occlusion_completion: tools/evaluate_occlusion_completion.cpp src/vulkan_stereo_pipeline.o include/occlusion_completion.hpp include/occlusion_surface.hpp | dirs $(SPV)
	$(CXX) $(CXXFLAGS) tools/evaluate_occlusion_completion.cpp src/vulkan_stereo_pipeline.o -lvulkan -o $@

bin/test_vulkan_stereo: tools/test_vulkan_stereo.cpp src/vulkan_stereo_pipeline.o | dirs $(SPV)
	$(CXX) $(CXXFLAGS) tools/test_vulkan_stereo.cpp src/vulkan_stereo_pipeline.o -lvulkan -o $@

bin/capture_rectified_pair: tools/capture_rectified_pair.cpp src/dma_capture.o src/vk_detile.o src/depth_pipeline.o | dirs
	$(CXX) $(CXXFLAGS) tools/capture_rectified_pair.cpp src/dma_capture.o src/vk_detile.o src/depth_pipeline.o -lvulkan -lpthread -o $@

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
OCCLUSION_TEST_SUPPORT = src/vulkan_guarded_stereo_pipeline.o src/occlusion_renderer.o src/cloud_worker.o src/dma_capture.o src/vk_detile.o src/depth_pipeline.o src/vulkan_stereo_pipeline.o
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

bin/test_occlusion_capture_generation: tools/test_occlusion_capture_generation.cpp src/vulkan_guarded_stereo_pipeline.o src/cloud_worker.o src/dma_capture.o src/vk_detile.o src/depth_pipeline.o src/vulkan_stereo_pipeline.o | dirs $(SPV)
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

bin/preview_stereo_guarded: tools/preview_stereo_guarded.cpp src/overlay_renderer.o src/dma_capture.o src/vk_detile.o src/depth_pipeline.o src/vulkan_stereo_pipeline.o src/vulkan_guarded_stereo_pipeline.o include/stereo_matcher_experiment.hpp include/depth_speckle_filter.hpp | dirs $(SPV) $(GUARDED_SPV)
	$(CXX) $(CXXFLAGS) tools/preview_stereo_guarded.cpp src/overlay_renderer.o src/dma_capture.o src/vk_detile.o src/depth_pipeline.o src/vulkan_stereo_pipeline.o src/vulkan_guarded_stereo_pipeline.o -L/opt/steamvr/bin/linuxarm64 -lopenvr_api -Wl,-rpath,/opt/steamvr/bin/linuxarm64 -lpthread -lm -lvulkan -o $@

bin/test_vulkan_guarded: tools/test_vulkan_guarded.cpp src/vulkan_guarded_stereo_pipeline.o include/stereo_matcher_experiment.hpp include/vulkan_guarded_stereo_pipeline.hpp | dirs $(SPV) $(DIAGNOSTIC_SPV) $(GUARDED_SPV) $(GUARDED_DIAGNOSTIC_SPV)
	$(CXX) $(CXXFLAGS) tools/test_vulkan_guarded.cpp src/vulkan_guarded_stereo_pipeline.o -lvulkan -o $@

-include src/vulkan_guarded_stereo_pipeline.d

$(OPENXR_TARGET): | $(GUARDED_SPV)

$(OCCLUSION_TARGET): | $(GUARDED_SPV)

bin/benchmark_stereo_guarded: tools/benchmark_stereo_guarded.cpp src/vulkan_guarded_stereo_pipeline.o | dirs $(SPV) $(GUARDED_SPV)
	$(CXX) $(CXXFLAGS) $^ -lvulkan -o $@


bin/test_model_depth_geometry: tools/test_model_depth_geometry.cpp include/model_depth_geometry.hpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

bin/test_model_depth_pipeline: tools/test_model_depth_pipeline.cpp src/model_depth_pipeline.o src/vulkan_stereo_pipeline.o | dirs $(SPV)
	$(CXX) $(CXXFLAGS) tools/test_model_depth_pipeline.cpp src/model_depth_pipeline.o src/vulkan_stereo_pipeline.o -lvulkan -lpthread -o $@
