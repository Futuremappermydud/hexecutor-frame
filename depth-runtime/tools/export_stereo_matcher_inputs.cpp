// Extract reproducible matcher-only inputs from a complete v2 recording.
// Runs the unchanged GPU matcher; no completion, history, capture or live layer.
#include "occlusion_benchmark.hpp"
#include "occlusion_stage_diagnostics.hpp"
#include "depth_speckle_filter.hpp"
#include <filesystem>
#include <set>
#include <iomanip>

namespace B = FrameSpatial::OcclusionBenchmark;
using namespace FrameSpatial;

int main(int argc, char** argv) try {
    if (argc != 5) {
        std::cerr << "Usage: export_stereo_matcher_inputs recording.bin shader-dir output-dir frame-ids\n";
        return 2;
    }
    std::set<uint64_t> selected, found;
    std::istringstream ids(argv[4]);
    std::string token;
    while (std::getline(ids, token, ',')) {
        size_t used = 0;
        auto id = std::stoull(token, &used);
        if (!id || used != token.size()) throw std::runtime_error("Invalid frame ID");
        selected.insert(id);
    }
    if (selected.empty()) throw std::runtime_error("No frame IDs");
    // Validate the entire file before exporting a partial selection.
    B::Reader validation(argv[1]); B::Frame frame;
    uint64_t previous = 0, previousTime = 0;
    while (validation.next(frame)) {
        if (previous && (frame.meta.frame != previous + 1 || frame.meta.captureBeginNs <= previousTime))
            throw std::runtime_error("Recording contains gaps or unordered timestamps");
        previous = frame.meta.frame; previousTime = frame.meta.captureBeginNs;
        if (selected.count(previous)) found.insert(previous);
    }
    if (!validation.complete || validation.dropped || found != selected)
        throw std::runtime_error("Incomplete recording or requested frames missing");
    if (validation.header.width != 616 || validation.header.height != 616)
        throw std::runtime_error("GPU matcher requires 616x616 inputs");
    VulkanStereoPipeline production, diagnostic;
    if (!production.initialize(argv[2]) || !diagnostic.initialize(argv[2], true))
        throw std::runtime_error(production.lastError() + diagnostic.lastError());
    std::filesystem::create_directories(argv[3]);
    const std::string output = argv[3];
    std::ofstream metrics(output + "/inputs.csv");
    metrics << "frame,phase,gpu_supported_pixels,recorded_speckle_pixels,diagnostic_mask_mismatch,diagnostic_max_depth_delta_m,recorded_measured_mismatch,gpu_process_ms_p50,gpu_process_ms_p95\n";
    metrics << std::setprecision(9);
    B::Reader reader(argv[1]);
    while (reader.next(frame)) {
        if (!selected.count(frame.meta.frame)) continue;
        std::vector<float> depth, diagnosticDepth;
        StereoDiagnostics details;
        std::vector<double> timings;
        for (int run = 0; run < 6; ++run) {
            const auto begin = std::chrono::steady_clock::now();
            if (!production.process(frame.left, frame.right, depth))
                throw std::runtime_error(production.lastError());
            const auto end = std::chrono::steady_clock::now();
            if (run) timings.push_back(std::chrono::duration<double,std::milli>(end-begin).count());
        }
        if (!diagnostic.process(frame.left, frame.right, diagnosticDepth, &details))
            throw std::runtime_error(diagnostic.lastError());
        size_t diagnosticMaskMismatch = 0; float diagnosticMaxDelta = 0;
        for (size_t i = 0; i < depth.size(); ++i) {
            diagnosticMaskMismatch += (depth[i] > 0) != (diagnosticDepth[i] > 0);
            diagnosticMaxDelta = std::max(diagnosticMaxDelta, std::abs(depth[i] - diagnosticDepth[i]));
        }
        if (diagnosticMaskMismatch || diagnosticMaxDelta > 1e-5f)
            throw std::runtime_error("Diagnostic shader changed production depth");
        auto measured = depth;
        removeDepthSpeckles(measured, 616, 616);
        size_t measuredMismatch = 0;
        for (size_t i = 0; i < measured.size(); ++i) measuredMismatch += measured[i] != frame.measured[i];
        if (measuredMismatch) throw std::runtime_error("GPU replay does not reproduce recorded measurements");
        const auto prefix = output + "/frame-" + std::to_string(frame.meta.frame);
        writeDiagnosticLuma(prefix + "-left.pgm", frame.left, 616, 616);
        writeDiagnosticLuma(prefix + "-right.pgm", frame.right, 616, 616);
        writeDiagnosticMask(prefix, "gpu-supported", depth, 616, 616);
        writeDiagnosticMask(prefix, "recorded-measured", frame.measured, 616, 616);
        writeStereoDiagnostics(prefix, details, 616, 616);
        std::sort(timings.begin(), timings.end());
        const auto count = [](const auto& d) { return std::count_if(d.begin(),d.end(),[](float z){return z>0;}); };
        metrics << frame.meta.frame << ',' << B::phaseName(frame.meta.phase) << ',' << count(depth) << ','
                << count(frame.measured) << ',' << diagnosticMaskMismatch << ',' << diagnosticMaxDelta << ','
                << measuredMismatch << ',' << timings[2] << ',' << timings[4] << '\n';
        std::cout << "Exported frame " << frame.meta.frame << " (" << B::phaseName(frame.meta.phase) << ")\n";
    }
    if (!metrics) throw std::runtime_error("Cannot write input metrics");
    std::ofstream manifest(output + "/export.txt");
    manifest << "recording=" << argv[1] << "\nshader_directory=" << argv[2]
             << "\nselected_frames=" << argv[4] << "\nvalidated_frames=" << validation.frames
             << "\nground_truth=none\nGPU timing=end-to-end host process, warmup1 then5samples; live runtime co-running\n";
    if (!manifest) throw std::runtime_error("Cannot write provenance");
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
}
