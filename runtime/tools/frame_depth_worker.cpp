#include "depth_model/deployment.hpp"
#include "depth_worker_protocol.hpp"
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#ifdef __linux__
#include <sched.h>
#endif

namespace {
std::atomic<bool> stopped{false};
void request_stop(int) { stopped = true; }
class NativeInference final : public FrameSpatial::DepthInference {
    hexecutor::Session session_;

  public:
    explicit NativeInference(const depth_model::Deployment &d) : session_(d.program, d.artifacts) {}
    std::vector<hexecutor::Bytes> run(const std::vector<hexecutor::Buffer> &buffers) override {
        return session_.run(buffers);
    }
    void close() override { session_.close(); }
};
} // namespace
int main(int argc, char **argv) {
    try {
        FrameSpatial::WorkerOptions options;
        std::string backend = "prepared-qnn", model = "primary", package;
        bool check = false;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            auto value = [&]() {
                if (++i == argc)
                    throw std::invalid_argument("missing value for " + arg);
                return std::string(argv[i]);
            };
            if (arg == "--help") {
                std::cout << "frame_depth_worker --fd FD --package PATH [--backend "
                             "prepared-qnn|prepared] [--model primary|alternate] [--host-temporal] "
                             "[--second-disparity] [--close-after-reply] [--check-package]\n";
                return 0;
            } else if (arg == "--fd")
                options.fd = std::stoi(value());
            else if (arg == "--package")
                package = value();
            else if (arg == "--model")
                model = value();
            else if (arg == "--backend")
                backend = value();
            else if (arg == "--spherical-deg") {
                (void)std::stod(value());
            } else if (arg == "--host-temporal")
                options.hostTemporal = true;
            else if (arg == "--second-disparity")
                options.secondDisparity = true;
            else if (arg == "--close-after-reply")
                options.closeAfterReply = true;
            else if (arg == "--check-package")
                check = true;
            else
                throw std::invalid_argument("unknown option: " + arg);
        }
        if (package.empty() || (!check && options.fd < 0))
            throw std::invalid_argument("--package and --fd are required");
        if (auto *value = std::getenv("FRAME_DEPTH_IDLE_CLOSE_S"))
            options.idleCloseSeconds = std::stod(value);
        auto deployment = depth_model::load_deployment(package, backend, model);
        if (check) {
            std::cout << "Depth package hashes verified; no hardware accessed\n";
            return 0;
        }
#ifdef __linux__
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        CPU_SET(7, &cpus);
        if (sched_setaffinity(0, sizeof(cpus), &cpus) != 0)
            throw std::runtime_error("cannot set worker CPU affinity");
#endif
        std::signal(SIGTERM, request_stop);
        std::signal(SIGINT, request_stop);
        std::signal(SIGPIPE, SIG_IGN);
        FrameSpatial::serve_depth(
            options, [&]() { return std::make_unique<NativeInference>(deployment); }, stopped);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[DepthWorker] " << error.what() << '\n';
        return 1;
    }
}
