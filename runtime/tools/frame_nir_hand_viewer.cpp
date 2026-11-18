// Native stereo NIR capture stream -> hand tracker -> transparent OpenXR layer.
#include "hand_viewer_xr.hpp"
#include "nirhands/pipeline.hpp"
#include "hexecutor/executor.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <pthread.h>
#include <sched.h>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {
constexpr int W = 1056, H = 1024;
constexpr size_t Pixels = size_t(W) * H;
volatile std::sig_atomic_t interrupted = 0;
std::atomic<bool> stop{false};
void interrupt(int) { interrupted = 1; }
uint64_t clock_ns(clockid_t id) {
    timespec t{};
    clock_gettime(id, &t);
    return uint64_t(t.tv_sec) * 1000000000ull + t.tv_nsec;
}
int64_t monotonic_minus_raw() {
    int64_t offset = 0, best = INT64_MAX;
    for (int i = 0; i < 3; ++i) {
        const auto a = clock_ns(CLOCK_MONOTONIC), r = clock_ns(CLOCK_MONOTONIC_RAW);
        const auto b = clock_ns(CLOCK_MONOTONIC);
        if (int64_t(b - a) < best) { best = int64_t(b - a); offset = int64_t((a + b) / 2) - int64_t(r); }
    }
    return offset;
}
struct Header { char magic[8]; int64_t left_ns, right_ns; uint32_t left_seq, right_seq; };
static_assert(sizeof(Header) == 32, "FRNIR001 header size");
struct Pair { Header header{}; std::array<std::vector<uint8_t>, 2> image{{std::vector<uint8_t>(Pixels), std::vector<uint8_t>(Pixels)}}; };
struct Shared {
    std::mutex mutex;
    std::shared_ptr<Pair> newest;
    std::shared_ptr<const HandScene> scene;
    uint64_t published = 0;
    std::atomic<bool> eof{false}, failed{false};
};
bool read_bytes(void* dst, size_t n) {
    auto* p=static_cast<uint8_t*>(dst);
    while (n && !stop) {
        pollfd fd{STDIN_FILENO,POLLIN,0};
        if (poll(&fd,1,100)<0) { if (errno == EINTR) continue; return false; }
        if (!fd.revents) continue;
        const auto got=read(STDIN_FILENO,p,n);
        if (got<0 && errno == EINTR) continue;
        if (got<=0) return false;
        p+=got; n-=size_t(got);
    }
    return n==0;
}
std::array<float,3> rotate(const std::array<float,4>& q, std::array<float,3> v) {
    const std::array<float,3> u{q[0], q[1], q[2]};
    const std::array<float,3> t{2*(u[1]*v[2]-u[2]*v[1]), 2*(u[2]*v[0]-u[0]*v[2]), 2*(u[0]*v[1]-u[1]*v[0])};
    for (int k=0;k<3;++k) v[k]+=q[3]*t[k]+u[(k+1)%3]*t[(k+2)%3]-u[(k+2)%3]*t[(k+1)%3];
    return v;
}
}
int main(int argc, char** argv) {
    std::string models, qnn16_dir, xrservice="/persist/xrservice.json", device="/persist/device_config.json";
    hexecutor::Artifacts artifacts;
    int threads=2, seconds=0;
    nirhands::TrackerConfig tracker;
    try {
        for (int i=1;i<argc;++i) {
            const std::string a=argv[i];
            auto value=[&]() -> std::string { if (++i>=argc) throw std::runtime_error("missing value for " + a); return argv[i]; };
            if (a=="--models") models=value();
            else if (a=="--qnn16-dir") qnn16_dir=value();
            else if (a=="--loader") artifacts.loader=value();
            else if (a=="--engine") artifacts.engine=value();
            else if (a=="--bootstrap") artifacts.bootstrap=value();
            else if (a=="--chunk-probe") artifacts.chunk_probe=value();
            else if (a=="--xrservice") xrservice=value();
            else if (a=="--device-config") device=value();
            else if (a=="--threads") threads=std::stoi(value());
            else if (a=="--seconds") seconds=std::stoi(value());
            else if (a=="--output") {
                const std::string o=value();
                if (o=="skeleton") tracker.output=nirhands::TrackerConfig::Output::Skeleton;
                else if (o=="heatmap") tracker.output=nirhands::TrackerConfig::Output::Heatmap;
                else if (o=="combined") tracker.output=nirhands::TrackerConfig::Output::Combined;
                else throw std::runtime_error("--output must be skeleton, heatmap or combined");
            }
            else if (a=="--heatmap-weight") tracker.heatmap_angle_weight=std::stod(value());
            else if (a=="--help") { std::cout << "frame_nir_hand_viewer --models DIR [--qnn16-dir DIR --loader ELF --engine ELF --bootstrap ELF --chunk-probe ELF] [--threads N] [--seconds N]\n"
                                                 "  [--output skeleton|heatmap|combined] [--heatmap-weight W]\n"
                                                 "  skeleton models only; combined = wearer's bones, finger angles (1-W) skeleton + W heatmap (W=0.5)\n"; return 0; }
            else throw std::runtime_error("unknown option " + a);
        }
        if (models.empty() || threads<1 || threads>8 || seconds<0) throw std::runtime_error("invalid --models, --threads or --seconds");
        if (!qnn16_dir.empty() && (artifacts.loader.empty() || artifacts.engine.empty() ||
            artifacts.bootstrap.empty() || artifacts.chunk_probe.empty()))
            throw std::runtime_error("NIR QNN16 requires --loader, --engine, --bootstrap and --chunk-probe");
        std::signal(SIGINT, interrupt); std::signal(SIGTERM, interrupt);
        auto rig=nirhands::load_rig(xrservice,device);
        // ORT creates its thread pools during session construction. Restore the main
        // thread afterward; ORT workers retain the inherited affinity mask.
        cpu_set_t original, inference;
        const bool have_affinity = pthread_getaffinity_np(pthread_self(), sizeof(original), &original) == 0;
        CPU_ZERO(&inference);
        if (have_affinity) for (int cpu=2; cpu<8; ++cpu) if (CPU_ISSET(cpu, &original)) CPU_SET(cpu, &inference);
        const bool pinned = qnn16_dir.empty() && have_affinity && CPU_COUNT(&inference) && pthread_setaffinity_np(pthread_self(), sizeof(inference), &inference) == 0;
        std::shared_ptr<nirhands::Models> nets;
        try { if (qnn16_dir.empty()) nets=std::make_shared<nirhands::Models>(models,threads); }
        catch (...) { if (pinned) pthread_setaffinity_np(pthread_self(), sizeof(original), &original); throw; }
        if (pinned) pthread_setaffinity_np(pthread_self(), sizeof(original), &original);
        else if (qnn16_dir.empty()) std::cerr << "Warning: could not pin ORT threads to CPUs 2–7\n";
        HandViewerXr viewer; viewer.initialize();
        Shared shared;
        std::thread reader([&] {
            while (!stop) {
                auto pair=std::make_shared<Pair>();
                if (!read_bytes(&pair->header,sizeof(Header))) break;
                if (std::memcmp(pair->header.magic,"FRNIR001",8)) { std::cerr << "expected FRNIR001 stream\n"; shared.failed=true; break; }
                if (!read_bytes(pair->image[0].data(),Pixels) || !read_bytes(pair->image[1].data(),Pixels)) break;
                std::lock_guard<std::mutex> lock(shared.mutex); shared.newest=std::move(pair);
            }
            shared.eof=true;
        });
        std::thread worker([&] {
            try {
                std::unique_ptr<hexecutor::Session> session;
                std::mutex session_mutex;
                if (!qnn16_dir.empty()) {
                    const auto root=std::filesystem::path(qnn16_dir);
                    std::vector<hexecutor::Program> programs;
                    for (const char* name : {"nirhands_detector", "nirhands_keypoints"}) {
                        const auto prefix=root/name;
                        programs.push_back(hexecutor::load_qnn16_program(prefix.string()+".q16p",prefix.string()+".q16c"));
                    }
                    session=std::make_unique<hexecutor::Session>(std::move(programs),artifacts);
                    nets=std::make_shared<nirhands::Models>(models,[&](size_t index,const std::vector<nirhands::Models::Buffer>& inputs,const nirhands::Geometry*) {
                        std::vector<hexecutor::Buffer> request;
                        for (const auto& input : inputs) request.push_back({input.data,input.bytes});
                        std::lock_guard<std::mutex> lock(session_mutex);
                        return session->run(request,index);
                    });
                }
                nirhands::Pipeline pipeline(rig,nets,tracker);
                std::shared_ptr<Pair> last;
                cpu_set_t inference; CPU_ZERO(&inference);
                for (int cpu=2; cpu<8; ++cpu) CPU_SET(cpu, &inference);
                if (qnn16_dir.empty()) pthread_setaffinity_np(pthread_self(), sizeof(inference), &inference);
                while (!stop) {
                    std::shared_ptr<Pair> pair;
                    { std::lock_guard<std::mutex> lock(shared.mutex); pair=shared.newest; }
                    if (!pair || pair==last) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
                    last=pair;
                    const auto result=pipeline.step({nirhands::Image{pair->image[0].data(),W,H},nirhands::Image{pair->image[1].data(),W,H}}, double(pair->header.left_ns)*1e-9);
                    auto scene=std::make_shared<HandScene>();
                    HeadPose head;
                    const int64_t captured=int64_t((pair->header.left_ns+pair->header.right_ns)/2)+monotonic_minus_raw();
                    scene->local_space=captured>0 && viewer.headPoseAt(uint64_t(captured),head);
                    for (const auto& hand:result.hands) {
                        HandSkeleton skeleton; skeleton.right=hand.is_right; skeleton.joint_count=21;
                        for (int j=0;j<21;++j) {
                            auto p=std::array<float,3>{float(hand.joints[j][0]),float(hand.joints[j][1]),float(hand.joints[j][2])};
                            if (scene->local_space) { p=rotate(head.orientation,p); for (int k=0;k<3;++k) p[k]+=head.position[k]; }
                            skeleton.joints[j]=p;
                        }
                        scene->hands.push_back(skeleton);
                    }
                    static uint64_t step_count = 0;
                    const uint64_t now_mono = clock_ns(CLOCK_MONOTONIC);
                    if (++step_count % 30 == 1) {
                        const double latency_ms = captured > 0 ? double(int64_t(now_mono) - captured) * 1e-6 : 0.0;
                        int combined=0;
                        for (const auto& hand:result.hands) combined+=hand.combined;
                        std::cout << "[viewer] step " << step_count << ": hands=" << result.hands.size()
                                  << " combined=" << combined
                                  << " local_space=" << scene->local_space << " latency=" << latency_ms << "ms\n" << std::flush;
                    }
                    { std::lock_guard<std::mutex> lock(shared.mutex); shared.scene=scene; shared.published=now_mono; }
                }
            } catch (const std::exception& e) { std::cerr << "NIR inference: " << e.what() << '\n'; shared.failed=true; stop=true; }
        });
        const uint64_t started=clock_ns(CLOCK_MONOTONIC);
        const HandScene empty;
        try {
        while (!stop && !interrupted && !shared.eof && viewer.pollEvents()) {
            if (seconds && clock_ns(CLOCK_MONOTONIC)-started>=uint64_t(seconds)*1000000000ull) break;
            std::shared_ptr<const HandScene> scene;
            { std::lock_guard<std::mutex> lock(shared.mutex); if (clock_ns(CLOCK_MONOTONIC)-shared.published<500000000ull) scene=shared.scene; }
            viewer.render(scene ? *scene : empty);
        }
        } catch (...) { stop=true; reader.join(); worker.join(); throw; }
        stop=true;
        reader.join(); worker.join();
        return shared.failed ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
