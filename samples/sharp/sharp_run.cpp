// SPDX-License-Identifier: LGPL-3.0-or-later
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
using Args = std::vector<std::string>;
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;

void on_signal(int) { interrupted = 1; }

class Child {
public:
    Child(const Args &args, const fs::path &log = {}) {
        std::vector<char *> argv;
        for (const auto &arg : args)
            argv.push_back(const_cast<char *>(arg.c_str()));
        argv.push_back(nullptr);
        pid_ = fork();
        if (pid_ < 0)
            throw std::runtime_error("could not start " + args.front());
        if (pid_ == 0) {
            if (!log.empty()) {
                const int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
                if (fd < 0 || dup2(fd, STDOUT_FILENO) < 0 || dup2(fd, STDERR_FILENO) < 0)
                    _exit(127);
                if (fd > STDERR_FILENO)
                    close(fd);
            }
            execv(argv.front(), argv.data());
            _exit(127);
        }
    }

    Child(const Child &) = delete;
    Child &operator=(const Child &) = delete;

    ~Child() {
        if (pid_ > 0) {
            kill(pid_, SIGTERM);
            while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {}
        }
    }

    bool exited() {
        if (pid_ <= 0)
            return true;
        const pid_t done = waitpid(pid_, &status_, WNOHANG);
        if (done < 0 && errno != EINTR)
            throw std::runtime_error("waiting for a SHARP process failed");
        if (done == pid_)
            pid_ = -1;
        return pid_ <= 0;
    }

    void finish(const std::string &label) {
        while (!exited())
            pause();
        if (!WIFEXITED(status_) || WEXITSTATUS(status_))
            throw std::runtime_error("SHARP phase failed: " + label);
    }

    static void pause() {
        if (interrupted)
            throw std::runtime_error("SHARP run interrupted");
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

private:
    pid_t pid_ = -1;
    int status_ = 0;
};

struct WorkLock {
    int fd;
    explicit WorkLock(const fs::path &work)
        : fd(open((work / "run.lock").c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0600)) {
        if (fd < 0)
            throw std::runtime_error("cannot open SHARP workspace lock");
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            close(fd);
            throw std::runtime_error("SHARP workspace is already in use");
        }
    }
    WorkLock(const WorkLock &) = delete;
    WorkLock &operator=(const WorkLock &) = delete;
    ~WorkLock() { close(fd); }
};

struct Graph {
    std::string name;
    fs::path image, constants;
};

Graph graph(std::string name, const fs::path &prefix, bool q16 = false) {
    return {std::move(name), prefix.string() + (q16 ? ".q16p" : ".hxfg"),
            prefix.string() + (q16 ? ".q16c" : ".bin")};
}

std::string block_id(int block) {
    return (block < 10 ? "0" : "") + std::to_string(block);
}

struct Pipeline {
    fs::path work, image, camera, models, launcher, clients, dsp, q16;
    fs::path socket() const { return work / "broker.sock"; }

    void bind_workspace() const {
        const auto path = work / "run-inputs.txt";
        std::string identity = models.string() + "\n";
        for (const auto &input : {image, camera}) {
            const auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                fs::last_write_time(input).time_since_epoch());
            identity += input.string() + "\n" + std::to_string(fs::file_size(input)) + "\n" +
                        std::to_string(timestamp.count()) + "\n";
        }
        if (fs::exists(path)) {
            std::ifstream in(path);
            const std::string saved((std::istreambuf_iterator<char>(in)), {});
            if (saved != identity)
                throw std::runtime_error("SHARP workspace belongs to a different input or model path");
            return;
        }
        if (done("tokens-embed"))
            throw std::runtime_error("existing SHARP checkpoints have no input identity");
        std::ofstream out(path);
        if (!(out << identity) || !out.flush())
            throw std::runtime_error("cannot record SHARP workspace input identity");
    }

    void client(Args args) const {
        args.front() = (clients / args.front()).string();
        Child command(args);
        command.finish(args.front());
    }

    void broker(const std::vector<Graph> &graphs, bool quantized,
                const std::function<void()> &execute) const {
        const auto started = Clock::now();
        if (fs::exists(socket()))
            throw std::runtime_error("SHARP broker socket already exists: " + socket().string());
        const auto log = work / "logs" / (graphs.front().name + ".log");
        Args args{launcher.string(), "--socket", socket().string(), "--loader", (dsp / "qd_loader.elf").string(),
                  "--engine", (quantized ? q16 : dsp / "engine.elf").string(),
                  "--bootstrap", (dsp / "bootstrap.elf").string(), "--chunk-probe", (dsp / "chunk_probe.elf").string()};
        for (const auto &g : graphs) {
            args.push_back(quantized ? "--qnn16-graph" : "--graph");
            args.push_back(g.name + "=" + g.image.string() + "," + g.constants.string());
        }
        Child daemon(args, log);
        const auto timeout = Clock::now() + std::chrono::seconds(10);
        while (!fs::is_socket(socket())) {
            if (daemon.exited() || Clock::now() >= timeout)
                throw std::runtime_error("SHARP broker failed to become ready: " + log.string());
            Child::pause();
        }
        const auto ready = Clock::now();
        execute();
        const auto finished = Clock::now();
        const auto milliseconds = [](Clock::duration duration) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        };
        std::cout << "broker " << graphs.front().name << " graphs " << graphs.size()
                  << " setup " << milliseconds(ready - started) << " ms, work "
                  << milliseconds(finished - ready) << " ms\n";
    }

    void broker(const std::vector<Graph> &graphs, bool quantized, Args command) const {
        broker(graphs, quantized, [&] { client(std::move(command)); });
    }

    bool done(const std::string &phase) const { return fs::exists(work / (phase + ".ok")); }

    void checkpoint(const std::string &phase) const {
        std::ofstream file(work / (phase + ".ok"));
        if (!(file << phase << '\n') || !file.flush())
            throw std::runtime_error("cannot record SHARP checkpoint " + phase);
        std::cout << "checkpoint " << phase << std::endl;
    }

    void run_phase(const std::string &phase, const std::vector<Graph> &graphs, bool q,
                   Args command) const {
        if (!done(phase)) {
            broker(graphs, q, std::move(command));
            checkpoint(phase);
        }
    }

    void region_tiles(const std::string &output, const std::string &graph_name,
                      const fs::path &packages, const std::string &name, const std::string &stage,
                      const std::string &tile, const std::string &mode, const fs::path &source) const {
        // Each row has three edge/interior graph shapes. Keep their session
        // open while the three regions are written, within the broker's limit.
        for (char row : {'0', '1', '2'}) {
            std::vector<Graph> graphs;
            std::vector<Args> commands;
            for (char col : {'0', '1', '2'}) {
                const std::string region{'r', row, 'c', col};
                if (done("decoder/" + output + "." + region))
                    continue;
                Args command{"sharp_disparity", socket().string(), source.string(), (work / "decoder").string(),
                             stage, std::string{row, col}, tile};
                if (!mode.empty())
                    command.push_back(mode);
                graphs.push_back(graph(graph_name + "-" + region, packages / region / name));
                commands.push_back(std::move(command));
            }
            if (!graphs.empty())
                broker(graphs, false, [&] {
                    for (auto &command : commands)
                        client(std::move(command));
                });
        }
    }

    void encode(const std::string &encoder) const {
        std::string previous = "tokens-embed";
        for (int block = 0; block <= 24; ++block) {
            const bool norm = block == 24;
            const auto id = norm ? "norm" : block_id(block);
            const std::string next = "tokens-" + encoder + "-" + id;
            const std::string label = "sharp-" + encoder + (norm ? "-norm" : "-block-" + id);
            const fs::path prefix = models / encoder / id;
            run_phase(next, {graph(label, norm ? prefix : prefix / "program", !norm)}, !norm,
                      {"sharp_blocks", socket().string(), encoder, std::to_string(block), "1",
                       (work / previous).string(), (work / next).string()});
            if (encoder == "pe" && block > 0 && !norm)
                for (int i = 0; i < 35; ++i)
                    fs::remove(work / previous / ("pe-" + std::to_string(i) + ".f32"));
            previous = next;
        }
    }

    void merge() const {
        if (done("merged"))
            return;
        const auto tokens = work / "encoder-tokens";
        fs::create_directories(tokens);
        const auto copy = [&](const std::string &from, const std::string &name) {
            fs::copy_file(work / from / name, tokens / name, fs::copy_options::overwrite_existing);
        };
        copy("tokens-ie-norm", "ie.f32");
        for (int i = 0; i < 35; ++i) {
            const auto name = "pe-" + std::to_string(i);
            copy("tokens-pe-norm", name + ".f32");
            if (i < 25) {
                copy("tokens-pe-05", name + "-latent0.f32");
                copy("tokens-pe-11", name + "-latent1.f32");
            }
        }
        client({"sharp_merge", tokens.string(), (work / "merged").string()});
        checkpoint("merged");
    }

    void heads() const {
        const auto decoder = work / "decoder";
        region_tiles("fusion.f32", "sharp-gaussian-head-fusion", models / "gaussian-head/fusion",
                     "phase", "fusion", "32", "--head", work);
        for (const std::string branch : {"geometry", "texture"}) {
            std::string residual = "fusion";
            for (int n = 0; n < 2; ++n) {
                std::string source = residual;
                for (int c = 0; c < 2; ++c) {
                    const auto phase = branch + "-" + std::to_string(n) + "-" + std::to_string(c);
                    const auto packages = models / "gaussian-head" / phase;
                    const auto normalized = decoder / (phase + ".normalized.f32");
                    if (!done("decoder/" + phase)) {
                        if (!fs::exists(normalized))
                            client({"sharp_groupnorm", (decoder / (source + ".f32")).string(),
                                    (packages / "affine.f32").string(), normalized.string(),
                                    "768", c ? "64" : "128", "8"});
                        region_tiles(phase + ".f32", "sharp-gaussian-head-" + phase, packages,
                                     "phase", phase, "32", "--head", work);
                        checkpoint("decoder/" + phase);
                        fs::remove(normalized);
                    }
                    source = phase;
                }
                residual = source;
            }
        }
        region_tiles("delta.f32", "sharp-gaussian-head-delta", models / "gaussian-head/delta",
                     "phase", "delta", "32", "--head", work);
    }

    void run() const {
        fs::create_directories(work / "logs");
        WorkLock lock(work);
        bind_workspace();
        if (setenv("HEXECUTOR_CHUNK_WINDOW_MB", "128", 0))
            throw std::runtime_error("cannot select the SHARP baseline CDSP window");
        run_phase("tokens-embed", {graph("sharp-pe-embed", models / "pe_embed"),
                                  graph("sharp-ie-embed", models / "ie_embed")}, false,
                  {"sharp_embed", image.string(), socket().string(), (work / "tokens-embed").string()});
        encode("ie");
        encode("pe");
        merge();
        const auto fusion = work / "fusion", decoder = work / "decoder";
        for (int i = 0; i < 5; ++i) {
            const auto branch = std::to_string(i), label = "sharp-fusion-" + branch;
            run_phase("fusion/branch-" + branch, {graph(label, models / "fusion" / branch / "fusion")}, false,
                      {"sharp_fusion", socket().string(), (work / "merged").string(), fusion.string(),
                       "--branch=" + branch});
        }
        for (int level = 4; level >= -1; --level) {
            const auto id = std::to_string(level);
            region_tiles(level == -1 ? "disparity.f32" : "disparity-stage-" + id + ".f32",
                         "sharp-disparity-" + id, models / "disparity" / id, "disparity", id,
                         level > 1 ? "16" : "32", "", fusion);
        }
        for (int level = 4; level >= 0; --level) {
            const auto id = std::to_string(level);
            region_tiles("gaussian-stage-" + id + ".f32", "sharp-gaussian-" + id, models / ("gaussian-" + id),
                         "gaussian", id, level > 1 ? "16" : "32", "--gaussian", fusion);
        }
        const auto rgb = (work / "tokens-embed/image.f32").string();
        const auto disparity = (decoder / "disparity.f32").string();
        if (!done("image-depth.f32")) {
            client({"sharp_compose", "--features", rgb, disparity, camera.string(), (work / "image-depth.f32").string()});
            checkpoint("image-depth.f32");
        }
        heads();
        if (!done("scene.ply")) {
            const auto metric = (work / "metric").string();
            client({"sharp_compose", rgb, disparity, (decoder / "delta.f32").string(), camera.string(), metric,
                    "--delta-hwc"});
            client({"sharp_ply", metric, camera.string(), (work / "scene.ply").string()});
            checkpoint("scene.ply");
        }
        std::cout << (work / "scene.ply").string() << '\n';
    }
};
} // namespace

int main(int argc, char **argv) {
    if (argc != 9) {
        std::cerr << "usage: sharp_run WORK IMAGE.ppm CAMERA.txt MODELS BROKER CLIENTS FLOAT_DSP Q16_ENGINE\n";
        return 2;
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    try {
        Pipeline{fs::absolute(argv[1]), fs::canonical(argv[2]), fs::canonical(argv[3]),
                 fs::canonical(argv[4]), fs::canonical(argv[5]), fs::canonical(argv[6]),
                 fs::canonical(argv[7]), fs::canonical(argv[8])}.run();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
