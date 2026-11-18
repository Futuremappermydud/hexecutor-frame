// hexecutord: the CDSP broker daemon (docs/hexecutor/broker.md, phase 1).
// One process owns the device session; clients submit tensor-graph jobs over a
// Unix socket. The optional stock-job adapter also uses this socket; it keeps
// the session warm and yields to stock firmware for requests it does not serve.
#include "hexecutor/broker.hpp"
#include "hexecutor/xrservice_featurenn.hpp"
#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>

using namespace hexecutor;

namespace {
struct NamedProgram {
    std::string name;
    Program program;
};

// Wraps one Session over every registered graph. After any device failure the
// backend refuses further work: a failed restoration must be recovered by hand
// (docs/hexecutor/operations.md), never by an automatic reopen.
class SessionBackend : public broker::Backend {
  public:
    SessionBackend(std::vector<NamedProgram> graphs, Artifacts artifacts)
        : graphs_(std::move(graphs)), artifacts_(std::move(artifacts)) {
        for (auto &g : graphs_)
            programs_.push_back(g.program);
    }
    std::vector<broker::GraphInfo> graphs() const override {
        std::vector<broker::GraphInfo> out;
        for (auto &g : graphs_) {
            broker::GraphInfo info{g.name, {}, {}};
            for (auto &t : g.program.inputs)
                info.input_bytes.push_back(t.bytes());
            for (auto &t : g.program.outputs)
                info.output_bytes.push_back(t.bytes());
            info.state_bytes = state_bytes();
            out.push_back(std::move(info));
        }
        return out;
    }
    bool has_persistent_state() const {
        if (session_)
            return session_->state_bytes() > 0;
        return artifacts_.persistent_state_bytes > 0;
    }
    size_t state_bytes() const override {
        return session_ ? session_->state_bytes() : artifacts_.persistent_state_bytes;
    }
    void reset_state() override {
        if (session_)
            session_->reset_state();
        state_lost_ = false;
    }
    // Called from the tripwire thread. The executor thread applies it: the open
    // session (if any) is closed by Server::release_device(), and every later
    // session uses the one-second wake rotation.
    void trip() { tripped_ = true; }
    void hold_session(bool hold) { hold_ = hold; }
    std::vector<Bytes> run(size_t graph, const std::vector<Buffer> &inputs,
                           double &dsp_ms) override {
        if (!failed_.empty())
            throw std::runtime_error("device session failed earlier: " + failed_);
        if (state_lost_)
            throw std::runtime_error("persistent state was lost: session was released or failed; reset required");
        if (tripped_ && artifacts_.max_wake_seconds > 1.0) {
            if (session_)
                release();
            artifacts_.max_wake_seconds = 1.0;
            std::cerr << "[hexecutord] tripwire: wake rotation back to 1 s\n";
        }
        try {
            if (!session_) {
                session_ = std::make_unique<Session>(programs_, artifacts_);
                std::cerr << "[hexecutord] device session opened\n";
            }
            auto out = session_->run(inputs, graph);
            dsp_ms = double(session_->profile().timer_ticks) / 19200.0;
            return out;
        } catch (const std::exception &e) {
            fail(e.what());
            throw;
        }
    }
    void idle() override {
        if (!hold_ && !has_persistent_state())
            close_session("idle");
    }
    void release() override {
        close_session("release");
        if (has_persistent_state())
            state_lost_ = true;
    }

  private:
    void close_session(const char *reason) {
        if (!session_)
            return;
        try {
            session_->close();
            std::cerr << "[hexecutord] " << reason << ": device session closed and restored\n";
        } catch (const std::exception &e) {
            fail(e.what());
        }
        session_.reset();
    }
    void fail(const std::string &what) {
        failed_ = what;
        state_lost_ = true;
        std::cerr << "[hexecutord] device failure, refusing further work: " << what << '\n';
        if (session_) {
            try {
                session_->close();
            } catch (const std::exception &e) {
                std::cerr << "[hexecutord] close after failure: " << e.what() << '\n';
            }
            session_.reset();
        }
    }
    std::vector<NamedProgram> graphs_;
    std::vector<Program> programs_;
    Artifacts artifacts_;
    std::unique_ptr<Session> session_;
    std::string failed_;
    std::atomic<bool> tripped_{false}, hold_{false}, state_lost_{false};
};

// Long wakes are only safe while XRService keeps controller FeatureNN off the
// CDSP (docs/hexecutor/operations.md). Require the setting, and require that
// the running XRService started after the settings file was last written.
const char *const xrservice_settings =
    "/home/steamos/.config/openvr/config/cv/xrservice/XRServiceSettings.json";
bool xrservice_guard(std::string &why) {
    std::ifstream f(xrservice_settings);
    if (!f) {
        why = "cannot read " + std::string(xrservice_settings);
        return false;
    }
    std::stringstream text;
    text << f.rdbuf();
    static const std::regex off(R"re("FeatureNN"\s*:\s*\{[^}]*"enableForLogic"\s*:\s*false)re");
    if (!std::regex_search(text.str(), off)) {
        why = "controller FeatureNN is not disabled (FeatureNN.enableForLogic)";
        return false;
    }
    struct stat settings {};
    stat(xrservice_settings, &settings);
    bool found = false;
    if (DIR *d = opendir("/proc")) {
        while (dirent *e = readdir(d)) {
            std::string pid = e->d_name;
            if (pid.find_first_not_of("0123456789") != std::string::npos)
                continue;
            std::ifstream cmd("/proc/" + pid + "/cmdline");
            std::string arg0;
            std::getline(cmd, arg0, '\0');
            if (arg0 != "XRService" && arg0.size() < 10)
                continue;
            if (arg0.compare(arg0.size() - 9, 9, "XRService") != 0)
                continue;
            found = true;
            struct stat proc {};
            if (stat(("/proc/" + pid).c_str(), &proc) == 0 &&
                proc.st_mtime < settings.st_mtime) {
                why = "XRService started before its settings were last written";
                closedir(d);
                return false;
            }
        }
        closedir(d);
    }
    if (!found) {
        why = "XRService is not running";
        return false;
    }
    return true;
}

// Follows XRService's journal. Any CDSP activity, or an XRService restart (which
// re-reads its settings), trips the broker back to safe behaviour.
void watch_xrservice(std::atomic<bool> &stop, const std::function<void(const std::string &)> &trip) {
    FILE *p = popen("exec journalctl -f -n 0 -o cat -t XRService 2>/dev/null", "r");
    if (!p) {
        trip("cannot follow the XRService journal");
        return;
    }
    char line[4096];
    while (!stop && fgets(line, sizeof line, p)) {
        std::string l(line);
        if (l.find("cDSP") != std::string::npos || l.find("CDSP") != std::string::npos ||
            l.find("XRService logging to") != std::string::npos) {
            while (!l.empty() && (l.back() == '\n' || l.back() == '\r'))
                l.pop_back();
            trip(l);
            break;
        }
    }
    pclose(p);
}

[[noreturn]] void usage(int code) {
    std::cerr
        << "usage: hexecutord --socket PATH --loader ELF --engine ELF --bootstrap ELF\n"
           "         --chunk-probe ELF --graph NAME=IMAGE.hxfg,CONSTANTS.bin [--graph ...]\n"
           "         [--xrservice-featurenn GRAPH]  (serve the built-in XRService stereoFeatureNN job 8)\n"
           "         [--xrservice-stereofeaturenn GRAPH]  (alias for --xrservice-featurenn)\n"
           "         [--xrservice-gazenet GRAPH]  (serve stock Gazenet image evaluation)\n"
           "         [--xrservice-gazenet-vector GRAPH]  (serve stock Gazenet float vectors)\n"
           "         [--idle-close-ms 2000] [--utilization 0.7] [--max-wake-seconds 1]\n"
           "         [--long-wake SECONDS]  (without the adapter, only if XRService keeps\n"
           "                                 FeatureNN off the CDSP; journal tripwire armed)\n"
           "         [--precise] [--f32]\n";
    std::exit(code);
}
double positive(const std::string &v, const char *what) {
    size_t n = 0;
    double x = std::stod(v, &n);
    if (n != v.size() || !(x > 0))
        throw std::runtime_error(std::string("invalid ") + what + ": " + v);
    return x;
}
} // namespace

int main(int argc, char **argv) {
    try {
        broker::ServerOptions options;
        Artifacts artifacts;
        std::vector<NamedProgram> graphs;
        xrservice::AdapterOptions adapter_options;
        double long_wake = 0;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc)
                    throw std::runtime_error("missing value for " + arg);
                return argv[i];
            };
            if (arg == "--socket")
                options.socket = value();
            else if (arg == "--loader")
                artifacts.loader = value();
            else if (arg == "--engine")
                artifacts.engine = value();
            else if (arg == "--bootstrap")
                artifacts.bootstrap = value();
            else if (arg == "--chunk-probe")
                artifacts.chunk_probe = value();
            else if (arg == "--graph") {
                auto v = value();
                auto eq = v.find('='), comma = v.find(',', eq);
                if (eq == std::string::npos || comma == std::string::npos || eq == 0)
                    throw std::runtime_error("--graph expects NAME=IMAGE,CONSTANTS");
                auto name = v.substr(0, eq);
                for (auto &g : graphs)
                    if (g.name == name)
                        throw std::runtime_error("duplicate graph name " + name);
                graphs.push_back({name, load_tensor_program(v.substr(eq + 1, comma - eq - 1),
                                                            v.substr(comma + 1))});
            } else if (arg == "--xrservice-featurenn" ||
                       arg == "--xrservice-stereofeaturenn" ||
                       arg == "--xrservice-stereo-featurenn")
                adapter_options.featurenn_graph = value();
            else if (arg == "--xrservice-gazenet")
                adapter_options.gazenet_graph = value();
            else if (arg == "--xrservice-gazenet-vector")
                adapter_options.gazenet_vector_graph = value();
            else if (arg == "--idle-close-ms")
                options.idle_close_ms = positive(value(), "idle close");
            else if (arg == "--utilization")
                options.utilization_limit = positive(value(), "utilization limit");
            else if (arg == "--max-wake-seconds")
                artifacts.max_wake_seconds = positive(value(), "wake limit");
            else if (arg == "--long-wake")
                long_wake = positive(value(), "long wake");
            else if (arg == "--precise")
                artifacts.tensor_hmx = false;
            else if (arg == "--f32")
                artifacts.tensor_fp16 = false;
            else if (arg == "--help")
                usage(0);
            else
                throw std::runtime_error("unknown option " + arg);
        }
        if (options.socket.empty() || artifacts.loader.empty() || artifacts.engine.empty() ||
            artifacts.bootstrap.empty() || artifacts.chunk_probe.empty() || graphs.empty())
            usage(2);
        if (graphs.size() > 8)
            throw std::runtime_error("at most eight graphs share one tensor session");
        auto require_registered = [&](const std::string &name, const char *what) {
            if (!name.empty() && std::none_of(graphs.begin(), graphs.end(),
                                              [&](const NamedProgram &g) { return g.name == name; }))
                throw std::runtime_error(std::string(what) + " graph is not registered");
        };
        require_registered(adapter_options.featurenn_graph, "XRService stereoFeatureNN");
        require_registered(adapter_options.gazenet_graph, "Gazenet image");
        require_registered(adapter_options.gazenet_vector_graph, "Gazenet vector");
        bool adapting = !adapter_options.featurenn_graph.empty() ||
                        !adapter_options.gazenet_graph.empty() ||
                        !adapter_options.gazenet_vector_graph.empty();
        artifacts.keep_eye_running = !adapter_options.gazenet_graph.empty() ||
                                     !adapter_options.gazenet_vector_graph.empty();

        // Handle SIGINT/SIGTERM on this thread only; the server threads inherit the mask.
        // A background launch can inherit SIGINT as ignored, and an ignored signal
        // never reaches sigwait, so restore the default disposition first.
        std::signal(SIGINT, SIG_DFL);
        std::signal(SIGTERM, SIG_DFL);
        sigset_t signals;
        sigemptyset(&signals);
        sigaddset(&signals, SIGINT);
        sigaddset(&signals, SIGTERM);
        pthread_sigmask(SIG_BLOCK, &signals, nullptr);

        std::vector<std::string> names;
        for (auto &g : graphs)
            names.push_back(g.name);
        bool watching = false;
        if (long_wake > 0) {
            std::string why;
            if (adapting) {
                artifacts.max_wake_seconds = long_wake;
                std::cerr << "[hexecutord] long wakes: up to " << long_wake
                          << " s while serving stock CDSP jobs\n";
            } else if (xrservice_guard(why)) {
                artifacts.max_wake_seconds = long_wake;
                watching = true;
                std::cerr << "[hexecutord] long wakes: up to " << long_wake
                          << " s, XRService journal tripwire armed\n";
            } else
                std::cerr << "[hexecutord] long wakes refused (" << why
                          << "); keeping " << artifacts.max_wake_seconds << " s rotation\n";
        }
        auto backend_owner = std::make_unique<SessionBackend>(std::move(graphs), artifacts);
        SessionBackend *backend = backend_owner.get();
        backend->hold_session(adapting);
        broker::Server server(std::move(backend_owner), options);
        server.start();
        std::unique_ptr<xrservice::Adapter> stock_adapter;
        if (adapting) {
            stock_adapter = std::make_unique<xrservice::Adapter>(
                options.socket, adapter_options, [&] { server.release_device(); });
            stock_adapter->start();
            std::cerr << "[hexecutord] serving built-in stock CDSP jobs through the broker\n";
        }
        std::atomic<bool> stop_watch{false};
        std::thread watcher;
        if (watching)
            watcher = std::thread([&] {
                watch_xrservice(stop_watch, [&](const std::string &line) {
                    std::cerr << "[hexecutord] tripwire: " << line << '\n';
                    backend->trip();
                    server.release_device();
                });
            });
        std::cerr << "[hexecutord] listening on " << options.socket.string() << " with graphs";
        for (auto &n : names)
            std::cerr << ' ' << n;
        std::cerr << '\n';
        int sig = 0;
        sigwait(&signals, &sig);
        std::cerr << "[hexecutord] signal " << sig << ", stopping\n";
        if (stock_adapter)
            stock_adapter->stop();
        backend->hold_session(false);
        auto stats = server.stats();
        server.stop();
        if (watcher.joinable()) {
            // journalctl -f only notices a closed pipe on its next line, so end it directly.
            stop_watch = true;
            std::system("pkill -f 'journalctl -f -n 0 -o cat -t XRService' >/dev/null 2>&1");
            watcher.join();
        }
        for (size_t i = 0; i < stats.size(); i++)
            std::cerr << "[hexecutord] " << names[i] << ": runs " << stats[i].runs << " mean run "
                      << stats[i].mean_run_ms << " ms, dsp " << stats[i].mean_dsp_ms << " ms\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "hexecutord: " << e.what() << '\n';
        return 1;
    }
}
