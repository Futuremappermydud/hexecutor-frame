// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
// CDSP broker, phase 1 (docs/hexecutor/broker.md): one process owns the device
// session and schedules every client's tensor-graph jobs through one queue.
#include "hexecutor/executor.hpp"
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hexecutor::broker {
using Clock = std::chrono::steady_clock;

// Lower value runs first when deadlines tie.
enum class Class : uint32_t { tracking = 0, interactive = 1, bulk = 2 };
enum class Status : uint32_t { ok = 0, replaced, expired, error };
const char *to_string(Status);

struct StreamSpec {
    std::string graph;        // graph name registered with the broker
    Class cls = Class::interactive;
    double period_ms = 0;     // expected arrival period; 0 = sporadic (not admitted by rate)
    double deadline_ms = 100; // relative to submit; a job not started by then is expired
    double cost_hint_ms = 0;  // used for admission until the broker has measured the graph
    bool latest_only = true;  // a newer submit replaces this stream's queued job
};

struct Result {
    Status status = Status::error;
    std::vector<Bytes> outputs;
    double queue_ms = 0, run_ms = 0, dsp_ms = 0;
    std::string error;
};

// What the broker runs. The daemon wraps a hexecutor::Session; tests inject fakes.
struct GraphInfo {
    std::string name;
    std::vector<size_t> input_bytes, output_bytes;
    size_t state_bytes = 0;
};
class Backend {
  public:
    virtual ~Backend() = default;
    virtual std::vector<GraphInfo> graphs() const = 0;
    // Runs one graph. dsp_ms is the device's own time when known, else 0.
    virtual std::vector<Bytes> run(size_t graph, const std::vector<Buffer> &inputs,
                                   double &dsp_ms) = 0;
    // Called after the queue has been empty for the idle timeout: release the device.
    virtual void idle() {}
    // Forced release for a stock firmware job; unlike idle(), cannot be pinned.
    virtual void release() { idle(); }
    virtual void reset_state() {}
    virtual size_t state_bytes() const { return 0; }
};

// Portable scheduling core: earliest deadline first; ties go to the more urgent
// class, then to the graph that ran last (a graph switch re-stages weights).
struct Job {
    uint64_t client = 0, ticket = 0;
    uint32_t stream = 0;
    size_t graph = 0;
    Class cls = Class::interactive;
    bool latest_only = true;
    Clock::time_point submitted, deadline;
    std::vector<Bytes> inputs;
};
class Scheduler {
  public:
    // Returns a job this submit displaced (latest_only), so it can be answered.
    std::optional<Job> submit(Job job);
    // Next job to run at `now`; jobs whose deadline passed are moved to `expired`.
    std::optional<Job> next(Clock::time_point now, std::vector<Job> &expired);
    // Remove every queued job of a client (disconnect) or of one of its streams.
    std::vector<Job> drop(uint64_t client, std::optional<uint32_t> stream = std::nullopt);
    size_t size() const { return queue_.size(); }
    void ran(size_t graph) { last_graph_ = graph; have_last_ = true; }

  private:
    std::deque<Job> queue_;
    size_t last_graph_ = 0;
    bool have_last_ = false;
};

// Utilization-based admission over periodic streams.
class Admission {
  public:
    explicit Admission(double limit = 0.7) : limit_(limit) {}
    // cost_ms: measured cost of the graph if known, else the stream's hint.
    bool admit(uint64_t key, double period_ms, double cost_ms);
    void release(uint64_t key);
    double utilization() const;

  private:
    double limit_;
    std::map<uint64_t, double> load_;
};

struct ServerOptions {
    std::filesystem::path socket;
    double idle_close_ms = 2000;  // release the device after this long with nothing queued
    double utilization_limit = 0.7;
};

// Unix-socket server: one I/O thread, one executor thread, one Backend.
class Server {
  public:
    Server(std::unique_ptr<Backend>, ServerOptions);
    ~Server();
    void start();
    void stop(); // drains nothing: queued jobs are answered with Status::error
    // Ask the executor to call Backend::release() before its next job, even if
    // work is queued. Safe from any thread; used when the device must be let go.
    void release_device();
    struct GraphStats {
        uint64_t runs = 0;
        double mean_run_ms = 0, mean_dsp_ms = 0;
        size_t state_bytes = 0;
    };
    std::vector<GraphStats> stats() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Client {
  public:
    explicit Client(const std::filesystem::path &socket);
    ~Client();
    Client(const Client &) = delete;
    Client &operator=(const Client &) = delete;
    struct Stream {
        uint32_t id = 0;
        std::vector<size_t> input_bytes, output_bytes;
    };
    Stream open(const StreamSpec &);
    void close(const Stream &);
    uint64_t submit(const Stream &, const std::vector<Buffer> &inputs);
    Result wait(uint64_t ticket);
    // submit + wait; throws unless the result is ok. Drop-in for Session::run.
    std::vector<Bytes> run(const Stream &, const std::vector<Buffer> &inputs);
    void reset_state();

  private:
    int fd_ = -1;
    uint64_t next_ticket_ = 1;
    std::map<uint64_t, Result> pending_;
};
} // namespace hexecutor::broker
