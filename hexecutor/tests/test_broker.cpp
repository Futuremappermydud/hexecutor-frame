// SPDX-License-Identifier: LGPL-3.0-or-later
// Portable broker tests: scheduling order, admission, and client/server over a
// real Unix socket with an injected backend (no device).
#include "hexecutor/broker.hpp"
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using namespace hexecutor;
using namespace hexecutor::broker;
using namespace std::chrono_literals;

static void require(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}

static Job job(uint64_t client, uint32_t stream, size_t graph, Class cls, Clock::time_point deadline,
               uint64_t ticket, bool latest = true) {
    Job j;
    j.client = client;
    j.stream = stream;
    j.graph = graph;
    j.cls = cls;
    j.deadline = deadline;
    j.ticket = ticket;
    j.latest_only = latest;
    j.submitted = Clock::now();
    return j;
}

static void scheduler_tests() {
    auto now = Clock::now();
    std::vector<Job> expired;
    {
        Scheduler s; // earliest deadline first
        s.submit(job(1, 1, 0, Class::tracking, now + 30ms, 1));
        s.submit(job(1, 2, 0, Class::bulk, now + 10ms, 2));
        s.submit(job(2, 1, 0, Class::interactive, now + 20ms, 3));
        require(s.next(now, expired)->ticket == 2, "EDF first");
        require(s.next(now, expired)->ticket == 3, "EDF second");
        require(s.next(now, expired)->ticket == 1, "EDF third");
        require(!s.next(now, expired), "EDF empty");
    }
    {
        Scheduler s; // equal deadlines: class, then the graph that ran last
        auto d = now + 10ms;
        s.submit(job(1, 1, 0, Class::bulk, d, 1));
        s.submit(job(1, 2, 1, Class::tracking, d, 2));
        s.submit(job(1, 3, 2, Class::tracking, d, 3));
        s.ran(2);
        require(s.next(now, expired)->ticket == 3, "same graph preferred among equal class");
        require(s.next(now, expired)->ticket == 2, "tracking before bulk");
        require(s.next(now, expired)->ticket == 1, "bulk last");
    }
    {
        Scheduler s; // latest_only replaces the stream's queued job
        require(!s.submit(job(1, 1, 0, Class::tracking, now + 10ms, 1)), "first submit");
        auto displaced = s.submit(job(1, 1, 0, Class::tracking, now + 10ms, 2));
        require(displaced && displaced->ticket == 1, "displaced job returned");
        require(!s.submit(job(1, 2, 0, Class::tracking, now + 10ms, 3)), "other stream kept");
        require(!s.submit(job(1, 3, 0, Class::tracking, now + 10ms, 4, false)), "queue mode a");
        require(!s.submit(job(1, 3, 0, Class::tracking, now + 10ms, 5, false)), "queue mode b");
        require(s.size() == 4, "replacement count");
        auto dropped = s.drop(1, 3u);
        require(dropped.size() == 2 && s.size() == 2, "drop one stream");
        require(s.drop(1).size() == 2 && s.size() == 0, "drop client");
    }
    {
        Scheduler s; // expired jobs are reported, not run
        s.submit(job(1, 1, 0, Class::tracking, now - 1ms, 1));
        s.submit(job(1, 2, 0, Class::tracking, now + 10ms, 2));
        expired.clear();
        auto j = s.next(now, expired);
        require(j && j->ticket == 2 && expired.size() == 1 && expired[0].ticket == 1, "expiry");
    }
    {
        Admission a(0.5);
        require(a.admit(1, 10, 2), "20% admitted");
        require(a.admit(2, 10, 2.5), "45% admitted");
        require(!a.admit(3, 10, 1), "55% refused");
        require(a.admit(4, 0, 100), "sporadic not rate-admitted");
        a.release(2);
        require(a.admit(3, 10, 1), "30% after release");
    }
}

// Graph 0 "twice": f32[2] -> f32[2] doubled. Graph 1 "sum": f32[2] -> f32[1].
// Runs block while `hold` is set, so tests can build a queue.
struct Fake : broker::Backend {
    std::mutex m;
    std::condition_variable cv;
    bool hold = false;
    std::atomic<int> runs{0}, idles{0}, entered{0};
    std::vector<GraphInfo> graphs() const override {
        return {{"twice", {8}, {8}}, {"sum", {8}, {4}}};
    }
    std::vector<Bytes> run(size_t graph, const std::vector<Buffer> &in, double &dsp_ms) override {
        entered++;
        {
            std::unique_lock<std::mutex> l(m);
            cv.wait(l, [&] { return !hold; });
        }
        float x[2];
        std::memcpy(x, in[0].data, 8);
        runs++;
        dsp_ms = 0.25;
        if (graph == 0) {
            float y[2] = {2 * x[0], 2 * x[1]};
            return {Bytes(reinterpret_cast<uint8_t *>(y), reinterpret_cast<uint8_t *>(y) + 8)};
        }
        float y = x[0] + x[1];
        return {Bytes(reinterpret_cast<uint8_t *>(&y), reinterpret_cast<uint8_t *>(&y) + 4)};
    }
    void idle() override { idles++; }
    void set_hold(bool h) {
        {
            std::lock_guard<std::mutex> l(m);
            hold = h;
        }
        cv.notify_all();
    }
};

// Bounded spin: a broken broker must fail the test, not hang it.
template <class F> static void wait_for(F done, const char *what) {
    auto until = Clock::now() + 10s;
    while (!done()) {
        if (Clock::now() > until)
            throw std::runtime_error(what);
        std::this_thread::sleep_for(100us);
    }
}

static float first(const Bytes &b) {
    float v;
    std::memcpy(&v, b.data(), 4);
    return v;
}

static void server_tests() {
    char dir[] = "/tmp/hxbrokerXXXXXX";
    require(::mkdtemp(dir) != nullptr, "temp dir");
    std::filesystem::path sock = std::string(dir) + "/b.sock";
    auto fake_owner = std::make_unique<Fake>();
    Fake &fake = *fake_owner;
    ServerOptions o;
    o.socket = sock;
    o.idle_close_ms = 50;
    o.utilization_limit = 0.5;
    Server server(std::move(fake_owner), o);
    server.start();
    {
        Client c(sock);
        // Admission: limit 0.5, so 30% + 30% is refused.
        StreamSpec rate;
        rate.graph = "sum";
        rate.period_ms = 10;
        rate.cost_hint_ms = 3; // no graph has run yet, so the hint is used
        bool threw = false;
        std::vector<Client::Stream> admitted;
        try {
            for (int i = 0; i < 1000; i++)
                admitted.push_back(c.open(rate));
        } catch (const std::runtime_error &e) {
            threw = std::string(e.what()).find("admission") != std::string::npos;
        }
        require(threw && !admitted.empty(), "admission limit");
        for (auto &a : admitted)
            c.close(a);

        StreamSpec spec;
        spec.graph = "twice";
        auto s = c.open(spec);
        require(s.input_bytes == std::vector<size_t>{8} && s.output_bytes == std::vector<size_t>{8},
                "stream contract");
        float x[2] = {1.5f, -3};
        auto out = c.run(s, {{x, 8}});
        require(out.size() == 1 && first(out[0]) == 3.0f, "run result");

        spec.graph = "sum";
        auto sum = c.open(spec);
        auto t = c.submit(sum, {{x, 8}});
        auto r = c.wait(t);
        require(r.status == Status::ok && first(r.outputs[0]) == -1.5f && r.dsp_ms == 0.25,
                "second graph");

        r = c.wait(c.submit(sum, {{x, 4}}));
        require(r.status == Status::error && r.error.find("contract") != std::string::npos,
                "contract checked");

        threw = false;
        try {
            spec.graph = "missing";
            c.open(spec);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        require(threw, "unknown graph refused");

        // latest_only: while the backend is held, a second submit replaces the first.
        fake.set_hold(true);
        int before = fake.entered.load();
        auto t0 = c.submit(s, {{x, 8}}); // starts running and blocks
        wait_for([&] { return fake.entered.load() != before; }, "held run never started");
        auto t1 = c.submit(s, {{x, 8}});
        auto t2 = c.submit(s, {{x, 8}});
        auto r1 = c.wait(t1);
        require(r1.status == Status::replaced, "replaced by newer submit");
        fake.set_hold(false);
        require(c.wait(t0).status == Status::ok && c.wait(t2).status == Status::ok,
                "running and newest complete");

        // Expiry: a job whose deadline passes while queued is not run.
        StreamSpec quick;
        quick.graph = "twice";
        quick.deadline_ms = 5;
        auto q = c.open(quick);
        fake.set_hold(true);
        before = fake.entered.load();
        auto blocker = c.submit(s, {{x, 8}});
        wait_for([&] { return fake.entered.load() != before; }, "blocker never started");
        auto late = c.submit(q, {{x, 8}});
        std::this_thread::sleep_for(20ms);
        fake.set_hold(false);
        require(c.wait(blocker).status == Status::ok, "blocker ran");
        require(c.wait(late).status == Status::expired, "late job expired");
    }
    // Idle release: the backend is told to let go of the device.
    wait_for([&] { return fake.idles.load() >= 1; }, "idle release");
    auto st = server.stats();
    require(st.size() == 2 && st[0].runs >= 4 && st[1].runs >= 1, "graph stats");
    server.stop();
    require(!std::filesystem::exists(sock), "socket removed");
    ::rmdir(dir);
}

struct PinnedFake : Fake {
    std::atomic<int> releases{0};
    void idle() override {}
    void release() override { releases++; }
};

static void release_test() {
    char dir[] = "/tmp/hxbrokerXXXXXX";
    require(::mkdtemp(dir) != nullptr, "temp dir");
    std::filesystem::path sock = std::string(dir) + "/r.sock";
    auto owner = std::make_unique<PinnedFake>();
    PinnedFake &fake = *owner;
    ServerOptions o;
    o.socket = sock;
    o.idle_close_ms = 60000; // the idle timeout alone would not release it
    Server server(std::move(owner), o);
    server.start();
    {
        Client c(sock);
        StreamSpec spec;
        spec.graph = "twice";
        auto s = c.open(spec);
        float x[2] = {1, 2};
        c.run(s, {{x, 8}});
        require(fake.releases.load() == 0, "no release before request");
        server.release_device();
        wait_for([&] { return fake.releases.load() >= 1; }, "release_device did not release");
        require(first(c.run(s, {{x, 8}})[0]) == 2.0f, "runs again after release");
    }
    server.stop();
    ::rmdir(dir);
}

static void single_context_benchmark_test() {
    char dir[] = "/tmp/hxbrokerXXXXXX";
    require(::mkdtemp(dir) != nullptr, "temp dir");
    std::filesystem::path sock = std::string(dir) + "/sc.sock";
    auto owner = std::make_unique<Fake>();
    Fake &fake = *owner;
    ServerOptions o;
    o.socket = sock;
    o.idle_close_ms = 60000;
    Server server(std::move(owner), o);
    server.start();
    {
        Client c(sock);
        StreamSpec spec;
        spec.graph = "twice";
        auto s = c.open(spec);
        float x[2] = {2.0f, 3.0f};
        const int rounds = 500;
        for (int i = 0; i < rounds; i++) {
            auto out = c.run(s, {{x, 8}});
            require(out.size() == 1 && first(out[0]) == 4.0f, "single context output");
        }
        require(fake.runs.load() == rounds, "all single context runs executed");
    }
    server.stop();
    ::rmdir(dir);
}

struct StatefulFake : broker::Backend {
    std::mutex m;
    float state_val = 0.0f;
    bool lost = false;
    size_t state_size = 1024;

    std::vector<GraphInfo> graphs() const override {
        return {{"accum", {4}, {4}, state_size}, {"scale", {4}, {4}, state_size}};
    }
    std::vector<Bytes> run(size_t graph, const std::vector<Buffer> &in, double &dsp_ms) override {
        std::lock_guard<std::mutex> l(m);
        if (lost)
            throw std::runtime_error("persistent state was lost: session was released/failed; reset required");
        float arg = 0.0f;
        std::memcpy(&arg, in[0].data, 4);
        dsp_ms = 0.1;
        if (graph == 0) {
            state_val += arg;
            float out = state_val;
            return {Bytes(reinterpret_cast<uint8_t *>(&out), reinterpret_cast<uint8_t *>(&out) + 4)};
        } else {
            float out = arg * state_val;
            return {Bytes(reinterpret_cast<uint8_t *>(&out), reinterpret_cast<uint8_t *>(&out) + 4)};
        }
    }
    void reset_state() override {
        std::lock_guard<std::mutex> l(m);
        state_val = 0.0f;
        lost = false;
    }
    size_t state_bytes() const override {
        return state_size;
    }
    void set_lost(bool l) {
        std::lock_guard<std::mutex> lock(m);
        lost = l;
    }
};

static void persistent_state_broker_test() {
    char dir[] = "/tmp/hxbrokerXXXXXX";
    require(::mkdtemp(dir) != nullptr, "temp dir");
    std::filesystem::path sock = std::string(dir) + "/ps.sock";
    auto owner = std::make_unique<StatefulFake>();
    StatefulFake &fake = *owner;
    ServerOptions o;
    o.socket = sock;
    o.idle_close_ms = 60000;
    Server server(std::move(owner), o);
    server.start();
    {
        Client c(sock);
        StreamSpec accum_spec;
        accum_spec.graph = "accum";
        auto accum = c.open(accum_spec);

        StreamSpec scale_spec;
        scale_spec.graph = "scale";
        auto scale = c.open(scale_spec);

        // 1. State survives across run() of the same graph (accumulate 2.0 then 3.0 then 10.0)
        float x1 = 2.0f;
        auto r1 = c.run(accum, {{&x1, 4}});
        require(first(r1[0]) == 2.0f, "step 1 accum");

        float x2 = 3.0f;
        auto r2 = c.run(accum, {{&x2, 4}});
        require(first(r2[0]) == 5.0f, "step 2 accum (2.0 + 3.0)");

        float x3 = 10.0f;
        auto r3 = c.run(accum, {{&x3, 4}});
        require(first(r3[0]) == 15.0f, "step 3 accum (5.0 + 10.0)");

        // 2. Running graph B (scale) after graph A (accum) while keeping persistent state
        float s1 = 2.0f;
        auto out_scaled = c.run(scale, {{&s1, 4}});
        require(first(out_scaled[0]) == 30.0f, "graph B reads shared state from graph A (15.0 * 2.0)");

        float s2 = 0.5f;
        out_scaled = c.run(scale, {{&s2, 4}});
        require(first(out_scaled[0]) == 7.5f, "graph B reads shared state (15.0 * 0.5)");

        // 3. Reset state request
        c.reset_state();
        float s_after_reset = 1.0f;
        auto out_reset = c.run(scale, {{&s_after_reset, 4}});
        require(first(out_reset[0]) == 0.0f, "state reset to 0");

        float x_new = 4.0f;
        auto r_new = c.run(accum, {{&x_new, 4}});
        require(first(r_new[0]) == 4.0f, "accum after reset starts from 0");

        // 4. Error handling / state loss refusal
        fake.set_lost(true);
        bool failed_as_expected = false;
        try {
            c.run(accum, {{&x_new, 4}});
        } catch (const std::exception &e) {
            failed_as_expected = std::string(e.what()).find("persistent state was lost") != std::string::npos;
        }
        require(failed_as_expected, "refuse further work after state lost");

        // Recover by resetting state
        c.reset_state();
        auto r_recovered = c.run(accum, {{&x_new, 4}});
        require(first(r_recovered[0]) == 4.0f, "recovered after reset_state");
    }

    // 5. Check stats report state_bytes
    auto stats = server.stats();
    require(stats.size() == 2, "2 graph stats");
    require(stats[0].state_bytes == 1024, "graph 0 state_bytes in stats");
    require(stats[1].state_bytes == 1024, "graph 1 state_bytes in stats");

    server.stop();
    ::rmdir(dir);
}

int main() {
    try {
        scheduler_tests();
        server_tests();
        release_test();
        single_context_benchmark_test();
        persistent_state_broker_test();
    } catch (const std::exception &e) {
        std::cerr << "test_broker: " << e.what() << '\n';
        return 1;
    }
    std::cout << "test_broker: ok\n";
    return 0;
}
