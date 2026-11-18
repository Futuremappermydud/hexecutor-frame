// CDSP broker phase 1: scheduler, admission, Unix-socket server and client.
#include "hexecutor/broker.hpp"
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace hexecutor::broker {
const char *to_string(Status s) {
    switch (s) {
    case Status::ok:
        return "ok";
    case Status::replaced:
        return "replaced";
    case Status::expired:
        return "expired";
    case Status::error:
        return "error";
    }
    return "unknown";
}

// ---------------------------------------------------------------- scheduling
std::optional<Job> Scheduler::submit(Job job) {
    std::optional<Job> displaced;
    if (job.latest_only && !queue_.empty())
        for (auto it = queue_.begin(); it != queue_.end(); ++it)
            if (it->client == job.client && it->stream == job.stream) {
                displaced = std::move(*it);
                queue_.erase(it);
                break;
            }
    queue_.push_back(std::move(job));
    return displaced;
}
std::optional<Job> Scheduler::next(Clock::time_point now, std::vector<Job> &expired) {
    if (queue_.empty())
        return std::nullopt;
    for (auto it = queue_.begin(); it != queue_.end();)
        if (it->deadline < now) {
            expired.push_back(std::move(*it));
            it = queue_.erase(it);
        } else
            ++it;
    if (queue_.empty())
        return std::nullopt;
    if (queue_.size() == 1) {
        Job job = std::move(queue_.front());
        queue_.pop_front();
        return job;
    }
    auto key = [&](const Job &j) {
        return std::make_tuple(j.deadline, uint32_t(j.cls), !(have_last_ && j.graph == last_graph_),
                               j.submitted);
    };
    auto best = std::min_element(queue_.begin(), queue_.end(),
                                 [&](const Job &a, const Job &b) { return key(a) < key(b); });
    Job job = std::move(*best);
    queue_.erase(best);
    return job;
}
std::vector<Job> Scheduler::drop(uint64_t client, std::optional<uint32_t> stream) {
    std::vector<Job> out;
    for (auto it = queue_.begin(); it != queue_.end();)
        if (it->client == client && (!stream || it->stream == *stream)) {
            out.push_back(std::move(*it));
            it = queue_.erase(it);
        } else
            ++it;
    return out;
}

bool Admission::admit(uint64_t key, double period_ms, double cost_ms) {
    if (period_ms <= 0) // sporadic: not rate-admitted
        return true;
    if (!(cost_ms >= 0) || !(period_ms > 0))
        return false;
    double load = cost_ms / period_ms;
    if (utilization() + load > limit_)
        return false;
    load_[key] = load;
    return true;
}
void Admission::release(uint64_t key) { load_.erase(key); }
double Admission::utilization() const {
    double sum = 0;
    for (auto &[_, v] : load_)
        sum += v;
    return sum;
}

// ---------------------------------------------------------------- wire format
namespace {
constexpr uint32_t magic = 0x31425848; // "HXB1"
constexpr size_t max_payload = 64u << 20, max_count = 16;
enum Type : uint32_t { OPEN = 1, OPENED, ERROR, SUBMIT, RESULT, CLOSE, RESET, RESET_DONE };
struct Header {
    uint32_t magic, type, stream, count;
    uint64_t ticket, value;
};
static_assert(sizeof(Header) == 32);
struct Frame {
    Header h{};
    std::vector<Bytes> parts;
};

bool read_all(int fd, void *p, size_t n) {
    auto *b = static_cast<uint8_t *>(p);
    size_t got = 0;
    while (got < n) {
        auto k = ::recv(fd, b + got, n - got, 0);
        if (k < 0 && errno == EINTR)
            continue;
        if (k == 0 && got == 0)
            return false;
        if (k <= 0)
            throw std::runtime_error("broker connection closed mid-frame");
        got += size_t(k);
    }
    return true;
}
void send_all_iov(int fd, struct iovec *iov, int iovcnt) {
    while (iovcnt > 0) {
        struct msghdr msg{};
        msg.msg_iov = iov;
        msg.msg_iovlen = iovcnt;
#ifdef MSG_NOSIGNAL
        auto k = ::sendmsg(fd, &msg, MSG_NOSIGNAL);
#else
        auto k = ::sendmsg(fd, &msg, 0);
#endif
        if (k < 0 && errno == EINTR)
            continue;
        if (k <= 0)
            throw std::runtime_error("broker connection closed while sending");
        size_t written = size_t(k);
        while (iovcnt > 0 && written >= iov[0].iov_len) {
            written -= iov[0].iov_len;
            iov++;
            iovcnt--;
        }
        if (written > 0 && iovcnt > 0) {
            iov[0].iov_base = static_cast<char *>(iov[0].iov_base) + written;
            iov[0].iov_len -= written;
        }
    }
}
void send_frame(int fd, Type type, uint32_t stream, uint64_t ticket, uint64_t value,
                const Buffer *parts, size_t num_parts) {
    if (num_parts > max_count)
        throw std::runtime_error("too many frame parts");
    Header h{magic, type, stream, uint32_t(num_parts), ticket, value};
    struct iovec iov[1 + max_count * 2];
    uint64_t lengths[max_count];
    int iovcnt = 0;
    iov[iovcnt++] = {&h, sizeof h};
    for (size_t i = 0; i < num_parts; i++) {
        if (parts[i].bytes > max_payload)
            throw std::runtime_error("frame part too large");
        lengths[i] = parts[i].bytes;
        iov[iovcnt++] = {&lengths[i], sizeof(uint64_t)};
        if (parts[i].bytes)
            iov[iovcnt++] = {const_cast<void *>(parts[i].data), parts[i].bytes};
    }
    send_all_iov(fd, iov, iovcnt);
}
void send_frame(int fd, Type type, uint32_t stream, uint64_t ticket, uint64_t value,
                const std::vector<Buffer> &parts) {
    send_frame(fd, type, stream, ticket, value, parts.data(), parts.size());
}
std::optional<Frame> read_frame(int fd) {
    Frame f;
    if (!read_all(fd, &f.h, sizeof f.h))
        return std::nullopt;
    if (f.h.magic != magic || f.h.count > max_count)
        throw std::runtime_error("malformed broker frame");
    f.parts.resize(f.h.count);
    for (auto &p : f.parts) {
        uint64_t n = 0;
        if (!read_all(fd, &n, sizeof n) || n > max_payload)
            throw std::runtime_error("malformed broker frame part");
        p.resize(size_t(n));
        if (n && !read_all(fd, p.data(), p.size()))
            throw std::runtime_error("broker connection closed mid-frame");
    }
    return f;
}
Buffer view(const Bytes &b) { return {b.data(), b.size()}; }
Buffer view(const std::string &s) { return {s.data(), s.size()}; }
template <class T> void put(Bytes &b, T v) {
    auto *p = reinterpret_cast<const uint8_t *>(&v);
    b.insert(b.end(), p, p + sizeof v);
}
template <class T> T take(const Bytes &b, size_t &at) {
    if (at + sizeof(T) > b.size())
        throw std::runtime_error("truncated broker record");
    T v;
    std::memcpy(&v, b.data() + at, sizeof v);
    at += sizeof v;
    return v;
}
void no_sigpipe(int fd) {
#ifdef SO_NOSIGPIPE
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#else
    (void)fd;
#endif
}
sockaddr_un address(const std::filesystem::path &path) {
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    auto s = path.string();
    if (s.empty() || s.size() >= sizeof a.sun_path)
        throw std::runtime_error("broker socket path is empty or too long");
    std::memcpy(a.sun_path, s.c_str(), s.size() + 1);
    return a;
}
double ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
} // namespace

// ---------------------------------------------------------------- server
struct Server::Impl {
    struct Stream {
        StreamSpec spec;
        size_t graph = 0;
    };
    struct Conn {
        int fd = -1;
        std::mutex send;
        std::map<uint32_t, Stream> streams;
        uint32_t next_stream = 1;
        bool dead = false;
    };
    std::unique_ptr<Backend> backend;
    ServerOptions options;
    std::vector<GraphInfo> graphs;
    int listen_fd = -1, wake[2] = {-1, -1};
    std::thread io, exec;
    mutable std::mutex mutex;
    std::condition_variable changed;
    bool stopping = false, started = false, release = false;
    Scheduler scheduler;
    Admission admission;
    std::map<uint64_t, std::shared_ptr<Conn>> conns;
    uint64_t next_conn = 1;
    std::vector<GraphStats> stats;

    Impl(std::unique_ptr<Backend> b, ServerOptions o)
        : backend(std::move(b)), options(std::move(o)), admission(options.utilization_limit) {
        if (!backend)
            throw std::runtime_error("broker needs a backend");
        graphs = backend->graphs();
        stats.resize(graphs.size());
        for (size_t i = 0; i < graphs.size(); i++)
            stats[i].state_bytes = graphs[i].state_bytes ? graphs[i].state_bytes : backend->state_bytes();
    }
    void reply(const std::shared_ptr<Conn> &c, const Job &job, Status status, const Result &r) {
        std::lock_guard<std::mutex> g(c->send);
        if (c->dead)
            return;
        struct {
            double queue_ms, run_ms, dsp_ms;
        } timing{r.queue_ms, r.run_ms, r.dsp_ms};
        Buffer parts[max_count];
        parts[0] = {&timing, sizeof timing};
        parts[1] = view(r.error);
        size_t count = 2;
        for (auto &o : r.outputs) {
            if (count >= max_count)
                throw std::runtime_error("too many frame parts in reply");
            parts[count++] = view(o);
        }
        try {
            send_frame(c->fd, RESULT, job.stream, job.ticket, uint64_t(status), parts, count);
        } catch (const std::exception &) {
            c->dead = true;
        }
    }
    void reply_status(const Job &job, Status status, const std::string &error = {}) {
        std::shared_ptr<Conn> c;
        {
            std::lock_guard<std::mutex> g(mutex);
            auto it = conns.find(job.client);
            if (it == conns.end())
                return;
            c = it->second;
        }
        Result r;
        r.status = status;
        r.error = error;
        r.queue_ms = ms(Clock::now() - job.submitted);
        reply(c, job, status, r);
    }
    void send_error(const std::shared_ptr<Conn> &c, uint32_t stream, uint64_t ticket,
                    const std::string &text) {
        std::lock_guard<std::mutex> g(c->send);
        try {
            send_frame(c->fd, ERROR, stream, ticket, 0, {view(text)});
        } catch (const std::exception &) {
            c->dead = true;
        }
    }
    double graph_cost(size_t graph) const {
        return stats[graph].runs ? stats[graph].mean_run_ms : -1;
    }
    void handle(uint64_t id, const std::shared_ptr<Conn> &c, Frame &f) {
        if (f.h.type == OPEN) {
            if (f.parts.size() != 2)
                throw std::runtime_error("malformed OPEN");
            StreamSpec spec;
            spec.graph.assign(f.parts[0].begin(), f.parts[0].end());
            size_t at = 0;
            auto cls = take<uint32_t>(f.parts[1], at);
            spec.latest_only = take<uint32_t>(f.parts[1], at) != 0;
            spec.period_ms = take<double>(f.parts[1], at);
            spec.deadline_ms = take<double>(f.parts[1], at);
            spec.cost_hint_ms = take<double>(f.parts[1], at);
            if (cls > uint32_t(Class::bulk) || !(spec.deadline_ms > 0) || !(spec.period_ms >= 0))
                return send_error(c, 0, f.h.ticket, "invalid stream specification");
            spec.cls = Class(cls);
            auto g = std::find_if(graphs.begin(), graphs.end(),
                                  [&](const GraphInfo &x) { return x.name == spec.graph; });
            if (g == graphs.end())
                return send_error(c, 0, f.h.ticket, "unknown graph " + spec.graph);
            size_t graph = size_t(g - graphs.begin());
            uint32_t sid;
            {
                std::lock_guard<std::mutex> lock(mutex);
                sid = c->next_stream++;
                double cost = graph_cost(graph);
                if (cost < 0)
                    cost = spec.cost_hint_ms;
                if (!admission.admit((id << 32) | sid, spec.period_ms, cost))
                    return send_error(c, 0, f.h.ticket,
                                      "admission refused: utilization would exceed the limit");
                c->streams[sid] = {spec, graph};
            }
            Bytes sizes;
            sizes.reserve(sizeof(uint32_t) * 2 +
                          sizeof(uint64_t) * (g->input_bytes.size() + g->output_bytes.size()));
            put(sizes, uint32_t(g->input_bytes.size()));
            put(sizes, uint32_t(g->output_bytes.size()));
            for (auto n : g->input_bytes)
                put(sizes, uint64_t(n));
            for (auto n : g->output_bytes)
                put(sizes, uint64_t(n));
            std::lock_guard<std::mutex> lock(c->send);
            Buffer part = view(sizes);
            send_frame(c->fd, OPENED, sid, f.h.ticket, 0, &part, 1);
        } else if (f.h.type == SUBMIT) {
            Job job;
            job.client = id;
            job.stream = f.h.stream;
            job.ticket = f.h.ticket;
            job.submitted = Clock::now();
            std::optional<Job> displaced;
            std::string error;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto s = (c->streams.size() == 1 && c->streams.begin()->first == f.h.stream)
                             ? c->streams.begin()
                             : c->streams.find(f.h.stream);
                if (s == c->streams.end())
                    error = "unknown stream";
                else {
                    auto &info = graphs[s->second.graph];
                    bool ok = f.parts.size() == info.input_bytes.size();
                    for (size_t i = 0; ok && i < f.parts.size(); i++)
                        ok = f.parts[i].size() == info.input_bytes[i];
                    if (!ok)
                        error = "inputs do not match the graph contract";
                    else {
                        auto &spec = s->second.spec;
                        job.graph = s->second.graph;
                        job.cls = spec.cls;
                        job.latest_only = spec.latest_only;
                        job.deadline =
                            job.submitted + std::chrono::duration_cast<Clock::duration>(
                                                std::chrono::duration<double, std::milli>(
                                                    spec.deadline_ms));
                        job.inputs = std::move(f.parts);
                        displaced = scheduler.submit(std::move(job));
                    }
                }
            }
            if (!error.empty()) {
                Result r;
                r.status = Status::error;
                r.error = error;
                reply(c, job, Status::error, r);
                return;
            }
            changed.notify_one();
            if (displaced)
                reply_status(*displaced, Status::replaced);
        } else if (f.h.type == CLOSE) {
            std::vector<Job> dropped;
            {
                std::lock_guard<std::mutex> lock(mutex);
                c->streams.erase(f.h.stream);
                admission.release((id << 32) | f.h.stream);
                dropped = scheduler.drop(id, f.h.stream);
            }
            for (auto &j : dropped)
                reply_status(j, Status::error, "stream closed");
        } else if (f.h.type == RESET) {
            std::string err;
            try {
                backend->reset_state();
            } catch (const std::exception &e) {
                err = e.what();
            }
            std::lock_guard<std::mutex> lock(c->send);
            if (!err.empty()) {
                send_frame(c->fd, ERROR, 0, f.h.ticket, 0, {view(err)});
            } else {
                send_frame(c->fd, RESET_DONE, 0, f.h.ticket, 0, nullptr, 0);
            }
        } else
            throw std::runtime_error("unexpected frame type from client");
    }
    void disconnect(uint64_t id) {
        std::vector<Job> dropped;
        std::shared_ptr<Conn> c;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = conns.find(id);
            if (it == conns.end())
                return;
            c = it->second;
            for (auto &[sid, _] : c->streams)
                admission.release((id << 32) | sid);
            dropped = scheduler.drop(id);
            conns.erase(it);
        }
        {
            std::lock_guard<std::mutex> g(c->send);
            c->dead = true;
            ::close(c->fd);
        }
    }
    void io_loop() {
        std::vector<pollfd> fds;
        std::vector<std::pair<uint64_t, std::shared_ptr<Conn>>> order;
        fds.reserve(8);
        order.reserve(8);
        for (;;) {
            fds.clear();
            order.clear();
            fds.push_back({wake[0], POLLIN, 0});
            fds.push_back({listen_fd, POLLIN, 0});
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (stopping)
                    return;
                for (auto &[id, c] : conns) {
                    fds.push_back({c->fd, POLLIN, 0});
                    order.emplace_back(id, c);
                }
            }
            if (::poll(fds.data(), fds.size(), -1) < 0) {
                if (errno == EINTR)
                    continue;
                return;
            }
            if (fds[0].revents)
                return;
            if (fds[1].revents & POLLIN) {
                int fd = ::accept(listen_fd, nullptr, nullptr);
                if (fd >= 0) {
                    no_sigpipe(fd);
                    auto c = std::make_shared<Conn>();
                    c->fd = fd;
                    std::lock_guard<std::mutex> lock(mutex);
                    conns[next_conn++] = c;
                }
            }
            for (size_t i = 2; i < fds.size(); i++) {
                if (!fds[i].revents)
                    continue;
                auto [id, c] = order[i - 2];
                try {
                    auto f = read_frame(c->fd);
                    if (!f) {
                        disconnect(id);
                        continue;
                    }
                    handle(id, c, *f);
                } catch (const std::exception &) {
                    disconnect(id);
                }
            }
        }
    }
    void exec_loop() {
        auto last_work = Clock::now();
        bool device_held = false;
        std::vector<Buffer> in;
        in.reserve(16);
        std::vector<Job> expired;
        expired.reserve(4);
        for (;;) {
            std::optional<Job> job;
            expired.clear();
            {
                std::unique_lock<std::mutex> lock(mutex);
                auto idle_after = std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double, std::milli>(options.idle_close_ms));
                for (;;) {
                    if (stopping || release)
                        break;
                    job = scheduler.next(Clock::now(), expired);
                    if (job || !expired.empty())
                        break;
                    if (device_held) {
                        if (changed.wait_until(lock, last_work + idle_after) ==
                                std::cv_status::timeout &&
                            scheduler.size() == 0)
                            break;
                    } else
                        changed.wait(lock);
                }
            }
            for (auto &e : expired)
                reply_status(e, Status::expired);
            if (stopping_now())
                break;
            if (take_release()) {
                try {
                    backend->release();
                } catch (const std::exception &) {
                }
                device_held = false;
                continue;
            }
            if (!job) {
                if (device_held && Clock::now() - last_work >=
                                       std::chrono::duration_cast<Clock::duration>(
                                           std::chrono::duration<double, std::milli>(
                                               options.idle_close_ms))) {
                    try {
                        backend->idle();
                    } catch (const std::exception &) {
                    }
                    device_held = false;
                }
                continue;
            }
            Result r;
            auto t0 = Clock::now();
            r.queue_ms = ms(t0 - job->submitted);
            in.clear();
            for (auto &b : job->inputs)
                in.push_back(view(b));
            try {
                r.outputs = backend->run(job->graph, in, r.dsp_ms);
                r.status = Status::ok;
                device_held = true;
            } catch (const std::exception &e) {
                r.status = Status::error;
                r.error = e.what();
            }
            auto t1 = Clock::now();
            r.run_ms = ms(t1 - t0);
            last_work = t1;
            std::shared_ptr<Conn> c;
            {
                std::lock_guard<std::mutex> lock(mutex);
                scheduler.ran(job->graph);
                if (r.status == Status::ok) {
                    auto &s = stats[job->graph];
                    double a = s.runs ? 0.1 : 1.0;
                    s.mean_run_ms += a * (r.run_ms - s.mean_run_ms);
                    s.mean_dsp_ms += a * (r.dsp_ms - s.mean_dsp_ms);
                    s.runs++;
                }
                if (conns.size() == 1 && conns.begin()->first == job->client)
                    c = conns.begin()->second;
                else {
                    auto it = conns.find(job->client);
                    if (it != conns.end())
                        c = it->second;
                }
            }
            if (c)
                reply(c, *job, r.status, r);
        }
    }
    bool stopping_now() {
        std::lock_guard<std::mutex> lock(mutex);
        return stopping;
    }
    bool take_release() {
        std::lock_guard<std::mutex> lock(mutex);
        bool r = release;
        release = false;
        return r;
    }
};

Server::Server(std::unique_ptr<Backend> b, ServerOptions o)
    : impl_(std::make_unique<Impl>(std::move(b), std::move(o))) {}
Server::~Server() { stop(); }
void Server::start() {
    auto &m = *impl_;
    if (m.started)
        throw std::runtime_error("broker already started");
    auto a = address(m.options.socket);
    std::error_code ec;
    if (std::filesystem::is_socket(m.options.socket, ec))
        std::filesystem::remove(m.options.socket, ec);
    m.listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (m.listen_fd < 0)
        throw std::runtime_error("cannot create broker socket");
    auto old = ::umask(0077); // owner-only socket
    int bound = ::bind(m.listen_fd, reinterpret_cast<sockaddr *>(&a), sizeof a);
    ::umask(old);
    if (bound < 0 || ::listen(m.listen_fd, 16) < 0) {
        ::close(m.listen_fd);
        m.listen_fd = -1;
        throw std::runtime_error("cannot listen on " + m.options.socket.string() + ": " +
                                 std::strerror(errno));
    }
    if (::pipe(m.wake) < 0)
        throw std::runtime_error("cannot create broker wake pipe");
    m.started = true;
    m.io = std::thread([&m] { m.io_loop(); });
    m.exec = std::thread([&m] { m.exec_loop(); });
}
void Server::stop() {
    if (!impl_ || !impl_->started)
        return;
    auto &m = *impl_;
    std::vector<Job> pending;
    {
        std::lock_guard<std::mutex> lock(m.mutex);
        if (m.stopping)
            return;
        m.stopping = true;
        for (auto &[id, _] : m.conns) {
            auto d = m.scheduler.drop(id);
            pending.insert(pending.end(), std::make_move_iterator(d.begin()),
                           std::make_move_iterator(d.end()));
        }
    }
    for (auto &j : pending)
        m.reply_status(j, Status::error, "broker stopping");
    m.changed.notify_all();
    char x = 0;
    (void)!::write(m.wake[1], &x, 1);
    if (m.io.joinable())
        m.io.join();
    if (m.exec.joinable())
        m.exec.join();
    try {
        m.backend->idle();
    } catch (const std::exception &) {
    }
    for (auto &[_, c] : m.conns)
        ::close(c->fd);
    m.conns.clear();
    ::close(m.listen_fd);
    ::close(m.wake[0]);
    ::close(m.wake[1]);
    std::error_code ec;
    std::filesystem::remove(m.options.socket, ec);
}
void Server::release_device() {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->release = true;
    }
    impl_->changed.notify_all();
}
std::vector<Server::GraphStats> Server::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->stats;
}

// ---------------------------------------------------------------- client
Client::Client(const std::filesystem::path &socket) {
    auto a = address(socket);
    fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd_ < 0)
        throw std::runtime_error("cannot create broker client socket");
    no_sigpipe(fd_);
    if (::connect(fd_, reinterpret_cast<sockaddr *>(&a), sizeof a) < 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("cannot connect to broker at " + socket.string());
    }
}
Client::~Client() {
    if (fd_ >= 0)
        ::close(fd_);
}
namespace {
Result decode_result(Frame &f) {
    if (f.parts.size() < 2)
        throw std::runtime_error("malformed broker result");
    Result r;
    r.status = Status(f.h.value);
    size_t at = 0;
    r.queue_ms = take<double>(f.parts[0], at);
    r.run_ms = take<double>(f.parts[0], at);
    r.dsp_ms = take<double>(f.parts[0], at);
    r.error.assign(f.parts[1].begin(), f.parts[1].end());
    r.outputs.reserve(f.parts.size() - 2);
    for (size_t i = 2; i < f.parts.size(); i++)
        r.outputs.push_back(std::move(f.parts[i]));
    return r;
}
} // namespace
Client::Stream Client::open(const StreamSpec &spec) {
    Bytes record;
    record.reserve(sizeof(uint32_t) * 2 + sizeof(double) * 3);
    put(record, uint32_t(spec.cls));
    put(record, uint32_t(spec.latest_only));
    put(record, spec.period_ms);
    put(record, spec.deadline_ms);
    put(record, spec.cost_hint_ms);
    uint64_t request = next_ticket_++;
    Buffer parts[2] = {view(spec.graph), view(record)};
    send_frame(fd_, OPEN, 0, request, 0, parts, 2);
    for (;;) {
        auto f = read_frame(fd_);
        if (!f)
            throw std::runtime_error("broker closed the connection");
        if (f->h.type == RESULT) {
            pending_[f->h.ticket] = decode_result(*f);
            continue;
        }
        if (f->h.ticket != request)
            throw std::runtime_error("unexpected broker reply");
        if (f->h.type == ERROR)
            throw std::runtime_error(
                "broker: " + std::string(f->parts.empty() ? "" : std::string(f->parts[0].begin(),
                                                                             f->parts[0].end())));
        if (f->h.type != OPENED || f->parts.size() != 1)
            throw std::runtime_error("malformed broker OPENED");
        Stream s;
        s.id = f->h.stream;
        size_t at = 0;
        auto nin = take<uint32_t>(f->parts[0], at), nout = take<uint32_t>(f->parts[0], at);
        s.input_bytes.reserve(nin);
        s.output_bytes.reserve(nout);
        for (uint32_t i = 0; i < nin; i++)
            s.input_bytes.push_back(size_t(take<uint64_t>(f->parts[0], at)));
        for (uint32_t i = 0; i < nout; i++)
            s.output_bytes.push_back(size_t(take<uint64_t>(f->parts[0], at)));
        return s;
    }
}
void Client::close(const Stream &s) { send_frame(fd_, CLOSE, s.id, 0, 0, {}); }
uint64_t Client::submit(const Stream &s, const std::vector<Buffer> &inputs) {
    uint64_t t = next_ticket_++;
    send_frame(fd_, SUBMIT, s.id, t, 0, inputs.data(), inputs.size());
    return t;
}
Result Client::wait(uint64_t ticket) {
    auto it = pending_.find(ticket);
    if (it != pending_.end()) {
        Result r = std::move(it->second);
        pending_.erase(it);
        return r;
    }
    for (;;) {
        auto f = read_frame(fd_);
        if (!f)
            throw std::runtime_error("broker closed the connection");
        if (f->h.type != RESULT)
            throw std::runtime_error("unexpected broker frame while waiting");
        if (f->h.ticket == ticket)
            return decode_result(*f);
        pending_[f->h.ticket] = decode_result(*f);
    }
}
std::vector<Bytes> Client::run(const Stream &s, const std::vector<Buffer> &inputs) {
    auto r = wait(submit(s, inputs));
    if (r.status != Status::ok)
        throw std::runtime_error(std::string("broker job ") + to_string(r.status) +
                                 (r.error.empty() ? "" : ": " + r.error));
    return std::move(r.outputs);
}
void Client::reset_state() {
    uint64_t request = next_ticket_++;
    send_frame(fd_, RESET, 0, request, 0, nullptr, 0);
    for (;;) {
        auto f = read_frame(fd_);
        if (!f)
            throw std::runtime_error("broker closed the connection");
        if (f->h.type == RESULT) {
            pending_[f->h.ticket] = decode_result(*f);
            continue;
        }
        if (f->h.ticket != request)
            throw std::runtime_error("unexpected broker reply");
        if (f->h.type == ERROR)
            throw std::runtime_error(
                "broker: " + std::string(f->parts.empty() ? "" : std::string(f->parts[0].begin(),
                                                                             f->parts[0].end())));
        if (f->h.type == RESET_DONE)
            return;
        throw std::runtime_error("unexpected broker reply to reset_state");
    }
}
} // namespace hexecutor::broker
