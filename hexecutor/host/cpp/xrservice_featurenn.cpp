// SPDX-License-Identifier: LGPL-3.0-or-later
#include "hexecutor/xrservice_featurenn.hpp"
#include "hexecutor/broker.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#ifdef __linux__
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#endif

namespace hexecutor::xrservice {
namespace {
constexpr uint32_t channels = 2, side = 128, gemm_floats = 552;
constexpr size_t head_floats = 69, head_bytes = head_floats * sizeof(float);
constexpr size_t gazenet_image_floats = channels * side * side;
constexpr size_t gazenet_output_floats = 64;
constexpr float input_mean = 0.449f, input_std = 0.226f;

Bytes join_float_outputs(const std::vector<Bytes> &parts, size_t min_floats, size_t max_floats) {
    Bytes joined;
    joined.reserve(max_floats * sizeof(float));
    for (auto &part : parts) {
        if (part.size() % sizeof(float) ||
            joined.size() + part.size() > max_floats * sizeof(float))
            throw std::runtime_error("stock adapter graph returned an invalid float output");
        joined.insert(joined.end(), part.begin(), part.end());
    }
    size_t floats = joined.size() / sizeof(float);
    if (floats < min_floats || floats > max_floats)
        throw std::runtime_error("stock adapter graph returned the wrong number of floats");
    auto *values = reinterpret_cast<const float *>(joined.data());
    for (size_t i = 0; i < floats; i++)
        if (!std::isfinite(values[i]))
            throw std::runtime_error("stock adapter graph returned a non-finite output");
    return joined;
}
using Clock = std::chrono::steady_clock;

template <class T> T get(const void *base, size_t offset) {
    T value;
    std::memcpy(&value, static_cast<const uint8_t *>(base) + offset, sizeof value);
    return value;
}
uint32_t load(const uint8_t *memory, size_t offset) {
    return __atomic_load_n(reinterpret_cast<const volatile uint32_t *>(memory + offset),
                           __ATOMIC_SEQ_CST);
}
void store(uint8_t *memory, size_t offset, uint32_t value) {
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(memory + offset), value,
                     __ATOMIC_SEQ_CST);
}
#ifdef __linux__
constexpr off_t dma_offset = 0x80000000;
void send_all(int fd, const void *data, size_t size) {
    auto *p = static_cast<const uint8_t *>(data);
    while (size) {
        auto n = ::send(fd, p, size, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("dsp_service send failed");
        p += n;
        size -= size_t(n);
    }
}
#endif
} // namespace

size_t FeatureNNRequest::input_bytes() const {
    return size_t(eyes) * controllers * height * width;
}
size_t FeatureNNRequest::output_bytes() const {
    return size_t(eyes) * controllers * featurenn_output_record_bytes;
}
FeatureNNRequest read_featurenn_request(const void *shared, size_t bytes) {
    if (!shared || bytes < featurenn_input_offset)
        throw std::runtime_error("XRService FeatureNN shared block is truncated");
    FeatureNNRequest r;
    auto at = static_cast<const uint8_t *>(shared) + 0x80;
    std::memcpy(&r.eyes, at, sizeof(uint32_t));
    uint32_t views;
    std::memcpy(&views, at + 4, sizeof(uint32_t));
    std::memcpy(&r.height, at + 12, sizeof(uint32_t));
    std::memcpy(&r.width, at + 8, sizeof(uint32_t));
    if (r.eyes != channels || views % r.eyes || (r.controllers = views / r.eyes) == 0 ||
        r.controllers > 2 || r.height != side || r.width != side)
        throw std::runtime_error("unsupported XRService FeatureNN request shape");
    if (r.input_bytes() > featurenn_output_offset - featurenn_input_offset)
        throw std::runtime_error("XRService FeatureNN input exceeds the stock shared block");
    return r;
}
Bytes make_featurenn_input(const FeatureNNRequest &r, const void *shared, size_t bytes,
                           uint32_t controller) {
    if (controller >= r.controllers || bytes < featurenn_input_offset + r.input_bytes())
        throw std::runtime_error("XRService FeatureNN input is out of range");
    auto *raw = static_cast<const uint8_t *>(shared) + featurenn_input_offset +
                size_t(controller) * r.eyes * r.height * r.width;
    Bytes out(side * side * 3 * sizeof(float));
    auto *dst = reinterpret_cast<float *>(out.data());
    for (uint32_t eye = 0; eye < r.eyes; eye++)
        for (size_t pixel = 0; pixel < side * side; pixel++) {
            float x = float(raw[size_t(eye) * side * side + pixel]) / 255.0f;
            dst[pixel * 3 + eye] = (x - input_mean) / input_std;
        }
    return out;
}
void write_featurenn_output(const FeatureNNRequest &r, const std::vector<Bytes> &graph_outputs,
                            uint32_t controller, void *shared, size_t bytes) {
    if (controller >= r.controllers || bytes < featurenn_output_offset + r.output_bytes())
        throw std::runtime_error("XRService FeatureNN output is out of range");
    Bytes joined = join_float_outputs(graph_outputs, gemm_floats, gemm_floats);
    auto *gemm = reinterpret_cast<const float *>(joined.data());
    auto *out = static_cast<uint8_t *>(shared) + featurenn_output_offset +
                size_t(controller) * r.eyes * featurenn_output_record_bytes;
    for (uint32_t eye = 0; eye < r.eyes; eye++) {
        auto *record = out + size_t(eye) * featurenn_output_record_bytes;
        for (size_t head = 0; head < 4; head++) {
            size_t source = (size_t(eye) * 92 + head * 23) * 3;
            if (head != 1) {
                std::memcpy(record + head * head_bytes, gemm + source, head_bytes);
                continue;
            }
            float uncertainty[head_floats];
            for (size_t i = 0; i < head_floats; i++)
                uncertainty[i] = 2.0f * std::min(gemm[source + i], 10.0f);
            std::memcpy(record + head_bytes, uncertainty, head_bytes);
        }
    }
}
void run_featurenn_request(
    const FeatureNNRequest &r, void *shared, size_t bytes,
    const std::function<std::vector<Bytes>(const Bytes &input)> &run) {
    Bytes staged(featurenn_output_offset + r.output_bytes());
    for (uint32_t controller = 0; controller < r.controllers; controller++) {
        auto input = make_featurenn_input(r, shared, bytes, controller);
        auto outputs = run(input);
        write_featurenn_output(r, outputs, controller, staged.data(), staged.size());
    }
    std::memcpy(static_cast<uint8_t *>(shared) + featurenn_output_offset,
                staged.data() + featurenn_output_offset, r.output_bytes());
}

Bytes make_gazenet_input(const void *shared, size_t bytes) {
    if (!shared || bytes < gazenet_input_offset + gazenet_image_floats * sizeof(float))
        throw std::runtime_error("stock Gazenet image input is out of range");
    auto *src = static_cast<const uint8_t *>(shared) + gazenet_input_offset;
    Bytes out(side * side * 3 * sizeof(float));
    auto *dst = reinterpret_cast<float *>(out.data());
    for (size_t pixel = 0; pixel < side * side; pixel++) {
        for (size_t eye = 0; eye < channels; eye++)
            std::memcpy(&dst[pixel * 3 + eye], src + (eye * side * side + pixel) * sizeof(float),
                        sizeof(float));
        dst[pixel * 3 + 2] = 0;
    }
    return out;
}
void write_gazenet_output(const std::vector<Bytes> &graph_outputs, void *shared, size_t bytes) {
    if (!shared || bytes < gazenet_output_offset + gazenet_output_floats * sizeof(float))
        throw std::runtime_error("stock Gazenet image output is out of range");
    Bytes joined = join_float_outputs(graph_outputs, gazenet_output_floats, gazenet_output_floats);
    std::memcpy(static_cast<uint8_t *>(shared) + gazenet_output_offset, joined.data(), joined.size());
}
void run_gazenet_request(
    void *shared, size_t bytes,
    const std::function<std::vector<Bytes>(const Bytes &input)> &run) {
    auto outputs = run(make_gazenet_input(shared, bytes));
    Bytes staged(gazenet_output_offset + gazenet_output_floats * sizeof(float));
    write_gazenet_output(outputs, staged.data(), staged.size());
    std::memcpy(static_cast<uint8_t *>(shared) + gazenet_output_offset,
                staged.data() + gazenet_output_offset, gazenet_output_floats * sizeof(float));
}
Bytes make_gazenet_vector_input(const void *shared, size_t bytes, uint32_t floats) {
    if (!shared || floats == 0 || floats > gazenet_vector_floats ||
        bytes < gazenet_vector_offset + size_t(floats) * sizeof(float))
        throw std::runtime_error("stock Gazenet vector input is out of range");
    auto *at = static_cast<const uint8_t *>(shared) + gazenet_vector_offset;
    return Bytes(at, at + size_t(floats) * sizeof(float));
}
size_t write_gazenet_vector_output(const std::vector<Bytes> &graph_outputs, void *shared,
                                   size_t bytes) {
    if (!shared || bytes < gazenet_vector_offset + gazenet_vector_floats * sizeof(float))
        throw std::runtime_error("stock Gazenet vector output is out of range");
    Bytes joined = join_float_outputs(graph_outputs, 1, gazenet_vector_floats);
    std::memcpy(static_cast<uint8_t *>(shared) + gazenet_vector_offset, joined.data(), joined.size());
    return joined.size() / sizeof(float);
}

struct Adapter::Impl {
    std::filesystem::path socket;
    AdapterOptions options;
    std::function<void()> yield;
    std::atomic<bool> stopping{false};
    std::thread thread;
    void *provided_memory = nullptr;
    size_t provided_bytes = 0;
    size_t dma_len = dma_bytes;
    int service = -1, descriptors[2] = {-1, -1};
    uint8_t *memory = nullptr;
    uint8_t *featurenn = nullptr, *gazenet = nullptr;
    std::unique_ptr<broker::Client> client;
    broker::Client::Stream featurenn_stream, gazenet_stream, vector_stream;

    void map_service() {
#ifdef __linux__
        service = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (service < 0)
            throw std::runtime_error("cannot create stock service adapter socket");
        timeval timeout{2, 0};
        setsockopt(service, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout);
        setsockopt(service, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof timeout);
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path + 1, "dsp_service", 11);
        if (::connect(service, reinterpret_cast<sockaddr *>(&address),
                      offsetof(sockaddr_un, sun_path) + 12) < 0)
            throw std::runtime_error("stock adapter cannot connect to dsp_service");
        uint32_t request = 4;
        send_all(service, &request, sizeof request);
        char response[64];
        alignas(cmsghdr) char ancillary[CMSG_SPACE(2 * sizeof(int))]{};
        iovec iov{response, sizeof response};
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = ancillary;
        msg.msg_controllen = sizeof ancillary;
        auto received = ::recvmsg(service, &msg, MSG_CMSG_CLOEXEC);
        size_t count = 0;
        if (received >= 0)
            for (auto *c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c))
                if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
                    auto *fds = reinterpret_cast<int *>(CMSG_DATA(c));
                    auto n = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
                    for (size_t i = 0; i < n && count < 2; i++)
                        descriptors[count++] = fds[i];
                }
        if (received != 4 || std::memcmp(response, "ADSP", 4) || count != 2 ||
            (msg.msg_flags & MSG_CTRUNC))
            throw std::runtime_error("unexpected dsp_service handshake in stock adapter");
        struct stat info{};
        if (fstat(descriptors[1], &info) || info.st_size != dma_bytes)
            throw std::runtime_error("unexpected stock shared DMA size");
        void *p = mmap(nullptr, dma_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptors[1],
                       dma_offset);
        if (p == MAP_FAILED)
            throw std::runtime_error("cannot map stock shared DMA");
        memory = static_cast<uint8_t *>(p);
        dma_len = dma_bytes;
#else
        throw std::runtime_error("stock service adapter requires Linux to map dsp_service");
#endif
    }
    void check_header() {
        if (!memory)
            throw std::runtime_error("stock adapter shared memory is null");
        if (get<uint64_t>(memory, 0xa90000) != 0x1614df651d03f542 ||
            get<uint64_t>(memory, 0xa90008) != 9)
            throw std::runtime_error("unsupported stock CDSP protocol header");
        featurenn = memory + featurenn_control_offset;
        gazenet = memory + gazenet_control_offset;
    }
    broker::Client::Stream open_stream(const std::string &graph, size_t input_bytes,
                                       size_t min_output_floats, size_t max_output_floats) {
        broker::StreamSpec spec;
        spec.graph = graph;
        spec.cls = broker::Class::tracking;
        spec.period_ms = 0;
        spec.deadline_ms = 50;
        spec.latest_only = false;
        auto stream = client->open(spec);
        if (stream.input_bytes != std::vector<size_t>{input_bytes})
            throw std::runtime_error("stock adapter graph " + graph + " has the wrong input contract");
        size_t outputs = 0;
        for (auto n : stream.output_bytes) {
            if (n % sizeof(float) || n / sizeof(float) > max_output_floats - outputs)
                throw std::runtime_error("stock adapter graph " + graph +
                                         " has the wrong output contract");
            outputs += n / sizeof(float);
        }
        if (outputs < min_output_floats)
            throw std::runtime_error("stock adapter graph " + graph + " has the wrong output contract");
        return stream;
    }
    void open_streams() {
        client = std::make_unique<broker::Client>(socket);
        if (!options.featurenn_graph.empty())
            featurenn_stream = open_stream(options.featurenn_graph, side * side * 3 * sizeof(float),
                                           gemm_floats, gemm_floats);
        if (!options.gazenet_graph.empty())
            gazenet_stream = open_stream(options.gazenet_graph, side * side * 3 * sizeof(float),
                                         gazenet_output_floats, gazenet_output_floats);
        if (!options.gazenet_vector_graph.empty())
            vector_stream = open_stream(options.gazenet_vector_graph,
                                        gazenet_vector_floats * sizeof(float), 1,
                                        gazenet_vector_floats);
    }
    void warm_one(const broker::Client::Stream &stream, size_t bytes, bool featurenn_input) {
        Bytes input(bytes);
        if (featurenn_input) {
            auto *p = reinterpret_cast<float *>(input.data());
            for (size_t i = 0; i < bytes / sizeof(float); i++)
                p[i] = -input_mean / input_std;
        }
        (void)client->run(stream, {{input.data(), input.size()}});
    }
    void warm_graphs() {
        if (featurenn_stream.id)
            warm_one(featurenn_stream, side * side * 3 * sizeof(float), true);
        if (gazenet_stream.id)
            warm_one(gazenet_stream, side * side * 3 * sizeof(float), false);
        if (vector_stream.id)
            warm_one(vector_stream, gazenet_vector_floats * sizeof(float), false);
    }
    bool stock_busy() {
        const bool gaze_enabled = !options.gazenet_graph.empty() ||
                                  !options.gazenet_vector_graph.empty();
        return load(featurenn, 4) ||
               (gaze_enabled && (load(gazenet, 4) || load(gazenet, 0xc)));
    }
    void warm_up() {
        auto deadline = Clock::now() + std::chrono::seconds(5);
        while (stock_busy() && Clock::now() < deadline) {
            yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (stock_busy())
            throw std::runtime_error("stock CDSP job did not finish before adapter startup");
        warm_graphs();
    }
    void run() {
        const bool gaze_enabled = !options.gazenet_graph.empty() ||
                                  !options.gazenet_vector_graph.empty();
        bool owned = false, failed = false, gaze_owned = false, gaze_failed = false;
        bool yielded = false, reacquire = false, grab_pending = false, vector_failed = false;
        bool gaze_grabbed = gaze_enabled && load(gazenet, 8) == gazenet_job && !load(gazenet, 4);
        while (!stopping) {
            uint32_t f_state = load(featurenn, 0), f_active = load(featurenn, 4);
            uint32_t f_taken = load(featurenn, 8);
            uint32_t g_state = gaze_enabled ? load(gazenet, 0) : 0;
            uint32_t g_active = gaze_enabled ? load(gazenet, 4) : 0;
            uint32_t g_taken = gaze_enabled ? load(gazenet, 8) : 0;
            uint32_t g_ready = gaze_enabled ? load(gazenet, 0xc) : 0;
            if (owned && !f_active) {
                store(featurenn, 8, 0);
                store(featurenn, 0, 1);
                owned = failed = false;
                continue;
            }
            if (gaze_owned && !g_active) {
                store(gazenet, 0, 1);
                gaze_owned = gaze_failed = false;
                continue;
            }
            if (g_active && g_taken == 2) {
                gaze_grabbed = false;
                grab_pending = false;
            }
            if (!g_active && grab_pending) {
                grab_pending = false;
                gaze_grabbed = g_taken == gazenet_job && (g_state == 3 || g_state == 1);
            }
            if (!g_ready)
                vector_failed = false;

            bool featurenn_served = f_active && f_taken == featurenn_job && featurenn_stream.id;
            bool gazenet_served = g_active && g_taken == gazenet_job && gazenet_stream.id &&
                                  gaze_grabbed && !grab_pending;
            bool vector_shape = g_ready && g_taken != gazenet_job && g_taken > 0 &&
                                g_taken <= gazenet_vector_floats && vector_stream.id &&
                                vector_stream.input_bytes[0] == size_t(g_taken) * sizeof(float);
            bool stock = (f_active && !featurenn_served) || (g_active && !gazenet_served) ||
                         (g_ready && (!vector_shape || vector_failed));
            if (stock) {
                reacquire = true;
                if (g_active && g_taken == gazenet_job && !gaze_grabbed)
                    grab_pending = true;
                if (!yielded) {
                    std::cerr << "[hexecutord] yielding to stock CDSP job FeatureNN=" << f_taken
                              << " Gazenet=" << g_taken << '\n';
                    yield();
                    yielded = true;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            yielded = false;
            if (reacquire) {
                try {
                    warm_graphs();
                    reacquire = false;
                } catch (const std::exception &e) {
                    std::cerr << "[hexecutord] cannot reacquire after stock CDSP job: "
                              << e.what() << '\n';
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
            }
            if (f_active && f_taken == featurenn_job && featurenn_stream.id && !owned && !failed &&
                f_state != 3) {
                try {
                    auto request = read_featurenn_request(featurenn, dma_len - featurenn_control_offset);
                    run_featurenn_request(request, featurenn, dma_len - featurenn_control_offset,
                                          [&](const Bytes &input) {
                                              return client->run(featurenn_stream,
                                                                 {{input.data(), input.size()}});
                                          });
                    __atomic_thread_fence(__ATOMIC_SEQ_CST);
                    store(featurenn, 0, 3);
                    owned = true;
                } catch (const std::exception &e) {
                    std::cerr << "[hexecutord] XRService FeatureNN job failed: " << e.what() << '\n';
                    failed = owned = true;
                }
                continue;
            }
            if (g_active && g_taken == gazenet_job && gazenet_stream.id && gaze_grabbed &&
                !gaze_owned && !gaze_failed && g_state != 3) {
                try {
                    run_gazenet_request(gazenet, dma_len - gazenet_control_offset,
                                        [&](const Bytes &input) {
                                            return client->run(gazenet_stream,
                                                               {{input.data(), input.size()}});
                                        });
                    __atomic_thread_fence(__ATOMIC_SEQ_CST);
                    store(gazenet, 0, 3);
                    gaze_owned = true;
                } catch (const std::exception &e) {
                    std::cerr << "[hexecutord] stock Gazenet image job failed: " << e.what() << '\n';
                    gaze_failed = gaze_owned = true;
                }
                continue;
            }
            if (g_ready && vector_shape && !vector_failed) {
                try {
                    auto input = make_gazenet_vector_input(gazenet, dma_len - gazenet_control_offset,
                                                           g_taken);
                    auto outputs = client->run(vector_stream, {{input.data(), input.size()}});
                    auto count = write_gazenet_vector_output(outputs, gazenet,
                                                             dma_len - gazenet_control_offset);
                    store(gazenet, 8, uint32_t(count));
                    __atomic_thread_fence(__ATOMIC_SEQ_CST);
                    store(gazenet, 0xc, 0);
                } catch (const std::exception &e) {
                    std::cerr << "[hexecutord] stock Gazenet vector job failed: " << e.what() << '\n';
                    vector_failed = true;
                }
                continue;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    void close() {
        stopping = true;
        if (thread.joinable())
            thread.join();
        client.reset();
        if (memory && !provided_memory) {
#ifdef __linux__
            munmap(memory, dma_bytes);
#endif
        }
        memory = nullptr;
        if (service >= 0) {
            ::close(service);
            service = -1;
        }
        for (int &fd : descriptors) {
            if (fd >= 0) {
                ::close(fd);
                fd = -1;
            }
        }
    }
};

Adapter::Adapter(std::filesystem::path socket, AdapterOptions options, std::function<void()> yield,
                 void *shared_memory, size_t shared_bytes)
    : impl_(std::make_unique<Impl>()) {
    impl_->socket = std::move(socket);
    impl_->options = std::move(options);
    impl_->yield = std::move(yield);
    impl_->provided_memory = shared_memory;
    impl_->provided_bytes = shared_bytes;
}
Adapter::~Adapter() { stop(); }
void Adapter::start() {
    if (!impl_->yield)
        throw std::runtime_error("stock service adapter needs a device-release callback");
    if (impl_->options.featurenn_graph.empty() && impl_->options.gazenet_graph.empty() &&
        impl_->options.gazenet_vector_graph.empty())
        throw std::runtime_error("stock service adapter needs at least one graph");
    if (impl_->provided_memory) {
        impl_->memory = static_cast<uint8_t *>(impl_->provided_memory);
        impl_->dma_len = impl_->provided_bytes ? impl_->provided_bytes : dma_bytes;
        impl_->check_header();
    } else {
        impl_->map_service();
        impl_->check_header();
    }
    impl_->open_streams();
    impl_->warm_up();
    impl_->thread = std::thread([this] { impl_->run(); });
}
void Adapter::stop() {
    impl_->close();
}
} // namespace hexecutor::xrservice
