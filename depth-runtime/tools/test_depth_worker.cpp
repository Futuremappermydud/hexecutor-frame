#include "depth_worker_protocol.hpp"
#include <array>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {
constexpr size_t pixels = 320 * 256;
void check(bool condition) {
    if (!condition)
        throw std::runtime_error("protocol assertion failed");
}
void write_all(int fd, const void *p, size_t n) {
    auto *d = static_cast<const char *>(p);
    while (n) {
        auto k = send(fd, d, n, 0);
        check(k > 0);
        d += k;
        n -= size_t(k);
    }
}
void read_all(int fd, void *p, size_t n) {
    auto *d = static_cast<char *>(p);
    while (n) {
        auto k = recv(fd, d, n, 0);
        check(k > 0);
        d += k;
        n -= size_t(k);
    }
}
class Fake final : public FrameSpatial::DepthInference {
    unsigned &closes_;
    bool closed_ = false;
    bool temporal_;

  public:
    Fake(unsigned &closes, bool temporal) : closes_(closes), temporal_(temporal) {}
    std::vector<hexecutor::Bytes> run(const std::vector<hexecutor::Buffer> &in) override {
        check(in.size() == 4 && in[0].bytes == pixels && in[2].bytes == 2 * pixels * 4 &&
              in[3].bytes == 128);
        check(static_cast<const uint8_t *>(in[0].data)[0] == 123);
        check(static_cast<const float *>(in[2].data)[0] == (temporal_ ? 5.f : 0.f));
        check(static_cast<const float *>(in[3].data)[0] == (temporal_ ? 2.f : 0.f));
        std::vector<hexecutor::Bytes> out = {hexecutor::Bytes(2 * pixels, 2),
                                             hexecutor::Bytes(2 * pixels * 4)};
        for (size_t i = 0; i < 2 * pixels; i++) {
            float v = i < pixels ? 3.f : 7.f;
            std::memcpy(out[1].data() + i * 4, &v, 4);
        }
        return out;
    }
    void close() override {
        if (!closed_) {
            ++closes_;
            closed_ = true;
        }
    }
};
void exercise(bool temporal, bool second, bool close_each) {
    int sockets[2];
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    unsigned opens = 0, closes = 0;
    std::exception_ptr failure;
    std::atomic<bool> stop{false};
    FrameSpatial::WorkerOptions options{sockets[1], temporal, second, close_each, 15};
    std::thread worker([&] {
        try {
            FrameSpatial::serve_depth(
                options,
                [&] {
                    ++opens;
                    return std::make_unique<Fake>(closes, temporal);
                },
                stop);
        } catch (...) {
            failure = std::current_exception();
        }
        ::close(sockets[1]);
    });
    uint32_t hello[4];
    read_all(sockets[0], hello, sizeof(hello));
    check(hello[0] == 0x36524451 && hello[3] == (second ? 4u : 3u));
    for (unsigned sequence = 1; sequence <= 2; sequence++) {
        uint32_t header[4] = {0x36504451, sequence, 256, 320};
        write_all(sockets[0], header, sizeof(header));
        uint8_t motion[64]{};
        write_all(sockets[0], motion, sizeof(motion));
        std::vector<uint8_t> pair(2 * pixels, 123);
        write_all(sockets[0], pair.data(), pair.size());
        if (temporal) {
            std::vector<float> h(2 * pixels, 5), t(32, 2);
            write_all(sockets[0], h.data(), h.size() * 4);
            write_all(sockets[0], t.data(), t.size() * 4);
        }
        uint32_t reply[4];
        read_all(sockets[0], reply, sizeof(reply));
        check(reply[0] == 0x364f4451 && reply[1] == sequence && reply[2] == pixels);
        std::vector<float> disparity(pixels);
        std::vector<uint8_t> uncertainty(pixels);
        read_all(sockets[0], disparity.data(), pixels * 4);
        read_all(sockets[0], uncertainty.data(), pixels);
        check(disparity.front() == 3 && disparity.back() == 3 && uncertainty.front() == 2);
        if (second) {
            read_all(sockets[0], disparity.data(), pixels * 4);
            check(disparity.front() == 7 && disparity.back() == 7);
        }
    }
    shutdown(sockets[0], SHUT_RDWR);
    ::close(sockets[0]);
    worker.join();
    if (failure)
        std::rethrow_exception(failure);
    check(opens == (close_each ? 2u : 1u) && closes == opens);
}
} // namespace
int main() {
    exercise(false, false, false);
    exercise(true, true, false);
    exercise(false, true, true);
    std::cout
        << "Native depth protocol v3/v4, temporal inputs, and session reuse/close checks passed\n";
}
