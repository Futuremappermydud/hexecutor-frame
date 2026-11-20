#include "depth_worker_protocol.hpp"
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>

namespace FrameSpatial {
namespace {
constexpr size_t pixels = 320 * 256;
using Clock = std::chrono::steady_clock;
void send_all(int fd, const void *buffer, size_t size) {
    auto *data = static_cast<const uint8_t *>(buffer);
    while (size) {
        auto n = send(fd, data, size, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("depth reply disconnected");
        data += n;
        size -= size_t(n);
    }
}
bool receive(int fd, void *buffer, size_t size, const std::atomic<bool> &stop, bool eof = false) {
    auto *data = static_cast<uint8_t *>(buffer);
    size_t offset = 0;
    while (offset < size) {
        if (stop)
            return false;
        pollfd p{fd, POLLIN, 0};
        auto ready = poll(&p, 1, 1000);
        if (ready < 0 && errno == EINTR)
            continue;
        if (ready == 0)
            continue;
        if (ready < 0)
            throw std::runtime_error("depth request poll failed");
        auto n = recv(fd, data + offset, size - offset, 0);
        if (n < 0 && errno == EINTR)
            continue;
        if (n == 0 && offset == 0 && eof)
            return false;
        if (n <= 0)
            throw std::runtime_error("incomplete depth request");
        offset += size_t(n);
    }
    return true;
}
} // namespace
void serve_depth(const WorkerOptions &options, const DepthFactory &factory,
                 const std::atomic<bool> &stop) {
    if (options.fd < 0 || !std::isfinite(options.idleCloseSeconds) ||
        options.idleCloseSeconds <= 0 || options.idleCloseSeconds > 30)
        throw std::invalid_argument("invalid worker socket or idle-close interval");
    auto session = factory();
    uint32_t hello[4] = {0x36524451, 256, 320, options.secondDisparity ? 4u : 3u};
    send_all(options.fd, hello, sizeof(hello));
    std::vector<uint8_t> images(2 * pixels);
    std::vector<float> hidden(2 * pixels), transforms(32);
    auto last = Clock::now();
    try {
        while (!stop) {
            pollfd p{options.fd, POLLIN, 0};
            auto ready = poll(&p, 1, 1000);
            if (ready < 0 && errno == EINTR)
                continue;
            if (ready < 0)
                throw std::runtime_error("depth request poll failed");
            if (!ready) {
                if (session && std::chrono::duration<double>(Clock::now() - last).count() >=
                                   options.idleCloseSeconds) {
                    session->close();
                    session.reset();
                }
                continue;
            }
            uint32_t header[4];
            if (!receive(options.fd, header, sizeof(header), stop, true))
                break;
            if (header[0] != 0x36504451 || header[2] != 256 || header[3] != 320)
                throw std::runtime_error("invalid depth request header");
            uint8_t motion[64];
            if (!receive(options.fd, motion, sizeof(motion), stop))
                break;
            if (!receive(options.fd, images.data(), images.size(), stop))
                break;
            if (options.hostTemporal) {
                if (!receive(options.fd, hidden.data(), hidden.size() * 4, stop) ||
                    !receive(options.fd, transforms.data(), transforms.size() * 4, stop))
                    break;
            }
            auto begin = Clock::now();
            if (!session)
                session = factory();
            auto output = session->run({{images.data(), pixels},
                                        {images.data() + pixels, pixels},
                                        {hidden.data(), hidden.size() * 4},
                                        {transforms.data(), transforms.size() * 4}});
            if (output.size() != 2 || output[0].size() != 2 * pixels ||
                output[1].size() != 2 * pixels * 4)
                throw std::runtime_error("depth output shape mismatch");
            struct Reply {
                uint32_t magic, sequence, count;
                float ms;
            };
            static_assert(sizeof(Reply) == 16);
            Reply reply{
                0x364f4451, header[1], uint32_t(pixels),
                float(std::chrono::duration<double, std::milli>(Clock::now() - begin).count())};
            send_all(options.fd, &reply, sizeof(reply));
            send_all(options.fd, output[1].data(), pixels * 4);
            send_all(options.fd, output[0].data(), pixels);
            if (options.secondDisparity)
                send_all(options.fd, output[1].data() + pixels * 4, pixels * 4);
            last = Clock::now();
            if (options.closeAfterReply) {
                session->close();
                session.reset();
            }
        }
    } catch (...) {
        if (session) {
            try {
                session->close();
            } catch (...) {
            }
        }
        throw;
    }
    if (session)
        session->close();
}
} // namespace FrameSpatial
