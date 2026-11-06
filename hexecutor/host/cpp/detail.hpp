#pragma once
#include "hexecutor/executor.hpp"
#include <array>
#include <atomic>
#include <cstring>
#include <stdexcept>

namespace hexecutor::detail {
inline void require(bool yes, const std::string &message) {
    if (!yes)
        throw std::runtime_error(message);
}
template <class T> T get(const void *base, size_t offset) {
    T value;
    std::memcpy(&value, static_cast<const uint8_t *>(base) + offset, sizeof(value));
    return value;
}
template <class T> void put(void *base, size_t offset, T value) {
    std::memcpy(static_cast<uint8_t *>(base) + offset, &value, sizeof(value));
}
constexpr uint32_t ctrl = 0xa91000, work = 0xb12000, arena_offset = 0x1800000;
constexpr uint32_t arena_bytes = 0x800000, dma_bytes = 0x2000000;
constexpr uint32_t code_base = 0x87606000, payload_base = 0x87610000;
constexpr uint32_t code_offset = code_base - 0x87000000, payload_offset = payload_base - 0x87000000;
constexpr uint32_t code_bytes = 4096, payload_bytes = 0x8000, exited = 0x54495845;
class Transport {
  public:
    Transport(const Artifacts &, const std::filesystem::path &payload, bool fourth_worker = false);
    ~Transport();
    uint8_t *arena() const;
    uint32_t sequence() const;
    void submit(const std::function<void(uint32_t)> &prepare);
    void invoke(const std::function<void(uint32_t)> &prepare, const std::function<void()> &collect);
    void stop();
    void retain(const std::string &reason);
    void close();
    bool has_run() const;
    bool worker_errors() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
void verify_chunk_map(const Artifacts &);
} // namespace hexecutor::detail
