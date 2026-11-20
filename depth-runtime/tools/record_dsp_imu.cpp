// Passive raw head-IMU recorder for Steam Frame.
//
// The head IMU is sampled by the aDSP firmware and written into a ring in aDSP shared memory.
// dsp_service (a root process in the SteamVR cv driver) hands that memory to clients over the
// abstract unix socket "\0dsp_service". This tool connects as one more client and reads the ring
// the same way driver_cv.so's IMU reader does, without writing to shared memory.
//
// Layout (reverse engineered from driver_cv.so / dsp_service, protocol version 22, 2026-11-16):
//   handshake: connect; recvmsg -> 4 bytes "ADSP" + SCM_RIGHTS fd(s), first fd = aDSP memory;
//              send u32 request mask (bit0 = IMU streaming; this tool sends 0 by default so the
//              DSP's IMU client count is left alone and an existing stream is only observed).
//   mapping:   mmap(fd, 16 MiB, offset 0x80000000) is the uncached alias of the aDSP memory.
//              +0x210000: u64 magic 0x1614df651d03f541, u64 version 22.
//              +0x211000: IMU block; +0x108 u32 monotonic write index; +0x140 2048 x 64-byte slots.
//   slot:      +0x00 f32[3], +0x0c f32[3], +0x18 u64 time, +0x20 f32, +0x24 u8 valid, +0x28 u64.
//              Field meaning and units are still to be confirmed from recordings.
//
// Output file: 64-byte header dump of the IMU block's first 0x140 bytes is written to OUT.hdr; OUT
// gets one 80-byte record per slot: u64 CLOCK_MONOTONIC ns at copy, u32 ring index, u32 flags
// (bit0 = overwritten during copy), then the raw 64-byte slot.
//
// Usage: record_dsp_imu OUT.bin DURATION_MS [--request-imu]
// Build: c++ -O2 -std=c++17 -Wall -Wextra -o record_dsp_imu record_dsp_imu.cpp
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kMagic = 0x1614df651d03f541ull;
constexpr uint64_t kVersion = 22;
constexpr size_t kMapSize = 16u << 20;
constexpr off_t kUncachedOffset = 0x80000000ll;
constexpr size_t kMagicOffset = 0x210000;
constexpr size_t kImuOffset = 0x211000;
constexpr size_t kWriteIndexOffset = 0x108;
constexpr size_t kSlotsOffset = 0x140;
constexpr uint32_t kSlots = 2048;
constexpr size_t kSlotSize = 64;

volatile sig_atomic_t running = 1;
void stopSignal(int) { running = 0; }

int64_t nowNs(clockid_t c) {
  timespec t{};
  clock_gettime(c, &t);
  return int64_t(t.tv_sec) * 1000000000ll + t.tv_nsec;
}

uint32_t loadIndex(const uint8_t *imu) {
  return __atomic_load_n(reinterpret_cast<const uint32_t *>(imu + kWriteIndexOffset), __ATOMIC_ACQUIRE);
}

struct Record {
  int64_t monoNs;
  uint32_t index;
  uint32_t flags;
  uint8_t slot[kSlotSize];
};
static_assert(sizeof(Record) == 80, "record layout");

int connectService(uint32_t mask, int *memFd) {
  int s = socket(AF_UNIX, SOCK_STREAM, 0);
  if (s < 0) { perror("socket"); return -1; }
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  static const char name[] = "\0dsp_service";  // abstract name, no trailing NUL
  memcpy(addr.sun_path, name, sizeof(name) - 1);
  socklen_t len = socklen_t(offsetof(sockaddr_un, sun_path) + sizeof(name) - 1);
  if (connect(s, reinterpret_cast<sockaddr *>(&addr), len) < 0) { perror("connect dsp_service"); close(s); return -1; }
  timeval tv{3, 0};
  setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  char data[256] = {};
  alignas(cmsghdr) char control[CMSG_SPACE(2 * sizeof(int))] = {};
  iovec iov{data, sizeof(data)};
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);
  ssize_t n = recvmsg(s, &msg, 0);
  if (n < 4 || memcmp(data, "ADSP", 4) != 0) { fprintf(stderr, "unexpected handshake (%zd bytes)\n", n); close(s); return -1; }
  cmsghdr *c = CMSG_FIRSTHDR(&msg);
  if (!c || c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS) { fprintf(stderr, "no fd in handshake\n"); close(s); return -1; }
  int fds[2] = {-1, -1};
  size_t nfds = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
  memcpy(fds, CMSG_DATA(c), std::min<size_t>(nfds, 2) * sizeof(int));
  for (size_t i = 1; i < nfds && i < 2; i++) close(fds[i]);  // second fd is cDSP memory, unused
  *memFd = fds[0];
  if (send(s, &mask, sizeof(mask), MSG_NOSIGNAL) != ssize_t(sizeof(mask))) { perror("send mask"); close(s); return -1; }
  return s;
}

void drainSocket(int s) {
  char buf[256];
  while (recv(s, buf, sizeof(buf), MSG_DONTWAIT) > 0) {}
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: record_dsp_imu OUT.bin DURATION_MS [--request-imu]\n"); return 2; }
  const long long durationMs = atoll(argv[2]);
  const bool requestImu = argc > 3 && strcmp(argv[3], "--request-imu") == 0;
  std::signal(SIGINT, stopSignal);
  std::signal(SIGTERM, stopSignal);

  int memFd = -1;
  int sock = connectService(requestImu ? 1u : 0u, &memFd);
  if (sock < 0) return 1;
  void *map = mmap(nullptr, kMapSize, PROT_READ, MAP_SHARED, memFd, kUncachedOffset);
  if (map == MAP_FAILED) { perror("mmap aDSP memory"); return 1; }
  const uint8_t *base = static_cast<const uint8_t *>(map);
  uint64_t magic = 0, version = 0;
  memcpy(&magic, base + kMagicOffset, 8);
  memcpy(&version, base + kMagicOffset + 8, 8);
  if (magic != kMagic || version != kVersion) {
    fprintf(stderr, "aDSP layout mismatch: magic %016llx version %llu (expected %016llx / %llu)\n",
            (unsigned long long)magic, (unsigned long long)version, (unsigned long long)kMagic, (unsigned long long)kVersion);
    return 1;
  }
  const uint8_t *imu = base + kImuOffset;

  std::string hdrPath = std::string(argv[1]) + ".hdr";
  if (FILE *h = fopen(hdrPath.c_str(), "wb")) { fwrite(imu, 1, kSlotsOffset, h); fclose(h); }
  FILE *out = fopen(argv[1], "wb");
  if (!out) { perror("open output"); return 1; }

  uint32_t cursor = loadIndex(imu);
  const int64_t end = nowNs(CLOCK_MONOTONIC) + durationMs * 1000000ll;
  std::vector<Record> batch(kSlots);
  uint64_t total = 0, overwritten = 0, skipped = 0, invalid = 0;
  double sumA[3] = {}, sumB[3] = {};
  uint64_t firstT = 0, lastT = 0;
  while (running && (durationMs == 0 || nowNs(CLOCK_MONOTONIC) < end)) {
    drainSocket(sock);
    uint32_t write = loadIndex(imu);
    uint32_t avail = write - cursor;
    if (avail >= kSlots) {  // fell behind: same recovery as the driver
      skipped += avail - (kSlots - 1);
      cursor = write - (kSlots - 1);
      avail = kSlots - 1;
    }
    if (avail == 0) { usleep(2000); continue; }
    const int64_t mono = nowNs(CLOCK_MONOTONIC);
    for (uint32_t i = 0; i < avail; i++) {
      Record &r = batch[i];
      r.monoNs = mono;
      r.index = cursor + i;
      r.flags = 0;
      memcpy(r.slot, imu + kSlotsOffset + size_t((cursor + i) & (kSlots - 1)) * kSlotSize, kSlotSize);
    }
    // Slots older than (newest - 2047) may have been rewritten while copying.
    uint32_t after = loadIndex(imu);
    for (uint32_t i = 0; i < avail; i++)
      if (after - batch[i].index >= kSlots) { batch[i].flags |= 1; overwritten++; }
    fwrite(batch.data(), sizeof(Record), avail, out);
    for (uint32_t i = 0; i < avail; i++) {
      const Record &r = batch[i];
      if (!r.slot[0x24] || (r.flags & 1)) { invalid++; continue; }
      float a[3], b[3];
      uint64_t t;
      memcpy(a, r.slot, 12);
      memcpy(b, r.slot + 0x0c, 12);
      memcpy(&t, r.slot + 0x18, 8);
      for (int k = 0; k < 3; k++) { sumA[k] += a[k]; sumB[k] += b[k]; }
      if (!firstT) firstT = t;
      lastT = t;
    }
    total += avail;
    cursor += avail;
    usleep(2000);
  }
  fclose(out);
  const double valid = double(total - invalid);
  printf("%llu slots (%llu invalid/overwritten, %llu skipped by overrun), %s\n",
         (unsigned long long)total, (unsigned long long)(invalid), (unsigned long long)skipped, argv[1]);
  if (valid > 0) {
    printf("mean f32@0x00 = %.4f %.4f %.4f  |.| = %.4f\n", sumA[0] / valid, sumA[1] / valid, sumA[2] / valid,
           std::sqrt(sumA[0] * sumA[0] + sumA[1] * sumA[1] + sumA[2] * sumA[2]) / valid);
    printf("mean f32@0x0c = %.4f %.4f %.4f  |.| = %.4f\n", sumB[0] / valid, sumB[1] / valid, sumB[2] / valid,
           std::sqrt(sumB[0] * sumB[0] + sumB[1] * sumB[1] + sumB[2] * sumB[2]) / valid);
    printf("time@0x18 first %llu last %llu (span %llu ticks over %.0f samples)\n", (unsigned long long)firstT,
           (unsigned long long)lastT, (unsigned long long)(lastT - firstT), valid);
  }
  (void)overwritten;
  munmap(map, kMapSize);
  close(memFd);
  close(sock);
  return 0;
}
