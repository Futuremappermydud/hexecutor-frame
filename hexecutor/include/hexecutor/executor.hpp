#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hexecutor {
using Bytes = std::vector<uint8_t>;
enum class ScalarType { u8, i8, u16, i16, i32, f32 };
struct TensorSpec {
    std::string name;
    ScalarType type;
    std::vector<size_t> shape;
    size_t bytes() const;
};
struct Buffer {
    const void *data;
    size_t bytes;
};
enum class Backend { qdrx_v1, tensor_f32, qnn16_f16 };
struct Program {
    std::filesystem::path image, constants;
    std::vector<TensorSpec> inputs, outputs;
    Backend backend = Backend::qdrx_v1;
    void validate() const;
};
struct Artifacts {
    std::filesystem::path loader, engine, bootstrap, chunk_probe;
    std::filesystem::path firmware = "/opt/steamvr/drivers/cv/bin/linuxarm64/cdsp_sm8650.mbn";
    std::string firmware_sha256 =
        "4f9fd567eb582c24f1f960a6525fafb0bd75733223ffa7ce8d6541cfd2a62b1c";
    std::filesystem::path prepared_program;
    double timeout_seconds = 120;
    // Longest time one firmware wake may stay held before the transport lets
    // the workers exit and wakes them again. Sessions up to 2.45 s have always
    // exited cleanly; 17.9 s and 46.5 s holds left the firmware at (2,0,8)
    // until reboot. Values above 2 s are for threshold experiments only.
    double max_wake_seconds = 1.0;
    bool tensor_hmx = true;
    bool tensor_fp16 = true; // Persistent blocked intermediates when the graph is supported.
    // Only the stock Gazenet broker adapter may keep the eye client running.
    bool keep_eye_running = false;
    // Memory reservation for CDSP: 0 = automatic based on workload requirements (min 128 MiB).
    size_t cdsp_memory_mb = 0;
    // Configurable persistent state region (0 = auto-allocated based on graph state tensors)
    size_t persistent_state_bytes = 0;
    uint32_t persistent_state_offset = 0;
};
// Read and preflight an HXFG image; the external I/O is compact float32.
Program load_tensor_program(const std::filesystem::path &image,
                            const std::filesystem::path &constants);
// Read and preflight a Q16P image (a lowered QNN FP16 run list) and its
// constant pool; external I/O is compact float32 NHWC. Runs on the tensor
// bootstrap with the qnn16 engine (Artifacts::engine).
Program load_qnn16_program(const std::filesystem::path &image,
                           const std::filesystem::path &constants);
Bytes read_file(const std::filesystem::path &path);
std::string sha256(const Bytes &bytes);
Bytes load_elf(const Bytes &elf, uint32_t base, size_t capacity);
void verify_file(const std::filesystem::path &path, const std::string &expected);

// QDRX retains its four-input/two-output ABI. The tensor backend supports
// multiple graphs under one device lease, with graph-specific I/O contracts.
struct ExecutionProfile {
    uint64_t cycles = 0;
    uint64_t timer_ticks = 0; // Frame architectural timer: 19.2 MHz.
    uint32_t matrix_ops = 0;
    bool blocked_fp16 = false;
    std::vector<uint64_t> op_cycles;
};
class Session {
  public:
    Session(Program program, Artifacts artifacts);
    Session(std::vector<Program> programs, Artifacts artifacts);
    ~Session();
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    std::vector<Bytes> run(const std::vector<Buffer> &inputs, size_t program_index = 0);
    const ExecutionProfile &profile() const;
    void reset_state();
    size_t state_bytes() const;
    void close();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace hexecutor
