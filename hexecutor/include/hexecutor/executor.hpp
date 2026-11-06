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
struct Program {
    std::filesystem::path image, constants;
    std::vector<TensorSpec> inputs, outputs;
    void validate() const;
};
struct Artifacts {
    std::filesystem::path loader, engine, bootstrap, chunk_probe;
    std::filesystem::path firmware = "/opt/steamvr/drivers/cv/bin/linuxarm64/cdsp_sm8650.mbn";
    std::string firmware_sha256 =
        "4f9fd567eb582c24f1f960a6525fafb0bd75733223ffa7ce8d6541cfd2a62b1c";
    std::filesystem::path prepared_program;
    double timeout_seconds = 120;
};
Bytes read_file(const std::filesystem::path &path);
std::string sha256(const Bytes &bytes);
Bytes load_elf(const Bytes &elf, uint32_t base, size_t capacity);
void verify_file(const std::filesystem::path &path, const std::string &expected);

// The qdrx-v1 backend has a fixed four-input/two-output wire ABI. Names,
// dimensions, types, artifact identities, and model semantics are caller-owned.
class Session {
  public:
    Session(Program program, Artifacts artifacts);
    ~Session();
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    std::vector<Bytes> run(const std::vector<Buffer> &inputs);
    void close();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace hexecutor
