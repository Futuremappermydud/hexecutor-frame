// SPDX-License-Identifier: LGPL-3.0-or-later
// HexecutorBackend: an ExecuTorch BackendInterface that runs a delegate blob
// (format v1, see blob.hpp) on the hexecutor tensor_f32 backend.
//
// Device path (default): every delegate handle in the process shares one
// multi-program hexecutor::Session (the CDSP has one owner per process).
// Handles whose HXFG images are identical and stateless share one program
// slot; a new distinct program reopens the session with all slots. Artifacts
// are explicit and required:
//   HEXECUTOR_LOADER, HEXECUTOR_ENGINE, HEXECUTOR_BOOTSTRAP, HEXECUTOR_CHUNK_PROBE
// Optional, same meaning as hexecutor_tensor_run:
//   HEXECUTOR_TENSOR_F32, HEXECUTOR_TENSOR_PRECISE, HEXECUTOR_MAX_WAKE_SECONDS
//
// Portable reference (explicit only): HEXECUTOR_ET_REFERENCE=1 runs the CPU
// reference from tensor_reference.cpp. There is no automatic fallback.
#include "hexecutor_backend.hpp"

#include "blob.hpp"
#include "hexecutor/executor.hpp"
#include "hexecutor/broker.hpp"
#include "tensor/graph.h"

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/log.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

// Defined in hexecutor/host/cpp/tensor_reference.cpp.
std::vector<hexecutor::Bytes> tensor_reference(const hexecutor::Program &,
                                               const std::vector<hexecutor::Buffer> &);

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::ArrayRef;

namespace hexecutor_et {
namespace {

namespace fs = std::filesystem;

const char *env_or_null(const char *name) { return std::getenv(name); }

bool env_flag_is_one(const char *name) {
    const char *v = env_or_null(name);
    return v && std::strcmp(v, "1") == 0;
}

bool write_file(const fs::path &path, const std::vector<uint8_t> &bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        return false;
    if (!bytes.empty())
        f.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
    f.close();
    return bool(f);
}

// A private temp directory holding one program's image and constants,
// removed when its last owner (a delegate or a device slot) lets go.
struct TempDir {
    fs::path path;
    TempDir() = default;
    TempDir(const TempDir &) = delete;
    TempDir &operator=(const TempDir &) = delete;
    ~TempDir() {
        if (!path.empty()) {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    }
};

// True when the image declares a persistent (state) tensor. Such a program keeps
// state between runs, so two handles must not share its slot.
bool image_is_stateful(const std::vector<uint8_t> &image) {
    HxHeader h;
    if (image.size() < sizeof h)
        return true; // load_tensor_program rejects it anyway; do not share
    std::memcpy(&h, image.data(), sizeof h);
    const uint64_t end = uint64_t(h.tensor_offset) + uint64_t(h.tensors) * sizeof(HxTensor);
    if (end > image.size())
        return true;
    for (uint32_t i = 0; i < h.tensors; i++) {
        HxTensor t;
        std::memcpy(&t, image.data() + h.tensor_offset + size_t(i) * sizeof t, sizeof t);
        if (t.flags & HX_TENSOR_PERSISTENT)
            return true;
    }
    return false;
}

bool same_artifacts(const hexecutor::Artifacts &a, const hexecutor::Artifacts &b) {
    return a.loader == b.loader && a.engine == b.engine && a.bootstrap == b.bootstrap &&
           a.chunk_probe == b.chunk_probe && a.tensor_hmx == b.tensor_hmx &&
           a.tensor_fp16 == b.tensor_fp16 && a.max_wake_seconds == b.max_wake_seconds;
}

// The process-wide device: one Session over every loaded program. All runs and
// (re)opens are serialized by one mutex; the DSP runs one graph at a time anyway.
class Device {
  public:
    // Returns the slot for `program`, opening or reopening the session as needed.
    Result<size_t> acquire(const std::vector<uint8_t> &image, const hexecutor::Program &program,
                           const std::shared_ptr<TempDir> &files,
                           const hexecutor::Artifacts &artifacts) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!slots_.empty() && !same_artifacts(artifacts_, artifacts)) {
            ET_LOG(Error, "hexecutor: HEXECUTOR_* artifacts changed while a session is open");
            return Error::InvalidArgument;
        }
        const bool stateful = image_is_stateful(image);
        for (size_t i = 0; i < slots_.size(); i++) {
            Slot &s = slots_[i];
            if (s.image != image)
                continue;
            if (stateful || s.stateful) {
                ET_LOG(Error, "hexecutor: a stateful program can be loaded only once per process");
                return Error::NotSupported;
            }
            s.users++;
            return i;
        }
        if (slots_.size() >= HXFG_MAX_GRAPHS) {
            ET_LOG(Error, "hexecutor: more than %u distinct programs in one process",
                   unsigned(HXFG_MAX_GRAPHS));
            return Error::NotSupported;
        }
        slots_.push_back(Slot{image, program, files, stateful, 1});
        artifacts_ = artifacts;
        if (!reopen()) {
            slots_.pop_back();
            if (!slots_.empty() && !reopen())
                ET_LOG(Error, "hexecutor: could not restore the previous session");
            return Error::DelegateInvalidCompatibility;
        }
        return slots_.size() - 1;
    }

    std::vector<hexecutor::Bytes> run(size_t slot, const std::vector<hexecutor::Buffer> &inputs) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!session_)
            throw std::runtime_error("no open session");
        return session_->run(inputs, slot);
    }

    // Slots stay in the session until every handle is gone (removing one would
    // renumber the others), then the session closes and the files are freed.
    void release(size_t slot) {
        std::lock_guard<std::mutex> lock(mu_);
        if (slot >= slots_.size() || slots_[slot].users == 0)
            return;
        slots_[slot].users--;
        for (const Slot &s : slots_)
            if (s.users)
                return;
        close();
        slots_.clear();
    }

  private:
    struct Slot {
        std::vector<uint8_t> image;
        hexecutor::Program program;
        std::shared_ptr<TempDir> files; // keeps program.image/constants on disk for reopens
        bool stateful = false;
        size_t users = 0;
    };

    void close() {
        if (!session_)
            return;
        try {
            session_->close();
        } catch (const std::exception &e) {
            ET_LOG(Error, "hexecutor: session close failed: %s", e.what());
        }
        session_.reset();
    }

    bool reopen() {
        close();
        std::vector<hexecutor::Program> programs;
        for (const Slot &s : slots_)
            programs.push_back(s.program);
        try {
            session_ = std::make_unique<hexecutor::Session>(std::move(programs), artifacts_);
        } catch (const std::exception &e) {
            ET_LOG(Error, "hexecutor: session open failed: %s", e.what());
            return false;
        }
        return true;
    }

    std::mutex mu_;
    std::vector<Slot> slots_;
    hexecutor::Artifacts artifacts_;
    std::unique_ptr<hexecutor::Session> session_;
};

Device &device() {
    static Device *d = new Device; // never destroyed: delegates may outlive static destruction
    return *d;
}

// Owns one delegate: the parsed contract, the private temp files and either a
// slot of the shared device session or (reference mode) a CPU reference.
struct Delegate {
    Blob blob;
    std::shared_ptr<TempDir> files;
    hexecutor::Program program;
    bool reference = false;
    bool has_slot = false;
    size_t slot = 0;
    std::unique_ptr<hexecutor::broker::Client> broker;
    hexecutor::broker::Client::Stream stream;
    // Reference mode: CPU execution with a persistent state region kept between execute() calls.
    std::unique_ptr<hexecutor::TensorReference> ref;
    std::mutex mu; // TensorReference::run is not reentrant.

    Delegate() = default;
    Delegate(const Delegate &) = delete;
    Delegate &operator=(const Delegate &) = delete;

    ~Delegate() {
        if (broker) {
            try { broker->close(stream); } catch (const std::exception &) {}
        }
        if (has_slot)
            device().release(slot);
    }
};

// Shape a tensor must have for a contract: the PyTorch dims for layout 0, the
// runtime NHWC dims for layout 1.
std::vector<size_t> runtime_shape(const TensorDesc &t) {
    std::vector<size_t> shape;
    if (t.layout == kLayoutNchw)
        shape = {t.dims[0], t.dims[2], t.dims[3], t.dims[1]};
    else
        shape.assign(t.dims.begin(), t.dims.end());
    return shape;
}

// Shapes without their size-one dimensions. The program reports every tensor as [1, h, w, c], so a
// rank-2 [1, K] or rank-3 [1, L, C] contract is compared the way Session compares I/O.
std::vector<size_t> squeezed(const std::vector<size_t> &shape) {
    std::vector<size_t> out;
    for (size_t d : shape)
        if (d != 1)
            out.push_back(d);
    return out;
}

bool tensor_matches(const Tensor &t, const TensorDesc &desc) {
    if (t.scalar_type() != ScalarType::Float)
        return false;
    if (t.dim() != ssize_t(desc.dims.size()))
        return false;
    for (size_t i = 0; i < desc.dims.size(); i++)
        if (t.size(ssize_t(i)) != ssize_t(desc.dims[i]))
            return false;
    return true;
}

// Parses HEXECUTOR_* artifact variables. Nothing is defaulted.
bool artifacts_from_env(hexecutor::Artifacts *a, std::string *missing) {
    static const char *const required[] = {
        "HEXECUTOR_LOADER", "HEXECUTOR_ENGINE", "HEXECUTOR_BOOTSTRAP", "HEXECUTOR_CHUNK_PROBE"};
    for (const char *name : required) {
        if (!env_or_null(name)) {
            *missing = name;
            return false;
        }
    }
    a->loader = env_or_null("HEXECUTOR_LOADER");
    a->engine = env_or_null("HEXECUTOR_ENGINE");
    a->bootstrap = env_or_null("HEXECUTOR_BOOTSTRAP");
    a->chunk_probe = env_or_null("HEXECUTOR_CHUNK_PROBE");
    a->tensor_hmx = env_or_null("HEXECUTOR_TENSOR_PRECISE") == nullptr;
    a->tensor_fp16 = env_or_null("HEXECUTOR_TENSOR_F32") == nullptr;
    if (const char *v = env_or_null("HEXECUTOR_MAX_WAKE_SECONDS")) {
        try {
            a->max_wake_seconds = std::stod(v);
        } catch (const std::exception &) {
            *missing = "HEXECUTOR_MAX_WAKE_SECONDS (not a number)";
            return false;
        }
    }
    return true;
}

class HexecutorBackend final : public BackendInterface {
  public:
    bool is_available() const override { return true; }

    Result<DelegateHandle *> init(BackendInitContext &context, FreeableBuffer *processed,
                                  ArrayRef<CompileSpec> compile_specs) const override {
        (void)context;
        (void)compile_specs;
        if (!processed || !processed->data() || processed->size() == 0) {
            ET_LOG(Error, "hexecutor: empty delegate blob");
            return Error::InvalidArgument;
        }
        auto d = std::make_unique<Delegate>();
        std::string err;
        if (!parse_blob(static_cast<const uint8_t *>(processed->data()), processed->size(),
                        &d->blob, &err)) {
            ET_LOG(Error, "hexecutor: invalid delegate blob: %s", err.c_str());
            return Error::InvalidProgram;
        }
        d->reference = env_flag_is_one("HEXECUTOR_ET_REFERENCE");

        // Private temp directory (mkdtemp creates it with mode 0700).
        std::string tmpl = (fs::temp_directory_path() / "hexecutor-et-XXXXXX").string();
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        if (!mkdtemp(buf.data())) {
            ET_LOG(Error, "hexecutor: mkdtemp failed: %s", std::strerror(errno));
            return Error::AccessFailed;
        }
        d->files = std::make_shared<TempDir>();
        d->files->path = buf.data();
        const fs::path image_path = d->files->path / "program.hxfg";
        const fs::path constants_path = d->files->path / "constants.bin";
        if (!write_file(image_path, d->blob.image) ||
            !write_file(constants_path, d->blob.constants)) {
            ET_LOG(Error, "hexecutor: cannot write temp program files");
            return Error::AccessFailed;
        }
        // The image and constants now live on disk; keep only the contract (and the
        // image until the device has compared it with the loaded programs).
        std::vector<uint8_t> image;
        image.swap(d->blob.image);
        std::vector<uint8_t>().swap(d->blob.constants);

        try {
            d->program = hexecutor::load_tensor_program(image_path, constants_path);
        } catch (const std::exception &e) {
            ET_LOG(Error, "hexecutor: program rejected: %s", e.what());
            return Error::InvalidProgram;
        }
        if (d->program.backend != hexecutor::Backend::tensor_f32) {
            ET_LOG(Error, "hexecutor: program is not a tensor_f32 image");
            return Error::InvalidProgram;
        }
        if (d->program.inputs.size() != d->blob.inputs.size() ||
            d->program.outputs.size() != d->blob.outputs.size()) {
            ET_LOG(Error, "hexecutor: program I/O count differs from blob contract");
            return Error::InvalidProgram;
        }
        for (size_t i = 0; i < d->blob.inputs.size(); i++) {
            const auto &spec = d->program.inputs[i];
            if (spec.type != hexecutor::ScalarType::f32 ||
                squeezed(spec.shape) != squeezed(runtime_shape(d->blob.inputs[i]))) {
                ET_LOG(Error, "hexecutor: input %zu shape differs from blob contract", i);
                return Error::InvalidProgram;
            }
        }
        for (size_t i = 0; i < d->blob.outputs.size(); i++) {
            const auto &spec = d->program.outputs[i];
            if (spec.type != hexecutor::ScalarType::f32 ||
                squeezed(spec.shape) != squeezed(runtime_shape(d->blob.outputs[i]))) {
                ET_LOG(Error, "hexecutor: output %zu shape differs from blob contract", i);
                return Error::InvalidProgram;
            }
        }

        if (d->reference) {
            try {
                d->ref = std::make_unique<hexecutor::TensorReference>(d->program);
            } catch (const std::exception &e) {
                ET_LOG(Error, "hexecutor: reference open failed: %s", e.what());
                return Error::DelegateInvalidCompatibility;
            }
        } else {
            const char *socket_env = env_or_null("HEXECUTORD_SOCKET");
            fs::path socket = socket_env && *socket_env ? socket_env
                                                        : "/run/hexecutor/hexecutord.sock";
            if ((socket_env && *socket_env) || fs::exists(socket)) {
                try {
                    d->broker = std::make_unique<hexecutor::broker::Client>(socket);
                    std::ifstream image_in(image_path, std::ios::binary);
                    std::ifstream constants_in(constants_path, std::ios::binary);
                    if (!image_in || !constants_in)
                        throw std::runtime_error("cannot open delegate graph for broker");
                    hexecutor::Bytes image_bytes((std::istreambuf_iterator<char>(image_in)), {});
                    hexecutor::Bytes constant_bytes((std::istreambuf_iterator<char>(constants_in)), {});
                    if (image_in.bad() || constants_in.bad())
                        throw std::runtime_error("cannot read delegate graph for broker");
                    std::string name = d->broker->register_graph(image_bytes, constant_bytes);
                    hexecutor::broker::StreamSpec spec;
                    spec.graph = name;
                    spec.latest_only = false;
                    spec.deadline_ms = 60000;
                    d->stream = d->broker->open(spec);
                } catch (const std::exception &e) {
                    ET_LOG(Error, "hexecutor: broker registration failed: %s", e.what());
                    return Error::DelegateInvalidCompatibility;
                }
                return static_cast<DelegateHandle *>(d.release());
            }
            hexecutor::Artifacts artifacts;
            std::string missing;
            if (!artifacts_from_env(&artifacts, &missing)) {
                ET_LOG(Error, "hexecutor: missing or invalid %s", missing.c_str());
                return Error::InvalidArgument;
            }
            Result<size_t> slot = device().acquire(image, d->program, d->files, artifacts);
            if (!slot.ok())
                return slot.error();
            d->slot = slot.get();
            d->has_slot = true;
        }
        return static_cast<DelegateHandle *>(d.release());
    }

    Error execute(BackendExecutionContext &context, DelegateHandle *handle,
                  Span<EValue *> args) const override {
        (void)context;
        if (!handle)
            return Error::InvalidArgument;
        auto *d = static_cast<Delegate *>(handle);
        const size_t n_in = d->blob.inputs.size();
        const size_t n_out = d->blob.outputs.size();
        if (args.size() != n_in + n_out) {
            ET_LOG(Error, "hexecutor: expected %zu args, got %zu", n_in + n_out, size_t(args.size()));
            return Error::InvalidArgument;
        }

        // Validate every tensor before any device work.
        for (size_t i = 0; i < n_in; i++) {
            if (!args[i]->isTensor() || !tensor_matches(args[i]->toTensor(), d->blob.inputs[i])) {
                ET_LOG(Error, "hexecutor: input %zu is not a float32 tensor of the contract shape", i);
                return Error::InvalidArgument;
            }
        }
        for (size_t j = 0; j < n_out; j++) {
            const size_t k = n_in + j;
            if (!args[k]->isTensor() || !tensor_matches(args[k]->toTensor(), d->blob.outputs[j])) {
                ET_LOG(Error, "hexecutor: output %zu is not a float32 tensor of the contract shape", j);
                return Error::InvalidArgument;
            }
        }

        std::vector<std::vector<float>> packed(n_in);
        std::vector<hexecutor::Buffer> inputs;
        inputs.reserve(n_in);
        for (size_t i = 0; i < n_in; i++) {
            const auto &desc = d->blob.inputs[i];
            Tensor &t = args[i]->toTensor();
            const float *src = t.const_data_ptr<float>();
            if (desc.layout == kLayoutNchw) {
                packed[i].resize(size_t(desc.elements));
                nchw_to_nhwc(src, packed[i].data(), desc.dims.data());
                inputs.push_back({packed[i].data(), size_t(desc.bytes)});
            } else {
                inputs.push_back({src, size_t(desc.bytes)});
            }
        }

        std::vector<hexecutor::Bytes> outputs;
        try {
            if (d->reference) {
                std::lock_guard<std::mutex> lock(d->mu);
                outputs = d->ref->run(inputs);
            } else if (d->broker) {
                std::lock_guard<std::mutex> lock(d->mu);
                outputs = d->broker->run(d->stream, inputs);
            } else {
                outputs = device().run(d->slot, inputs);
            }
        } catch (const std::exception &e) {
            ET_LOG(Error, "hexecutor: run failed: %s", e.what());
            return Error::Internal;
        }
        if (outputs.size() != n_out) {
            ET_LOG(Error, "hexecutor: run returned %zu outputs, expected %zu", outputs.size(), n_out);
            return Error::Internal;
        }

        for (size_t j = 0; j < n_out; j++) {
            const auto &desc = d->blob.outputs[j];
            if (outputs[j].size() != size_t(desc.bytes)) {
                ET_LOG(Error, "hexecutor: output %zu has %zu bytes, expected %zu", j,
                       outputs[j].size(), size_t(desc.bytes));
                return Error::Internal;
            }
            Tensor &t = args[n_in + j]->toTensor();
            float *dst = t.mutable_data_ptr<float>();
            const float *src = reinterpret_cast<const float *>(outputs[j].data());
            if (desc.layout == kLayoutNchw)
                nhwc_to_nchw(src, dst, desc.dims.data());
            else
                std::memcpy(dst, src, size_t(desc.bytes));
        }
        return Error::Ok;
    }

    void destroy(DelegateHandle *handle) const override {
        delete static_cast<Delegate *>(handle);
    }
};

HexecutorBackend g_backend;

struct Registrar {
    Registrar() {
        const Error e = executorch::runtime::register_backend(Backend{"HexecutorBackend", &g_backend});
        if (e != Error::Ok)
            ET_LOG(Error, "hexecutor: backend registration failed (%d)", int(e));
    }
};
Registrar g_registrar;

} // namespace

Error ensure_registered() {
    // Calling this from another translation unit forces this object file to be
    // linked from a static archive; the registrar above then has run.
    return executorch::runtime::get_backend_class("HexecutorBackend") ? Error::Ok
                                                                       : Error::NotFound;
}

} // namespace hexecutor_et
