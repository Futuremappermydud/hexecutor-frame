// SPDX-License-Identifier: LGPL-3.0-or-later
// HexecutorBackend: an ExecuTorch BackendInterface that runs a delegate blob
// (format v1, see blob.hpp) on the hexecutor tensor_f32 backend.
//
// Device path (default): init() opens one hexecutor::Session per delegate
// handle. Artifacts are explicit and required:
//   HEXECUTOR_LOADER, HEXECUTOR_ENGINE, HEXECUTOR_BOOTSTRAP, HEXECUTOR_CHUNK_PROBE
// Optional, same meaning as hexecutor_tensor_run:
//   HEXECUTOR_TENSOR_F32, HEXECUTOR_TENSOR_PRECISE, HEXECUTOR_MAX_WAKE_SECONDS
//
// Portable reference (explicit only): HEXECUTOR_ET_REFERENCE=1 runs the CPU
// reference from tensor_reference.cpp. There is no automatic fallback.
#include "hexecutor_backend.hpp"

#include "blob.hpp"
#include "hexecutor/executor.hpp"

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/log.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
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

// Owns one delegate: the parsed contract, the private temp files and the
// device session (absent in reference mode). Destruction closes the session
// and removes the temp directory.
struct Delegate {
    Blob blob;
    fs::path dir;
    hexecutor::Program program;
    bool reference = false;
    std::unique_ptr<hexecutor::Session> session;
    std::mutex mu; // Session::run is not reentrant.

    Delegate() = default;
    Delegate(const Delegate &) = delete;
    Delegate &operator=(const Delegate &) = delete;

    ~Delegate() {
        if (session) {
            try {
                session->close();
            } catch (const std::exception &e) {
                ET_LOG(Error, "hexecutor: session close failed: %s", e.what());
            }
            session.reset();
        }
        if (!dir.empty()) {
            std::error_code ec;
            fs::remove_all(dir, ec);
        }
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
        d->dir = buf.data();
        const fs::path image_path = d->dir / "program.hxfg";
        const fs::path constants_path = d->dir / "constants.bin";
        if (!write_file(image_path, d->blob.image) ||
            !write_file(constants_path, d->blob.constants)) {
            ET_LOG(Error, "hexecutor: cannot write temp program files");
            return Error::AccessFailed;
        }
        // The image and constants now live on disk; keep only the contract.
        std::vector<uint8_t>().swap(d->blob.image);
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
                spec.shape != runtime_shape(d->blob.inputs[i])) {
                ET_LOG(Error, "hexecutor: input %zu shape differs from blob contract", i);
                return Error::InvalidProgram;
            }
        }
        for (size_t i = 0; i < d->blob.outputs.size(); i++) {
            const auto &spec = d->program.outputs[i];
            if (spec.type != hexecutor::ScalarType::f32 ||
                spec.shape != runtime_shape(d->blob.outputs[i])) {
                ET_LOG(Error, "hexecutor: output %zu shape differs from blob contract", i);
                return Error::InvalidProgram;
            }
        }

        if (!d->reference) {
            hexecutor::Artifacts artifacts;
            std::string missing;
            if (!artifacts_from_env(&artifacts, &missing)) {
                ET_LOG(Error, "hexecutor: missing or invalid %s", missing.c_str());
                return Error::InvalidArgument;
            }
            try {
                d->session = std::make_unique<hexecutor::Session>(d->program, artifacts);
            } catch (const std::exception &e) {
                ET_LOG(Error, "hexecutor: session open failed: %s", e.what());
                return Error::DelegateInvalidCompatibility;
            }
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

        std::lock_guard<std::mutex> lock(d->mu);
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
            if (d->reference)
                outputs = tensor_reference(d->program, inputs);
            else
                outputs = d->session->run(inputs, 0);
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
