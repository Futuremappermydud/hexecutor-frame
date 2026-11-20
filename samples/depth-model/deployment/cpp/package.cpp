#include "depth_model/deployment.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace depth_model {
namespace fs = std::filesystem;
namespace {
fs::path configured(const char *key, const fs::path &fallback) {
    const char *value = std::getenv(key);
    return value && *value ? fs::path(value) : fallback;
}
} // namespace
Deployment load_deployment(const fs::path &package, const std::string &backend,
                           const std::string &model) {
    if (backend != "prepared" && backend != "prepared-qnn")
        throw std::invalid_argument("backend must be prepared or prepared-qnn");
    if (model != "primary" && model != "alternate")
        throw std::invalid_argument("unknown depth model");
    if (backend == "prepared-qnn" && model != "primary")
        throw std::invalid_argument("prepared-qnn supports primary only");
    auto root = fs::absolute(package);
    bool legacy = fs::is_directory(root / "frame");
    auto build =
        configured("FRAME_DEPTH_ARTIFACTS",
                   legacy ? root / "frame/build" : root.parent_path() / "hexecutor/native/build");
    auto programs =
        configured("FRAME_DEPTH_PROGRAMS",
                   legacy ? root / "frame/native_program/examples" : root / "deployment/programs");
    bool qnn = backend == "prepared-qnn";
    auto engine =
        build / (qnn ? "ab/claim_20261102i/eng-claim.elf"
                     : "qd_graph_dsp-scalar-dynamic-tcm-fallback-l2-blocked-glue-fast-spill-native-"
                       "1x1-native-3x3-direct-range-prepared-reference-vtcm-lease.elf");
    auto boot = build / (qnn ? "ab/claim_20261102i/boot-claim.elf"
                             : "qd_graph_boot-c0-preserve-vtcm-guarded-entry.elf");
    auto loader = build / "qd_loader.elf", image = build / ("resident-" + model + ".qdrx"),
         program = programs / (model + "-v2.nprg");
    hexecutor::verify_file(
        engine, qnn ? "678611d0d66eeeb28b01d76f403494cfe206babd2cd3b7fe4192417608629068"
                    : "c432a7140ac633d9e20366ac8d801de1b42acdd2bf21dc2fdde65d415e9ce13b");
    hexecutor::verify_file(
        boot, qnn ? "cc30661b9496c489fcf6aad3fd79ad6009ab320f76e69edc3bc0aa319e0f798e"
                  : "628e362de7563a2c3b0b5cf2a02cae49493b80931e47e52d652fedae449fa147");
    hexecutor::verify_file(loader,
                           "828ca0638750a6d8b69b3aaa536cb6c23508ce821b2333e270b455c321414572");
    hexecutor::verify_file(
        image, model == "primary"
                   ? "dc8fcddcf4876d11f7a717ec4f111e1d3d3a40d4f3163758b590234e0b50b92b"
                   : "54b2949bbb25655ee459c903432f67cacd26e66519854dcb9745a91777eb7298");
    hexecutor::verify_file(
        program, model == "primary"
                     ? "c6ae3022dbbdab04f474751bd987e9bfc0547337d5e9b95becb2e100eed86013"
                     : "21953104d4960e19dff0f61116dd787e2eb533ef1e4c8eeeedfba197787f69a4");
    using T = hexecutor::ScalarType;
    Deployment out;
    out.program = {
        image,
        root / "recovered" / model / "constants/constant_pool_5.bin",
        {{"x", T::u8, {320, 256}},
         {"y", T::u8, {320, 256}},
         {"hidden", T::f32, {1, 2, 320, 256}},
         {"t", T::f32, {1, 2, 4, 4}}},
        {{"uncertainty", T::u8, {1, 2, 320, 256}}, {"disparity", T::f32, {1, 2, 320, 256}}}};
    out.artifacts.loader = loader;
    out.artifacts.engine = engine;
    out.artifacts.bootstrap = boot;
    out.artifacts.chunk_probe =
        configured("FRAME_DEPTH_CHUNK_PROBE", build / "qd_chunk_map_probe.elf");
    out.artifacts.prepared_program = program;
    // Package checks include the private constant pool, without taking a DSP lease.
    auto header = hexecutor::read_file(image);
    if (header.size() < 256)
        throw std::runtime_error("truncated depth image");
    std::string constant_hash;
    constexpr char hex[] = "0123456789abcdef";
    for (size_t i = 64; i < 96; i++) {
        constant_hash += hex[header[i] >> 4];
        constant_hash += hex[header[i] & 15];
    }
    hexecutor::verify_file(out.program.constants, constant_hash);
    hexecutor::load_elf(hexecutor::read_file(out.artifacts.chunk_probe), 0x87610000, 0x8000);
    if (auto *seconds = std::getenv("FRAME_DEPTH_TIMEOUT_S")) {
        out.artifacts.timeout_seconds = std::stod(seconds);
        if (!(out.artifacts.timeout_seconds > 0 && out.artifacts.timeout_seconds <= 3600))
            throw std::invalid_argument("invalid DSP timeout");
    }
    return out;
}
} // namespace depth_model
