// SPDX-License-Identifier: LGPL-3.0-or-later
// Opt-in hook for the stripped 2026-11-03 Frame XRService binary. The stock
// function takes CDSP ownership before publishing job 8, so a mailbox poller
// cannot redirect it while Hexecutor owns the device.
#include "hexecutor/xrservice_broker_shim.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__linux__) && defined(__aarch64__)
#include <link.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {
constexpr uintptr_t dispatch_offset = 0x2db8630;
constexpr uint8_t expected_prologue[16] = {
    0xff, 0x03, 0x03, 0xd1, 0xeb, 0x2b, 0x04, 0x6d,
    0xe9, 0x23, 0x05, 0x6d, 0xfd, 0x7b, 0x06, 0xa9};
constexpr uint8_t expected_mailbox_code[16] = {
    0x09, 0x00, 0x84, 0x52, 0x49, 0x2a, 0xa0, 0x72,
    0x08, 0x01, 0x09, 0x8b, 0x68, 0x06, 0x00, 0xf9};
constexpr size_t output_record_bytes = 0x450;
constexpr size_t head_floats = 69;

extern "C" int hooked_featurenn_dispatch(void *, uint32_t controllers, uint32_t eyes,
                                           const void *input, uint32_t width,
                                           uint32_t height, void *output) {
    if (eyes != 2 || controllers == 0 || controllers > 2)
        return 2;
    int result = broker_shim_dispatch_stereofeaturenn(
        input, output, controllers * eyes, width, height);
    if (result != 0)
        return 1;
    // The stock dispatch applies expf after copying the raw head from DMA.
    // The public shim returns that raw head, so reproduce the final step here.
    auto *records = static_cast<uint8_t *>(output);
    for (uint32_t view = 0; view < controllers * eyes; ++view) {
        auto *uncertainty = reinterpret_cast<float *>(
            records + size_t(view) * output_record_bytes + head_floats * sizeof(float));
        for (size_t i = 0; i < head_floats; ++i)
            uncertainty[i] = std::exp(uncertainty[i]);
    }
    return 0;
}

int install_hook(dl_phdr_info *info, size_t, void *) {
    if (info->dlpi_name && *info->dlpi_name)
        return 0;
    char executable[256]{};
    auto length = ::readlink("/proc/self/exe", executable, sizeof executable - 1);
    if (length < 0 || std::strcmp(executable,
            "/opt/steamvr/drivers/cv/bin/linuxarm64/XRService") != 0)
        return 1;
    auto *target = reinterpret_cast<uint8_t *>(info->dlpi_addr + dispatch_offset);
    if (std::memcmp(target, expected_prologue, sizeof expected_prologue) != 0 ||
        std::memcmp(reinterpret_cast<void *>(info->dlpi_addr + 0x2db8a9c),
                    expected_mailbox_code, sizeof expected_mailbox_code) != 0 ||
        std::memcmp(reinterpret_cast<void *>(info->dlpi_addr + 0xaa814f),
                    "dispatchStereoFeatureNNOnCDSP", 29) != 0) {
        std::fprintf(stderr, "[xrservice_broker_shim] XRService dispatch signature mismatch; hook disabled\n");
        return 1;
    }
    const auto page_size = static_cast<uintptr_t>(::sysconf(_SC_PAGESIZE));
    const auto page = reinterpret_cast<uintptr_t>(target) & ~(page_size - 1);
    if (::mprotect(reinterpret_cast<void *>(page), page_size, PROT_READ | PROT_WRITE | PROT_EXEC)) {
        std::perror("[xrservice_broker_shim] mprotect dispatch");
        return 1;
    }
    const uint32_t jump[] = {0x58000050u, 0xd61f0200u}; // ldr x16, #8; br x16
    std::memcpy(target, jump, sizeof jump);
    const auto destination = reinterpret_cast<uintptr_t>(&hooked_featurenn_dispatch);
    std::memcpy(target + sizeof jump, &destination, sizeof destination);
    __builtin___clear_cache(reinterpret_cast<char *>(target),
                            reinterpret_cast<char *>(target + 16));
    ::mprotect(reinterpret_cast<void *>(page), page_size, PROT_READ | PROT_EXEC);
    std::fprintf(stderr, "[xrservice_broker_shim] installed internal FeatureNN dispatch hook\n");
    return 1;
}

__attribute__((constructor)) void install_if_enabled() {
    const char *enabled = std::getenv("HEXECUTOR_XRSERVICE_INLINE_HOOK");
    if (enabled && std::strcmp(enabled, "1") == 0)
        ::dl_iterate_phdr(install_hook, nullptr);
}
} // namespace
#endif
