// SPDX-License-Identifier: LGPL-3.0-or-later
#include "detail.hpp"
#include <iostream>
// Dumps the CDSP's 128 JTLB entries (read-only probe). Usage: hexecutor_jtlb_probe LOADER PROBE_ELF
static void fence() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: hexecutor_jtlb_probe LOADER PROBE_ELF\n";
        return 2;
    }
    try {
        using namespace hexecutor::detail;
        hexecutor::Artifacts a;
        a.loader = argv[1];
        a.chunk_probe = argv[2];
        a.timeout_seconds = 30;
        Transport probe(a, argv[2]);
        auto *arena = probe.arena();
        probe.invoke(
            [&](uint32_t seq) {
                std::memset(arena, 0, 4096);
                put<uint32_t>(arena, 8, seq);
                fence();
            },
            [&] {
                require(get<uint32_t>(arena, 2048) == 0x4254314a, "jtlb probe failed");
                for (unsigned i = 0; i < 128; i++) {
                    uint32_t lo = get<uint32_t>(arena, 8192 + i * 8), hi = get<uint32_t>(arena, 8192 + i * 8 + 4);
                    std::cout << i << ' ' << std::hex << hi << ' ' << lo << std::dec << (hi >> 31 ? " valid" : "") << '\n';
                }
            });
        probe.close();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "jtlb probe failed: " << e.what() << '\n';
        return 1;
    }
}
