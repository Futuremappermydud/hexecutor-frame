/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Offline ABI probe for QAIRT's pure integer stride1 cfg builder. */
#include <stdint.h>
#include <stdio.h>
#include <hexagon_standalone.h>

extern void qd_set_hmx_params_stride1(uint32_t *cfg, uint32_t a, uint32_t b,
                                     uint32_t c, uint32_t d, uint32_t e,
                                     uint32_t f, uint32_t g);

static void run(const char *name, uint32_t flags, uint32_t b, uint32_t c,
                uint32_t d, uint32_t e, uint32_t f, uint32_t g) {
    uint32_t cfg[16] = {0};
    qd_set_hmx_params_stride1(cfg, flags, b, c, d, e, f, g);
    printf("%s args=%08x,%u,%u,%u,%u,%u,%u cfg=", name,
           flags, b, c, d, e, f, g);
    for (unsigned i = 0; i < 13; ++i) printf("%s%08x", i ? "," : "", cfg[i]);
    puts("");
}

int main(void) {
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < 0x10000; i += 4096)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" :: "r"(ssr) : "memory");

    /* Sweep b/c across common kernel sizes; keep other args fixed. */
    run("k1x1", 0x700, 1, 1, 32, 128, 0, 0);
    run("k3x3", 0x700, 3, 3, 32, 128, 0, 0);
    run("k3x3_altflags", 0x70b, 3, 3, 32, 128, 0, 0);
    run("k3x3_g7", 0x700, 3, 3, 32, 128, 0, 7);
    run("k1x1_flags7e0", 0x7e0, 1, 1, 32, 128, 0, 0);
    run("k1x1_flags7ff", 0x7ff, 1, 1, 32, 128, 0, 0);
    return 0;
}
