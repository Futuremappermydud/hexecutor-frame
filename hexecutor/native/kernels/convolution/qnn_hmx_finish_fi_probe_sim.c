/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Simulator-only map of qd_hmx_finish's mxmem2 read footprint. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>

extern void qd_hmx_finish(void *, void *);

int main(void) {
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < 0x80000u; i += 4096u)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" :: "r"(ssr) : "memory");

    int32_t *fi = (int32_t *)(base + 0x1000u);
    uint8_t *done = (uint8_t *)(base + 0x2000u);
    uint32_t *guard = (uint32_t *)(base + 0x3000u);
    for (uint32_t i = 0; i < 64u; ++i) {
        memset((void *)base, 0, 0x80000u);
        for (uint32_t j = 0; j < 64u; ++j) fi[j] = 0;
        fi[i] = 0x7fffffff;
        memset(done, 0xa5, 2048u);
        for (uint32_t j = 0; j < 512u; ++j) guard[j] = 0xdecafbad;
        qd_hmx_finish(fi, done);
        uint32_t changed = 0, first = 0;
        for (uint32_t j = 0; j < 2048u; ++j) {
            if (done[j] != 0) {
                if (!changed) first = j;
                ++changed;
            }
        }
        uint32_t guard_bad = 0;
        for (uint32_t j = 0; j < 512u; ++j)
            if (guard[j] != 0xdecafbad) ++guard_bad;
        printf("onehot_word=%u changed_bytes=%u first=%u guard_bad=%u\n",
               i, changed, first, guard_bad);
    }
    return 0;
}
