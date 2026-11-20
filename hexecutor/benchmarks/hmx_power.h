/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Shared power sequence copied from the hardware-validated bring-up. */
#pragma once
#include <stdint.h>
static uint64_t cycles(void) {
    uint64_t value;
    __asm__ volatile("%0 = upcycle" : "=r"(value));
    return value;
}

/* Exact HMX subsequences recovered from this device's stock QuRT ELF.
 * Source/provenance is recorded in build/hmx-qurt-sequence.json.
 * These do not include the neighboring whole-core sleep sequences. */
static const uint32_t hmx_power_words[17] = {
    0x503d502du,0x70275024u,0x50364024u,0x403d7020u,
    0x5039503cu,0x7025403cu,0x503a4030u,0x10005026u,
    0x70285025u,0x503c503bu,0x403c5038u,0x40255037u,
    0x4038403bu,0x60204037u,0x70265023u,0x502e4023u,0x1000402eu
};

static uint32_t power_trigger(volatile uint32_t *rsc, uint32_t address) {
    rsc[0x410/4] = (rsc[0x410/4] & 1u) | 0x80000000u;
    rsc[0x464/4] = address;
    rsc[0x460/4] &= ~1u;
    __asm__ volatile("syncht" ::: "memory");
    rsc[0x460/4] |= 1u;
    __asm__ volatile("syncht" ::: "memory");
    uint64_t start = cycles();
    /* QuRT waits for the completion latch, then clears it via bit 31.
     * Merely observing BUSY=0 can race the start of the sequence. */
    while (!(rsc[0x410/4] & 1u)) {
        if (cycles() - start > 200000000u) return 1;
    }
    rsc[0x460/4] &= ~1u;
    rsc[0x410/4] = (rsc[0x410/4] & 1u) | 0x80000000u;
    __asm__ volatile("syncht" ::: "memory");
    return 0;
}

