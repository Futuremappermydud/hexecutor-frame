/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_block_zap.h"

#include <stdint.h>
#include <string.h>

enum { BLOCK_BYTES = 2048, MAX_COPIES = 8 };

typedef struct {
    uint8_t *pool;
    uint32_t src[MAX_COPIES], dst[MAX_COPIES];
    uint32_t count, zero_bytes;
    uint8_t fill;
} BlockZapPlan;

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static int make_plan(const Nprg2ExecCall *c, BlockZapPlan *p) {
    if (!c || !p || !c->bound || !c->pools ||
        c->command.opcode != QDR_BLOCK_ZAP || c->command.variant != 5 ||
        c->command.input_count || c->command.output_count || c->command.call_count ||
        c->command.duplicate_pad || c->command.io_index != UINT16_MAX ||
        !c->pools[1].base || !c->bound->vectors.data ||
        c->command.vector_first > c->bound->vectors.bytes / 4u ||
        c->command.vector_count > c->bound->vectors.bytes / 4u - c->command.vector_first ||
        c->command.vector_count < 5u) return 0;

    const uint8_t *v = c->bound->vectors.data + (size_t)c->command.vector_first * 4u;
    uint32_t header = rd32(v + 4u);
    /* These are the only three headers in the two serialized schedules. */
    if (header != 0x00c48002u && header != 0x00c48008u &&
        header != 0x00c68004u) return 0;
    uint32_t count = header & 0x7fffu;
    if (count > MAX_COPIES || c->command.vector_count != 3u + 2u * count ||
        rd32(v + (size_t)(2u + 2u * count) * 4u) != 0u) return 0;

    uint32_t limit = c->pool1_writable_bytes;
    if (limit > c->pools[1].bytes) return 0;
    p->pool = c->pools[1].base;
    p->count = count;
    p->zero_bytes = ((header >> 16) & 7u) * 256u;
    p->fill = (uint8_t)rd32(v);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t src = rd32(v + (size_t)(2u + 2u * i) * 4u);
        uint32_t dst = rd32(v + (size_t)(3u + 2u * i) * 4u);
        if ((src & (BLOCK_BYTES - 1u)) || (dst & (BLOCK_BYTES - 1u)) ||
            src > limit || BLOCK_BYTES > limit - src ||
            dst > limit || BLOCK_BYTES > limit - dst) return 0;
        p->src[i] = src;
        p->dst[i] = dst;
    }
    return 1;
}

static int validate(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
    BlockZapPlan p;
    return make_plan(c, &p);
}

static int run(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
    BlockZapPlan p;
    uint8_t tile[BLOCK_BYTES];
    if (!make_plan(c, &p)) return 0;
    for (uint32_t i = 0; i < p.count; i++) {
        memcpy(tile, p.pool + p.src[i], BLOCK_BYTES);
        memset(tile + BLOCK_BYTES - p.zero_bytes, p.fill, p.zero_bytes);
        memcpy(p.pool + p.dst[i], tile, BLOCK_BYTES);
    }
    return 1;
}

int nprg2_executor_register_block_zap(Nprg2Executor *e) {
    return nprg2_executor_register(e, QDR_BLOCK_ZAP, 5, validate, run, NULL);
}
