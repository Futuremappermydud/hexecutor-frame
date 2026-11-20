/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_s2d.h"

#include "../kernels/layout/space_to_depth_u8_direct.h"
#include <string.h>

enum { STORAGE_BLOCKED = 2, DTYPE_U8 = 1, GEOMETRY_SHAPE = 1,
       GEOMETRY_CROUTON = 64, OPCODE_SPACE_TO_DEPTH = 7, VARIANT_S2D = 31 };

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)p[1] << 8;
}

static int dims(const uint8_t *g, int32_t d[4], int32_t m[4], int32_t pad[4]) {
    if (!g || g[0] != DTYPE_U8 || !(g[1] & GEOMETRY_SHAPE) ||
        !(g[1] & GEOMETRY_CROUTON) || memcmp(g + 2, "CB", 2)) return 0;
    for (int i = 0; i < 4; ++i) {
        d[i] = (int16_t)rd16(g + 6 + 2 * i);
        m[i] = (int16_t)rd16(g + 14 + 2 * i);
        pad[i] = (int16_t)rd16(g + 22 + 2 * i);
        if (d[i] < 0 || m[i] < 0 || pad[i] < 0 || d[i] > m[i]) return 0;
    }
    return 1;
}

static uint8_t *block_access(uint32_t encoded, int store, void *ctx) {
    (void)store;
    const Nprg2ExecCall *c = (const Nprg2ExecCall *)ctx;
    uint32_t pool = encoded & 127u, offset = encoded & ~127u;
    if (!c || pool == 0 || pool >= 7 || !c->pools[pool].base ||
        offset > c->pools[pool].bytes ||
        2048u > c->pools[pool].bytes - offset) return NULL;
    return c->pools[pool].base + offset;
}

static int prepare(const Nprg2ExecCall *c, QdLayoutS2dPlan *plan) {
    if (!c || !plan || c->command.opcode != OPCODE_SPACE_TO_DEPTH ||
        c->command.variant != VARIANT_S2D || c->command.duplicate_pad ||
        c->command.input_count < 1 || c->command.output_count != 1 ||
        c->command.vector_count || c->command.call_count ||
        c->command.io_index != UINT16_MAX) return 0;

    uint32_t input_id, output_id;
    Nprg2TensorView in, out;
    if (nprg2_exec_input(c, 0, &input_id, &in) != NPRG2_EXEC_OK ||
        nprg2_exec_output(c, 0, &output_id, &out) != NPRG2_EXEC_OK) return 0;
    (void)input_id; (void)output_id;
    if (in.storage_kind != STORAGE_BLOCKED || out.storage_kind != STORAGE_BLOCKED ||
        !in.blocks || !out.blocks || !in.block_count || !out.block_count ||
        in.block_count > 65535u || out.block_count > 65535u) return 0;

    int32_t in_d[4], in_m[4], in_p[4], out_d[4], out_m[4], out_p[4];
    if (!dims(in.geometry_record, in_d, in_m, in_p) ||
        !dims(out.geometry_record, out_d, out_m, out_p)) return 0;

    uint8_t *pool_base[7];
    size_t pool_bytes[7];
    for (uint32_t i = 0; i < 7; ++i) {
        pool_base[i] = c->pools[i].base;
        pool_bytes[i] = c->pools[i].bytes;
    }
    if (!qd_layout_s2d_prepare(in_m, in_p, out_d, out_m, out_p,
            in.blocks, in.block_count, in.slice_origin,
            out.blocks, out.block_count, out.slice_origin,
            pool_base, pool_bytes, plan)) return 0;

    /* The audited helper proves full 2 KiB block ranges and input/output
     * disjointness. Keep graph writes inside the executor's pool-1 ceiling. */
    for (uint32_t ti = 0; ti < plan->tile_count; ++ti)
        for (uint32_t j = 0; j < 4; ++j) {
            uint32_t encoded = plan->tile[ti].output[j], pool = encoded & 127u;
            uint32_t offset = encoded & ~127u;
            if (pool != 1 || offset > c->pool1_writable_bytes ||
                2048u > c->pool1_writable_bytes - offset) return 0;
        }
    return 1;
}

static int validate(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
    QdLayoutS2dPlan plan;
    return prepare(c, &plan);
}

static int run(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
    QdLayoutS2dPlan plan;
    if (!prepare(c, &plan) || !c->scratch ||
        c->scratch_bytes < QD_LAYOUT_S2D_TILE_BYTES) return 0;
    uint8_t *pool_base[7];
    size_t pool_bytes[7];
    for (uint32_t i = 0; i < 7; ++i) {
        pool_base[i] = c->pools[i].base;
        pool_bytes[i] = c->pools[i].bytes;
    }
    return qd_layout_s2d_apply(&plan, pool_base, pool_bytes,
            (uint8_t *)c->scratch, c->scratch_bytes, block_access, (void *)c);
}

int nprg2_executor_register_s2d(Nprg2Executor *e) {
    if (!e) return NPRG2_EXEC_ARGUMENT;
    return nprg2_executor_register_scratch(e, OPCODE_SPACE_TO_DEPTH, VARIANT_S2D,
            validate, run, NULL, QD_LAYOUT_S2D_TILE_BYTES, 0);
}
