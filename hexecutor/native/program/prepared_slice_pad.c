/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_slice_pad.h"

#include "../kernels/layout/input_slice_pad_direct.h"
#include <limits.h>
#include <stdint.h>
#include <string.h>

enum { IO_SHAPE=1, IO_CROUTON=64, IO_SHAPE_VALUE=8, IO_SCALAR=16, IO_DIRECT=1, IO_NONE=0 };
typedef struct {
    const uint8_t *src;
    uint8_t *dst;
    uint32_t src_bytes, dst_bytes;
    int32_t in[4], out[4], start[4], pad[4];
    uint8_t fill;
} SlicePadPlan;

static uint16_t rd16io(const uint8_t *p){return (uint16_t)p[0]|(uint16_t)p[1]<<8;}
static int32_t rd_i32io(const uint8_t *p){return (int32_t)((uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24);}
static int text4(const uint8_t *g, const char *s){return !memcmp(g+2, s, strlen(s));}

static int shape_values(const Nprg2TensorView *v, int32_t d[4]){
    const uint8_t *g = v->geometry_record;
    if (v->storage_kind != IO_NONE || g[0] != 0 || g[1] != IO_SHAPE_VALUE || !text4(g, "s4")) return 0;
    for (uint32_t i = 0; i < 4; i++) d[i] = rd_i32io(g + 6 + i * 4u);
    return 1;
}

static int shape_product(const int32_t d[4], uint32_t *bytes){
    uint64_t n = 1;
    for (uint32_t i = 0; i < 4; i++) {
        if (d[i] < 0) return 0;
        n *= (uint32_t)d[i];
        if (n > UINT32_MAX) return 0;
    }
    *bytes = (uint32_t)n;
    return 1;
}

static int prepare_slice_pad(const Nprg2ExecCall *c, SlicePadPlan *p){
    if (!c || !p || c->command.opcode != QDR_INPUT_SLICE_PAD ||
        (c->command.variant != 2 && c->command.variant != 8) ||
        c->command.duplicate_pad ||
        c->command.input_count != 7 || c->command.output_count != 1 ||
        c->command.vector_count || c->command.call_count ||
        c->command.io_index >= 2 || !c->inputs ||
        c->command.io_index >= c->input_count || !c->inputs[c->command.io_index].data) {
        return 0;
    }

    uint32_t ids[7], out_id;
    Nprg2TensorView t[7], o;
    for (uint32_t i = 0; i < 7; i++) {
        if (nprg2_exec_input(c, i, &ids[i], &t[i]) != NPRG2_EXEC_OK) return 0;
    }
    if (nprg2_exec_output(c, 0, &out_id, &o) != NPRG2_EXEC_OK) return 0;
    (void)ids; (void)out_id;

    /* Four runtime shape-value tensors carry pad/start/output/input geometry;
     * slots 4 and 5 are also shape-value metadata in this source family. */
    if (!shape_values(&t[0], p->pad) || !shape_values(&t[1], p->start) ||
        !shape_values(&t[2], p->out) || !shape_values(&t[4], (int32_t[4]){0}) ||
        !shape_values(&t[5], (int32_t[4]){0}) || !shape_values(&t[6], p->in)) {
        return 0;
    }

    const uint8_t *fg = t[3].geometry_record;
    int32_t fill;
    if (t[3].storage_kind != IO_NONE) return 0;
    if (fg[1] == IO_SCALAR && fg[0] == 5u && text4(fg, "ni")) {
        fill = rd_i32io(fg + 6);
    } else if (fg[1] == IO_SHAPE_VALUE && fg[0] == 0 && text4(fg, "s4")) {
        for (uint32_t i = 0; i < 4; i++) {
            if (rd_i32io(fg + 6 + i * 4u)) return 0;
        }
        fill = 0;
    } else {
        return 0;
    }

    if (o.storage_kind != IO_DIRECT || o.pool != 1 || !c->pools[1].base) return 0;
    const uint8_t *og = o.geometry_record;
    if (og[0] != 1u || !(og[1] & IO_SHAPE) || (og[1] & ~(IO_SHAPE | 32u)) ||
        !text4(og, "FB") || (og[1] & IO_CROUTON)) {
        return 0;
    }

    uint64_t out_count = 1;
    for (uint32_t i = 0; i < 4; i++) {
        int32_t d = (int16_t)rd16io(og + 6 + i * 2u);
        int32_t m = (int16_t)rd16io(og + 14 + i * 2u);
        int32_t pad = (int16_t)rd16io(og + 22 + i * 2u);
        if (d < 0 || m < 0 || pad != 0 || d != m || d != p->out[i]) return 0;
        out_count *= (uint32_t)d;
        if (out_count > UINT32_MAX) return 0;
    }

    uint32_t input_bytes;
    if (!shape_product(p->in, &input_bytes) || !input_bytes ||
        input_bytes != c->inputs[c->command.io_index].bytes ||
        out_count > UINT32_MAX || out_count > c->pool1_writable_bytes ||
        o.offset > c->pool1_writable_bytes - (uint32_t)out_count ||
        !out_count || o.offset > c->pools[1].bytes ||
        (uint32_t)out_count > c->pools[1].bytes - o.offset) {
        return 0;
    }

    p->src = c->inputs[c->command.io_index].data;
    p->src_bytes = input_bytes;
    p->dst = c->pools[1].base + o.offset;
    p->dst_bytes = (uint32_t)out_count;
    p->fill = (uint8_t)fill;

    uintptr_t sb = (uintptr_t)p->src, db = (uintptr_t)p->dst;
    if (p->src_bytes > UINTPTR_MAX - sb || p->dst_bytes > UINTPTR_MAX - db ||
        (db < sb + p->src_bytes && sb < db + p->dst_bytes)) {
        return 0;
    }

    for (int i = 0; i < 4; i++) {
        int64_t l = (int64_t)p->start[i] - p->pad[i], end = l + p->out[i];
        if (l < INT32_MIN || l > INT32_MAX) return 0;
        if (end < p->in[i] && (end < INT32_MIN || end > INT32_MAX)) return 0;
    }

    return 1;
}

static int validate_slice_pad(const Nprg2ExecCall *c, void *ctx){
    (void)ctx;
    SlicePadPlan p;
    return prepare_slice_pad(c, &p);
}

static int run_slice_pad(const Nprg2ExecCall *c, void *ctx){
    (void)ctx;
    SlicePadPlan p;
    if (!prepare_slice_pad(c, &p)) return 0;
    return qd_layout_input_slice_pad_direct_u8(p.dst, p.dst_bytes, p.src, p.src_bytes,
                                               p.in, p.out, p.start, p.pad, p.fill);
}

int nprg2_executor_register_slice_pad(Nprg2Executor *e){
    if (!e) return NPRG2_EXEC_ARGUMENT;
    int r = nprg2_executor_register(e, QDR_INPUT_SLICE_PAD, 2, validate_slice_pad, run_slice_pad, NULL);
    if (r != NPRG2_EXEC_OK) return r;
    return nprg2_executor_register(e, QDR_INPUT_SLICE_PAD, 8, validate_slice_pad, run_slice_pad, NULL);
}
