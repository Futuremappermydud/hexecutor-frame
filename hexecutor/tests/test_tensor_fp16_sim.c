/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "../native/tensor/graph.h"
#include "../native/tensor/fp16.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hexagon_standalone.h>
/* Window offsets select the depthwise loop's column parity. */
static uint32_t pad_shift;
/* VTCM bytes the plan may use; a small value leaves a small workspace, so a packed
 * convolution needs several batches. */
static uint32_t vtcm_size = 0x800000;
/* z of the next convolution/depthwise check(): HXFG_NONE or (dh << 8) | dw. */
static uint32_t dilation_z = HXFG_NONE;
static uint32_t slice_channel_offset = 5;
/* Simulator pcycles spent in the FP16 operator calls of the last check(), summed over parts. */
static uint64_t op_pcycles;
/* Extra ROPE fields of the next check(): theta float bits (0: 10000), rotated width per head
 * (0: all) and 1 to add the row index to the position. */
static uint32_t rope_theta_bits, rope_rot, rope_row_pos;
/* Simulator processor cycles: the service call behind hexagon_sim_read_pcycles, inlined because
 * the tensor build does not link libhexagon. */
static uint64_t pcycles(void) {
    uint64_t c;
    __asm__ volatile("r0 = #0x52\n\ttrap0(#0x0)\n\t%0 = combine(r1,r0)" : "=r"(c) : : "r0", "r1");
    return c;
}
static void import_four_parts(const HxFp16Job *job) {
    for (uint32_t part = 0; part < 4; part++)
        hx_fp16_job(job, part, 4);
}
static int check(uint32_t base, uint32_t kind, uint32_t h, uint32_t w, uint32_t ci, uint32_t co,
                 uint32_t kh, uint32_t step, uint32_t axis) {
    uint8_t *constants = aligned_alloc(128, 0x100000);
    float *reference = aligned_alloc(128, 0x200000), *actual = aligned_alloc(128, 0x200000);
    uint8_t *arena = (uint8_t *)base;
    if (!constants || !reference || !actual)
        return 1;
    memset(constants, 0, 0x100000);
    HxHeader header = {0};
    header.tensors = 3;
    header.inputs = 2;
    header.outputs = 1;
    header.ops = 1;
    uint32_t io[3] = {0, 1, 2};
    HxTensor t[3] = {{0, h, w, ci, (ci + 31) & ~31u, {0}},
                     {0x80000, h, w, ci, (ci + 31) & ~31u, {0}},
                     {0x100000, h, w, co, (co + 31) & ~31u, {0}}};
    HxOp o = {0};
    o.kind = kind;
    o.x = 0;
    o.y = o.z = HXFG_NONE;
    if (kind == HX_CONV || kind == HX_DEPTHWISE || kind == HX_MAXPOOL || kind == HX_AVGPOOL)
        o.z = dilation_z;
    o.out = 2;
    if (kind == HX_ADD || kind == HX_CONCAT || kind == HX_MATMUL ||
        (kind == HX_BINARY && step < 4))
        o.y = 1;
    if (kind == HX_CONV || kind == HX_DEPTHWISE || kind == HX_MAXPOOL || kind == HX_AVGPOOL) {
        o.p[0] = o.p[1] = kh;
        o.p[2] = o.p[3] = step;
        o.p[4] = o.p[5] = kh / 2 - ((kind == HX_DEPTHWISE || kind == HX_CONV) ? pad_shift : 0);
        o.p[6] = 0;
        o.p[7] = 0x50000;
        o.p[8] = 3;
        t[2].h = (h + step - 1) / step;
        t[2].w = (w + step - 1) / step;
        if (kind == HX_CONV && kh == 3u && step == 1u && pad_shift == UINT32_MAX) {
            t[2].h = h + 2u;
            t[2].w = w + 2u;
        }
        if (kind == HX_MAXPOOL || kind == HX_AVGPOOL) {
            o.p[6] = o.p[7] = o.p[8] = 0;
            if (kind == HX_AVGPOOL)
                o.p[6] = axis; /* count_include_pad */
        }
        if (kind == HX_CONV) {
            o.p[9] = 0x60000;
            o.p[10] = (kh * kh * ci + 31) / 32;
        }
    } else if (kind == HX_PAD) {
        o.p[0] = 1;
        o.p[1] = 2;
        o.p[2] = 3;
        t[2].h = h + 2;
        t[2].w = w + 4;
        t[2].c = ci + 7;
        t[2].stride = (t[2].c + 31) & ~31u;
    } else if (kind == HX_RESHAPE) {
        /* axis is the p[0] flag: 0 HWC copy, 1 NCHW flatten to 1x1xN, 2 NCHW unflatten from 1x1xN
         * (the source is then 1x1x(h*w*ci) and the output h x w x ci). */
        o.p[0] = axis;
        if (axis == 1) {
            t[2].h = t[2].w = 1;
            t[2].c = h * w * ci;
        } else if (axis == 2) {
            t[0].h = t[0].w = 1;
            t[0].c = h * w * ci;
            t[0].stride = (t[0].c + 31) & ~31u;
            t[2].h = h;
            t[2].w = w;
            t[2].c = ci;
        } else {
            t[2].h = 1;
            t[2].w = h * w * ci / co;
        }
        t[2].stride = (t[2].c + 31) & ~31u;
    } else if (kind == HX_TRANSPOSE) {
        /* axis selects one of the six permutations of (h, w, c); co is ignored. */
        static const uint32_t perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                             {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        const uint32_t ad[3] = {h, w, ci};
        for (uint32_t k = 0; k < 3; k++)
            o.p[k] = perms[axis][k];
        t[2].h = ad[o.p[0]];
        t[2].w = ad[o.p[1]];
        t[2].c = ad[o.p[2]];
        t[2].stride = (t[2].c + 31) & ~31u;
    } else if (kind == HX_CONCAT) {
        o.p[0] = axis;
        if (axis == 0)
            t[2].h *= 2;
        else if (axis == 1)
            t[2].w *= 2;
        else {
            t[2].c = ci * 2;
            t[2].stride = (t[2].c + 31) & ~31u;
        }
    } else if (kind == HX_RESIZE) {
        t[2].h = h * 2;
        t[2].w = w * 2;
        if (axis == 1)
            o.p[2] = 1; /* nearest */
        else if (axis == 2) {
            o.p[0] = 1; /* bilinear align_corners */
            o.p[3] = 1;
        } else if (axis == 3) {
            o.p[5] = 1; /* bicubic */
            o.p[3] = 2;
        } else
            o.p[1] = 1; /* bilinear half-pixel */
    } else if (kind == HX_ATTN_SOFTMAX) {
        /* step bits: causal, additive mask, scalar-per-pixel mask. */
        o.p[0] = 2;
        o.p[1] = step & 1u;
        if (step & 8u) { /* temperature 4: the scores are multiplied by it before the mask */
            const float temperature = 4.0f;
            memcpy(&o.p[3], &temperature, sizeof temperature);
        }
        if (step & 2u) {
            o.y = 1;
            if (step & 4u) {
                t[1].c = 1;
                t[1].stride = 32;
            }
        }
    } else if (kind == HX_ZEROINSERT) {
        /* Stride step, pad 1 on top and left, one trailing row/column: output (h-1)*step + 3. */
        o.p[0] = o.p[1] = step;
        o.p[2] = o.p[3] = 1;
        t[2].h = (h - 1) * step + 3;
        t[2].w = (w - 1) * step + 3;
    } else if (kind == HX_SLICE) {
        /* Sub-box starting at (1, 2, 5): the output keeps co channels (co + 5 <= ci). */
        o.p[0] = 1;
        o.p[1] = 2;
        o.p[2] = slice_channel_offset;
        t[2].h = h - 1;
        t[2].w = w - 2;
    } else if (kind == HX_MEAN)
        t[2].h = t[2].w = 1;
    else if (kind == HX_REDUCE) {
        /* kh: HWC bitmask of reduced axes, axis: HX_RED_* operation; co is ignored. */
        o.p[0] = axis;
        o.p[1] = kh;
        t[2].h = kh & 1u ? 1 : h;
        t[2].w = kh & 2u ? 1 : w;
        t[2].c = kh & 4u ? 1 : ci;
        t[2].stride = (t[2].c + 31) & ~31u;
    }
    else if (kind == HX_ARGMAX || kind == HX_ARGMIN) {
        o.p[0] = step;
        o.p[1] = axis;
        t[2].c = 1;
        t[2].stride = 32;
    }
    else if (kind == HX_ACT) {
        o.p[0] = axis;
        if (axis == HX_ACT_LEAKY)
            o.p[1] = 0x3dcccccdu; /* 0.1f */
        if (axis == HX_ACT_HARDSIGMOID) {
            o.p[1] = 0x3e2aaaabu; /* 1/6 */
            o.p[2] = 0x3f000000u; /* 0.5 */
        }
    } else if (kind == HX_BINARY) {
        /* axis: operation. step: 0 same-shape y, 1 y is 1x1xC, 2 y is Hx1xC, 3 y is 1x1x1,
         * 4 per-channel constant. kh: reversed. */
        o.p[0] = axis;
        o.p[3] = kh;
        if (step == 1 || step == 3)
            t[1].h = t[1].w = 1;
        if (step == 2)
            t[1].w = 1;
        if (step == 3) {
            t[1].c = 1;
            t[1].stride = 32;
        }
        if (step == 4) {
            o.p[1] = 1;
            o.p[2] = 0x70000;
            for (uint32_t c = 0; c < 64; c++)
                ((float *)(constants + 0x70000))[c] = ((int)(c % 5) - 2) * .5f;
        }
        if (step == 5) {
            o.p[1] = 2;
            o.p[2] = 0x70000;
            for (uint32_t c = 0; c < h * w * t[2].stride; c++)
                ((float *)(constants + 0x70000))[c] = ((int)(c % 5) - 2) * .5f;
        }
    } else if (kind == HX_MATMUL) {
        /* h x 1 x ci is A and co is N; step is transB (B is [ci,co], or [co,ci] when 1); axis 1
         * scales the product by 0.5, otherwise p[1] stays zero (1.0). */
        o.p[0] = step;
        o.p[1] = axis ? 0x3f000000u : 0u;
        t[0].w = t[1].w = t[2].w = 1;
        t[1].h = step ? co : ci;
        t[1].c = step ? ci : co;
        t[1].stride = (t[1].c + 31) & ~31u;
        t[2].h = h;
        t[2].c = co;
        t[2].stride = (co + 31) & ~31u;
    } else if (kind == HX_NORM) {
        /* axis: mode (HX_NORM_CHANNEL or HX_NORM_GROUP). step: group count (GROUP only).
         * Epsilon 1e-5; gamma and beta are per-channel vectors at 0x90000 and 0x98000. */
        o.p[0] = axis;
        o.p[1] = axis == HX_NORM_GROUP ? step : 0;
        o.p[2] = 0x3727c5acu; /* 1e-5f */
        o.p[3] = 0x90000;
        o.p[4] = 0x98000;
        for (uint32_t c = 0; c < ci; c++) {
            ((float *)(constants + 0x90000))[c] = .5f + .25f * (float)(c % 3);
            ((float *)(constants + 0x98000))[c] = .125f * (float)((int)(c % 5) - 2);
        }
    }
    else if (kind == HX_RMSNORM) {
        /* axis 0: gamma only; axis 1: gamma + beta; axis 2: no gamma, no beta. Gamma at 0x90000, beta at 0x98000, epsilon 1e-5. */
        o.p[0] = 0x3727c5acu;
        o.p[1] = axis == 2 ? HXFG_NONE : 0x90000;
        o.p[2] = axis == 1 ? 0x98000 : HXFG_NONE;
        for (uint32_t c = 0; c < ci; c++) {
            ((float *)(constants + 0x90000))[c] = .5f + .25f * (float)(c % 3);
            ((float *)(constants + 0x98000))[c] = .125f * (float)((int)(c % 5) - 2);
        }
    } else if (kind == HX_ROPE) {
        /* step is head_dim; axis packs mode (bit 0) and position (axis >> 1) for p[2]; kh stays 1
         * because the generic weight fill below writes kh*kh rows. x and out are h x 1 x ci. */
        o.p[0] = step;
        o.p[1] = ci / step;
        o.p[2] = axis >> 1;
        o.p[3] = axis & 1u;
        o.p[4] = rope_theta_bits;
        o.p[5] = rope_rot;
        o.p[6] = rope_row_pos;
    } else if (kind == HX_LINEAR_Q) {
        /* One token: ci is K, co is N, kh is the format (0 Q4_0, 1 Q8_0), axis the activation.
         * Blocks of 32 weights per output row, fp16 scale last; bias at 0x98000. */
        const uint32_t block_bytes = kh ? 34u : 18u, blocks = ci / 32u;
        uint8_t *w = constants + 0xa0000;
        for (uint32_t r = 0; r < co; r++)
            for (uint32_t b = 0; b < blocks; b++) {
                uint8_t *blk = w + ((uint64_t)r * blocks + b) * block_bytes;
                for (uint32_t j = 0; j < block_bytes - 2u; j++)
                    blk[j] = (uint8_t)((r * 131u + b * 17u + j * 29u + 7u) & 0xffu);
                blk[block_bytes - 2u] = 0x00;
                /* Q8 weights reach +-128, Q4 +-8: scale 2^-9 and 2^-6 keep outputs near 5. */
                blk[block_bytes - 1u] = kh ? 0x18 : 0x24;
            }
        for (uint32_t c = 0; c < co; c++)
            ((float *)(constants + 0x98000))[c] = .125f * (float)((int)(c % 5) - 2);
        o.p[0] = kh;
        o.p[1] = 0xa0000;
        o.p[2] = 0x98000;
        o.p[3] = ci;
        o.p[4] = co;
        o.p[5] = axis;
    }
    uint32_t stride = t[2].stride;
    float *wf = (float *)constants;
    for (uint32_t k = 0; k < kh * kh * (kind == HX_CONV ? ci : 1); k++)
        for (uint32_t c = 0; c < stride; c++)
            wf[k * stride + c] = c < co ? ((k + c) % 2 ? -.0625f : .0625f) : 0;
    for (uint32_t c = 0; c < stride; c++)
        ((float *)(constants + 0x50000))[c] = c < co ? .25f : 0;
    if (kind == HX_PRELU)
        for (uint32_t c = 0; c < t[0].stride; c++)
            wf[c] = .25f;
    if (kind == HX_CONV) {
        uint16_t *wp = (uint16_t *)(constants + o.p[9]);
        uint32_t kb = o.p[10];
        for (uint32_t k = 0; k < kh * kh * ci; k++)
            for (uint32_t c = 0; c < co; c++)
                wp[(c / 32) * kb * 1024 + (k / 32) * 1024 + ((k % 32) / 2) * 64 + (c % 32) * 2 +
                   k % 2] = (k + c) % 2 ? 0xac00 : 0x2c00;
    }
    HxGraph g = {&header, t, &o, io, constants, arena, NULL};
    static uint32_t tables[4096];
    static struct HxFp16Plan plan;
    if (!hx_fp16_plan(&g, &plan, tables, arena + vtcm_size)) {
        printf("FAIL plan kind%u\n", (unsigned)kind);
        return 1;
    }
    g.fp16 = &plan;
    hx_fp16_stage(&g, NULL);
    for (uint32_t n = 0; n < 2; n++) {
        float *v = reference + t[n].offset / 4;
        for (uint32_t i = 0; i < t[n].h * t[n].w; i++)
            for (uint32_t c = 0; c < t[n].stride; c++)
                /* Matrix products use small values so the FP16 result stays within .004. */
                v[i * t[n].stride + c] =
                    c >= t[n].c ? 0
                    : kind == HX_MATMUL ? ((int)((i * 7 + c * 3 + n) % 5) - 2) * .125f
                                        : ((int)((i * 7 + c * 3 + n) % 23) - 11) * .25f;
        /* HX_ACT_SQRT and HX_ACT_LOG take positive inputs; HX_ACT_RECIPROCAL avoids zero. */
        if (kind == HX_ACT && (axis == HX_ACT_SQRT || axis == HX_ACT_LOG))
            for (uint32_t j = 0; j < t[n].h * t[n].w * t[n].stride; j++)
                v[j] = fabsf(v[j]) + 0.25f;
        if (kind == HX_ACT && axis == HX_ACT_RECIPROCAL)
            for (uint32_t j = 0; j < t[n].h * t[n].w * t[n].stride; j++)
                if (v[j] == 0.0f)
                    v[j] = 0.25f;
        if (kind == HX_BINARY && axis == HX_BIN_POW)
            for (uint32_t j = 0; j < t[n].h * t[n].w * t[n].stride; j++)
                v[j] = fabsf(v[j]) * 0.25f + 0.5f;
        hx_fp16_import(&g, n, v, import_four_parts);
        hx_fp16_export(&g, n, actual, NULL);
        if (memcmp(actual, v, t[n].h * t[n].w * t[n].stride * 4)) {
            printf("FAIL round trip kind%u c%u\n", (unsigned)kind, (unsigned)ci);
            return 1;
        }
    }
    g.arena = (uint8_t *)reference;
    for (uint32_t p = 0; p < 3; p++)
        hx_execute_op(&g, 0, p, 3);
    g.arena = arena;
    const uint64_t op_begin = pcycles();
    if (kind == HX_CONV)
        hx_fp16_conv(&g, 0, NULL);
    else
        for (uint32_t p = 0; p < 3; p++)
            hx_fp16_op(&g, 0, p, 3);
    op_pcycles = pcycles() - op_begin;
    hx_fp16_complete(&g, 0);
    if (t[2].w & 1u) {
        /* The pad column must read as zero for any later border tap. */
        uint32_t tile_count = (t[2].h * (t[2].w + 1u) + 31u) / 32u;
        const uint16_t *d = (const uint16_t *)(arena + plan.offset[2]);
        for (uint32_t b = 0; b < t[2].stride / 32u; b++)
            for (uint32_t y = 0; y < t[2].h; y++)
                for (uint32_t c = 0; c < 32u; c++)
                    if (d[b * tile_count * 1024u + (y * (t[2].w + 1u) / 2u + t[2].w / 2u) * 64u +
                          c * 2u + 1u]) {
                        printf("FAIL pad kind%u\n", (unsigned)kind);
                        return 1;
                    }
    }
    hx_fp16_export(&g, 2, actual, NULL);
    float *expected = reference + t[2].offset / 4;
    for (uint32_t i = 0; i < t[2].h * t[2].w; i++)
        for (uint32_t c = 0; c < t[2].c; c++) {
            float a = actual[i * stride + c], b = expected[i * stride + c];
            float tol = (kind == HX_BINARY && axis == HX_BIN_POW) ? .02f : .004f;
            if (fabsf(a - b) > tol) {
                printf("FAIL blocked kind%u shape%ux%ux%u to%u k%u s%u axis%u pixel%u c%u got%f "
                       "expected%f\n",
                       (unsigned)kind, (unsigned)h, (unsigned)w, (unsigned)ci, (unsigned)co,
                       (unsigned)kh, (unsigned)step, (unsigned)axis, (unsigned)i, (unsigned)c, a,
                       b);
                return 1;
            }
        }
    free(actual);
    free(reference);
    free(constants);
    return 0;
}
/* Persistent state in the FP16 plan: a causal short convolution (conv ring, optional gate, SiLU or
 * none) followed by a KV write at a position that changes every step. The FP32 reference and the
 * FP16 plan each keep their own state buffer; outputs and both state buffers must agree after
 * every step, so the ring shift and the cache append are checked across run() calls. */
static int check_state(uint32_t base, uint32_t c, uint32_t taps, uint32_t steps, uint32_t gated,
                       uint32_t act) {
    const uint32_t stride = (c + 31u) & ~31u, ring = taps - 1u, cache_rows = steps + 2u;
    uint8_t *constants = aligned_alloc(128, 0x100000);
    float *reference = aligned_alloc(128, 0x200000), *actual = aligned_alloc(128, 0x200000);
    float *state_ref = aligned_alloc(128, 0x40000), *state_dsp = aligned_alloc(128, 0x40000);
    uint8_t *arena = (uint8_t *)base;
    if (!constants || !reference || !actual || !state_ref || !state_dsp)
        return 1;
    memset(constants, 0, 0x100000);
    memset(state_ref, 0, 0x40000);
    memset(state_dsp, 0, 0x40000);
    float *w = (float *)constants;
    for (uint32_t j = 0; j < taps; j++)
        for (uint32_t ch = 0; ch < stride; ch++)
            w[j * stride + ch] = ch < c ? ((int)((j * 5 + ch * 3) % 7) - 3) * .125f : 0;
    for (uint32_t ch = 0; ch < stride; ch++)
        ((float *)(constants + 0x50000))[ch] = ch < c ? .0625f * (float)(ch % 3) : 0;
    HxHeader header = {0};
    header.tensors = 5;
    header.inputs = 2;
    header.outputs = 1;
    header.ops = 2;
    uint32_t io[3] = {0, 4, 1};
    HxTensor t[5] = {{0, 1, 1, c, stride, {0}},
                     {0x80000, 1, 1, c, stride, {0}},
                     {0, ring, 1, c, stride, {HX_TENSOR_PERSISTENT, 0x1001, 0}},
                     {0x20000, cache_rows, 1, c, stride, {HX_TENSOR_PERSISTENT, 0x1002, 0}},
                     {0x100000, 1, 1, c, stride, {0}}};
    HxOp ops[2];
    memset(ops, 0, sizeof ops);
    ops[0].kind = HX_SHORTCONV;
    ops[0].x = 0;
    ops[0].y = 2;
    ops[0].z = gated ? 4 : HXFG_NONE;
    ops[0].out = 1;
    ops[0].p[0] = taps;
    ops[0].p[1] = 0;
    ops[0].p[2] = 0x50000;
    ops[0].p[3] = act;
    ops[1].kind = HX_KV_WRITE;
    ops[1].x = 0;
    ops[1].y = ops[1].z = HXFG_NONE;
    ops[1].out = 3;
    HxGraph g = {&header, t, ops, io, constants, arena, NULL};
    static uint32_t tables[4096];
    static struct HxFp16Plan plan;
    g.state = (uint8_t *)state_dsp;
    g.state_bytes = 0x40000;
    if (!hx_fp16_plan(&g, &plan, tables, arena + vtcm_size)) {
        printf("FAIL plan state c%u taps%u\n", (unsigned)c, (unsigned)taps);
        return 1;
    }
    g.fp16 = &plan;
    hx_fp16_stage(&g, NULL);
    for (uint32_t step = 0; step < steps; step++) {
        float *xr = reference + t[0].offset / 4, *gr = reference + t[4].offset / 4;
        for (uint32_t ch = 0; ch < stride; ch++) {
            xr[ch] = ch < c ? ((int)((step * 11 + ch * 7) % 23) - 11) * .125f : 0;
            gr[ch] = ch < c ? ((int)((step * 3 + ch * 5) % 9) - 4) * .25f : 0;
        }
        for (uint32_t k = 0; k < 2; k++) {
            ops[1].p[0] = step;
            if (k == 0) {
                g.arena = (uint8_t *)reference;
                g.state = (uint8_t *)state_ref;
                g.fp16 = NULL;
                for (uint32_t o = 0; o < 2; o++)
                    for (uint32_t p = 0; p < 3; p++)
                        hx_execute_op(&g, o, p, 3);
            } else {
                g.arena = arena;
                g.state = (uint8_t *)state_dsp;
                g.fp16 = &plan;
                hx_fp16_import(&g, 0, xr, NULL);
                if (gated)
                    hx_fp16_import(&g, 4, gr, NULL);
                for (uint32_t o = 0; o < 2; o++) {
                    for (uint32_t p = 0; p < 3; p++)
                        hx_fp16_op(&g, o, p, 3);
                    hx_fp16_complete(&g, o);
                }
                hx_fp16_export(&g, 1, actual, NULL);
            }
        }
        const float *expected = reference + t[1].offset / 4;
        for (uint32_t ch = 0; ch < c; ch++)
            if (fabsf(actual[ch] - expected[ch]) > .004f) {
                printf("FAIL state output c%u taps%u gate%u act%u step%u ch%u got%f expected%f\n",
                       (unsigned)c, (unsigned)taps, (unsigned)gated, (unsigned)act,
                       (unsigned)step, (unsigned)ch, actual[ch], expected[ch]);
                return 1;
            }
        /* The persistent buffers hold float32 in both runs; the FP16 path may differ only by
         * the rounding of its half-precision input. */
        for (uint32_t r = 0; r < ring + cache_rows + 4; r++) {
            const uint32_t off = r < ring ? (t[2].offset / 4 + r * stride)
                                          : (t[3].offset / 4 + (r - ring) * stride);
            if (r >= ring + cache_rows)
                break;
            for (uint32_t ch = 0; ch < c; ch++)
                if (fabsf(state_dsp[off + ch] - state_ref[off + ch]) > .004f) {
                    printf("FAIL state buffer c%u taps%u gate%u step%u row%u ch%u got%f "
                           "expected%f\n",
                           (unsigned)c, (unsigned)taps, (unsigned)gated, (unsigned)step,
                           (unsigned)r, (unsigned)ch, state_dsp[off + ch], state_ref[off + ch]);
                    return 1;
                }
        }
    }
    free(state_dsp);
    free(state_ref);
    free(actual);
    free(reference);
    free(constants);
    return 0;
}
/* The FP16 plan holds activations as half-precision tiles but persistent state as float32 rows, so a
 * kernel that reads a persistent tensor other than the state ops must make the plan refuse, leaving
 * the graph to the FP32 path (which reads the caches directly). */
static int check_persistent_rejected(uint32_t base) {
    HxHeader header = {0};
    header.tensors = 4;
    header.inputs = 1;
    header.outputs = 1;
    header.ops = 1;
    uint32_t io[2] = {0, 3};
    HxTensor t[4] = {{0, 1, 1, 64, 64, {0}},
                     {0, 8, 1, 64, 64, {HX_TENSOR_PERSISTENT, 1, 0}},
                     {0x4000, 8, 1, 64, 64, {HX_TENSOR_PERSISTENT, 2, 0}},
                     {0x1000, 1, 1, 64, 64, {0}}};
    HxOp o = {0};
    o.kind = HX_SDPA;
    o.x = 0;
    o.y = 1;
    o.z = 2;
    o.out = 3;
    o.p[0] = HXFG_NONE;
    o.p[3] = 1;
    HxGraph g = {&header, t, &o, io, NULL, (uint8_t *)base, NULL};
    static uint32_t tables[4096];
    static struct HxFp16Plan plan;
    if (hx_fp16_plan(&g, &plan, tables, (uint8_t *)base + vtcm_size)) {
        printf("FAIL FP16 plan accepted SDPA over a persistent KV cache\n");
        return 1;
    }
    return 0;
}
/* Compact RGB entry: staged image, parallel-shaped pack, direct matrix output. */
static int check_rgb(uint32_t base, uint32_t ih, uint32_t iw, uint32_t kh, uint32_t co) {
    uint8_t *constants = aligned_alloc(128, 0x100000);
    float *reference = aligned_alloc(128, 0x200000), *actual = aligned_alloc(128, 0x200000);
    uint8_t *arena = (uint8_t *)base;
    if (!constants || !reference || !actual)
        return 1;
    memset(constants, 0, 0x100000);
    HxHeader header = {0};
    header.tensors = 2;
    header.inputs = 1;
    header.outputs = 1;
    header.ops = 1;
    uint32_t io[2] = {0, 1}, stride = (co + 31) & ~31u;
    HxTensor t[2] = {{0, ih, iw, 3, 3, {0}},
                     {0x100000, (ih + 1) / 2, (iw + 1) / 2, co, stride, {0}}};
    HxOp o = {0};
    o.kind = HX_CONV;
    o.x = 0;
    o.y = o.z = HXFG_NONE;
    o.out = 1;
    o.p[0] = o.p[1] = kh;
    o.p[2] = o.p[3] = 2;
    o.p[4] = o.p[5] = kh / 2;
    o.p[7] = 0x50000;
    o.p[8] = 3;
    o.p[9] = 0x60000;
    o.p[10] = (kh * kh * 3 + 31) / 32;
    float *wf = (float *)constants;
    uint16_t *wp = (uint16_t *)(constants + o.p[9]);
    for (uint32_t k = 0; k < kh * kh * 3; k++)
        for (uint32_t c = 0; c < co; c++) {
            wf[k * stride + c] = (k + c) % 2 ? -.0625f : .0625f;
            wp[(c / 32) * o.p[10] * 1024 + (k / 32) * 1024 + ((k % 32) / 2) * 64 + (c % 32) * 2 +
               k % 2] = (k + c) % 2 ? 0xac00 : 0x2c00;
        }
    for (uint32_t c = 0; c < co; c++)
        ((float *)(constants + o.p[7]))[c] = .25f;
    for (uint32_t i = 0; i < ih * iw * 3; i++)
        reference[i] = ((int)(i % 9) - 4) * .25f;
    HxGraph g = {&header, t, &o, io, constants, (uint8_t *)reference, NULL};
    for (uint32_t p = 0; p < 3; p++)
        hx_execute_op(&g, 0, p, 3);
    static uint32_t tables[4096];
    static struct HxFp16Plan plan;
    g.arena = arena;
    if (!hx_fp16_plan(&g, &plan, tables, arena + vtcm_size)) {
        puts("FAIL RGB plan");
        return 1;
    }
    g.fp16 = &plan;
    hx_fp16_stage(&g, NULL);
    hx_fp16_import(&g, 0, reference, NULL);
    hx_fp16_conv(&g, 0, NULL);
    hx_fp16_complete(&g, 0);
    hx_fp16_export(&g, 1, actual, NULL);
    const float *expected = reference + t[1].offset / 4;
    for (uint32_t i = 0; i < t[1].h * t[1].w * stride; i++)
        if (fabsf(actual[i] - expected[i]) > .004f) {
            printf("FAIL RGB %ux%u input %ux%u element %u got %f expected %f\n", (unsigned)kh,
                   (unsigned)kh, (unsigned)ih, (unsigned)iw, (unsigned)i, actual[i], expected[i]);
            return 1;
        }
    free(actual);
    free(reference);
    free(constants);
    return 0;
}

static int check_indexing(uint32_t base) {
    uint8_t *constants = (uint8_t *)aligned_alloc(128, 0x100000);
    float *reference = (float *)aligned_alloc(128, 0x200000);
    float *actual = (float *)aligned_alloc(128, 0x200000);
    uint8_t *arena = (uint8_t *)base;
    if (!constants || !reference || !actual)
        return 1;
    static uint32_t tables[4096];
    static struct HxFp16Plan plan;
    int errors = 0;

    /* 1. HX_GATHER Mode 1 (embedding table in constants) */
    {
        memset(constants, 0, 0x100000);
        memset(reference, 0, 0x200000);
        memset(actual, 0, 0x200000);
        HxHeader header = {0};
        header.tensors = 2;
        header.inputs = 1;
        header.outputs = 1;
        header.ops = 1;
        uint32_t io[2] = {0, 1};
        uint32_t vocab = 16, dim = 64;
        HxTensor t[2] = {
            {0, 4, 1, 1, 32, {0}}, /* indices [4, 1, 1] */
            {0x40000, 4, 1, dim, dim, {0}} /* output [4, 1, 64] */
        };
        HxOp o = {0};
        o.kind = HX_GATHER;
        o.x = 0;
        o.y = o.z = HXFG_NONE;
        o.out = 1;
        o.p[0] = 1; /* Mode 1: constant table */
        o.p[1] = 0x20000; /* offset */
        o.p[2] = vocab;
        o.p[3] = dim;
        o.p[4] = 0; /* axis */
        o.p[5] = dim; /* inner */
        o.p[6] = 1; /* outer */
        o.p[7] = dim; /* stride */
        float *tbl = (float *)(constants + 0x20000);
        for (uint32_t i = 0; i < vocab * dim; i++)
            tbl[i] = ((int)(i % 23) - 11) * 0.125f;
        float *idx_in = reference + t[0].offset / 4;
        idx_in[0] = 3.0f;
        idx_in[32] = 0.0f;
        idx_in[64] = 7.0f;
        idx_in[96] = 2.0f;
        HxGraph g = {&header, t, &o, io, constants, (uint8_t *)reference, NULL};
        for (uint32_t p = 0; p < 3; p++)
            hx_execute_op(&g, 0, p, 3);
        g.arena = arena;
        if (!hx_fp16_plan(&g, &plan, tables, arena + vtcm_size)) {
            printf("FAIL plan gather mode 1\n");
            return 1;
        }
        g.fp16 = &plan;
        hx_fp16_stage(&g, NULL);
        hx_fp16_import(&g, 0, idx_in, NULL);
        for (uint32_t p = 0; p < 3; p++)
            hx_fp16_op(&g, 0, p, 3);
        hx_fp16_complete(&g, 0);
        hx_fp16_export(&g, 1, actual, NULL);
        const float *expected = reference + t[1].offset / 4;
        int mismatch = 0;
        for (uint32_t i = 0; i < t[1].h * t[1].w * t[1].stride; i++) {
            if (fabsf(actual[i] - expected[i]) > 0.004f) {
                printf("FAIL gather mode 1 mismatch at %u: got %f exp %f\n", (unsigned)i, actual[i], expected[i]);
                mismatch = 1;
                break;
            }
        }
        errors += mismatch;
    }

    /* 2. HX_GATHER Mode 2 (constant indices, runtime table) */
    {
        memset(constants, 0, 0x100000);
        memset(reference, 0, 0x200000);
        memset(actual, 0, 0x200000);
        HxHeader header = {0};
        header.tensors = 2;
        header.inputs = 1;
        header.outputs = 1;
        header.ops = 1;
        uint32_t io[2] = {0, 1};
        uint32_t vocab = 8, dim = 32;
        HxTensor t[2] = {
            {0, vocab, 1, dim, dim, {0}}, /* table in arena [8, 1, 32] */
            {0x40000, 3, 1, dim, dim, {0}} /* output [3, 1, 32] */
        };
        HxOp o = {0};
        o.kind = HX_GATHER;
        o.x = 0;
        o.y = o.z = HXFG_NONE;
        o.out = 1;
        o.p[0] = 2; /* Mode 2: constant indices */
        o.p[2] = vocab;
        o.p[3] = dim;
        o.p[4] = 0; /* axis */
        o.p[5] = dim; /* inner */
        o.p[6] = 1; /* outer */
        o.p[7] = dim; /* stride */
        o.p[8] = 0x10000; /* const_idx_off */
        o.p[9] = 3; /* const_idx_num */
        int32_t *c_idx = (int32_t *)(constants + 0x10000);
        c_idx[0] = 5; c_idx[1] = 1; c_idx[2] = 6;
        float *tbl_in = reference + t[0].offset / 4;
        for (uint32_t i = 0; i < vocab * dim; i++)
            tbl_in[i] = ((int)(i % 17) - 8) * 0.25f;
        HxGraph g = {&header, t, &o, io, constants, (uint8_t *)reference, NULL};
        for (uint32_t p = 0; p < 3; p++)
            hx_execute_op(&g, 0, p, 3);
        g.arena = arena;
        if (!hx_fp16_plan(&g, &plan, tables, arena + vtcm_size)) {
            printf("FAIL plan gather mode 2\n");
            return 1;
        }
        g.fp16 = &plan;
        hx_fp16_stage(&g, NULL);
        hx_fp16_import(&g, 0, tbl_in, NULL);
        for (uint32_t p = 0; p < 3; p++)
            hx_fp16_op(&g, 0, p, 3);
        hx_fp16_complete(&g, 0);
        hx_fp16_export(&g, 1, actual, NULL);
        const float *expected = reference + t[1].offset / 4;
        int mismatch = 0;
        for (uint32_t i = 0; i < t[1].h * t[1].w * t[1].stride; i++) {
            if (fabsf(actual[i] - expected[i]) > 0.004f) {
                printf("FAIL gather mode 2 mismatch at %u: got %f exp %f\n", (unsigned)i, actual[i], expected[i]);
                mismatch = 1;
                break;
            }
        }
        errors += mismatch;
    }

    /* 3. HX_GATHER_ELEMENTS */
    {
        memset(constants, 0, 0x100000);
        memset(reference, 0, 0x200000);
        memset(actual, 0, 0x200000);
        HxHeader header = {0};
        header.tensors = 3;
        header.inputs = 2;
        header.outputs = 1;
        header.ops = 1;
        uint32_t io[3] = {0, 1, 2};
        HxTensor t[3] = {
            {0, 4, 4, 32, 32, {0}}, /* data */
            {0x20000, 4, 4, 32, 32, {0}}, /* indices */
            {0x40000, 4, 4, 32, 32, {0}}  /* output */
        };
        HxOp o = {0};
        o.kind = HX_GATHER_ELEMENTS;
        o.x = 0;
        o.y = 1;
        o.z = HXFG_NONE;
        o.out = 2;
        o.p[0] = 1; /* axis = 1 */
        o.p[1] = 1; /* index_dtype = 1 (float32) */
        o.p[2] = 0; /* mode = 0 (tensors in arena) */
        o.p[3] = 0;
        float *data = reference + t[0].offset / 4;
        float *idx = reference + t[1].offset / 4;
        for (uint32_t i = 0; i < 4 * 4 * 32; i++) {
            data[i] = ((int)(i % 19) - 9) * 0.25f;
            idx[i] = (float)((i / 32) % 4); /* column indices 0..3 */
        }
        HxGraph g = {&header, t, &o, io, constants, (uint8_t *)reference, NULL};
        for (uint32_t p = 0; p < 3; p++)
            hx_execute_op(&g, 0, p, 3);
        g.arena = arena;
        if (!hx_fp16_plan(&g, &plan, tables, arena + vtcm_size)) {
            printf("FAIL plan gather_elements\n");
            return 1;
        }
        g.fp16 = &plan;
        hx_fp16_stage(&g, NULL);
        hx_fp16_import(&g, 0, data, NULL);
        hx_fp16_import(&g, 1, idx, NULL);
        for (uint32_t p = 0; p < 3; p++)
            hx_fp16_op(&g, 0, p, 3);
        hx_fp16_complete(&g, 0);
        hx_fp16_export(&g, 2, actual, NULL);
        const float *expected = reference + t[2].offset / 4;
        int mismatch = 0;
        for (uint32_t i = 0; i < t[2].h * t[2].w * t[2].stride; i++) {
            if (fabsf(actual[i] - expected[i]) > 0.004f) {
                printf("FAIL gather_elements mismatch at %u: got %f exp %f\n", (unsigned)i, actual[i], expected[i]);
                mismatch = 1;
                break;
            }
        }
        errors += mismatch;
    }

    /* 4. HX_GATHER_ND */
    {
        memset(constants, 0, 0x100000);
        memset(reference, 0, 0x200000);
        memset(actual, 0, 0x200000);
        HxHeader header = {0};
        header.tensors = 3;
        header.inputs = 2;
        header.outputs = 1;
        header.ops = 1;
        uint32_t io[3] = {0, 1, 2};
        HxTensor t[3] = {
            {0, 4, 4, 32, 32, {0}}, /* data */
            {0x20000, 2, 1, 2, 32, {0}}, /* indices: 2 slices, k=2 */
            {0x40000, 2, 1, 32, 32, {0}}  /* output */
        };
        HxOp o = {0};
        o.kind = HX_GATHER_ND;
        o.x = 0;
        o.y = 1;
        o.z = HXFG_NONE;
        o.out = 2;
        o.p[0] = 0; /* batch_dims */
        o.p[1] = 1; /* index_dtype = 1 (float32) */
        o.p[2] = 0; /* mode = 0 (tensors in arena) */
        o.p[3] = 0; /* const offset */
        o.p[4] = 2; /* k */
        o.p[5] = 32; /* slice_size */
        float *data = reference + t[0].offset / 4;
        float *idx = reference + t[1].offset / 4;
        for (uint32_t i = 0; i < 4 * 4 * 32; i++)
            data[i] = ((int)(i % 23) - 10) * 0.25f;
        /* Slice 0: row 1, col 2 -> (1, 2) */
        idx[0] = 1.0f; idx[1] = 2.0f;
        /* Slice 1: row 3, col 0 -> (3, 0) */
        idx[32] = 3.0f; idx[33] = 0.0f;
        HxGraph g = {&header, t, &o, io, constants, (uint8_t *)reference, NULL};
        for (uint32_t p = 0; p < 3; p++)
            hx_execute_op(&g, 0, p, 3);
        g.arena = arena;
        if (!hx_fp16_plan(&g, &plan, tables, arena + vtcm_size)) {
            printf("FAIL plan gather_nd\n");
            return 1;
        }
        g.fp16 = &plan;
        hx_fp16_stage(&g, NULL);
        hx_fp16_import(&g, 0, data, NULL);
        hx_fp16_import(&g, 1, idx, NULL);
        for (uint32_t p = 0; p < 3; p++)
            hx_fp16_op(&g, 0, p, 3);
        hx_fp16_complete(&g, 0);
        hx_fp16_export(&g, 2, actual, NULL);
        const float *expected = reference + t[2].offset / 4;
        int mismatch = 0;
        for (uint32_t i = 0; i < t[2].h * t[2].w * t[2].stride; i++) {
            if (fabsf(actual[i] - expected[i]) > 0.004f) {
                printf("FAIL gather_nd mismatch at %u: got %f exp %f\n", (unsigned)i, actual[i], expected[i]);
                mismatch = 1;
                break;
            }
        }
        errors += mismatch;
    }

    /* 5. HX_SCATTER_ND */
    {
        memset(constants, 0, 0x100000);
        memset(reference, 0, 0x200000);
        memset(actual, 0, 0x200000);
        HxHeader header = {0};
        header.tensors = 3;
        header.inputs = 2;
        header.outputs = 1;
        header.ops = 1;
        uint32_t io[3] = {0, 1, 2};
        HxTensor t[3] = {
            {0, 4, 4, 32, 32, {0}}, /* data */
            {0x20000, 2, 1, 32, 32, {0}}, /* updates */
            {0x40000, 4, 4, 32, 32, {0}}  /* output */
        };
        HxOp o = {0};
        o.kind = HX_SCATTER_ND;
        o.x = 0;
        o.y = HXFG_NONE;
        o.z = 1; /* updates is z */
        o.out = 2;
        o.p[0] = 0; /* mode: out distinct from x */
        o.p[1] = 0; /* index_dtype: 0 int32 */
        o.p[2] = 2; /* k */
        o.p[3] = 32; /* slice_size */
        o.p[4] = 0; /* reduction: none */
        o.p[5] = 1; /* const_mask: bit 0 = indices in constants */
        o.p[6] = 0x10000; /* idx_offset in constants */
        o.p[7] = 0; /* const_upd_off */
        o.p[8] = 2; /* num_idx */
        int32_t *c_idx = (int32_t *)(constants + 0x10000);
        c_idx[0] = 1; c_idx[1] = 2; /* (1, 2) */
        c_idx[2] = 2; c_idx[3] = 3; /* (2, 3) */
        float *data = reference + t[0].offset / 4;
        float *updates = reference + t[1].offset / 4;
        for (uint32_t i = 0; i < 4 * 4 * 32; i++)
            data[i] = 1.0f;
        for (uint32_t i = 0; i < 2 * 32; i++)
            updates[i] = 5.0f;
        HxGraph g = {&header, t, &o, io, constants, (uint8_t *)reference, NULL};
        for (uint32_t p = 0; p < 3; p++)
            hx_execute_op(&g, 0, p, 3);
        g.arena = arena;
        if (!hx_fp16_plan(&g, &plan, tables, arena + vtcm_size)) {
            printf("FAIL plan scatter_nd\n");
            return 1;
        }
        g.fp16 = &plan;
        hx_fp16_stage(&g, NULL);
        hx_fp16_import(&g, 0, data, NULL);
        hx_fp16_import(&g, 1, updates, NULL);
        for (uint32_t p = 0; p < 3; p++)
            hx_fp16_op(&g, 0, p, 3);
        hx_fp16_complete(&g, 0);
        hx_fp16_export(&g, 2, actual, NULL);
        const float *expected = reference + t[2].offset / 4;
        int mismatch = 0;
        for (uint32_t i = 0; i < t[2].h * t[2].w * t[2].stride; i++) {
            if (fabsf(actual[i] - expected[i]) > 0.004f) {
                printf("FAIL scatter_nd mismatch at %u: got %f exp %f\n", (unsigned)i, actual[i], expected[i]);
                mismatch = 1;
                break;
            }
        }
        errors += mismatch;
    }

    free(actual);
    free(reference);
    free(constants);
    return errors;
}

static int check_sdpa_vector(uint32_t base, uint32_t causal, uint32_t dk, uint32_t dv) {
    uint8_t *constants = aligned_alloc(128, 0x100000);
    float *reference = aligned_alloc(128, 0x100000), *actual = aligned_alloc(128, 0x100000);
    if (!constants || !reference || !actual) return 1;
    memset(constants, 0, 0x100000);
    memset(reference, 0, 0x100000);
    HxHeader header = {0};
    header.tensors = 4; header.inputs = 3; header.outputs = 1; header.ops = 1;
    uint32_t io[4] = {0, 1, 2, 3};
    const uint32_t sd = (dk + 31u) & ~31u, sv = (dv + 31u) & ~31u;
    HxTensor t[4] = {{0, 7, 1, dk, sd, {0}},
                     {0x20000, 11, 1, dk, sd, {0}},
                     {0x40000, 11, 1, dv, sv, {0}},
                     {0x60000, 7, 1, dv, sv, {0}}};
    HxOp o = {0};
    o.kind = HX_SDPA; o.x = 0; o.y = 1; o.z = 2; o.out = 3;
    o.p[0] = HXFG_NONE; o.p[2] = causal;
    HxGraph g = {&header, t, &o, io, constants, (uint8_t *)reference, NULL};
    static uint32_t tables[4096];
    static struct HxFp16Plan plan;
    if (!hx_fp16_plan(&g, &plan, tables, (uint8_t *)base + 0x800000)) {
        printf("FAIL SDPA plan dk%u dv%u\n", (unsigned)dk, (unsigned)dv);
        return 1;
    }
    g.fp16 = &plan;
    hx_fp16_stage(&g, NULL);
    for (uint32_t n = 0; n < 3; n++) {
        float *data = reference + t[n].offset / 4u;
        for (uint32_t row = 0; row < t[n].h; row++)
            for (uint32_t c = 0; c < t[n].stride; c++)
                data[row * t[n].stride + c] = c < t[n].c ?
                    ((int)((row * 7u + c * 3u + n) % 9u) - 4) * 0.125f : 0.0f;
        g.arena = (uint8_t *)base;
        hx_fp16_import(&g, n, data, NULL);
    }
    g.arena = (uint8_t *)reference;
    for (uint32_t p = 0; p < 3; p++) hx_execute_op(&g, 0, p, 3);
    g.arena = (uint8_t *)base;
    for (uint32_t p = 0; p < 3; p++) hx_fp16_op(&g, 0, p, 3);
    hx_fp16_export(&g, 3, actual, NULL);
    const float *expected = reference + t[3].offset / 4u;
    int bad = 0;
    for (uint32_t row = 0; row < t[3].h && !bad; row++)
        for (uint32_t c = 0; c < dv; c++)
            if (fabsf(actual[row * sv + c] - expected[row * sv + c]) > .004f) {
                printf("FAIL SDPA causal%u dk%u dv%u row%u c%u got%f expected%f\n",
                       (unsigned)causal, (unsigned)dk, (unsigned)dv, (unsigned)row,
                       (unsigned)c, actual[row * sv + c], expected[row * sv + c]);
                bad = 1; break;
            }
    free(actual); free(reference); free(constants);
    return bad;
}

int main(void) {
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < 0x800000; i += 4096)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" ::"r"(ssr) : "memory");
    unsigned errors = 0;
#ifdef HX_MERCURY_QUICK
    errors += check(base, HX_BINARY, 16, 16, 32, 32, 0, 3, HX_BIN_MUL);
    printf("Mercury scalar multiply 16x16x32: %llu pcycles\n", (unsigned long long)op_pcycles);
    errors += check(base, HX_ZEROINSERT, 10, 10, 63, 63, 1, 2, 0);
    printf("Mercury zero insertion 10x10x63 to 21x21x63: %llu pcycles\n",
           (unsigned long long)op_pcycles);
    pad_shift = UINT32_MAX;
    errors += check(base, HX_CONV, 8, 8, 32, 32, 3, 1, 0);
    printf("Mercury pad-two 3x3 conv 8x8x32 to 10x10x32: %llu pcycles\n",
           (unsigned long long)op_pcycles);
    pad_shift = 0;
    puts(errors ? "FAIL Mercury quick" : "PASS Mercury quick");
    return errors != 0;
#elif defined(HX_VECTOR_QUICK)
    errors += check_sdpa_vector(base, 0, 64, 32);
    errors += check_sdpa_vector(base, 1, 40, 33);
    errors += check(base, HX_DEPTHWISE, 9, 11, 40, 40, 7, 1, 0);
    errors += check(base, HX_DEPTHWISE, 9, 11, 33, 33, 3, 3, 0);
    dilation_z = (2u << 8) | 1u;
    errors += check(base, HX_DEPTHWISE, 9, 11, 40, 40, 3, 1, 0);
    dilation_z = HXFG_NONE;
    errors += check(base, HX_CONV, 8, 8, 32, 32, 1, 2, 0);
    errors += check(base, HX_CONV, 16, 12, 64, 40, 1, 2, 0);
    slice_channel_offset = 32;
    errors += check(base, HX_SLICE, 7, 9, 64, 32, 1, 1, 0);
    errors += check(base, HX_SLICE, 8, 10, 96, 40, 1, 1, 0);
    errors += check(base, HX_NORM, 7, 9, 40, 40, 1, 1, HX_NORM_CHANNEL);
    errors += check(base, HX_NORM, 5, 6, 64, 64, 1, 1, HX_NORM_CHANNEL);
    errors += check(base, HX_ATTN_SOFTMAX, 7, 9, 40, 40, 1, 0, 0);
    for (uint32_t mode = 1; mode <= 3; mode++)
        errors += check(base, HX_ATTN_SOFTMAX, 7, 9, 40, 40, 1, mode, 0);
    errors += check(base, HX_ATTN_SOFTMAX, 7, 9, 40, 40, 1, 7, 0);
    errors += check(base, HX_ATTN_SOFTMAX, 7, 9, 40, 40, 1, 8, 0);
    errors += check(base, HX_ATTN_SOFTMAX, 7, 9, 40, 40, 1, 11, 0);
    for (uint32_t axes = 1; axes <= 3; axes++)
        for (uint32_t red = 0; red < HX_RED_COUNT; red++)
            errors += check(base, HX_REDUCE, 7, 9, 40, 40, axes, 1, red);
    for (uint32_t fn = 0; fn < HX_ACT_COUNT; fn++)
        errors += check(base, HX_ACT, 7, 9, 40, 40, 1, 1, fn);
    if (!errors) puts("PASS vector quick");
    return errors ? 1 : 0;
#endif
    for (uint32_t ci = 24; ci <= 64; ci += 8) {
        errors += check(base, HX_CONV, 7, 9, ci, 35, 1, 1, 0);
        for (uint32_t k = 3; k <= 5; k += 2)
            for (uint32_t s = 1; s <= 2; s++)
                errors += check(base, HX_DEPTHWISE, 7, 9, ci, ci, k, s, 0);
        errors += check(base, HX_MAXPOOL, 7, 9, ci, ci, 2, 2, 0);
        errors += check(base, HX_ADD, 7, 9, ci, ci, 1, 1, 0);
        errors += check(base, HX_PRELU, 7, 9, ci, ci, 1, 1, 0);
        errors += check(base, HX_PAD, 7, 9, ci, ci, 1, 1, 0);
        errors += check(base, HX_RESIZE, 7, 9, ci, ci, 1, 1, 0);
        errors += check(base, HX_RESIZE, 7, 9, ci, ci, 1, 1, 1);
        errors += check(base, HX_RESIZE, 7, 9, ci, ci, 1, 1, 2);
        errors += check(base, HX_RESIZE, 7, 9, ci, ci, 1, 1, 3);
        errors += check(base, HX_MEAN, 7, 9, ci, ci, 1, 1, 0);
        for (uint32_t axis = 0; axis < 3; axis++)
            errors += check(base, HX_CONCAT, 7, 9, ci, ci, 1, 1, axis);
    }
    for (uint32_t k = 3; k <= 5; k += 2)
        for (uint32_t step = 1; step <= 2; step++) {
            errors += check(base, HX_DEPTHWISE, 16, 18, 24, 24, k, step, 0);
            errors += check(base, HX_DEPTHWISE, 16, 18, 64, 64, k, step, 0);
        }
    /* Every kernel, stride and parity, with and without a padded window. */
    pad_shift = 1;
    for (uint32_t k = 3; k <= 5; k += 2)
        for (uint32_t step = 1; step <= 2; step++) {
            errors += check(base, HX_DEPTHWISE, 7, 9, 24, 24, k, step, 0);
            errors += check(base, HX_DEPTHWISE, 16, 18, 64, 64, k, step, 0);
        }
    pad_shift = 0;
    errors += check(base, HX_RESHAPE, 7, 9, 36, 18, 1, 1, 0);
    errors += check(base, HX_RESHAPE, 3, 3, 108, 18, 1, 1, 0);
    errors += check(base, HX_RESHAPE, 7, 9, 36, 18, 1, 1, 1); /* NCHW flatten, 2268 channels */
    errors += check(base, HX_RESHAPE, 5, 6, 33, 33, 1, 1, 1);
    errors += check(base, HX_RESHAPE, 7, 9, 36, 18, 1, 1, 2); /* NCHW unflatten */
    errors += check(base, HX_RESHAPE, 5, 6, 33, 33, 1, 1, 2);
    for (uint32_t perm = 0; perm < 6; perm++) {
        errors += check(base, HX_TRANSPOSE, 7, 9, 40, 40, 1, 1, perm);
        errors += check(base, HX_TRANSPOSE, 5, 6, 33, 33, 1, 1, perm);
    }
    errors += check(base, HX_CONV, 8, 8, 24, 24, 1, 1, 0);
    errors += check(base, HX_CONV, 8, 8, 32, 32, 1, 1, 0);
    errors += check(base, HX_LOGISTIC, 1, 1, 1, 1, 1, 1, 0);
    errors += check(base, HX_SLICE, 7, 9, 40, 20, 1, 1, 0);
    errors += check(base, HX_SLICE, 8, 8, 33, 17, 1, 1, 0);
    slice_channel_offset = 32;
    errors += check(base, HX_SLICE, 7, 9, 64, 32, 1, 1, 0);
    errors += check(base, HX_SLICE, 8, 10, 96, 40, 1, 1, 0);
    slice_channel_offset = 5;
    errors += check(base, HX_ZEROINSERT, 7, 9, 40, 40, 1, 2, 0);
    errors += check(base, HX_ZEROINSERT, 8, 8, 33, 33, 1, 3, 0);
    errors += check(base, HX_ZEROINSERT, 5, 6, 24, 24, 1, 1, 0);
    errors += check(base, HX_ZEROINSERT, 10, 10, 63, 63, 1, 2, 0);
    printf("Mercury zero insertion 10x10x63 to 21x21x63: %llu pcycles\n",
           (unsigned long long)op_pcycles);
    pad_shift = UINT32_MAX;
    errors += check(base, HX_CONV, 8, 8, 32, 32, 3, 1, 0);
    pad_shift = 0;
    /* Normalisation: channel mode; LayerNorm over C,H,W (1 group); two groups; InstanceNorm
     * (one group per channel); three groups over two channel blocks. */
    errors += check(base, HX_NORM, 7, 9, 40, 40, 1, 1, HX_NORM_CHANNEL);
    errors += check(base, HX_NORM, 5, 6, 64, 64, 1, 1, HX_NORM_CHANNEL);
    errors += check(base, HX_NORM, 7, 9, 40, 40, 1, 1, HX_NORM_GROUP);
    errors += check(base, HX_NORM, 7, 9, 40, 40, 1, 2, HX_NORM_GROUP);
    errors += check(base, HX_NORM, 7, 9, 40, 40, 1, 40, HX_NORM_GROUP);
    errors += check(base, HX_NORM, 8, 8, 33, 33, 1, 3, HX_NORM_GROUP);
    /* General reductions: every HWC subset (kh bitmask) and operation (axis), odd width. */
    for (uint32_t mask = 1; mask < 8; mask++)
        for (uint32_t red = 0; red < HX_RED_COUNT; red++)
            errors += check(base, HX_REDUCE, 7, 9, 40, 40, mask, 1, red);
    for (uint32_t last = 0; last <= 1; last++) {
        errors += check(base, HX_ARGMAX, 7, 9, 40, 1, 1, last, 0);
        errors += check(base, HX_ARGMIN, 7, 9, 40, 1, 1, last, 0);
    }
    /* Softmax over channels per pixel: odd width (pad column), one and two channel blocks. */
    errors += check(base, HX_SOFTMAX, 7, 9, 40, 40, 1, 1, 0);
    errors += check(base, HX_SOFTMAX, 7, 9, 3, 3, 1, 1, 0);
    errors += check(base, HX_SOFTMAX, 8, 8, 64, 64, 1, 1, 0);
    errors += check(base, HX_ATTN_SOFTMAX, 7, 9, 40, 40, 1, 0, 0);
    errors += check(base, HX_ATTN_SOFTMAX, 8, 8, 64, 64, 1, 0, 0);
    for (uint32_t mode = 1; mode <= 3; mode++)
        errors += check(base, HX_ATTN_SOFTMAX, 7, 9, 40, 40, 1, mode, 0);
    errors += check(base, HX_ATTN_SOFTMAX, 7, 9, 40, 40, 1, 7, 0);
    errors += check(base, HX_CONV, 1, 1, 1152, 63, 1, 1, 0);
    /* Dense convolutions packed as im2col: kernel, stride, padding, channel blocks and tails. */
    static const struct {
        uint32_t h, w, ci, co, k, s;
    } convs[] = {
        {7, 9, 24, 35, 3, 1}, {7, 9, 40, 35, 3, 1}, {7, 9, 5, 8, 3, 1},  {8, 8, 32, 32, 3, 2},
        {8, 8, 32, 32, 1, 2}, {16, 12, 64, 40, 1, 2},
        {7, 9, 24, 35, 3, 2}, {9, 7, 33, 4, 3, 2},  {7, 9, 24, 35, 5, 1}, {16, 18, 32, 32, 5, 2},
        {16, 18, 16, 32, 7, 2}, {7, 9, 40, 35, 1, 2}, {5, 5, 64, 40, 3, 1}, {1, 1, 32, 32, 3, 1},
    };
    for (unsigned i = 0; i < sizeof convs / sizeof convs[0]; i++) {
        errors += check(base, HX_CONV, convs[i].h, convs[i].w, convs[i].ci, convs[i].co,
                        convs[i].k, convs[i].s, 0);
        pad_shift = convs[i].k > 1; /* no leading padding: SAME_UPPER style */
        errors += check(base, HX_CONV, convs[i].h, convs[i].w, convs[i].ci, convs[i].co,
                        convs[i].k, convs[i].s, 0);
        pad_shift = 0;
    }
    /* Several batches: 18 K-tiles per output tile, so a workspace near 0x42000 holds about seven output tiles of nine. */
    vtcm_size = 0x70000;
    errors += check(base, HX_CONV, 16, 18, 64, 40, 3, 1, 0);
    vtcm_size = 0x800000;
    /* Depthwise shapes outside the hand-scheduled kernels use the scalar lanes. */
    for (uint32_t step = 1; step <= 3; step++) {
        errors += check(base, HX_DEPTHWISE, 9, 11, 40, 40, 7, step, 0);
        errors += check(base, HX_DEPTHWISE, 9, 11, 33, 33, 3, 3, 0);
        errors += check(base, HX_DEPTHWISE, 9, 11, 33, 33, 5, 4, 0);
    }
    /* Standalone activations, broadcasting binary operations and average pooling. */
    for (uint32_t fn = 0; fn < HX_ACT_COUNT; fn++)
        errors += check(base, HX_ACT, 7, 9, 40, 40, 1, 1, fn);
    for (uint32_t opcode = 0; opcode < HX_BIN_COUNT; opcode++)
        for (uint32_t mode = 0; mode <= 5; mode++)
            for (uint32_t reversed = 0; reversed <= 1; reversed++)
                errors += check(base, HX_BINARY, 7, 9, 40, 40, reversed, mode, opcode);
    errors += check(base, HX_BINARY, 16, 16, 32, 32, 0, 3, HX_BIN_MUL);
    printf("Mercury scalar multiply 16x16x32: %llu pcycles\n", (unsigned long long)op_pcycles);
    for (uint32_t include = 0; include <= 1; include++) {
        errors += check(base, HX_AVGPOOL, 7, 9, 40, 40, 3, 1, include);
        errors += check(base, HX_AVGPOOL, 8, 8, 33, 33, 2, 2, include);
    }
    /* Dilated convolution, depthwise, and pooling. */
    dilation_z = (2u << 8) | 2u;
    errors += check(base, HX_MAXPOOL, 9, 11, 40, 40, 3, 1, 0);
    errors += check(base, HX_AVGPOOL, 9, 11, 40, 40, 3, 1, 0);
    dilation_z = (2u << 8) | 2u;
    errors += check(base, HX_CONV, 9, 11, 40, 40, 3, 1, 0);
    errors += check(base, HX_CONV, 10, 12, 33, 33, 3, 2, 0);
    dilation_z = (1u << 8) | 3u;
    errors += check(base, HX_CONV, 9, 11, 24, 24, 3, 1, 0);
    dilation_z = (3u << 8) | 3u;
    errors += check(base, HX_CONV, 11, 9, 32, 40, 5, 2, 0); /* 25*32*64 floats stay below the bias at 0x50000 */
    dilation_z = (2u << 8) | 1u;
    errors += check(base, HX_DEPTHWISE, 9, 11, 40, 40, 3, 1, 0);
    dilation_z = (3u << 8) | 3u;
    errors += check(base, HX_DEPTHWISE, 16, 18, 64, 64, 5, 2, 0);
    dilation_z = HXFG_NONE;
    for (uint32_t kh = 3; kh <= 5; kh += 2) {
        errors += check_rgb(base, 15, 18, kh, 35);
        errors += check_rgb(base, 16, 16, kh, 24);
        errors += check_rgb(base, 13, 17, kh, 64);
    }
    /* LFM ops in the FP16 plan: RMSNorm (HVX and scalar), RoPE (HVX head_dim 64 and generic),
     * quantised linear (Q4_0 and Q8_0). */
    errors += check(base, HX_RMSNORM, 3, 1, 64, 64, 1, 1, 0);
    errors += check(base, HX_RMSNORM, 5, 1, 2048, 2048, 1, 1, 1);
    {
        const int e = check(base, HX_RMSNORM, 1, 1, 2048, 2048, 1, 1, 1);
        errors += e;
        printf("RMSNORM D2048 1 token: %s, %llu pcycles (3 parts)\n", e ? "FAIL" : "ok",
               (unsigned long long)op_pcycles);
    }
    errors += check(base, HX_RMSNORM, 4, 1, 33, 33, 1, 1, 1);
    errors += check(base, HX_RMSNORM, 3, 2, 40, 40, 1, 1, 0);
    errors += check(base, HX_RMSNORM, 3, 1, 64, 64, 1, 1, 2);
    errors += check(base, HX_RMSNORM, 4, 1, 33, 33, 1, 1, 2);
    errors += check(base, HX_ROPE, 3, 1, 128, 128, 1, 64, 5 << 1);
    {
        const int e = check(base, HX_ROPE, 1, 1, 2048, 2048, 1, 64, 17 << 1);
        errors += e;
        printf("ROPE 32 heads x 64, 1 token: %s, %llu pcycles (3 parts)\n", e ? "FAIL" : "ok",
               (unsigned long long)op_pcycles);
    }
    errors += check(base, HX_ROPE, 2, 1, 96, 96, 1, 32, 3 << 1);
    errors += check(base, HX_ROPE, 2, 1, 128, 128, 1, 64, (9 << 1) | 1);
    /* Theta 1e6 (0x49742400), partial rotation and a position that advances with the row. */
    rope_theta_bits = 0x49742400u;
    errors += check(base, HX_ROPE, 1, 1, 128, 128, 1, 64, 11 << 1);
    rope_rot = 32;
    errors += check(base, HX_ROPE, 1, 1, 64, 64, 1, 64, 5 << 1);
    errors += check(base, HX_ROPE, 2, 1, 256, 256, 1, 128, (9 << 1) | 1);
    rope_row_pos = 1;
    errors += check(base, HX_ROPE, 5, 1, 128, 128, 1, 64, 3 << 1);
    rope_rot = 0;
    errors += check(base, HX_ROPE, 5, 1, 64, 64, 1, 64, 0);
    rope_theta_bits = 0;
    errors += check(base, HX_ROPE, 4, 1, 64, 64, 1, 64, 7 << 1);
    rope_row_pos = 0;
    /* Several token rows through one quantized linear. */
    for (uint32_t fmt = 0; fmt <= 1; fmt++) {
        errors += check(base, HX_LINEAR_Q, 3, 1, 256, 64, fmt, 1, 0);
        errors += check(base, HX_LINEAR_Q, 5, 1, 96, 32, fmt, 1, 3);
    }
    for (uint32_t fmt = 0; fmt <= 1; fmt++)
        for (uint32_t act = 0; act <= 3; act += 3) {
            errors += check(base, HX_LINEAR_Q, 1, 1, 256, 64, fmt, 1, act);
            errors += check(base, HX_LINEAR_Q, 1, 1, 96, 32, fmt, 1, act);
        }
    /* Persistent state across steps: conv ring plus KV append, FP16 plan versus reference. */
    errors += check_state(base, 64, 3, 6, 0, 0);
    errors += check_state(base, 96, 3, 4, 0, 0);
    errors += check_state(base, 64, 7, 4, 0, 0);
    errors += check_state(base, 64, 3, 4, 1, 0);
    errors += check_state(base, 64, 3, 4, 0, 3);
    errors += check_state(base, 96, 7, 9, 1, 3);
    errors += check_state(base, 33, 4, 6, 0, 3);
    errors += check_state(base, 2048, 7, 8, 1, 0);
    errors += check_persistent_rejected(base);
    /* Matrix product of two runtime tensors: transB 0 and 1, scale 1 and 0.5, odd sizes. */
    for (uint32_t trans = 0; trans <= 1; trans++)
        for (uint32_t scaled = 0; scaled <= 1; scaled++) {
            errors += check(base, HX_MATMUL, 7, 1, 33, 19, 1, trans, scaled);
            errors += check(base, HX_MATMUL, 40, 1, 64, 40, 1, trans, scaled);
            errors += check(base, HX_MATMUL, 1, 1, 96, 65, 1, trans, scaled);
        }
    /* Attention-sized products: Q.K^T (K = 64) and attn.V (K = 197 tokens), with timing. */
    static const struct {
        uint32_t l, k, n;
    } attn[] = {{197, 64, 197}, {197, 197, 64}};
    for (unsigned i = 0; i < sizeof attn / sizeof attn[0]; i++)
        for (uint32_t trans = 0; trans <= 1; trans++)
            for (uint32_t scaled = 0; scaled <= 1; scaled++) {
                const int e = check(base, HX_MATMUL, attn[i].l, 1, attn[i].k, attn[i].n, 1, trans,
                                    scaled);
                printf("MATMUL L%u K%u N%u transB%u scale%u: %s, %llu pcycles\n",
                       (unsigned)attn[i].l, (unsigned)attn[i].k, (unsigned)attn[i].n,
                       (unsigned)trans, (unsigned)scaled, e ? "FAIL" : "ok",
                       (unsigned long long)op_pcycles);
                errors += e;
            }
    /* Workspace slices too small for the operands: the scalar-lane product (N 64) and the
     * per-element loop (N 197) run instead, and must match the reference. */
    vtcm_size = 0x58000;
    for (uint32_t trans = 0; trans <= 1; trans++)
        errors += check(base, HX_MATMUL, 8, 1, 197, 64, 1, trans, 1);
    vtcm_size = 0x80000;
    for (uint32_t trans = 0; trans <= 1; trans++)
        errors += check(base, HX_MATMUL, 8, 1, 197, 197, 1, trans, 0);
    vtcm_size = 0x800000;
    errors += check_sdpa_vector(base, 0, 64, 32);
    errors += check_sdpa_vector(base, 1, 40, 33);
    errors += check_indexing(base);
    if (errors)
        return 1;
    puts("PASS blocked FP16 RGB entry, convolution, depthwise, elementwise, pooling, resize, transforms, "
         "reductions, matrix products and channel/pixel tails");
    return 0;
}
