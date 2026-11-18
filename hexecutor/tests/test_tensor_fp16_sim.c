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
/* Simulator pcycles spent in the FP16 operator calls of the last check(), summed over parts. */
static uint64_t op_pcycles;
/* Simulator processor cycles: the service call behind hexagon_sim_read_pcycles, inlined because
 * the tensor build does not link libhexagon. */
static uint64_t pcycles(void) {
    uint64_t c;
    __asm__ volatile("r0 = #0x52\n\ttrap0(#0x0)\n\t%0 = combine(r1,r0)" : "=r"(c) : : "r0", "r1");
    return c;
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
    if (kind == HX_CONV || kind == HX_DEPTHWISE)
        o.z = dilation_z;
    o.out = 2;
    if (kind == HX_ADD || kind == HX_CONCAT || kind == HX_MATMUL ||
        (kind == HX_BINARY && step != 4))
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
        if (axis)
            o.p[2] = 1; /* nearest; axis 0 is bilinear, half-pixel centres */
        else
            o.p[1] = 1;
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
        o.p[2] = 5;
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
        /* HX_ACT_SQRT takes |x|: signed inputs are NaN, which check() cannot compare. */
        if (kind == HX_ACT && axis == HX_ACT_SQRT)
            for (uint32_t j = 0; j < t[n].h * t[n].w * t[n].stride; j++)
                v[j] = fabsf(v[j]);
        hx_fp16_import(&g, n, v, NULL);
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
            if (fabsf(a - b) > .004f) {
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
int main(void) {
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < 0x800000; i += 4096)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" ::"r"(ssr) : "memory");
    unsigned errors = 0;
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
    errors += check(base, HX_ZEROINSERT, 7, 9, 40, 40, 1, 2, 0);
    errors += check(base, HX_ZEROINSERT, 8, 8, 33, 33, 1, 3, 0);
    errors += check(base, HX_ZEROINSERT, 5, 6, 24, 24, 1, 1, 0);
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
    /* Softmax over channels per pixel: odd width (pad column), one and two channel blocks. */
    errors += check(base, HX_SOFTMAX, 7, 9, 40, 40, 1, 1, 0);
    errors += check(base, HX_SOFTMAX, 7, 9, 3, 3, 1, 1, 0);
    errors += check(base, HX_SOFTMAX, 8, 8, 64, 64, 1, 1, 0);
    errors += check(base, HX_CONV, 1, 1, 1152, 63, 1, 1, 0);
    /* Dense convolutions packed as im2col: kernel, stride, padding, channel blocks and tails. */
    static const struct {
        uint32_t h, w, ci, co, k, s;
    } convs[] = {
        {7, 9, 24, 35, 3, 1}, {7, 9, 40, 35, 3, 1}, {7, 9, 5, 8, 3, 1},  {8, 8, 32, 32, 3, 2},
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
        for (uint32_t mode = 0; mode <= 4; mode++)
            for (uint32_t reversed = 0; reversed <= 1; reversed++)
                errors += check(base, HX_BINARY, 7, 9, 40, 40, reversed, mode, opcode);
    for (uint32_t include = 0; include <= 1; include++) {
        errors += check(base, HX_AVGPOOL, 7, 9, 40, 40, 3, 1, include);
        errors += check(base, HX_AVGPOOL, 8, 8, 33, 33, 2, 2, include);
    }
    /* Dilated convolution (im2col, incl. a strided one) and depthwise (scalar lanes). */
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
    if (errors)
        return 1;
    puts("PASS blocked FP16 RGB entry, convolution, depthwise, elementwise, pooling, resize, transforms, "
         "reductions, matrix products and channel/pixel tails");
    return 0;
}
