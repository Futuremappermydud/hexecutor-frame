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
    o.out = 2;
    if (kind == HX_ADD || kind == HX_CONCAT)
        o.y = 1;
    if (kind == HX_CONV || kind == HX_DEPTHWISE || kind == HX_MAXPOOL) {
        o.p[0] = o.p[1] = kh;
        o.p[2] = o.p[3] = step;
        o.p[4] = o.p[5] = kh / 2 - (kind == HX_DEPTHWISE ? pad_shift : 0);
        o.p[6] = 0;
        o.p[7] = 0x50000;
        o.p[8] = 3;
        t[2].h = (h + step - 1) / step;
        t[2].w = (w + step - 1) / step;
        if (kind == HX_MAXPOOL)
            o.p[6] = o.p[7] = o.p[8] = 0;
        if (kind == HX_CONV) {
            o.p[9] = 0x60000;
            o.p[10] = (ci + 31) / 32;
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
        t[2].h = 1;
        t[2].w = h * w * ci / co;
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
        o.p[1] = 1;
    } else if (kind == HX_MEAN)
        t[2].h = t[2].w = 1;
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
        for (uint32_t k = 0; k < ci; k++)
            for (uint32_t c = 0; c < co; c++)
                wp[(c / 32) * kb * 1024 + (k / 32) * 1024 + ((k % 32) / 2) * 64 + (c % 32) * 2 +
                   k % 2] = (k + c) % 2 ? 0xac00 : 0x2c00;
    }
    HxGraph g = {&header, t, &o, io, constants, arena, NULL};
    static uint32_t tables[4096];
    static struct HxFp16Plan plan;
    if (!hx_fp16_plan(&g, &plan, tables, arena + 0x800000)) {
        printf("FAIL plan kind%u\n", (unsigned)kind);
        return 1;
    }
    g.fp16 = &plan;
    hx_fp16_stage(&g, NULL);
    for (uint32_t n = 0; n < 2; n++) {
        float *v = reference + t[n].offset / 4;
        for (uint32_t i = 0; i < t[n].h * t[n].w; i++)
            for (uint32_t c = 0; c < t[n].stride; c++)
                v[i * t[n].stride + c] =
                    c < t[n].c ? ((int)((i * 7 + c * 3 + n) % 23) - 11) * .25f : 0;
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
    if (kind == HX_CONV)
        hx_fp16_conv(&g, 0, NULL);
    else
        for (uint32_t p = 0; p < 3; p++)
            hx_fp16_op(&g, 0, p, 3);
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
    if (!hx_fp16_plan(&g, &plan, tables, arena + 0x800000)) {
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
    errors += check(base, HX_CONV, 8, 8, 24, 24, 1, 1, 0);
    errors += check(base, HX_CONV, 8, 8, 32, 32, 1, 1, 0);
    errors += check(base, HX_LOGISTIC, 1, 1, 1, 1, 1, 1, 0);
    errors += check(base, HX_CONV, 1, 1, 1152, 63, 1, 1, 0);
    for (uint32_t kh = 3; kh <= 5; kh += 2) {
        errors += check_rgb(base, 15, 18, kh, 35);
        errors += check_rgb(base, 16, 16, kh, 24);
        errors += check_rgb(base, 13, 17, kh, 64);
    }
    if (errors)
        return 1;
    puts("PASS blocked FP16 RGB entry, convolution, depthwise, elementwise, pooling, resize, transforms, "
         "reductions and channel/pixel tails");
    return 0;
}
