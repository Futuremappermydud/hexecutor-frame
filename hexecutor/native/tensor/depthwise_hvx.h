/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef HEXECUTOR_TENSOR_DEPTHWISE_HVX_H
#define HEXECUTOR_TENSOR_DEPTHWISE_HVX_H
/* Keep each channel block's filter in vector registers across the pixel
 * loop. Interior pixels have no tap-boundary branches or QF/SF conversions
 * between multiply-accumulates. Border pixels share the same zero padding. */
#define HX_DW_WEIGHT(n) HVX_Vector w##n = *(const HVX_Vector *)(weights + (n) * stride + c);
#define HX_DW_TAP(n, y, x)                                                                         \
    acc = Q6_Vqf32_vadd_Vqf32Vqf32(                                                                \
        acc, Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)(src + ((y) * iw + (x)) * stride), w##n));
static void depthwise3(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t ih = a->h, iw = a->w, ow = b->w, stride = a->stride;
    const uint32_t sh = o->p[2], sw = o->p[3], ph = o->p[4], pw = o->p[5], activation = o->p[8];
    const uint32_t pixels = b->h * b->w, lo = pixels * part / parts,
                   hi = pixels * (part + 1u) / parts;
    const float *input = (const float *)(g->arena + a->offset);
    float *out = (float *)(g->arena + b->offset);
    const float *weights = (const float *)(g->constants + o->p[6]),
                *bias = (const float *)(g->constants + o->p[7]);
    for (uint32_t c = 0; c < stride; c += 32u) {
        HX_DW_WEIGHT(0)
        HX_DW_WEIGHT(1) HX_DW_WEIGHT(2) HX_DW_WEIGHT(3) HX_DW_WEIGHT(4) HX_DW_WEIGHT(5)
            HX_DW_WEIGHT(6) HX_DW_WEIGHT(7) HX_DW_WEIGHT(8) HVX_Vector initial =
                Q6_Vqf32_vadd_VsfVsf(*(const HVX_Vector *)(bias + c), Q6_V_vzero());
        for (uint32_t i = lo; i < hi; i++) {
            int32_t iy = (int32_t)((i / ow) * sh) - (int32_t)ph,
                    ix = (int32_t)((i % ow) * sw) - (int32_t)pw;
            HVX_Vector acc = initial;
            if (iy >= 0 && ix >= 0 && iy + 3 <= (int32_t)ih && ix + 3 <= (int32_t)iw) {
                const float *src = input + ((uint32_t)iy * iw + (uint32_t)ix) * stride + c;
                HX_DW_TAP(0, 0, 0)
                HX_DW_TAP(1, 0, 1) HX_DW_TAP(2, 0, 2) HX_DW_TAP(3, 1, 0) HX_DW_TAP(4, 1, 1)
                    HX_DW_TAP(5, 1, 2) HX_DW_TAP(6, 2, 0) HX_DW_TAP(7, 2, 1) HX_DW_TAP(8, 2, 2)
            } else {
                for (uint32_t ky = 0; ky < 3; ky++)
                    for (uint32_t kx = 0; kx < 3; kx++) {
                        int32_t yy = iy + (int32_t)ky, xx = ix + (int32_t)kx;
                        if (yy < 0 || xx < 0 || yy >= (int32_t)ih || xx >= (int32_t)iw)
                            continue;
                        HVX_Vector v =
                            *(const HVX_Vector *)(input +
                                                  ((uint32_t)yy * iw + (uint32_t)xx) * stride + c);
                        HVX_Vector w = *(const HVX_Vector *)(weights + (ky * 3 + kx) * stride + c);
                        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(v, w));
                    }
            }
            *(HVX_Vector *)(out + i * stride + c) = act(Q6_Vsf_equals_Vqf32(acc), activation);
        }
    }
}
static void depthwise5(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t ih = a->h, iw = a->w, ow = b->w, stride = a->stride;
    const uint32_t sh = o->p[2], sw = o->p[3], ph = o->p[4], pw = o->p[5], activation = o->p[8];
    const uint32_t pixels = b->h * b->w, lo = pixels * part / parts,
                   hi = pixels * (part + 1u) / parts;
    const float *input = (const float *)(g->arena + a->offset);
    float *out = (float *)(g->arena + b->offset);
    const float *weights = (const float *)(g->constants + o->p[6]),
                *bias = (const float *)(g->constants + o->p[7]);
    for (uint32_t c = 0; c < stride; c += 32u) {
        HX_DW_WEIGHT(0)
        HX_DW_WEIGHT(1) HX_DW_WEIGHT(2) HX_DW_WEIGHT(3) HX_DW_WEIGHT(4) HX_DW_WEIGHT(5)
            HX_DW_WEIGHT(6) HX_DW_WEIGHT(7) HX_DW_WEIGHT(8) HX_DW_WEIGHT(9) HX_DW_WEIGHT(10)
                HX_DW_WEIGHT(11) HX_DW_WEIGHT(12) HX_DW_WEIGHT(13) HX_DW_WEIGHT(14) HX_DW_WEIGHT(15)
                    HX_DW_WEIGHT(16) HX_DW_WEIGHT(17) HX_DW_WEIGHT(18) HX_DW_WEIGHT(19)
                        HX_DW_WEIGHT(20) HX_DW_WEIGHT(21) HX_DW_WEIGHT(22) HX_DW_WEIGHT(23)
                            HX_DW_WEIGHT(24) HVX_Vector initial =
                                Q6_Vqf32_vadd_VsfVsf(*(const HVX_Vector *)(bias + c), Q6_V_vzero());
        for (uint32_t i = lo; i < hi; i++) {
            int32_t iy = (int32_t)((i / ow) * sh) - (int32_t)ph,
                    ix = (int32_t)((i % ow) * sw) - (int32_t)pw;
            HVX_Vector acc = initial;
            if (iy >= 0 && ix >= 0 && iy + 5 <= (int32_t)ih && ix + 5 <= (int32_t)iw) {
                const float *src = input + ((uint32_t)iy * iw + (uint32_t)ix) * stride + c;
                HX_DW_TAP(0, 0, 0)
                HX_DW_TAP(1, 0, 1) HX_DW_TAP(2, 0, 2) HX_DW_TAP(3, 0, 3) HX_DW_TAP(4, 0, 4)
                    HX_DW_TAP(5, 1, 0) HX_DW_TAP(6, 1, 1) HX_DW_TAP(7, 1, 2) HX_DW_TAP(8, 1, 3)
                        HX_DW_TAP(9, 1, 4) HX_DW_TAP(10, 2, 0) HX_DW_TAP(11, 2, 1)
                            HX_DW_TAP(12, 2, 2) HX_DW_TAP(13, 2, 3) HX_DW_TAP(14, 2, 4)
                                HX_DW_TAP(15, 3, 0) HX_DW_TAP(16, 3, 1) HX_DW_TAP(17, 3, 2)
                                    HX_DW_TAP(18, 3, 3) HX_DW_TAP(19, 3, 4) HX_DW_TAP(20, 4, 0)
                                        HX_DW_TAP(21, 4, 1) HX_DW_TAP(22, 4, 2) HX_DW_TAP(23, 4, 3)
                                            HX_DW_TAP(24, 4, 4)
            } else {
                for (uint32_t ky = 0; ky < 5; ky++)
                    for (uint32_t kx = 0; kx < 5; kx++) {
                        int32_t yy = iy + (int32_t)ky, xx = ix + (int32_t)kx;
                        if (yy < 0 || xx < 0 || yy >= (int32_t)ih || xx >= (int32_t)iw)
                            continue;
                        HVX_Vector v =
                            *(const HVX_Vector *)(input +
                                                  ((uint32_t)yy * iw + (uint32_t)xx) * stride + c);
                        HVX_Vector w = *(const HVX_Vector *)(weights + (ky * 5 + kx) * stride + c);
                        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(v, w));
                    }
            }
            *(HVX_Vector *)(out + i * stride + c) = act(Q6_Vsf_equals_Vqf32(acc), activation);
        }
    }
}
#undef HX_DW_WEIGHT
#undef HX_DW_TAP
#endif
