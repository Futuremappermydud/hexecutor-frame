/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Q16P interpreter: one validated operation at a time, serially.
 *
 * Matrix 1x1 convolutions run through the injected HMX leaf on QNN's own
 * block tables. Other operations gather FP16 operands into logical NHWC
 * scratch, run the scalar handler and scatter the result back to QNN's
 * placement, so every tensor keeps the address QNN gave it. */
#include "program.h"
#include "qnn16_f32.h"
#ifdef __HVX__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "qnn16_tile.h"
#endif
#include <string.h>
#include <math.h>
#ifndef Q16_PARALLEL_COPY_BYTES
#define Q16_PARALLEL_COPY_BYTES 32768u
#endif

static Qnn16Tensor view(const Q16Exec *e, uint32_t id) {
    const Q16Program *p = e->program;
    const Q16Tensor *t = &p->tensors[id];
    Qnn16Tensor v;
    v.base = t->direct == 2u ? e->scratch : e->vtcm;
    v.block_offsets = t->direct ? 0 : p->words + t->block_first;
    v.block_count = t->block_count;
    v.base_bytes = t->direct == 2u ? p->header->reserved[0] : Q16P_COMPLETION_OFFSET;
    v.direct_offset = t->offset;
    v.dims = t->dims;
    v.full = t->full;
    v.padding = t->padding;
    v.direct = t->direct != 0;
    return v;
}
static uint32_t lbytes(const Q16Tensor *t) {
    return ((t->dims.b * t->dims.h * t->dims.w * t->dims.c * 2u) + 127u) & ~127u;
}
static uint16_t *take(uint8_t **cursor, const Q16Tensor *t) {
    uint16_t *at = (uint16_t *)*cursor;
    *cursor += lbytes(t);
    return at;
}
static int gather(const Q16Exec *e, uint16_t *dst, uint32_t id) {
    Qnn16Tensor v = view(e, id);
    return qnn16_gather_f16(dst, &v);
}
static int scatter(const Q16Exec *e, uint32_t id, const uint16_t *src) {
    Qnn16Tensor v = view(e, id);
    return qnn16_scatter_f16(&v, src);
}
static void *at(const Q16Exec *e, uint32_t id) {
    const Q16Tensor *t = &e->program->tensors[id];
    return (t->direct == 2u ? e->scratch : e->vtcm) + t->offset;
}

static uint8_t *bump(uint8_t **cursor, uint64_t bytes) {
    uint8_t *at = *cursor;
    *cursor += (bytes + 127u) & ~(uint64_t)127u;
    return at;
}

typedef struct {
    Q16Exec *exec;
    const Q16Op *op;
} U16PointwiseJob;

static uint16_t pw_u16_requant(int64_t value, float scale, uint32_t zero) {
    double code = value * (double)scale + zero + 0.5;
    return code <= 0 ? 0 : code >= UINT16_MAX ? UINT16_MAX : (uint16_t)code;
}

#ifdef __HVX__
typedef union {
    HVX_Vector vector;
    int32_t lane[32];
} U16DotLanes;

/* Split four U16 activations into two unsigned byte dot products. Keeping
 * the byte sums separate avoids 32-bit overflow even at K=4096. */
static void pw_accum_four(U16DotLanes sums[4], const uint16_t *in, const int8_t *packed) {
    HVX_Vector weights = *(const HVX_UVector *)packed;
    for (uint32_t token = 0; token < 2u; ++token) {
        uint32_t lo = 0, hi = 0;
        for (uint32_t k = 0; k < 4u; ++k) {
            uint32_t code = in[2u * k + token];
            lo |= (code & 255u) << (8u * k);
            hi |= (code >> 8) << (8u * k);
        }
        sums[token * 2u].vector = Q6_Vw_vrmpyacc_VwVubVb(
            sums[token * 2u].vector, Q6_V_vsplat_R(lo), weights);
        sums[token * 2u + 1u].vector = Q6_Vw_vrmpyacc_VwVubVb(
            sums[token * 2u + 1u].vector, Q6_V_vsplat_R(hi), weights);
    }
}

static int64_t pw_dot(const U16DotLanes sums[4], uint32_t token, uint32_t lane) {
    return sums[token * 2u].lane[lane] + (int64_t)sums[token * 2u + 1u].lane[lane] * 256;
}
#endif

static void pw_xsb_task(void *ctx, uint32_t part, uint32_t parts) {
    const U16PointwiseJob *job = ctx;
    Q16Exec *e = job->exec;
    const Q16Op *o = job->op;
    const Q16Tensor *t = e->program->tensors;
    const uint32_t *a = e->program->words + t[o->x].block_first;
    const uint32_t *c = e->program->words + t[o->out].block_first;
    const int8_t *weights = at(e, o->y);
    const int32_t *epilogue = at(e, o->z);
    uint32_t kb = t[o->x].dims.c / 32u, nb = t[o->out].dims.c / 32u;
    float scale;
    memcpy(&scale, &o->p[0], sizeof(scale));
    /* Fi lane scales are constant across the 19 token tiles. Decode them once
     * per output block, not once for every tile assigned to this worker. */
    for (uint32_t n = 0; n < nb; ++n) {
        const int32_t *fi = epilogue + n * 128u;
        float scales[32];
        for (uint32_t j = 0; j < 32u; ++j) {
            uint32_t lane = (j & 1u) * 64u + (j & ~1u);
            const int exponent = ((uint32_t)fi[lane] >> 10) & 31u;
            const uint32_t mantissa = 1024u + ((uint32_t)fi[lane + 1u] & 1023u);
            scales[j] = o->p[2] ? ldexpf((float)mantissa, exponent - 26) : scale;
        }
        for (uint32_t m = part; m < 19u; m += parts) {
            uint16_t *out = (uint16_t *)(e->vtcm + c[m * nb + n]);
            for (uint32_t pair = 0; pair < 16u; ++pair) {
#ifdef __HVX__
                U16DotLanes sums[4];
                for (uint32_t i = 0; i < 4u; ++i)
                    sums[i].vector = Q6_V_vzero();
                for (uint32_t k = 0; k < kb; ++k) {
                    const uint16_t *in = (const uint16_t *)(e->vtcm + a[m * kb + k]) + pair * 64u;
                    const int8_t *w = weights + (k * nb + n) * 1024u;
                    for (uint32_t g = 0; g < 8u; ++g)
                        pw_accum_four(sums, in + g * 8u, w + g * 128u);
                }
                for (uint32_t j = 0; j < 32u; ++j) {
                    int32_t bias = fi[(j & 1u) * 64u + 32u + (j & ~1u)];
                    for (uint32_t token = 0; token < 2u; ++token)
                        out[pair * 64u + j * 2u + token] =
                            pw_u16_requant((pw_dot(sums, token, j) >> 8) + bias, scales[j], o->p[1]);
                }
#else
                int64_t dot0[32] = {0}, dot1[32] = {0};
                for (uint32_t k = 0; k < kb; ++k) {
                    const uint16_t *in = (const uint16_t *)(e->vtcm + a[m * kb + k]) + pair * 64u;
                    const int8_t *w = weights + (k * nb + n) * 1024u;
                    int32_t tile0[32] = {0}, tile1[32] = {0};
                    for (uint32_t g = 0; g < 8u; ++g)
                        for (uint32_t j = 0; j < 32u; ++j)
                            for (uint32_t r = 0; r < 4u; ++r) {
                                int32_t b = w[g * 128u + j * 4u + r];
                                tile0[j] += (int32_t)in[g * 8u + r * 2u] * b;
                                tile1[j] += (int32_t)in[g * 8u + r * 2u + 1u] * b;
                            }
                    for (uint32_t j = 0; j < 32u; ++j) {
                        dot0[j] += tile0[j];
                        dot1[j] += tile1[j];
                    }
                }
                for (uint32_t j = 0; j < 32u; ++j) {
                    int32_t bias = fi[(j & 1u) * 64u + 32u + (j & ~1u)];
                    out[pair * 64u + j * 2u] = pw_u16_requant((dot0[j] >> 8) + bias, scales[j], o->p[1]);
                    out[pair * 64u + j * 2u + 1u] = pw_u16_requant((dot1[j] >> 8) + bias, scales[j], o->p[1]);
                }
#endif
            }
        }
    }
}

static void pw_xpb_task(void *ctx, uint32_t part, uint32_t parts) {
    const U16PointwiseJob *job = ctx;
    Q16Exec *e = job->exec;
    const Q16Op *o = job->op;
    const Q16Tensor *t = e->program->tensors;
    const uint32_t *a = e->program->words + t[o->x].block_first;
    const uint32_t *c = e->program->words + t[o->out].block_first;
    const int8_t *weights = at(e, o->y);
    const int32_t *epilogue = at(e, o->z);
    uint32_t kb = t[o->x].full.c / 32u, nb = t[o->out].full.c / 32u;
    uint32_t k_groups = t[o->y].full.w / 4u, n_groups = t[o->y].full.c / 32u;
    float scale;
    memcpy(&scale, &o->p[0], sizeof(scale));
    for (uint32_t m = part; m < t[o->x].dims.w / 4u; m += parts)
        for (uint32_t n = 0; n < n_groups; ++n) {
            uint16_t *out = (uint16_t *)(e->vtcm + c[m * nb + n / 2u]) + (n % 2u) * 32u;
            const int32_t *fi = epilogue + n * 64u + 32u;
            for (uint32_t pair = 0; pair < 16u; ++pair) {
#ifdef __HVX__
                U16DotLanes sums[4];
                for (uint32_t i = 0; i < 4u; ++i)
                    sums[i].vector = Q6_V_vzero();
                for (uint32_t k = 0; k < k_groups; ++k) {
                    const uint16_t *in = (const uint16_t *)(e->vtcm + a[m * kb + k / 8u]) +
                                         pair * 64u + (k % 8u) * 8u;
                    pw_accum_four(sums, in, weights + (n * k_groups + k) * 128u);
                }
                /* Xpb interleaves signed low and high weight bytes in adjacent vector lanes. */
                for (uint32_t j = 0; j < 16u; ++j)
                    for (uint32_t token = 0; token < 2u; ++token) {
                        int64_t dot = pw_dot(sums, token, j * 2u) +
                                      pw_dot(sums, token, j * 2u + 1u) * 256;
                        out[pair * 64u + j * 2u + token] =
                            pw_u16_requant((dot >> 16) + fi[j * 2u], scale, o->p[1]);
                    }
#else
                int64_t dot0[16] = {0}, dot1[16] = {0};
                for (uint32_t k = 0; k < k_groups; ++k) {
                    const uint16_t *in = (const uint16_t *)(e->vtcm + a[m * kb + k / 8u]) + pair * 64u + (k % 8u) * 8u;
                    const int8_t *w = weights + (n * k_groups + k) * 128u;
                    for (uint32_t j = 0; j < 16u; ++j)
                        for (uint32_t r = 0; r < 4u; ++r) {
                            int32_t b = w[j * 8u + r] + w[j * 8u + r + 4u] * 256;
                            dot0[j] += (int64_t)in[r * 2u] * b;
                            dot1[j] += (int64_t)in[r * 2u + 1u] * b;
                        }
                }
                for (uint32_t j = 0; j < 16u; ++j) {
                    int32_t bias = fi[j * 2u];
                    out[pair * 64u + j * 2u] = pw_u16_requant((dot0[j] >> 16) + bias, scale, o->p[1]);
                    out[pair * 64u + j * 2u + 1u] = pw_u16_requant((dot1[j] >> 16) + bias, scale, o->p[1]);
                }
#endif
            }
        }
}

static void softmax_u16(Q16Exec *e, const Q16Op *o) {
    const Q16Program *p = e->program;
    const uint32_t *a = p->words + p->tensors[o->x].block_first;
    const uint32_t *b = p->words + p->tensors[o->out].block_first;
    float scale;
    memcpy(&scale, &o->p[0], sizeof(scale));
    const uint32_t groups = 19u, channels = 577u;
    float row[577];
    for (uint32_t strip = 0; strip < p->tensors[o->x].dims.w / 4u; strip++)
        for (uint32_t token = 0; token < 32u; token++) {
            uint32_t lane = (token / 2u) * 64u + (token % 2u);
            float max = 0;
            for (uint32_t c = 0; c < channels; c++) {
                const uint16_t *in = (const uint16_t *)(e->vtcm + a[strip * groups + c / 32u]);
                float value = in[lane + (c % 32u) * 2u];
                row[c] = value;
                if (value > max)
                    max = value;
            }
            /* Reference softmax with the captured input encoding. QAIRT's
             * fixed-point exponential and reciprocal are approximate. */
            double sum = 0;
            for (uint32_t c = 0; c < channels; c++)
                sum += row[c] = expf((row[c] - max) * scale);
            double ratio = 65536.0 / sum;
            for (uint32_t c = 0; c < groups * 32u; c++) {
                uint16_t *out = (uint16_t *)(e->vtcm + b[strip * groups + c / 32u]);
                double v = c < channels ? row[c] * ratio + 0.5 : 0;
                out[lane + (c % 32u) * 2u] = v >= UINT16_MAX ? UINT16_MAX : (uint16_t)v;
            }
        }
}

static void ew_u16(Q16Exec *e, const Q16Op *o) {
    const Q16Program *p = e->program;
    const Q16Tensor *x = &p->tensors[o->x], *out = &p->tensors[o->out];
    const uint32_t *a = p->words + x->block_first, *d = p->words + out->block_first;
    const uint32_t *b = 0;
    uint16_t broadcast[64];
    float scales[2];
    if (o->kind == Q16_GELU_U16)
        memcpy(scales, o->p, sizeof(scales));
    else
        b = p->words + p->tensors[o->y].block_first;
    if (o->kind == Q16_MUL_U16)
        for (uint32_t c = 0; c < x->dims.c; c++) {
            const uint16_t *tile = (const uint16_t *)(e->vtcm + b[c / 32u]);
            broadcast[c] = tile[(c % 32u) * 2u];
        }

    for (uint32_t block = 0; block < out->block_count; block++) {
        const uint16_t *in = (const uint16_t *)(e->vtcm + a[block]);
        uint16_t *dest = (uint16_t *)(e->vtcm + d[block]);
        if (o->kind == Q16_ADD_U16) {
            const uint16_t *other = (const uint16_t *)(e->vtcm + b[block]);
            uint32_t shift = o->p[2];
            int64_t offset = (int64_t)(int32_t)o->p[3] * 2 + ((int64_t)1 << shift);
            for (uint32_t i = 0; i < 1024u; i++) {
                int64_t v = ((int64_t)in[i] * o->p[0] + (int64_t)other[i] * o->p[1] + offset) >> (shift + 1u);
                dest[i] = v < 0 ? 0 : v > UINT16_MAX ? UINT16_MAX : (uint16_t)v;
            }
        } else if (o->kind == Q16_MUL_U16) {
            const uint16_t *channels = broadcast + (block % (x->dims.c / 32u)) * 32u;
            uint32_t lo = o->p[0] & 65535u, hi = o->p[0] >> 16;
            for (uint32_t i = 0; i < 1024u; i++) {
                uint32_t u = in[i], v = channels[(i / 2u) % 32u];
                int32_t centered = (int32_t)(u * v + o->p[3] - (v * o->p[1] + u * o->p[2]));
                int64_t mixed = ((int64_t)centered * lo) >> 16;
                int64_t q = ((mixed + (int64_t)centered * hi) >> 15) + (int32_t)o->p[4];
                q = (q + 4) >> 3;
                dest[i] = q < 0 ? 0 : q > UINT16_MAX ? UINT16_MAX : (uint16_t)q;
            }
        } else {
            for (uint32_t i = 0; i < 1024u; i++) {
                double v = ((int32_t)in[i] - (int32_t)o->p[2]) * (double)scales[0];
                /* Exact GELU formula, compared with QAIRT's approximation. */
                double q = 0.5 * v * (1.0 + erff((float)(v * 0.70710678118654752))) * scales[1] + o->p[3] + 0.5;
                dest[i] = q < 0 ? 0 : q > UINT16_MAX ? UINT16_MAX : (uint16_t)q;
            }
        }
    }
}

typedef struct {
    const Q16Op *o;
    Qnn16Tensor xin, yout;
    Qnn16Shape xs;
    float *xf, *yf, *w;
    Qnn16Epilogue ep;
    uint32_t stage;
} ConvJob;
static void conv_task(void *ctx, uint32_t part, uint32_t parts) {
    ConvJob *j = (ConvJob *)ctx;
    const Qnn16Shape ys = j->yout.dims;
    if (j->stage == 0) {
        qnn16_gather_f32(j->xf, &j->xin, part, parts);
        return;
    }
    if (j->stage == 2) {
        qnn16_scatter_f32(&j->yout, j->yf, part, parts);
        return;
    }
    switch (j->o->kind) {
    case Q16_DW3X3: qnn16_dw3x3_f32(j->yf, j->xf, j->xs, j->w, j->ep, part, parts); break;
    case Q16_DW3X3_S2: qnn16_dw3x3_s2_f32(j->yf, j->xf, j->xs, ys.h, ys.w, ys.c, j->w, j->ep, part, parts); break;
    case Q16_CONV3X3_S2: qnn16_conv3x3_s2_f32(j->yf, j->xf, j->xs, ys.h, ys.w, ys.c, j->w, j->ep, part, parts); break;
    case Q16_DW1X1: qnn16_dw1x1_f32(j->yf, j->xf, j->xs, ys.c, j->w, j->ep, part, parts); break;
    case Q16_CONV3X3: qnn16_conv3x3_f32(j->yf, j->xf, j->xs, ys.c, j->w, j->ep, part, parts); break;
    case Q16_PW_SWISH: qnn16_pw_swish_f32(j->yf, j->xf, j->xs, ys.c, j->w, j->ep, part, parts); break;
    case Q16_PW_TANH: qnn16_pw_tanh_f32(j->yf, j->xf, j->xs, ys.c, j->w, j->ep, part, parts); break;
    case Q16_PW1X1: qnn16_pw_f32(j->yf, j->xf, j->xs, ys.c, j->w, j->ep, part, parts); break;
    default: qnn16_s2d_f32(j->yf, j->xf, j->xs, ys.h, ys.w, part, parts); break;
    }
}
static void run_parts(Q16Exec *e, ConvJob *j) {
    if (e->parallel)
        e->parallel(conv_task, j);
    else
        conv_task(j, 0, 1);
}
#ifdef __HVX__
typedef struct {
    Qnn16TileJob job;
    void (*fn)(const Qnn16TileJob *, uint32_t, uint32_t);
} TileTask;
static void tile_task(void *ctx, uint32_t part, uint32_t parts) {
    const TileTask *t = (const TileTask *)ctx;
    t->fn(&t->job, part, parts);
}
/* Depthwise and space-to-depth directly on VTCM tiles, when supported. */
static int tile_op(Q16Exec *e, const Q16Op *o) {
    Qnn16Tensor x = view(e, o->x), y = view(e, o->out);
    TileTask t;
    switch (o->kind) {
    case Q16_DW3X3: t.fn = qnn16_tile_dw3x3_ok(&x, &y) ? qnn16_tile_dw3x3 : 0; break;
    case Q16_DW3X3_S2: t.fn = qnn16_tile_dw3x3_s2_ok(&x, &y) ? qnn16_tile_dw3x3_s2 : 0; break;
    case Q16_DW1X1: t.fn = qnn16_tile_dw1x1_ok(&x, &y) ? qnn16_tile_dw1x1 : 0; break;
    case Q16_S2D: t.fn = qnn16_tile_s2d_ok(&x, &y) ? qnn16_tile_s2d : 0; break;
    case Q16_CONV3X3: t.fn = qnn16_tile_conv3x3_ok(&x, &y, o->p[1]) ? qnn16_tile_conv3x3 : 0; break;
    case Q16_PW1X1: t.fn = qnn16_tile_pw_ok(&x, &y, o->p[1]) ? qnn16_tile_pw : 0; break;
    case Q16_PW_SWISH: {
        const Q16Tensor *wt = &e->program->tensors[o->y], *ft = &e->program->tensors[o->z];
        /* Aligned vector loads of the weights and tables: both in VTCM on 128-byte offsets. */
        t.fn = wt->direct == 1u && ft->direct == 1u && !(wt->offset & 127u) && !(ft->offset & 127u) &&
               qnn16_tile_pw_swish_ok(&x, &y, o->p[0]) ? qnn16_tile_pw_swish : 0;
        break;
    }
    default: t.fn = 0; break;
    }
    if (!t.fn)
        return 0;
    t.job.x = &x;
    t.job.out = &y;
    t.job.weights = o->kind == Q16_S2D ? 0 : (const uint16_t *)at(e, o->y);
    t.job.fi = o->kind == Q16_S2D ? 0 : (const uint8_t *)at(e, o->z);
    t.job.tables = o->kind == Q16_S2D ? 0 : o->p[0];
    t.job.packed_ic = o->kind == Q16_CONV3X3 || o->kind == Q16_PW1X1 ? o->p[1] : o->kind == Q16_PW_SWISH ? o->p[0] : 0;
    t.job.compact = o->kind == Q16_DW3X3_S2 ? o->p[1] : 0;
    /* Even small operations gain from the workers (a serial threshold measured slower). */
    if (e->parallel)
        e->parallel(tile_task, &t);
    else
        tile_task(&t, 0, 1);
    return 1;
}
#endif
#ifdef __HVX__
/* Small tile ops split by rows over the workers (AVGPOOL runs in part 0). */
typedef struct {
    const Q16Op *o;
    Qnn16Tensor x, y, z;
    float lo, hi, v[4];
    const float *moments, *bias; /* INSTANCE_NORM */
    const uint16_t *scale;
    volatile uint32_t parts, ok[8];
} SmallJob;
static void small_task(void *ctx, uint32_t part, uint32_t parts) {
    SmallJob *j = (SmallJob *)ctx;
    const Q16Op *o = j->o;
    const int hw = o->p[0] == 0 && o->p[1] == 2 && o->p[2] == 1 && o->p[3] == 3;
    const int wc = o->p[0] == 0 && o->p[1] == 1 && o->p[2] == 3 && o->p[3] == 2;
    int ok = 0;
    switch (o->kind) {
    case Q16_MUL:
        ok = qnn16_tile_mul_scalar(&j->x, &j->z, &j->y, part, parts) || qnn16_tile_mul(&j->x, &j->z, &j->y, part, parts);
        break;
    case Q16_RESIZE_BILINEAR: ok = qnn16_tile_resize2x(&j->x, &j->y, j->v, part, parts); break;
    case Q16_RESIZE_NEAREST: ok = qnn16_tile_resize_nearest(&j->x, &j->y, o->p[0], o->p[1], part, parts); break;
    case Q16_DECIMATE: ok = qnn16_tile_decimate(&j->x, &j->y, o->p[0], o->p[1], part, parts); break;
    case Q16_SOFTMAX: ok = qnn16_tile_softmax(&j->x, &j->y, part, parts); break;
    case Q16_REDUCE_SUM: ok = part ? 1 : qnn16_tile_reduce_sum(&j->x, &j->y); break;
    case Q16_REDUCE_MEAN: ok = qnn16_tile_reduce_mean_h(&j->x, &j->y, part, parts); break;
    case Q16_L2NORM: ok = o->p[0] == 1u && qnn16_tile_l2norm(&j->x, &j->y, j->v[1], part, parts); break;
    case Q16_CAST_F16: ok = qnn16_tile_cast_f16(&j->x, &j->y, part, parts); break;
    case Q16_FORMAT: ok = qnn16_tile_format_fast(&j->x, &j->y, part, parts); break;
    case Q16_MINMAX: ok = qnn16_tile_minmax(&j->x, &j->y, j->lo, j->hi, part, parts); break;
    case Q16_SHUFFLE: ok = qnn16_tile_shuffle(&j->x, &j->y, o->p[0], part, parts); break;
    case Q16_X2S: ok = qnn16_tile_x2s(&j->x, &j->y, part, parts); break;
    case Q16_AVGPOOL: ok = part ? 1 : qnn16_tile_avgpool(&j->x, &j->y, o->p[0], o->p[1]); break;
    case Q16_SWISH: ok = qnn16_tile_swish(&j->x, &j->y, part, parts); break;
    case Q16_ADD_SWISH: ok = qnn16_tile_add_swish(&j->x, &j->y, part, parts); break;
    case Q16_INSTANCE_NORM:
        ok = qnn16_tile_instance_norm(&j->x, &j->y, j->moments, j->scale, j->bias, part, parts);
        break;
    case Q16_TRANSPOSE:
        ok = qnn16_tile_transpose(&j->x, &j->y, o->p, part, parts) ||
             (hw ? qnn16_tile_transpose_hw(&j->x, &j->y, part, parts)
                 : wc ? qnn16_tile_transpose_wc(&j->x, &j->y, part, parts) : 0);
        break;
    default: break;
    }
    if (part < 8u)
        j->ok[part] = (uint32_t)ok;
    if (!part)
        j->parts = parts;
}
/* Operands of a small tile op in j; 0 when the tile form cannot apply (nothing runs). */
static int small_prepare(Q16Exec *e, const Q16Op *o, SmallJob *j) {
    const Q16Tensor *T = e->program->tensors;
    /* HVX must stay in VTCM: no tensor in the scratch arena. */
    if (T[o->out].direct == 2u || (o->x != Q16P_NONE && T[o->x].direct == 2u) ||
        (o->kind == Q16_MUL && T[o->y].direct == 2u && q16_tensor_elements(&T[o->y]) > 64u))
        return 0; /* a few MUL factors in the arena are read as scalars (qnn16_tile_mul) */
    j->o = o;
    j->x = view(e, o->x);
    j->y = view(e, o->out);
    if (o->y != Q16P_NONE)
        j->z = view(e, o->y);
    if (o->kind == Q16_INSTANCE_NORM) {
        const Q16Tensor *zt = &e->program->tensors[o->z];
        /* The scale must be a flat FP16 run of two halves per channel. */
        if (!zt->direct || zt->padding.b || zt->padding.h || zt->padding.w || zt->padding.c ||
            zt->full.c != zt->dims.c || zt->dims.b * zt->dims.h * zt->dims.w != 1u || zt->dims.c != 2u * j->x.dims.c)
            return 0;
        if (e->program->tensors[o->y].direct != 1u || e->program->tensors[o->p[0]].direct != 1u)
            return 0; /* HVX reads must stay in VTCM */
        j->moments = (const float *)at(e, o->y);
        j->scale = (const uint16_t *)at(e, o->z);
        j->bias = (const float *)at(e, o->p[0]);
    }
    memcpy(&j->lo, &o->p[0], 4);
    memcpy(&j->hi, &o->p[1], 4);
    memcpy(j->v, o->p, sizeof j->v);
    j->parts = 1;
    return 1;
}
/* 1 when the tile form ran for every part recorded in j, 0 to use the generic path. */
static int small_ok(const SmallJob *j) {
    if (j->parts > 8u)
        return 0;
    for (uint32_t i = 0; i < j->parts; i++)
        if (!j->ok[i])
            return 0;
    return 1;
}
/* 1 when the tile form ran for every part, 0 to use the generic path. */
static int small_op(Q16Exec *e, const Q16Op *o) {
    SmallJob j;
    if (!small_prepare(e, o, &j))
        return 0;
    if (e->parallel)
        e->parallel(small_task, &j);
    else
        small_task(&j, 0, 1);
    return small_ok(&j);
}
/* Runs of independent operations of one kind run as one job. Two operations
 * are independent when neither writes a VTCM granule (2 KiB) that the other
 * reads or writes. Granules come from each operand's direct range or present
 * blocks, so tensor ids that share VTCM are caught. SWISH, ADD_SWISH and
 * INSTANCE_NORM split each operation by rows over the parts, as small_op does;
 * WELFORD_PARTIAL (serial in its tile form) gives whole operations to parts
 * p, p + parts, ... */
/* Batched small ops measured slower on the Frame (keypoints 17.2 vs 13.1 ms), so off by default. */
#ifndef Q16_NO_BATCH_FLAG
#define Q16_NO_BATCH_FLAG 1
#endif
enum { Q16_BATCH_MAX = 16, Q16_GRANULE = 2048, Q16_GRANULE_WORDS = Q16P_VTCM_BYTES / Q16_GRANULE / 32 };
typedef struct {
    uint32_t r[Q16_GRANULE_WORDS], w[Q16_GRANULE_WORDS];
} Q16Marks;
typedef struct {
    SmallJob j[Q16_BATCH_MAX];
    uint32_t n;
} SmallBatch;
typedef struct {
    Qnn16Tensor x[Q16_BATCH_MAX];
    float *out[Q16_BATCH_MAX];
    volatile uint32_t ok[Q16_BATCH_MAX];
    uint32_t n;
} WelfordBatch;
typedef struct {
    Q16Marks acc, op;
    SmallBatch small;
    WelfordBatch welford;
} RunState;
static RunState run_state; /* one program runs at a time; kept off the stack */
static int batch_kind(uint32_t kind) {
    return kind == Q16_SWISH || kind == Q16_ADD_SWISH || kind == Q16_INSTANCE_NORM || kind == Q16_WELFORD_PARTIAL;
}
static void small_batch_task(void *ctx, uint32_t part, uint32_t parts) {
    SmallBatch *b = (SmallBatch *)ctx;
    for (uint32_t k = 0; k < b->n; k++)
        small_task(&b->j[k], part, parts);
}
static void welford_batch_task(void *ctx, uint32_t part, uint32_t parts) {
    WelfordBatch *b = (WelfordBatch *)ctx;
    for (uint32_t k = part; k < b->n; k += parts)
        b->ok[k] = (uint32_t)qnn16_tile_welford(&b->x[k], b->out[k]);
}
/* Marks the granules of [lo, hi) in bm; 0 when the range leaves VTCM. */
static int mark_range(uint32_t *bm, uint64_t lo, uint64_t hi) {
    if (lo >= hi || hi > Q16P_VTCM_BYTES)
        return 0;
    for (uint64_t g = lo / Q16_GRANULE; g <= (hi - 1u) / Q16_GRANULE; g++)
        bm[g >> 5] |= 1u << (g & 31u);
    return 1;
}
/* Marks the granules tensor id occupies: its direct range, or each present block. 0 when not VTCM. */
static int mark_tensor(const Q16Program *p, uint32_t id, uint32_t *bm) {
    if (id == Q16P_NONE)
        return 0;
    const Q16Tensor *t = &p->tensors[id];
    if (t->direct == 1u)
        return mark_range(bm, t->offset, (uint64_t)t->offset + (t->bytes ? t->bytes : 1u));
    if (t->direct != 0u)
        return 0;
    for (uint32_t i = 0; i < t->block_count; i++) {
        const uint32_t off = p->words[t->block_first + i];
        if (off != Q16P_NONE && !mark_range(bm, off, (uint64_t)off + Q16_GRANULE))
            return 0;
    }
    return 1;
}
/* Marks what an operation reads (r) and writes (w). 0 when an operand is not in VTCM. */
static int op_marks(const Q16Program *p, const Q16Op *o, uint32_t *r, uint32_t *w) {
    uint32_t reads[4], n = 0;
    reads[n++] = o->x;
    if (o->kind == Q16_WELFORD_PARTIAL)
        reads[n++] = o->y;
    if (o->kind == Q16_INSTANCE_NORM) {
        reads[n++] = o->y;
        reads[n++] = o->z;
        reads[n++] = o->p[0];
    }
    for (uint32_t i = 0; i < n; i++)
        if (!mark_tensor(p, reads[i], r))
            return 0;
    return mark_tensor(p, o->out, w);
}
/* Runs the operations from `first` on, while they are one batchable kind and
 * independent (up to Q16_BATCH_MAX), as one job. An operation whose tile form
 * declines runs its generic path after the job; by independence the order does
 * not matter. *ran: operations done; on error, the index of the failing one + 1. */
static uint32_t tile_run(Q16Exec *e, uint32_t first, uint32_t *ran) {
    const Q16Program *p = e->program;
    const uint32_t kind = p->ops[first].kind, limit = p->header->ops;
    const int welford = kind == Q16_WELFORD_PARTIAL;
    RunState *s = &run_state;
    uint32_t n = 0;
    memset(s->acc.r, 0, sizeof s->acc.r);
    memset(s->acc.w, 0, sizeof s->acc.w);
    for (uint32_t k = 0; first + k < limit && k < Q16_BATCH_MAX; k++) {
        const Q16Op *o = &p->ops[first + k];
        if (o->kind != kind)
            break;
        if (welford) {
            if (o->p[0] != 1u || p->tensors[o->out].direct != 1u)
                break; /* qnn16_tile_welford's preconditions as in q16_exec_op */
        } else if (!small_prepare(e, o, &s->small.j[n]))
            break;
        memset(s->op.r, 0, sizeof s->op.r);
        memset(s->op.w, 0, sizeof s->op.w);
        if (!op_marks(p, o, s->op.r, s->op.w))
            break;
        int clash = 0;
        for (uint32_t g = 0; g < Q16_GRANULE_WORDS && !clash; g++)
            clash = ((s->op.w[g] & (s->acc.r[g] | s->acc.w[g])) | (s->op.r[g] & s->acc.w[g])) != 0;
        if (clash)
            break;
        for (uint32_t g = 0; g < Q16_GRANULE_WORDS; g++) {
            s->acc.r[g] |= s->op.r[g];
            s->acc.w[g] |= s->op.w[g];
        }
        if (welford) {
            s->welford.x[n] = view(e, o->x);
            s->welford.out[n] = (float *)at(e, o->out);
        }
        n++;
    }
    if (n < 2u) {
        *ran = 1;
        return q16_exec_op(e, first);
    }
    if (welford) {
        s->welford.n = n;
        if (e->parallel)
            e->parallel(welford_batch_task, &s->welford);
        else
            welford_batch_task(&s->welford, 0, 1);
    } else {
        s->small.n = n;
        if (e->parallel)
            e->parallel(small_batch_task, &s->small);
        else
            small_batch_task(&s->small, 0, 1);
    }
    for (uint32_t i = 0; i < n; i++) {
        const int ok = welford ? (int)s->welford.ok[i] : small_ok(&s->small.j[i]);
        if (!ok) {
            const uint32_t err = q16_exec_op(e, first + i);
            if (err) {
                *ran = i + 1u;
                return err;
            }
        }
    }
    *ran = n;
    return Q16_OK;
}
#endif
#ifdef __HVX__
static void stage_task(void *ctx, uint32_t part, uint32_t parts) {
    if (!qnn16_tile_stage3x3_fast((const Qnn16StageJob *)ctx, part, parts))
        qnn16_tile_stage3x3((const Qnn16StageJob *)ctx, part, parts);
}
/* Dense 3x3 (stride 1, or stride 2 over space-to-depth input) on HMX: workers
 * stage each output tile's nine tap windows per input group into dead VTCM
 * blocks (p[3], p[4]); the 1x1 leaf then accumulates K = 9 * groups tiles per
 * output tile against QNN's weight tiles, which are already in that order. */
static int hmx_conv3(Q16Exec *e, const Q16Op *o, uint8_t *cursor) {
    static const uint32_t acc[2][2] = {{2, 0}, {2, 6}};
    const Q16Program *p = e->program;
    const Q16Tensor *T = p->tensors;
    const uint32_t stride2 = o->kind == Q16_CONV3X3_S2, count = o->p[4], tables = o->p[0];
    if (!count || !e->conv1x1 || T[o->y].direct != 1u || (T[o->y].offset & 2047u) || T[o->z].direct != 1u ||
        tables < 1u || tables > 2u)
        return 0;
    Qnn16Tensor x = view(e, o->x), y = view(e, o->out);
    const uint32_t kg = stride2 ? T[o->x].dims.c / 128u : (o->p[1] + 31u) / 32u, K = kg * 9u;
    if (!qnn16_tile_stage3x3_ok(&x, &y, stride2, kg) || count < K)
        return 0;
    const uint32_t n = y.dims.c / 32u, tw = (y.dims.w + 3u) / 4u, tiles = y.dims.b * ((y.dims.h + 7u) / 8u) * tw;
    uint32_t chunk = count / K;
    if (chunk > tiles)
        chunk = tiles;
    const uint8_t **a = (const uint8_t **)cursor;
    uint8_t **scratch = (uint8_t **)(cursor + 8u * chunk * K);
    uint8_t **c = (uint8_t **)(cursor + 16u * chunk * K);
    /* Partial tiles: their output blocks are saved here and the lanes outside the tensor put back. */
    const uint32_t vh = y.dims.h & 7u, vw = y.dims.w & 3u;
    uint8_t *stash = cursor + 16u * chunk * K + 8u * chunk * n;
    if ((vh || vw) && 16u * chunk * K + 8u * chunk * n + (uint64_t)chunk * n * 2048u > q16_op_scratch(p, o))
        return 0;
    for (uint32_t i = 0; i < chunk * K; i++)
        scratch[i] = e->vtcm + p->words[o->p[3] + i];
    Qnn16StageJob job = {&x, &y, a, scratch, 0, 0, kg, stride2};
    for (uint32_t first = 0; first < tiles; first += chunk) {
        const uint32_t m = tiles - first < chunk ? tiles - first : chunk;
        job.first = first;
        job.tiles = m;
        if (e->parallel)
            e->parallel(stage_task, &job);
        else
            stage_task(&job, 0, 1);
        for (uint32_t i = 0; i < m * n; i++)
            c[i] = e->vtcm + y.block_offsets[(first + i / n) * n + i % n];
        if (vh || vw)
            for (uint32_t i = 0; i < m * n; i++)
                memcpy(stash + i * 2048u, c[i], 2048u);
        Qnn16Conv1x1 leaf = {a, (const uint8_t *)at(e, o->y), (const uint8_t *)at(e, o->z), (uint8_t *const *)c,
                             m, K, n, tables, acc[tables - 1u]};
        e->conv1x1(&leaf);
        if (e->finish)
            e->finish(e->vtcm + Q16P_COMPLETION_OFFSET);
        if (vh || vw)
            for (uint32_t i = 0; i < m * n; i++) {
                const uint32_t t = first + i / n, ty = (t / tw) % ((y.dims.h + 7u) / 8u), tx = t % tw;
                const uint32_t rows = ty * 8u + 8u <= y.dims.h ? 8u : y.dims.h - ty * 8u;
                const uint32_t cols = tx * 4u + 4u <= y.dims.w ? 4u : y.dims.w - tx * 4u;
                if (rows == 8u && cols == 4u)
                    continue;
                /* Block layout: row r is 256 bytes; columns (2k, 2k + 1) share 128 bytes, 4 per channel. */
                for (uint32_t r = 0; r < 8u; r++) {
                    uint8_t *dst = c[i] + r * 256u;
                    const uint8_t *src = stash + i * 2048u + r * 256u;
                    if (r >= rows) {
                        memcpy(dst, src, 256u);
                        continue;
                    }
                    for (uint32_t col = cols; col < 4u; col++) {
                        if (!(col & 1u) && col + 1u < 4u) {
                            memcpy(dst + (col / 2u) * 128u, src + (col / 2u) * 128u, 128u);
                            col++;
                            continue;
                        }
                        for (uint32_t ch = 0; ch < 32u; ch++)
                            memcpy(dst + (col / 2u) * 128u + ch * 4u + (col & 1u) * 2u,
                                   src + (col / 2u) * 128u + ch * 4u + (col & 1u) * 2u, 2u);
                    }
                }
            }
    }
    e->matrix_ops++;
    return 1;
}
#endif
#ifdef __HVX__
/* PW1X1 on HMX (qnn16_tile_pw_hmx.c). p[3]/p[4]: scratch blocks, the weight tiles first, then one temporary
 * block per output block of a row chunk. Returns 0 when the layout or the scratch does not fit. */
typedef struct {
    Qnn16Tensor *x, *y;
    const uint16_t *w;
    uint8_t *const *tiles, *const *temps;
    uint32_t kg, ng, packed, first, rows, phase;
} PwHmx;
static void pw_hmx_task(void *ctx, uint32_t part, uint32_t parts) {
    const PwHmx *j = (const PwHmx *)ctx;
    if (j->phase == 0)
        qnn16_tile_pw_hmx_tails(j->x, j->kg, part, parts);
    else if (j->phase == 1)
        qnn16_tile_pw_hmx_weights(j->w, j->y, j->packed, j->kg, j->ng, j->tiles, part, parts);
    else
        qnn16_tile_pw_hmx_store(j->y, j->ng, j->temps, j->first, j->rows, part, parts);
}
static void pw_hmx_run(Q16Exec *e, PwHmx *j, uint32_t phase) {
    j->phase = phase;
    if (e->parallel)
        e->parallel(pw_hmx_task, j);
    else
        pw_hmx_task(j, 0, 1);
}
static int hmx_pw(Q16Exec *e, const Q16Op *o, uint8_t *cursor) {
    static const uint32_t acc[2][2] = {{2, 0}, {2, 6}};
    const Q16Program *p = e->program;
    const Q16Tensor *T = p->tensors;
    const uint32_t tables = o->p[0];
    if (!o->p[4] || T[o->y].direct != 1u || T[o->z].direct != 1u || tables < 1u || tables > 2u)
        return 0;
    Qnn16Tensor x = view(e, o->x), y = view(e, o->out);
    uint32_t kg = 0, ng = 0;
    if (!qnn16_tile_pw_hmx_ok(&x, &y, o->p[1], &kg, &ng))
        return 0;
    const uint32_t tiles = kg * ng, blocks = y.block_count / ng;
    if (o->p[4] < tiles + ng)
        return 0;
    uint32_t rows = (o->p[4] - tiles) / ng;
    if (rows > blocks)
        rows = blocks;
    /* Pointer tables in the operation's scratch: tiles, temporaries, activation rows. */
    if (4u * (tiles + rows * (ng + kg)) > q16_op_scratch(p, o))
        return 0;
    uint8_t **blk = (uint8_t **)cursor;
    uint8_t **temps = blk + tiles;
    const uint8_t **a = (const uint8_t **)(temps + rows * ng);
    for (uint32_t i = 0; i < tiles + rows * ng; i++)
        blk[i] = e->vtcm + p->words[o->p[3] + i];
    PwHmx j = {&x, &y, (const uint16_t *)at(e, o->y), blk, temps, kg, ng, o->p[1], 0, rows, 0};
    pw_hmx_run(e, &j, 0);
    pw_hmx_run(e, &j, 1);
    for (uint32_t first = 0; first < blocks; first += rows) {
        const uint32_t m = blocks - first < rows ? blocks - first : rows;
        qnn16_tile_pw_hmx_rows(&x, kg, first, m, a);
        Qnn16Conv1x1 leaf = {a, (const uint8_t *)blk, (const uint8_t *)at(e, o->z), (uint8_t *const *)temps, m,
                             kg, ng, tables, acc[tables - 1u]};
        qnn16_conv1x1_wt(&leaf);
        if (e->finish)
            e->finish(e->vtcm + Q16P_COMPLETION_OFFSET);
        j.first = first;
        j.rows = m;
        pw_hmx_run(e, &j, 2);
    }
    e->matrix_ops++;
    return 1;
}
#endif
/* Gather to float32, prepare weights, compute and scatter, each stage split
 * by rows over the workers. Mirrors conv_scratch in program.c. */
static uint32_t conv_op(Q16Exec *e, const Q16Op *o, uint8_t *cursor) {
    const Q16Tensor *T = e->program->tensors;
#ifdef __HVX__
    if (!(e->flags & Q16_EXEC_NO_TILE) && (o->kind == Q16_CONV3X3 || o->kind == Q16_CONV3X3_S2) &&
        hmx_conv3(e, o, cursor))
        return Q16_OK;
    if (!(e->flags & Q16_EXEC_NO_TILE) && o->kind == Q16_PW1X1 && hmx_pw(e, o, cursor))
        return Q16_OK;
    if (!(e->flags & Q16_EXEC_NO_TILE) && tile_op(e, o))
        return Q16_OK;
#endif
    const Qnn16Shape xs = T[o->x].dims, ys = T[o->out].dims;
    uint32_t oc = ys.c, xc = xs.c, nw = 0;
    ConvJob j;
    j.o = o;
    j.xin = view(e, o->x);
    j.yout = view(e, o->out);
    j.xs = xs;
    switch (o->kind) {
    case Q16_DW3X3: case Q16_DW3X3_S2: nw = oc * 9u; break;
    case Q16_DW1X1: nw = oc * 2u; break;
    case Q16_CONV3X3: nw = oc * 9u * xc; break;
    case Q16_CONV3X3_S2: nw = oc * 9u * (xc / 4u); break;
    case Q16_PW1X1: case Q16_PW_SWISH: case Q16_PW_TANH: nw = oc * xc; break;
    default: oc = 0; break;
    }
    j.xf = (float *)bump(&cursor, (uint64_t)xs.b * xs.h * xs.w * xs.c * 4u);
    j.yf = (float *)bump(&cursor, (uint64_t)ys.b * ys.h * ys.w * ys.c * 4u);
    j.w = (float *)bump(&cursor, (uint64_t)nw * 4u);
    j.ep.bias = (float *)bump(&cursor, oc * 4u);
    j.ep.lo = (float *)bump(&cursor, oc * 4u);
    j.ep.hi = (float *)bump(&cursor, oc * 4u);
    if (o->kind != Q16_S2D) {
        const uint16_t *wt = (const uint16_t *)at(e, o->y);
        const uint8_t *fi = (const uint8_t *)at(e, o->z);
        uint32_t tables = o->p[0];
        if (o->kind == Q16_PW_SWISH)
            qnn16_prepare_pw_swish(j.w, j.ep, wt, fi, o->p[0], xc, oc);
        else if (o->kind == Q16_PW_TANH) {
            float unit;
            memcpy(&unit, &o->p[1], 4);
            qnn16_prepare_pw_tanh(j.w, j.ep, wt, fi, o->p[0], xc, oc, unit);
        } else
            qnn16_prepare_epilogue(j.ep, fi, tables, oc);
        if (o->kind == Q16_PW_SWISH || o->kind == Q16_PW_TANH) { /* prepared above */ }
        else if (o->kind == Q16_DW3X3)
            qnn16_prepare_dw3x3(j.w, wt, fi, tables, oc);
        else if (o->kind == Q16_DW3X3_S2)
            qnn16_prepare_dw3x3_s2(j.w, wt, fi, tables, oc, o->p[1]);
        else if (o->kind == Q16_DW1X1)
            qnn16_prepare_dw1x1(j.w, wt, fi, tables, oc);
        else if (o->kind == Q16_CONV3X3_S2)
            qnn16_prepare_conv3x3_s2(j.w, wt, fi, tables, xc / 4u, oc);
        else if (o->kind == Q16_CONV3X3)
            qnn16_prepare_conv3x3(j.w, wt, fi, tables, o->p[1], xc, oc);
        else
            qnn16_prepare_pw(j.w, wt, fi, tables, o->p[1], xc, oc);
    }
    for (j.stage = 0; j.stage < 3; j.stage++)
        run_parts(e, &j);
    return Q16_OK;
}

/* Work split over the workers for serial-only operations. */
typedef struct {
    uint8_t *dst;
    const uint8_t *src;
    uint32_t bytes;
} CopyJob;
static void copy_task(void *ctx, uint32_t part, uint32_t parts) {
    const CopyJob *j = (const CopyJob *)ctx;
    uint32_t chunk = ((j->bytes + parts - 1u) / parts + 127u) & ~127u, at = part * chunk;
    if (at < j->bytes)
        memcpy(j->dst + at, j->src + at, j->bytes - at < chunk ? j->bytes - at : chunk);
}
typedef struct {
    Qnn16Tensor out, x;
    const float *in;
    Qnn16Shape shape;
    int tile;
    volatile uint32_t bad;
} Im2colJob;
static void im2col_task(void *ctx, uint32_t part, uint32_t parts) {
    Im2colJob *j = (Im2colJob *)ctx;
#ifdef __HVX__
    if (j->tile) {
        qnn16_tile_im2col(&j->x, &j->out, part, parts);
        return;
    }
#endif
    if (qnn16_tensor_im2col_cast_rows(&j->out, j->in, j->shape, part, parts))
        j->bad = 1;
}
/* Compact direct tensor: contiguous NHWC with no padding. */
static int compact(const Q16Tensor *t) {
    return t->direct && !t->padding.b && !t->padding.h && !t->padding.w && !t->padding.c &&
           t->full.b == t->dims.b && t->full.h == t->dims.h && t->full.w == t->dims.w && t->full.c == t->dims.c;
}
/* CONCAT of compact tensors of one dtype: per outer index, one copy per part. */
static int concat_direct(Q16Exec *e, const Q16Op *o, uint32_t count) {
    const Q16Program *p = e->program;
    const Q16Tensor *T = p->tensors, *y = &T[o->out];
    const uint32_t axis = o->p[0];
    if (!compact(y) || axis > 3u)
        return 0;
    const uint32_t esz = y->dtype == Q16_F32 ? 4u : 2u;
    const uint32_t yd[4] = {y->dims.b, y->dims.h, y->dims.w, y->dims.c};
    uint32_t outer = 1, inner = 1;
    for (uint32_t a = 0; a < axis; a++)
        outer *= yd[a];
    for (uint32_t a = axis + 1u; a < 4u; a++)
        inner *= yd[a];
    for (uint32_t i = 0; i < count; i++) {
        const Q16Tensor *x = &T[p->words[o->p[1] + i]];
        if (!compact(x) || x->dtype != y->dtype)
            return 0;
    }
    uint8_t *dst = (uint8_t *)at(e, o->out);
    for (uint32_t r = 0; r < outer; r++)
        for (uint32_t i = 0; i < count; i++) {
            const Q16Tensor *x = &T[p->words[o->p[1] + i]];
            const uint32_t xd[4] = {x->dims.b, x->dims.h, x->dims.w, x->dims.c};
            const uint32_t chunk = xd[axis] * inner * esz;
            memcpy(dst, (const uint8_t *)at(e, p->words[o->p[1] + i]) + r * chunk, chunk);
            dst += chunk;
        }
    return 1;
}
#ifdef __HVX__
enum { CONCAT_TILE_MAX = 16 };
typedef struct {
    Qnn16Tensor in[CONCAT_TILE_MAX], out;
    uint32_t count;
} ConcatJob;
static void concat_task(void *ctx, uint32_t part, uint32_t parts) {
    const ConcatJob *j = (const ConcatJob *)ctx;
    qnn16_tile_concat_h(j->in, j->count, &j->out, part, parts);
}
/* CONCAT along H of tile tensors (qnn16_tile_concat.c); 0 to use the generic path. */
static int concat_tiles(Q16Exec *e, const Q16Op *o, uint32_t count) {
    const Q16Program *p = e->program;
    ConcatJob j;
    if (o->p[0] != 1u || count > CONCAT_TILE_MAX)
        return 0;
    for (uint32_t i = 0; i < count; i++)
        j.in[i] = view(e, p->words[o->p[1] + i]);
    j.out = view(e, o->out);
    j.count = count;
    if (!qnn16_tile_concat_h_ok(j.in, count, &j.out))
        return 0;
    if (e->parallel)
        e->parallel(concat_task, &j);
    else
        concat_task(&j, 0, 1);
    return 1;
}
#endif
enum { COPY_RUN_MAX = 64 };
typedef struct {
    uint8_t *dst[COPY_RUN_MAX];
    const uint8_t *src[COPY_RUN_MAX];
    uint32_t bytes[COPY_RUN_MAX], start[COPY_RUN_MAX]; /* start: 128-aligned position in the stream */
    uint32_t count, total;
} CopyRun;
/* Each part copies its share of the concatenated stream; copies start on
 * 128-byte stream positions, so share boundaries are vector offsets in a copy. */
static void copy_run_task(void *ctx, uint32_t part, uint32_t parts) {
    const CopyRun *r = (const CopyRun *)ctx;
    const uint32_t share = ((r->total + parts - 1u) / parts + 127u) & ~127u;
    const uint32_t lo = part * share, hi = lo + share < r->total ? lo + share : r->total;
    for (uint32_t i = 0; i < r->count && lo < hi; i++) {
        const uint32_t s0 = r->start[i], s1 = s0 + r->bytes[i];
        const uint32_t a = lo > s0 ? lo : s0, b = hi < s1 ? hi : s1;
        if (a < b)
            memcpy(r->dst[i] + (a - s0), r->src[i] + (a - s0), b - a);
    }
}
uint32_t q16_exec_copies(Q16Exec *e, uint32_t first, uint32_t *ran) {
    const Q16Program *p = e->program;
    CopyRun r;
    uint32_t n = 0, pos = 0;
    while (n < COPY_RUN_MAX && first + n < p->header->ops && p->ops[first + n].kind == Q16_COPY) {
        const Q16Op *o = &p->ops[first + n];
        r.dst[n] = (uint8_t *)at(e, o->out);
        r.src[n] = e->constants + o->p[0];
        r.bytes[n] = o->p[1];
        r.start[n] = pos;
        pos = (pos + o->p[1] + 127u) & ~127u;
        n++;
    }
    *ran = n ? n : 1u;
    if (!n)
        return q16_exec_op(e, first);
    int overlap = 0;
    for (uint32_t i = 0; i < n && !overlap; i++)
        for (uint32_t j = i + 1u; j < n; j++)
            if ((uintptr_t)r.dst[i] < (uintptr_t)r.dst[j] + r.bytes[j] &&
                (uintptr_t)r.dst[j] < (uintptr_t)r.dst[i] + r.bytes[i]) {
                overlap = 1;
                break;
            }
    if (overlap || !e->parallel || n == 1u) {
        for (uint32_t i = 0; i < n; i++) {
            uint32_t err = q16_exec_op(e, first + i);
            if (err) {
                *ran = i + 1u;
                return err;
            }
        }
        return Q16_OK;
    }
    r.count = n;
    r.total = pos;
    e->parallel(copy_run_task, &r);
    return Q16_OK;
}
uint32_t q16_exec_run(Q16Exec *e, uint32_t first, uint32_t *ran) {
    const uint32_t kind = e->program->ops[first].kind;
    *ran = 1;
    if (kind == Q16_COPY)
        return q16_exec_copies(e, first, ran);
#ifdef __HVX__
    if (!(e->flags & Q16_EXEC_NO_TILE) && batch_kind(kind) && !Q16_NO_BATCH_FLAG)
        return tile_run(e, first, ran);
#endif
    return q16_exec_op(e, first);
}
uint32_t q16_exec_op(Q16Exec *e, uint32_t index) {
    const Q16Program *p = e->program;
    const Q16Op *o = &p->ops[index];
    const Q16Tensor *T = p->tensors;
    uint8_t *cursor = e->scratch + (p->header->version >= Q16P_ARENA_VERSION ? p->header->reserved[0] : 0u);
    int bad = 0;
    switch (o->kind) {
    case Q16_COPY:
        if (e->parallel && o->p[1] >= Q16_PARALLEL_COPY_BYTES) {
            CopyJob j = {(uint8_t *)at(e, o->out), e->constants + o->p[0], o->p[1]};
            e->parallel(copy_task, &j);
        } else
            memcpy(at(e, o->out), e->constants + o->p[0], o->p[1]);
        break;
    case Q16_ZERO: {
        const Q16Tensor *t = &T[o->out];
        if (t->direct)
            memset(at(e, o->out), 0, t->bytes);
        else
            for (uint32_t i = 0; i < t->block_count; i++) {
                uint32_t off = p->words[t->block_first + i];
                if (off != Q16P_NONE)
                    memset(e->vtcm + off, 0, 2048);
            }
        break;
    }
    case Q16_CONV1X1: {
        static const uint32_t acc[2][2] = {{2, 0}, {2, 6}};
        uint32_t m = o->p[0], k = o->p[1], n = o->p[2];
        const uint8_t **a = (const uint8_t **)cursor;
        uint8_t **c = (uint8_t **)(cursor + 8u * m * k);
        const uint32_t *w = p->words + o->p[4];
        for (uint32_t i = 0; i < m * k; i++)
            a[i] = e->vtcm + w[i];
        for (uint32_t i = 0; i < m * n; i++)
            c[i] = e->vtcm + w[m * k + i];
        Qnn16Conv1x1 job = {a, (const uint8_t *)at(e, o->y), (const uint8_t *)at(e, o->z),
                            (uint8_t *const *)c, m, k, n, o->p[3], acc[o->p[3] - 1]};
        e->conv1x1(&job);
        if (e->finish)
            e->finish(e->vtcm + Q16P_COMPLETION_OFFSET);
        e->matrix_ops++;
        break;
    }
    case Q16_DW1X1: case Q16_DW3X3: case Q16_CONV3X3: case Q16_DW3X3_S2: case Q16_CONV3X3_S2: case Q16_S2D:
    case Q16_PW1X1: case Q16_PW_SWISH: case Q16_PW_TANH:
        return conv_op(e, o, cursor);
    case Q16_IM2COL: {
        Im2colJob j = {view(e, o->out), view(e, o->x), (const float *)at(e, o->x), T[o->x].dims, 0, 0};
#ifdef __HVX__
        j.tile = !(e->flags & Q16_EXEC_NO_TILE) && qnn16_tile_im2col_ok(&j.x, &j.out);
#endif
        if (e->parallel)
            e->parallel(im2col_task, &j);
        else
            im2col_task(&j, 0, 1);
        bad |= (int)j.bad;
        break;
    }
    case Q16_INPUT: {
        const Q16Io *io = &p->io[o->p[0]];
        Qnn16Shape source = io->dims; source.b = T[o->out].dims.b;
        const float *base = (const float *)(e->exchange + io->offset) + (uint64_t)o->p[3] * io->dims.h * io->dims.w * io->dims.c;
        bad |= qnn16_window_f32((float *)at(e, o->out), T[o->out].dims,
                                base, source, (int32_t)o->p[1],
                                (int32_t)o->p[2]);
        break;
    }
    case Q16_OUTPUT: {
        const Q16Io *io = &p->io[p->header->inputs + o->p[0]];
        const float *src = (const float *)at(e, o->x);
        if (T[o->x].dtype == Q16_F16) {
            Qnn16Tensor v = view(e, o->x);
            bad |= qnn16_tensor_f16_to_f32((float *)cursor, &v);
            src = (const float *)cursor;
        }
        Qnn16Shape dest = io->dims; dest.b = T[o->x].dims.b;
        float *base = (float *)(e->exchange + io->offset) + (uint64_t)o->p[3] * io->dims.h * io->dims.w * io->dims.c;
        bad |= qnn16_place_f32(base, dest, src, T[o->x].dims, o->p[1],
                               o->p[2]);
        break;
    }
    case Q16_INPUT_U16: {
        const Q16Io *io = &p->io[o->p[0]];
        const Qnn16Shape tile = T[o->out].dims;
        const float *src = (const float *)(e->exchange + io->offset);
        uint16_t *dst = (uint16_t *)at(e, o->out);
        float scale;
        memcpy(&scale, &o->p[4], sizeof scale);
        for (uint32_t b = 0; b < tile.b; b++)
            for (uint32_t y = 0; y < tile.h; y++) {
                size_t offset = (((size_t)(b + o->p[3]) * io->dims.h + y + o->p[1]) *
                                 io->dims.w + o->p[2]) * io->dims.c;
                for (size_t i = 0; i < (size_t)tile.w * tile.c; i++) {
                    /* Preserve QNN's U16 decision thresholds at large zero points. */
                    double scaled = (double)src[offset + i] / scale + o->p[5];
                    if (isnan(scaled))
                        scaled = o->p[5];
                    *dst++ = scaled <= 0 ? 0 : scaled >= UINT16_MAX ? UINT16_MAX : (uint16_t)(scaled + 0.5);
                }
            }
        break;
    }
    case Q16_OUTPUT_U16: {
        const Q16Io *io = &p->io[p->header->inputs + o->p[0]];
        const Qnn16Shape tile = T[o->x].dims;
        uint16_t *src = (uint16_t *)cursor;
        float *dst = (float *)(e->exchange + io->offset);
        float scale;
        memcpy(&scale, &o->p[4], sizeof scale);
        if (gather(e, src, o->x))
            return Q16_SHAPE;
        for (uint32_t b = 0; b < tile.b; b++)
            for (uint32_t y = 0; y < tile.h; y++) {
                size_t offset = (((size_t)(b + o->p[3]) * io->dims.h + y + o->p[1]) *
                                 io->dims.w + o->p[2]) * io->dims.c;
                for (size_t i = 0; i < (size_t)tile.w * tile.c; i++)
                    dst[offset + i] = ((int32_t)*src++ - (int32_t)o->p[5]) * scale;
            }
        break;
    }
    case Q16_FORMAT: {
        Qnn16Tensor out = view(e, o->out), in = view(e, o->x);
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && (small_op(e, o) || qnn16_tile_format(&in, &out)))
            break;
#endif
        bad |= qnn16_tensor_copy_format(&out, &in);
        break;
    }
    case Q16_CAST_F32: {
        Qnn16Tensor in = view(e, o->x);
#ifdef __HVX__
        Qnn16Tensor out = view(e, o->out);
        if (!(e->flags & Q16_EXEC_NO_TILE) && qnn16_tile_cast_f32(&in, &out))
            break;
#endif
        bad |= qnn16_tensor_f16_to_f32((float *)at(e, o->out), &in);
        break;
    }
    case Q16_CAST_F16: {
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && T[o->x].direct == 1u && small_op(e, o))
            break;
#endif
        uint16_t *y = take(&cursor, &T[o->out]);
        qnn16_f32_to_f16(y, (const float *)at(e, o->x), q16_tensor_elements(&T[o->x]));
        bad |= scatter(e, o->out, y);
        break;
    }
    case Q16_MUL: {
        Qnn16Tensor out = view(e, o->out), a = view(e, o->x), b = view(e, o->y);
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && small_op(e, o))
            break;
#endif
        bad |= qnn16_tensor_mul(&out, &a, &b, 0, 0);
        break;
    }
    case Q16_MINMAX: {
        float lo, hi;
        memcpy(&lo, &o->p[0], 4);
        memcpy(&hi, &o->p[1], 4);
        Qnn16Tensor out = view(e, o->out), in = view(e, o->x);
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && small_op(e, o))
            break;
#endif
        bad |= qnn16_tensor_minmax(&out, &in, 0, lo, hi);
        break;
    }
    case Q16_SIGMOID: {
        Qnn16Tensor out = view(e, o->out), in = view(e, o->x);
        bad |= qnn16_tensor_sigmoid(&out, &in, 0);
        break;
    }
    case Q16_REDUCE_SUM: case Q16_REDUCE_MEAN: {
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && small_op(e, o))
            break;
#endif
        uint16_t *x = take(&cursor, &T[o->x]);
        Qnn16Shape xs = T[o->x].dims, ys = T[o->out].dims;
        uint32_t n = q16_tensor_elements(&T[o->out]);
        const float mean = o->kind == Q16_REDUCE_MEAN ? (float)n / (float)q16_tensor_elements(&T[o->x]) : 1.f;
        float *acc = (float *)bump(&cursor, (uint64_t)n * 4u);
        memset(acc, 0, n * sizeof(float));
        bad |= gather(e, x, o->x);
        uint32_t i = 0;
        for (uint32_t b = 0; b < xs.b; b++) for (uint32_t h = 0; h < xs.h; h++)
            for (uint32_t w = 0; w < xs.w; w++) for (uint32_t c = 0; c < xs.c; c++) {
                uint32_t j = (((ys.b == 1 ? 0 : b) * ys.h + (ys.h == 1 ? 0 : h)) * ys.w +
                              (ys.w == 1 ? 0 : w)) * ys.c + (ys.c == 1 ? 0 : c);
                acc[j] += qnn16_half_to_float(x[i++]);
            }
        for (uint32_t j = 0; j < n; j++) x[j] = qnn16_float_to_half(acc[j] * mean);
        bad |= scatter(e, o->out, x);
        break;
    }
    case Q16_L2NORM: {
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && small_op(e, o))
            break;
#endif
        uint16_t *x = take(&cursor, &T[o->x]);
        const Qnn16Shape s = T[o->x].dims;
        const uint32_t d[4] = {s.b, s.h, s.w, s.c}, axis = o->p[0], count = q16_tensor_elements(&T[o->x]);
        uint32_t inner = 1;
        for (uint32_t a = axis + 1u; a < 4u; a++)
            inner *= d[a];
        const uint32_t span = d[axis] * inner; /* elements of one outer index */
        float *acc = (float *)bump(&cursor, (uint64_t)(count / d[axis]) * 4u), epsilon;
        memcpy(&epsilon, &o->p[1], 4);
        bad |= gather(e, x, o->x);
        for (uint32_t i = 0; i < count / d[axis]; i++)
            acc[i] = 0.f;
        for (uint32_t i = 0; i < count; i++) {
            const float v = qnn16_half_to_float(x[i]);
            acc[(i / span) * inner + i % inner] += v * v;
        }
        for (uint32_t i = 0; i < count / d[axis]; i++)
            acc[i] = 1.f / sqrtf(acc[i] > epsilon ? acc[i] : epsilon);
        for (uint32_t i = 0; i < count; i++)
            x[i] = qnn16_float_to_half(qnn16_half_to_float(x[i]) * acc[(i / span) * inner + i % inner]);
        bad |= scatter(e, o->out, x);
        break;
    }
    case Q16_AFFINE: {
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && T[o->out].dtype == Q16_F16 && small_op(e, o))
            break;
#endif
        uint16_t *x = take(&cursor, &T[o->x]);
        const uint32_t count = q16_tensor_elements(&T[o->x]);
        float scale, bias;
        memcpy(&scale, &o->p[0], 4);
        memcpy(&bias, &o->p[1], 4);
        bad |= gather(e, x, o->x);
        if (T[o->out].dtype == Q16_F32) {
            float *y = (float *)at(e, o->out);
            for (uint32_t i = 0; i < count; i++)
                y[i] = qnn16_half_to_float(x[i]) * scale + bias;
        } else {
            for (uint32_t i = 0; i < count; i++)
                x[i] = qnn16_float_to_half(qnn16_half_to_float(x[i]) * scale + bias);
            bad |= scatter(e, o->out, x);
        }
        break;
    }
    case Q16_INSTANCE_NORM: {
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && small_op(e, o))
            break;
#endif
        uint16_t *x = take(&cursor, &T[o->x]), *scale = take(&cursor, &T[o->z]);
        const float *moments = (const float *)at(e, o->y), *bias = (const float *)at(e, o->p[0]);
        uint32_t c = T[o->x].dims.c, count = q16_tensor_elements(&T[o->x]);
        bad |= gather(e, x, o->x); bad |= gather(e, scale, o->z);
        for (uint32_t i = 0; i < count; i++) {
            uint32_t j = i % c;
            float v = (qnn16_half_to_float(x[i]) * qnn16_half_to_float(scale[2 * j]) - moments[j]) *
                      moments[c + j] + bias[j];
            x[i] = qnn16_float_to_half(v);
        }
        bad |= scatter(e, o->out, x);
        break;
    }
    case Q16_GROUP_MOMENTS: {
        const float *x = (const float *)at(e, o->x), *factor = (const float *)at(e, o->y);
        const float *ratio = (const float *)at(e, o->z), *gamma = (const float *)at(e, o->p[0]);
        float *out = (float *)at(e, o->out), inv_count, epsilon;
        memcpy(&inv_count, &o->p[1], 4); memcpy(&epsilon, &o->p[2], 4);
        uint32_t c = T[o->x].dims.c, group = o->p[3];
        for (uint32_t start = 0; start < c; start += group) {
            float mean = x[start], m2 = x[c + start];
            for (uint32_t j = 1; j < group; j++) {
                float delta = x[start + j] - mean;
                m2 += x[c + start + j] + delta * delta * factor[j];
                mean += delta * ratio[j];
            }
            float inv_std = 1.f / sqrtf(m2 * inv_count + epsilon);
            for (uint32_t j = 0; j < group; j++) {
                out[start + j] = mean * gamma[start + j];
                out[c + start + j] = inv_std;
            }
        }
        break;
    }
    case Q16_MOVE:
        memmove(at(e, o->out), at(e, o->x), T[o->x].bytes);
        break;
    case Q16_WELFORD_MERGE: {
        const float *factor = (const float *)at(e, o->x), *ratio = (const float *)at(e, o->y);
        float *out = (float *)at(e, o->out);
        const uint32_t c = T[o->out].dims.c, count = o->p[1];
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && count && count <= 16u && T[o->x].direct == 1u && T[o->y].direct == 1u &&
            T[o->out].direct == 1u) {
            const float *parts[16];
            uint32_t vt = 1;
            for (uint32_t i = 0; i < count; i++) {
                vt &= T[p->words[o->p[0] + i]].direct == 1u;
                parts[i] = (const float *)at(e, p->words[o->p[0] + i]);
            }
            if (vt && qnn16_tile_welford_merge(out, parts, count, factor, ratio, c))
                break;
        }
#endif
        const float *first = (const float *)at(e, p->words[o->p[0]]);
        for (uint32_t j = 0; j < c; j++) {
            float mean = first[j], m2 = first[c + j];
            for (uint32_t i = 1; i < count; i++) {
                const float *next = (const float *)at(e, p->words[o->p[0] + i]);
                float delta = next[j] - mean;
                m2 += next[c + j] + delta * delta * factor[(i - 1) * 32 + j % 32];
                mean += delta * ratio[(i - 1) * 32 + j % 32];
            }
            out[j] = mean; out[c + j] = m2;
        }
        break;
    }
    case Q16_WELFORD_PARTIAL: {
#ifdef __HVX__
        {
            Qnn16Tensor tx = view(e, o->x);
            if (!(e->flags & Q16_EXEC_NO_TILE) && o->p[0] == 1u && T[o->out].direct == 1u &&
                qnn16_tile_welford(&tx, (float *)at(e, o->out)))
                break;
        }
#endif
        uint16_t *x = take(&cursor, &T[o->x]);
        const float *weights = (const float *)at(e, o->y);
        float *out = (float *)at(e, o->out);
        const uint32_t c = T[o->x].dims.c, count = T[o->x].dims.h * T[o->x].dims.w;
        bad |= gather(e, x, o->x);
        for (uint32_t j = 0; j < c; j++) {
            float mean = 0.f, m2 = 0.f;
            for (uint32_t i = 0; i < count; i++) {
                float v = qnn16_half_to_float(x[i * c + j]);
                float delta = v - mean;
                mean += delta * weights[i * T[o->y].dims.c + j % T[o->y].dims.c];
                m2 += delta * (v - mean);
            }
            out[j] = mean; out[c + j] = m2;
        }
        break;
    }
    case Q16_SOFTMAX: {
        Qnn16Tensor out = view(e, o->out), in = view(e, o->x);
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && small_op(e, o))
            break;
#endif
        bad |= qnn16_tensor_softmax(&out, &in);
        break;
    }
    case Q16_RESIZE_NEAREST: {
        Qnn16Tensor out = view(e, o->out), in = view(e, o->x);
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && small_op(e, o))
            break;
#endif
        bad |= qnn16_tensor_resize_nearest(&out, &in, o->p[0], o->p[1]);
        break;
    }
    case Q16_RESIZE_BILINEAR: {
        Qnn16Tensor out = view(e, o->out), in = view(e, o->x);
        float v[4]; memcpy(v, o->p, sizeof(v));
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && small_op(e, o))
            break;
#endif
        bad |= qnn16_tensor_resize(&out, &in, v[0], v[1], v[2], v[3]);
        break;
    }
    case Q16_DECIMATE: {
        Qnn16Tensor out = view(e, o->out), in = view(e, o->x);
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && small_op(e, o))
            break;
#endif
        bad |= qnn16_tensor_decimate(&out, &in, o->p[0], o->p[1]);
        break;
    }
    case Q16_SLICE_F32: {
        const float *src = (const float *)at(e, o->x);
        float *dst = (float *)at(e, o->out);
        uint32_t rows = T[o->out].dims.b * T[o->out].dims.h * T[o->out].dims.w;
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && T[o->x].direct == 1u && T[o->out].direct == 1u &&
            qnn16_tile_slice_f32(dst, src, rows, T[o->x].dims.c, T[o->out].dims.c, o->p[0]))
            break;
#endif
        for (uint32_t i = 0; i < rows; i++)
            for (uint32_t c = 0; c < T[o->out].dims.c; c++)
                dst[i * T[o->out].dims.c + c] = src[i * T[o->x].dims.c + o->p[0] + c];
        break;
    }
    case Q16_MOMENTS_U16: {
        const Q16Tensor *in = &T[o->x];
        const uint32_t *blocks = p->words + in->block_first;
        uint32_t *out = (uint32_t *)at(e, o->out), tokens = in->dims.w * 8u;
        uint32_t groups = in->dims.c / 32u;
        /* Each strip emits 32 tokens in physical row order. Each 128-byte
         * slice interleaves the channels of two adjacent pixels. */
        for (uint32_t strip = 0; strip < in->dims.w / 4u; strip++)
            for (uint32_t pair = 0; pair < 16u; pair++) {
                uint32_t sum0 = 0, sum1 = 0;
                uint64_t sq0 = 0, sq1 = 0;
                for (uint32_t g = 0; g < groups; g++) {
                    const uint16_t *values = (const uint16_t *)(e->vtcm + blocks[strip * groups + g] + pair * 128u);
                    for (uint32_t c = 0; c < 32u; c++) {
                        uint32_t a = values[2u * c], b = values[2u * c + 1u];
                        sum0 += a;
                        sum1 += b;
                        sq0 += a * a;
                        sq1 += b * b;
                    }
                }
                uint32_t lane = strip * 32u + pair * 2u;
                out[lane] = sum0;
                out[lane + 1u] = sum1;
                out[tokens + lane] = (uint32_t)sq0;
                out[tokens + lane + 1u] = (uint32_t)sq1;
                out[2u * tokens + lane] = (uint32_t)(sq0 >> 32);
                out[2u * tokens + lane + 1u] = (uint32_t)(sq1 >> 32);
            }
        break;
    }
    case Q16_LINCLIP_U16: {
        const Q16Tensor *in = &T[o->x], *out = &T[o->out];
        uint32_t blocks = in->direct ? 1u : in->block_count;
        uint32_t count = in->direct ? q16_tensor_elements(in) : 1024u;
        for (uint32_t block = 0; block < blocks; ++block) {
            const uint16_t *src = in->direct ? at(e, o->x) :
                (const uint16_t *)(e->vtcm + p->words[in->block_first + block]);
            uint16_t *dst = out->direct ? at(e, o->out) :
                (uint16_t *)(e->vtcm + p->words[out->block_first + block]);
            for (uint32_t i = 0; i < count; ++i) {
                int64_t v = ((int64_t)src[i] - o->p[2]) * o->p[0] + o->p[4];
                v = (v >> o->p[1]) + o->p[3];
                dst[i] = v < 0 ? 0 : v > UINT16_MAX ? UINT16_MAX : (uint16_t)v;
            }
        }
        break;
    }
    case Q16_PACK_XPB_U16: {
        const uint16_t *src = at(e, o->x);
        uint8_t *dst = at(e, o->out);
        uint32_t k = T[o->x].dims.w, n = T[o->x].dims.c;
        uint32_t full_k = T[o->out].full.w, full_n = T[o->out].full.c / 2u;
        for (uint32_t ng = 0; ng < full_n; ng += 16u)
            for (uint32_t kg = 0; kg < full_k; kg += 4u)
                for (uint32_t col = ng; col < ng + 16u; ++col) {
                    for (uint32_t row = 0; row < 4u; ++row) {
                        int32_t v = (kg + row < k && col < n) ? (int32_t)src[(kg + row) * n + col] - 32768 : 0;
                        /* The greatest code representable by two signed base-256 digits. */
                        if (v > 32639)
                            v = 32639;
                        int32_t hi = (v + 128) >> 8;
                        dst[row] = (uint8_t)(v - hi * 256);
                        dst[4u + row] = (uint8_t)hi;
                    }
                    dst += 8;
                }
        break;
    }
    case Q16_XPB_BIAS: {
        const int8_t *w = at(e, o->x);
        int32_t *bias = at(e, o->out);
        uint32_t full_n = T[o->x].full.c / 2u;
        for (uint32_t n = 0; n < full_n; n++) {
            int32_t sum = 0;
            const int8_t *col = w + (n / 16u * 16u * 16u + n % 16u) * 8u;
            for (uint32_t kg = 0; kg < 16u; ++kg) {
                const int8_t *group = col + kg * 128u;
                for (uint32_t j = 0; j < 4u; ++j)
                    sum += group[j] + group[j + 4u] * 256;
            }
            bias[n] = (int32_t)((-(int64_t)sum * o->p[0] + 32768) >> 16);
        }
        memset(bias + full_n, 0, (T[o->out].dims.c - full_n) * sizeof(*bias));
        break;
    }
    case Q16_XPB_FI: {
        const int32_t *bias = at(e, o->x);
        const uint32_t *scales = p->words + o->p[0];
        uint32_t *fi = at(e, o->out);
        for (uint32_t n = 0; n < T[o->x].dims.c; n += 16u) {
            memcpy(fi, scales + n * 2u, 32u * sizeof(*fi));
            for (uint32_t i = 0; i < 16u; ++i) {
                fi[32u + i * 2u] = (uint32_t)bias[n + i];
                fi[33u + i * 2u] = 0;
            }
            fi += 64;
        }
        break;
    }
    case Q16_NORMVALS_U16: {
        const uint32_t *in[] = {at(e, o->x), at(e, o->y), at(e, o->z), at(e, o->p[0])};
        uint32_t lanes = T[o->out].dims.c;
        int32_t *out = at(e, o->out);
        float params[4];
        memcpy(params, &o->p[1], sizeof(params));
        double epsilon = (double)params[0] / ((double)params[1] * params[1]);
        double gain_scale = (double)params[3] / params[2];
        const uint64_t max_sum = 256u * UINT16_MAX;
        const uint64_t max_square = max_sum * UINT16_MAX;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            uint64_t sum = 0, square = 0;
            for (uint32_t i = 0; i < 4u; ++i) {
                uint32_t partial_sum = in[i][lane];
                uint64_t partial_square = (uint64_t)in[i][lanes * 2u + lane] << 32 |
                                          in[i][lanes + lane];
                if (partial_sum > max_sum || partial_square > max_square)
                    return Q16_OPERATOR;
                sum += partial_sum;
                square += partial_square;
            }
            if (sum * sum > square * 1024u)
                return Q16_OPERATOR;
            double variance = (double)(square * 1024u - sum * sum) / (1024u * 1024u);
            /* Reference arithmetic. QAIRT's Qf32 reciprocal sqrt and
             * conversions can differ in the least significant output bits. */
            double gain = gain_scale / sqrt(variance + epsilon);
            int exponent;
            double mantissa = frexp(gain, &exponent);
            if (exponent > 16 || exponent < -15)
                return Q16_OPERATOR;
            int32_t fixed_gain = (int32_t)(mantissa * 65536.0 + 0.5);
            if (fixed_gain == 65536) {
                fixed_gain = 32768;
                if (++exponent > 16)
                    return Q16_OPERATOR;
            }
            out[lane] = -(int32_t)(sum * 32u);
            out[lanes + lane] = fixed_gain;
            out[lanes * 2u + lane] = exponent;
        }
        break;
    }
    case Q16_PW_XSB_U16: case Q16_PW_XPB_U16: {
        U16PointwiseJob job = {e, o};
        void (*task)(void *, uint32_t, uint32_t) = o->kind == Q16_PW_XSB_U16 ? pw_xsb_task : pw_xpb_task;
        if (e->parallel)
            e->parallel(task, &job);
        else
            task(&job, 0, 1);
        break;
    }
    case Q16_SOFTMAX_U16:
        softmax_u16(e, o);
        break;
    case Q16_ADD_U16: case Q16_MUL_U16: case Q16_GELU_U16:
        ew_u16(e, o);
        break;
    case Q16_NORMALIZE_U16: {
        const Q16Tensor *src = &T[o->x], *dst = &T[o->out];
        const uint32_t *src_blocks = p->words + src->block_first, *dst_blocks = p->words + dst->block_first;
        const int32_t *norm = at(e, o->y), *beta = at(e, o->p[0]);
        const int16_t *gamma = at(e, o->z);
        uint32_t lanes = src->dims.w * 8u;
        for (uint32_t lane = 0; lane < lanes; ++lane)
            if (norm[lane] > 0 || norm[lanes + lane] < 0 || norm[lanes + lane] > 65535 ||
                norm[2u * lanes + lane] < -15 || norm[2u * lanes + lane] > 16)
                return Q16_OPERATOR;
        for (uint32_t strip = 0; strip < src->dims.w / 4u; ++strip)
            for (uint32_t group = 0; group < 8u; ++group) {
                const uint16_t *in = (const uint16_t *)(e->vtcm + src_blocks[strip * 8u + group]);
                uint16_t *out = (uint16_t *)(e->vtcm + dst_blocks[strip * 8u + group]);
                for (uint32_t pair = 0; pair < 16u; ++pair) {
                    uint32_t lane = strip * 32u + pair * 2u;
                    for (uint32_t pixel = 0; pixel < 2u; ++pixel) {
                        uint32_t token = lane + pixel;
                        uint32_t shift = 31 - norm[2u * lanes + token];
                        int64_t rounding = (int64_t)1 << (shift - 1u);
                        for (uint32_t c = 0; c < 32u; ++c) {
                            uint32_t index = pair * 64u + c * 2u + pixel;
                            uint32_t channel = group * 32u + c;
                            int64_t centered = (int64_t)in[index] * 32768 + norm[token];
                            int64_t scaled = centered * norm[lanes + token] * gamma[channel * 2u + pixel];
                            int64_t code = ((scaled + rounding) >> shift) + beta[channel];
                            out[index] = code < 0 ? 0 : code > 65535 ? 65535 : (uint16_t)code;
                        }
                    }
                }
            }
        break;
    }
    case Q16_REARRANGE_U16: {
        int inverse = T[o->out].dims.h == 1u;
        const Q16Tensor *narrow = &T[inverse ? o->out : o->x];
        const Q16Tensor *wide = &T[inverse ? o->x : o->out];
        uint32_t groups = narrow->dims.c / 32u, columns = narrow->full.w / 4u;
        const uint32_t *narrow_blocks = p->words + narrow->block_first;
        const uint32_t *wide_blocks = p->words + wide->block_first;
        for (uint32_t column = 0; column < columns; column++)
            for (uint32_t c = 0; c < groups; c++) {
                uint8_t *packed = e->vtcm + narrow_blocks[column * groups + c] + narrow->padding.h * 256u;
                uint8_t *spread = e->vtcm + wide_blocks[(column / 8u) * groups + c] + (column % 8u) * 256u;
                /* Inverse tail writes one pixel pair. Keep the other pair
                 * and the unused rows byte-identical to their prior contents. */
                if (inverse)
                    memcpy(packed, spread, column + 1u == columns ? 128u : 256u);
                else
                    memcpy(spread, packed, 256u);
            }
        break;
    }
    case Q16_SPILL_FILL: {
        if (!e->spill)
            return Q16_RANGE;
        const uint32_t *copy = p->words + o->p[0];
        for (uint32_t k = 0; k < o->p[1]; k++, copy += 3) {
            uint8_t *vtcm = e->vtcm + copy[0];
            uint8_t *ddr = e->spill + copy[1];
            if (o->p[2])
                memcpy(ddr, vtcm, copy[2]);
            else
                memcpy(vtcm, ddr, copy[2]);
        }
        break;
    }
    case Q16_SLICE_U16: {
        const Qnn16Shape in = T[o->x].dims, out = T[o->out].dims;
        const uint16_t *src = (const uint16_t *)at(e, o->x);
        uint16_t *dst = (uint16_t *)at(e, o->out);
        for (uint32_t b = 0; b < out.b; b++)
            for (uint32_t y = 0; y < out.h; y++)
                for (uint32_t x = 0; x < out.w; x++) {
                    size_t origin = (((size_t)(b + o->p[0]) * in.h + y + o->p[1]) *
                                     in.w + x + o->p[2]) * in.c + o->p[3];
                    memcpy(dst, src + origin, (size_t)out.c * sizeof(*dst));
                    dst += out.c;
                }
        break;
    }
    case Q16_ADD_SWISH: {
        Qnn16Tensor out = view(e, o->out), in = view(e, o->x);
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && small_op(e, o))
            break;
#endif
        bad |= qnn16_tensor_add_swish(&out, &in);
        break;
    }
    case Q16_SWISH: {
        Qnn16Tensor out = view(e, o->out), in = view(e, o->x);
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && small_op(e, o))
            break;
#endif
        bad |= qnn16_tensor_swish(&out, &in);
        break;
    }
    case Q16_RESHAPE: {
#ifdef __HVX__
        Qnn16Tensor tx = view(e, o->x), ty = view(e, o->out);
        if (!(e->flags & Q16_EXEC_NO_TILE) && qnn16_tile_to_compact(&tx, &ty))
            break;
#endif
        uint16_t *x = take(&cursor, &T[o->x]);
        bad |= gather(e, x, o->x);
        bad |= scatter(e, o->out, x);
        break;
    }
    case Q16_CONCAT: {
        enum { MAX_PARTS = 64 };
        const uint16_t *parts[MAX_PARTS];
        Qnn16Shape shapes[MAX_PARTS];
        uint32_t count = o->p[2];
        if (count > MAX_PARTS)
            return Q16_OPERATOR;
        if (concat_direct(e, o, count))
            break;
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) && concat_tiles(e, o, count))
            break;
#endif
        for (uint32_t i = 0; i < count; i++) {
            uint32_t id = p->words[o->p[1] + i];
            uint16_t *x = take(&cursor, &T[id]);
            bad |= gather(e, x, id);
            parts[i] = x;
            shapes[i] = T[id].dims;
        }
        uint16_t *y = take(&cursor, &T[o->out]);
        qnn16_concat_f16(y, parts, shapes, count, o->p[0]);
        bad |= scatter(e, o->out, y);
        break;
    }
    case Q16_SLICE: case Q16_SHUFFLE: case Q16_TRANSPOSE: case Q16_AVGPOOL: case Q16_X2S: {
#ifdef __HVX__
        if (!(e->flags & Q16_EXEC_NO_TILE) &&
            (o->kind == Q16_X2S || o->kind == Q16_TRANSPOSE || o->kind == Q16_AVGPOOL || o->kind == Q16_SHUFFLE)) {
            if (small_op(e, o))
                break;
        }
#endif
        uint16_t *x = take(&cursor, &T[o->x]), *y = take(&cursor, &T[o->out]);
        Qnn16Shape s = T[o->x].dims;
        bad |= gather(e, x, o->x);
        if (o->kind == Q16_SLICE)
            qnn16_slice_f16(y, x, s, o->p[0], o->p[1],
                            o->p[0] == 0 ? T[o->out].dims.b : o->p[0] == 1 ? T[o->out].dims.h
                                                          : o->p[0] == 2 ? T[o->out].dims.w
                                                                         : T[o->out].dims.c);
        else if (o->kind == Q16_SHUFFLE)
            qnn16_channel_shuffle_f16(y, x, s, o->p[0]);
        else if (o->kind == Q16_TRANSPOSE)
            qnn16_transpose_f16(y, x, s, o->p);
        else if (o->kind == Q16_AVGPOOL)
            qnn16_avgpool_valid_f16(y, x, s, o->p[0], o->p[1]);
        else
            qnn16_x2s_f16(y, x, s);
        bad |= scatter(e, o->out, y);
        break;
    }
    case Q16_ZAP: {
        uint8_t *blocks[64];
        for (uint32_t i = 0; i < o->p[1]; i += 64) {
            uint32_t n = o->p[1] - i < 64 ? o->p[1] - i : 64;
            for (uint32_t j = 0; j < n; j++)
                blocks[j] = e->vtcm + p->words[o->p[0] + i + j];
#ifdef __HVX__
            bad |= qnn16_tile_zap(blocks, n, o->p[2], o->p[3]);
#else
            bad |= qnn16_zap_blocks(blocks, n, o->p[2], o->p[3]);
#endif
        }
        break;
    }
    default:
        return Q16_OPERATOR;
    }
    return bad ? Q16_OPERATOR : Q16_OK;
}
