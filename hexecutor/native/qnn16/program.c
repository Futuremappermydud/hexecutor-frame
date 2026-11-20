/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Q16P structural validation, shared by the host preflight and the DSP. */
#include "program.h"
#include <stddef.h>

static int span(uint64_t at, uint64_t n, uint64_t size) { return at <= size && n <= size - at; }
static uint32_t dim(Qnn16Shape s, uint32_t a) { return a == 0 ? s.b : a == 1 ? s.h : a == 2 ? s.w : s.c; }
static uint64_t elements64(Qnn16Shape s) { return (uint64_t)s.b * s.h * s.w * s.c; }
static int same(Qnn16Shape a, Qnn16Shape b) { return a.b == b.b && a.h == b.h && a.w == b.w && a.c == b.c; }
static uint32_t align128(uint64_t n) { return (uint32_t)((n + 127u) & ~(uint64_t)127u); }
static uint32_t blocks_of(Qnn16Shape f) {
    return f.b * ((f.h + 7u) / 8u) * ((f.w + 3u) / 4u) * ((f.c + 31u) / 32u);
}

uint32_t q16_tensor_elements(const Q16Tensor *t) { return (uint32_t)elements64(t->dims); }

static uint32_t logical(const Q16Program *p, uint32_t id) {
    return align128(elements64(p->tensors[id].dims) * 2u);
}

static uint64_t f32(uint64_t n) { return ((n * 4u) + 127u) & ~(uint64_t)127u; }
/* Float path (exec.c): input, output, prepared weights, bias/lo/hi. */
static uint32_t conv_scratch(const Q16Program *p, const Q16Op *o) {
    const Qnn16Shape x = p->tensors[o->x].dims, y = p->tensors[o->out].dims;
    uint64_t oc = y.c, w;
    switch (o->kind) {
    case Q16_DW3X3: case Q16_DW3X3_S2: w = oc * 9u; break;
    case Q16_DW1X1: w = oc * 2u; break;
    case Q16_CONV3X3: w = oc * 9u * x.c; break;
    case Q16_CONV3X3_S2: w = oc * 9u * (x.c / 4u); break;
    case Q16_PW1X1: case Q16_PW_SWISH: case Q16_PW_TANH: w = oc * x.c; break;
    default: w = 0; oc = 0; break; /* S2D */
    }
    uint64_t n = f32(elements64(x)) + f32(elements64(y)) + f32(w) + 3u * f32(oc);
    return n > 0xffffffffu ? 0xffffffffu : (uint32_t)n;
}

uint32_t q16_op_scratch(const Q16Program *p, const Q16Op *o) {
    switch (o->kind) {
    case Q16_DW1X1: case Q16_DW3X3: case Q16_CONV3X3: case Q16_DW3X3_S2: case Q16_PW1X1: case Q16_PW_SWISH: case Q16_PW_TANH: case Q16_CONV3X3_S2: case Q16_S2D:
        return conv_scratch(p, o);
    case Q16_MINMAX: case Q16_SIGMOID: case Q16_SWISH: case Q16_RESHAPE: case Q16_SLICE:
    case Q16_SHUFFLE: case Q16_TRANSPOSE: case Q16_AVGPOOL: case Q16_X2S: case Q16_FORMAT: case Q16_ADD_SWISH:
        return logical(p, o->x) + logical(p, o->out);
    case Q16_REDUCE_SUM: case Q16_REDUCE_MEAN:
        return logical(p, o->x) + align128(elements64(p->tensors[o->out].dims) * 4u);
    case Q16_L2NORM: /* logical input, then one float32 sum per element (an upper bound on the sums) */
        return logical(p, o->x) + align128(elements64(p->tensors[o->x].dims) * 4u);
    case Q16_AFFINE:
        return logical(p, o->x);
    case Q16_INSTANCE_NORM:
        return logical(p, o->x) + logical(p, o->z);
    case Q16_WELFORD_PARTIAL:
        return logical(p, o->x);
    case Q16_MUL:
        return logical(p, o->x) + logical(p, o->y) + logical(p, o->out);
    case Q16_CONCAT: {
        uint64_t n = logical(p, o->out);
        for (uint32_t i = 0; i < o->p[2]; i++)
            n += logical(p, p->words[o->p[1] + i]);
        return n > 0xffffffffu ? 0xffffffffu : (uint32_t)n;
    }
    case Q16_CAST_F16:
        return logical(p, o->out);
    case Q16_OUTPUT:
        return align128(elements64(p->tensors[o->x].dims) * 4u);
    case Q16_CONV1X1: /* block pointer tables; 8 bytes per pointer on any host */
        return align128(8u * ((uint64_t)o->p[0] * o->p[1] + (uint64_t)o->p[0] * o->p[2])) + 128u;
    default:
        return 0;
    }
}

static int tensor_ok(const Q16Program *p, uint32_t id) { return id < p->header->tensors; }
static int is_f16(const Q16Program *p, uint32_t id) {
    return tensor_ok(p, id) && p->tensors[id].dtype == Q16_F16;
}
static int is_raw(const Q16Program *p, uint32_t id) {
    return tensor_ok(p, id) && p->tensors[id].direct && p->tensors[id].dtype == Q16_RAW;
}
/* Contiguous float32: direct, no padding, full == dims. */
static int is_f32(const Q16Program *p, uint32_t id) {
    if (!tensor_ok(p, id))
        return 0;
    const Q16Tensor *t = &p->tensors[id];
    return t->direct && t->dtype == Q16_F32 && same(t->dims, t->full) && !t->padding.b &&
           !t->padding.h && !t->padding.w && !t->padding.c;
}
static int weights(const Q16Program *p, uint32_t w, uint32_t fi, uint64_t wbytes, uint32_t groups,
                   uint32_t tables) {
    return is_raw(p, w) && is_raw(p, fi) && (tables == 1 || tables == 2) && groups &&
           p->tensors[w].bytes >= wbytes &&
           p->tensors[fi].bytes >= (uint64_t)groups * tables * 256u;
}
static int vtcm_block(uint32_t off) {
    return !(off & 2047u) && off <= Q16P_COMPLETION_OFFSET - 2048u;
}

/* Optional VTCM staging blocks of the HMX 3x3 forms: p[3] word index, p[4] count. */
static int stage_blocks(const Q16Program *p, const Q16Op *o) {
    if (!o->p[4])
        return 1;
    if (!span(o->p[3], o->p[4], p->header->words))
        return 0;
    for (uint32_t i = 0; i < o->p[4]; i++)
        if (!vtcm_block(p->words[o->p[3] + i]))
            return 0;
    return 1;
}

static int op_ok(const Q16Program *p, const Q16Op *o) {
    const Q16Header *h = p->header;
    const Q16Tensor *T = p->tensors;
#define TX T[o->x]
#define TO T[o->out]
    switch (o->kind) {
    case Q16_COPY:
        return tensor_ok(p, o->out) && TO.direct && o->p[1] && span(o->p[0], o->p[1], h->constant_bytes) &&
               o->p[1] <= TO.bytes;
    case Q16_ZERO:
        return tensor_ok(p, o->out);
    case Q16_CONV1X1: {
        uint32_t m = o->p[0], k = o->p[1], n = o->p[2], tables = o->p[3], at = o->p[4];
        if (!is_f16(p, o->x) || !is_f16(p, o->out) || TX.direct || TO.direct || !m || !k || !n ||
            !weights(p, o->y, o->z, (uint64_t)n * k * 2048u, n, tables) ||
            !span(at, (uint64_t)m * k + (uint64_t)m * n, h->words))
            return 0;
        for (uint64_t i = 0; i < (uint64_t)m * k + (uint64_t)m * n; i++)
            if (!vtcm_block(p->words[at + i]))
                return 0;
        return 1;
    }
    case Q16_DW1X1: {
        if (!is_f16(p, o->x) || !is_f16(p, o->out))
            return 0;
        uint32_t oc = TO.dims.c, groups = (oc + 31u) / 32u;
        return TX.dims.c >= ((oc - 1u) / 32u) * 64u + 32u + ((oc - 1u) % 32u) + 1u && TX.dims.b == TO.dims.b && TX.dims.h == TO.dims.h &&
               TX.dims.w == TO.dims.w && weights(p, o->y, o->z, (uint64_t)groups * 2048u, groups, o->p[0]);
    }
    case Q16_PW1X1: {
        if (!is_f16(p, o->x) || !is_f16(p, o->out))
            return 0;
        uint32_t ic = o->p[1], oc = TO.dims.c, groups = (oc + 31u) / 32u;
        return ic && TX.dims.c <= ic && TX.dims.b == TO.dims.b && TX.dims.h == TO.dims.h &&
               TX.dims.w == TO.dims.w && stage_blocks(p, o) &&
               weights(p, o->y, o->z, (uint64_t)groups * ((ic + 1u) / 2u) * 128u, groups, o->p[0]);
    }
    case Q16_DW3X3: {
        if (!is_f16(p, o->x) || !is_f16(p, o->out) || TX.dims.h < 3 || TX.dims.w < 3)
            return 0;
        uint32_t groups = (TX.dims.c + 31u) / 32u;
        return TO.dims.b == TX.dims.b && TO.dims.h == TX.dims.h - 2 && TO.dims.w == TX.dims.w - 2 &&
               TO.dims.c == TX.dims.c && weights(p, o->y, o->z, (uint64_t)groups * 1152u, groups, o->p[0]);
    }
    case Q16_CONV3X3: {
        if (!is_f16(p, o->x) || !is_f16(p, o->out) || TX.dims.h < 3 || TX.dims.w < 3)
            return 0;
        uint32_t ic = o->p[1], oc = o->p[2], ng = (oc + 31u) / 32u, kg = (ic + 31u) / 32u;
        return ic && TX.dims.c <= ic && TO.dims.c == oc && TO.dims.b == TX.dims.b &&
               TO.dims.h == TX.dims.h - 2 && TO.dims.w == TX.dims.w - 2 && stage_blocks(p, o) &&
               weights(p, o->y, o->z, (uint64_t)ng * kg * 9u * 2048u, ng, o->p[0]);
    }
    case Q16_PW_SWISH: {
        uint32_t ic = o->p[0], oc = TO.dims.c, ng = (oc + 31u) / 32u;
        return is_f16(p, o->x) && is_f16(p, o->out) && is_raw(p, o->y) && is_raw(p, o->z) &&
               ic && TX.dims.c <= ic && TX.dims.b == TO.dims.b && TX.dims.h == TO.dims.h && TX.dims.w == TO.dims.w &&
               p->tensors[o->y].bytes >= (uint64_t)ng * ((ic + 1u) / 2u) * 128u &&
               p->tensors[o->z].bytes >= (uint64_t)ng * 5u * 256u;
    }
    case Q16_PW_TANH: {
        uint32_t ic = o->p[0], oc = TO.dims.c, ng = (oc + 31u) / 32u;
        return is_f16(p, o->x) && is_f16(p, o->out) && is_raw(p, o->y) && is_raw(p, o->z) &&
               ic && TX.dims.c <= ic && TX.dims.b == TO.dims.b && TX.dims.h == TO.dims.h && TX.dims.w == TO.dims.w &&
               (o->p[1] & 0x7fffffffu) && (o->p[1] & 0x7f800000u) != 0x7f800000u &&
               p->tensors[o->y].bytes >= (uint64_t)ng * ((ic + 1u) / 2u) * 128u &&
               p->tensors[o->z].bytes >= (uint64_t)ng * 6u * 256u;
    }
    case Q16_L2NORM:
        return is_f16(p, o->x) && is_f16(p, o->out) && same(TX.dims, TO.dims) && o->p[0] < 4u &&
               elements64(TX.dims);
    case Q16_AFFINE:
        return is_f16(p, o->x) && (is_f16(p, o->out) || is_f32(p, o->out)) && same(TX.dims, TO.dims);
    case Q16_RESIZE_NEAREST:
        return is_f16(p, o->x) && is_f16(p, o->out) && TX.dims.b == TO.dims.b && TX.dims.c == TO.dims.c &&
               o->p[0] < TX.dims.h && o->p[1] < TX.dims.w &&
               (TO.dims.h - 1u) / 2u < TX.dims.h - o->p[0] &&
               (TO.dims.w - 1u) / 2u < TX.dims.w - o->p[1];
    case Q16_RESIZE_BILINEAR:
        return is_f16(p, o->x) && is_f16(p, o->out) && TX.dims.b == TO.dims.b &&
               TX.dims.c == TO.dims.c && TX.dims.h && TX.dims.w;
    case Q16_DECIMATE:
        return is_f16(p, o->x) && is_f16(p, o->out) && o->p[0] && o->p[1] &&
               TO.dims.b == TX.dims.b && TO.dims.c == TX.dims.c &&
               TO.dims.h == (TX.dims.h + o->p[0] - 1u) / o->p[0] &&
               TO.dims.w == (TX.dims.w + o->p[1] - 1u) / o->p[1];
    case Q16_SLICE_F32:
        return is_f32(p, o->x) && is_f32(p, o->out) && TX.direct && TO.direct &&
               TX.dims.b == TO.dims.b && TX.dims.h == TO.dims.h && TX.dims.w == TO.dims.w &&
               o->p[0] <= TX.dims.c && TO.dims.c <= TX.dims.c - o->p[0];
    case Q16_ADD_SWISH:
        return is_f16(p, o->x) && is_f16(p, o->out) && TX.dims.b == TO.dims.b &&
               TX.dims.h == TO.dims.h && TX.dims.w == TO.dims.w &&
               TX.dims.c == 64u && TO.dims.c == 32u;
    case Q16_CONV3X3_S2: {
        if (!is_f16(p, o->x) || !is_f16(p, o->out)) return 0;
        uint32_t ic = o->p[1], oc = TO.dims.c, ng = (oc + 31u) / 32u, kg = (ic + 31u) / 32u;
        return ic && !(ic % 32u) && TX.dims.c == 4u * ic && TX.dims.b == TO.dims.b &&
               TO.dims.h + 1u <= TX.dims.h && TO.dims.w + 1u <= TX.dims.w && stage_blocks(p, o) &&
               weights(p, o->y, o->z, (uint64_t)ng * kg * 9u * 2048u, ng, o->p[0]);
    }
    case Q16_S2D:
        return is_f16(p, o->x) && is_f16(p, o->out) && TX.dims.c &&
               TO.dims.c == 4u * ((TX.dims.c + 31u) / 32u * 32u) && TO.dims.b == TX.dims.b;
    case Q16_DW3X3_S2: {
        if (!is_f16(p, o->x) || !is_f16(p, o->out))
            return 0;
        uint32_t c = TO.dims.c, groups = (c + 31u) / 32u;
        return c && TX.dims.c == 128u * groups && TX.dims.b == TO.dims.b &&
               TO.dims.h + 1 <= TX.dims.h && TO.dims.w + 1 <= TX.dims.w &&
               o->p[1] <= 1u &&
               weights(p, o->y, o->z, (uint64_t)groups * 9u * (o->p[1] ? 128u : 2048u), groups, o->p[0]);
    }
    case Q16_IM2COL:
        return is_f32(p, o->x) && is_f16(p, o->out) && TX.dims.c == 1 && TX.dims.h >= 3 &&
               TX.dims.w >= 3 && TO.dims.b == TX.dims.b && TO.dims.h == (TX.dims.h - 3) / 2 + 1 &&
               TO.dims.w == (TX.dims.w - 3) / 2 + 1 && TO.dims.c == 9;
    case Q16_INPUT:
        return is_f32(p, o->out) && o->p[0] < h->inputs && o->p[3] <= p->io[o->p[0]].dims.b &&
               TO.dims.b <= p->io[o->p[0]].dims.b - o->p[3] && p->io[o->p[0]].dims.c == TO.dims.c;
    case Q16_OUTPUT: {
        if (o->p[0] >= h->outputs || !(is_f32(p, o->x) || is_f16(p, o->x)))
            return 0;
        const Q16Io *io = &p->io[h->inputs + o->p[0]];
        return o->p[3] <= io->dims.b && TX.dims.b <= io->dims.b - o->p[3] && io->dims.c == TX.dims.c && o->p[1] <= io->dims.h &&
               TX.dims.h <= io->dims.h - o->p[1] && o->p[2] <= io->dims.w &&
               TX.dims.w <= io->dims.w - o->p[2];
    }
    case Q16_FORMAT: case Q16_MINMAX: case Q16_SIGMOID: case Q16_SWISH: case Q16_SOFTMAX:
        return is_f16(p, o->x) && is_f16(p, o->out) && same(TX.dims, TO.dims);
    case Q16_CAST_F32:
        return is_f16(p, o->x) && is_f32(p, o->out) && same(TX.dims, TO.dims);
    case Q16_CAST_F16:
        return is_f32(p, o->x) && is_f16(p, o->out) && same(TX.dims, TO.dims);
    case Q16_MUL:
        return is_f16(p, o->x) && is_f16(p, o->y) && is_f16(p, o->out) && same(TX.dims, TO.dims) &&
               (T[o->y].dims.b == 1 || T[o->y].dims.b == TX.dims.b) &&
               (T[o->y].dims.h == 1 || T[o->y].dims.h == TX.dims.h) &&
               (T[o->y].dims.w == 1 || T[o->y].dims.w == TX.dims.w) &&
               (T[o->y].dims.c == 1 || T[o->y].dims.c == TX.dims.c);
    case Q16_REDUCE_SUM: case Q16_REDUCE_MEAN:
        return is_f16(p, o->x) && is_f16(p, o->out) &&
               (TO.dims.b == 1 || TO.dims.b == TX.dims.b) &&
               (TO.dims.h == 1 || TO.dims.h == TX.dims.h) &&
               (TO.dims.w == 1 || TO.dims.w == TX.dims.w) &&
               (TO.dims.c == 1 || TO.dims.c == TX.dims.c);
    case Q16_INSTANCE_NORM:
        return is_f16(p, o->x) && is_f32(p, o->y) && is_f16(p, o->z) && is_f32(p, o->p[0]) &&
               is_f16(p, o->out) && same(TX.dims, TO.dims) &&
               T[o->y].dims.b == 1 && T[o->y].dims.h == 1 && T[o->y].dims.w == 2 &&
               T[o->y].dims.c == TX.dims.c && elements64(T[o->z].dims) == 2u * TX.dims.c &&
               elements64(T[o->p[0]].dims) == TX.dims.c;
    case Q16_GROUP_MOMENTS: {
        uint32_t group = o->p[3], gamma = o->p[0];
        return is_f32(p, o->x) && is_f32(p, o->y) && is_f32(p, o->z) && is_f32(p, gamma) &&
               is_f32(p, o->out) && same(TX.dims, TO.dims) && TX.dims.b == 1 && TX.dims.h == 1 &&
               TX.dims.w == 2 && group && !(TX.dims.c % group) &&
               elements64(T[o->y].dims) == group && same(T[o->y].dims, T[o->z].dims) &&
               elements64(T[gamma].dims) == TX.dims.c;
    }
    case Q16_MOVE:
        return tensor_ok(p, o->x) && tensor_ok(p, o->out) && TX.direct && TO.direct &&
               TX.dtype == TO.dtype && TX.bytes == TO.bytes && same(TX.dims, TO.dims) &&
               same(TX.full, TO.full) && same(TX.padding, TO.padding);
    case Q16_WELFORD_MERGE: {
        uint32_t first = o->p[0], count = o->p[1];
        if (!is_f32(p, o->x) || !is_f32(p, o->y) || !is_f32(p, o->out) || !count ||
            !span(first, count, h->words) || !same(TX.dims, T[o->y].dims) ||
            TX.dims.b != 1 || TX.dims.h != 1 || TX.dims.w != (count > 1 ? count - 1 : 1) || TX.dims.c != 32 ||
            TO.dims.b != 1 || TO.dims.h != 1 || TO.dims.w != 2 || TO.dims.c % 32)
            return 0;
        for (uint32_t i = 0; i < count; i++) {
            uint32_t id = p->words[first + i];
            if (!is_f32(p, id) || !same(T[id].dims, TO.dims)) return 0;
        }
        return 1;
    }
    case Q16_WELFORD_PARTIAL:
        return is_f16(p, o->x) && is_f32(p, o->y) && is_f32(p, o->out) &&
               TX.dims.b == 1 && T[o->y].dims.b == 1 && T[o->y].dims.h == 1 &&
               T[o->y].dims.w == TX.dims.h * TX.dims.w &&
               (T[o->y].dims.c == TX.dims.c || (T[o->y].dims.c == 32 && TX.dims.c % 32 == 0)) &&
               TO.dims.b == 1 && TO.dims.h == 1 && TO.dims.w == 2 && TO.dims.c == TX.dims.c;
    case Q16_RESHAPE:
        return is_f16(p, o->x) && is_f16(p, o->out) && elements64(TX.dims) == elements64(TO.dims);
    case Q16_CONCAT: {
        uint32_t axis = o->p[0], at = o->p[1], count = o->p[2];
        uint64_t sum = 0;
        if (!is_f16(p, o->out) || axis > 3 || !count || !span(at, count, h->words))
            return 0;
        for (uint32_t i = 0; i < count; i++) {
            uint32_t id = p->words[at + i];
            if (!is_f16(p, id))
                return 0;
            for (uint32_t a = 0; a < 4; a++)
                if (a != axis && dim(T[id].dims, a) != dim(TO.dims, a))
                    return 0;
            sum += dim(T[id].dims, axis);
        }
        return sum == dim(TO.dims, axis);
    }
    case Q16_SLICE: {
        uint32_t axis = o->p[0];
        if (!is_f16(p, o->x) || !is_f16(p, o->out) || axis > 3)
            return 0;
        for (uint32_t a = 0; a < 4; a++)
            if (a != axis && dim(TX.dims, a) != dim(TO.dims, a))
                return 0;
        return span(o->p[1], dim(TO.dims, axis), dim(TX.dims, axis));
    }
    case Q16_SHUFFLE:
        return is_f16(p, o->x) && is_f16(p, o->out) && same(TX.dims, TO.dims) && o->p[0] &&
               !(TX.dims.c % o->p[0]);
    case Q16_TRANSPOSE: {
        uint32_t seen = 0;
        if (!is_f16(p, o->x) || !is_f16(p, o->out))
            return 0;
        for (uint32_t a = 0; a < 4; a++) {
            if (o->p[a] > 3 || (seen & (1u << o->p[a])) || dim(TO.dims, a) != dim(TX.dims, o->p[a]))
                return 0;
            seen |= 1u << o->p[a];
        }
        return 1;
    }
    case Q16_AVGPOOL: {
        uint32_t k = o->p[0], s = o->p[1];
        return is_f16(p, o->x) && is_f16(p, o->out) && k && s && TX.dims.h >= k && TX.dims.w >= k &&
               TO.dims.b == TX.dims.b && TO.dims.c == TX.dims.c &&
               TO.dims.h == (TX.dims.h - k) / s + 1 && TO.dims.w == (TX.dims.w - k) / s + 1;
    }
    case Q16_X2S:
        return is_f16(p, o->x) && is_f16(p, o->out) && !(TX.dims.c % 4u) && TO.dims.b == TX.dims.b &&
               TO.dims.h == 2u * TX.dims.h && TO.dims.w == 2u * TX.dims.w && TO.dims.c == TX.dims.c / 4u;
    case Q16_ZAP:
        if (!span(o->p[0], o->p[1], h->words) || o->p[2] > 8 || o->p[3] > 4)
            return 0;
        for (uint32_t i = 0; i < o->p[1]; i++)
            if (!vtcm_block(p->words[o->p[0] + i]))
                return 0;
        return 1;
    default:
        return 0;
    }
#undef TX
#undef TO
}

uint32_t q16_program_open(Q16Program *p, const void *image, uint32_t image_bytes,
                          uint32_t constant_bytes, uint32_t *bad_op) {
    const uint8_t *b = (const uint8_t *)image;
    const Q16Header *h = (const Q16Header *)image;
    *bad_op = Q16P_NONE;
    if (!image || ((uintptr_t)image & 3u) || image_bytes < sizeof(Q16Header) || h->magic != Q16P_MAGIC ||
        (h->version != Q16P_VERSION && h->version != Q16P_ARENA_VERSION) || h->bytes != image_bytes || h->constant_bytes != constant_bytes ||
        h->vtcm_bytes != Q16P_VTCM_BYTES || !h->tensors || h->tensors > Q16P_MAX_TENSORS || !h->ops ||
        h->ops > Q16P_MAX_OPS || h->words > Q16P_MAX_WORDS || !h->inputs || h->inputs > Q16P_MAX_IO ||
        !h->outputs || h->outputs > Q16P_MAX_IO || !h->exchange_bytes || (h->scratch_bytes & 127u))
        return Q16_HEADER;
    if (h->version == Q16P_ARENA_VERSION &&
        ((h->reserved[0] & 127u) || h->reserved[0] > h->scratch_bytes))
        return Q16_HEADER;
    for (uint32_t i = h->version == Q16P_ARENA_VERSION ? 1u : 0u; i < 8; i++)
        if (h->reserved[i])
            return Q16_HEADER;
    if ((h->tensor_offset | h->op_offset | h->word_offset | h->io_offset) & 3u ||
        h->tensor_offset != sizeof(Q16Header) ||
        h->op_offset != h->tensor_offset + h->tensors * sizeof(Q16Tensor) ||
        h->word_offset != h->op_offset + h->ops * sizeof(Q16Op) ||
        h->io_offset != h->word_offset + h->words * 4u ||
        (uint64_t)h->io_offset + (uint64_t)(h->inputs + h->outputs) * sizeof(Q16Io) != image_bytes)
        return Q16_RANGE;
    p->header = h;
    p->tensors = (const Q16Tensor *)(b + h->tensor_offset);
    p->ops = (const Q16Op *)(b + h->op_offset);
    p->words = (const uint32_t *)(b + h->word_offset);
    p->io = (const Q16Io *)(b + h->io_offset);
    for (uint32_t i = 0; i < h->inputs + h->outputs; i++) {
        const Q16Io *io = &p->io[i];
        if (!elements64(io->dims) || io->bytes != elements64(io->dims) * 4u || (io->offset & 127u) ||
            !span(io->offset, io->bytes, h->exchange_bytes) || io->reserved[0] || io->reserved[1])
            return Q16_RANGE;
    }
    for (uint32_t i = 0; i < h->tensors; i++) {
        const Q16Tensor *t = &p->tensors[i];
        if (t->dtype > Q16_RAW || t->reserved[0] || t->reserved[1])
            return Q16_RANGE;
        for (uint32_t a = 0; a < 4; a++)
            if (dim(t->dims, a) > dim(t->full, a) ||
                (uint64_t)dim(t->padding, a) + dim(t->dims, a) > dim(t->full, a))
                return Q16_SHAPE;
        if (t->direct) {
            uint32_t esize = t->dtype == Q16_F16 ? 2u : t->dtype == Q16_F32 ? 4u : 1u;
            if (t->block_first || t->block_count || !t->bytes ||
                (t->direct != 1u && (t->direct != 2u || h->version != Q16P_ARENA_VERSION)) ||
                !span(t->offset, t->bytes, t->direct == 2u ? h->reserved[0] : Q16P_COMPLETION_OFFSET) ||
                (t->dtype != Q16_RAW && t->bytes < elements64(t->full) * esize))
                return Q16_RANGE;
        } else {
            if (t->dtype != Q16_F16 || t->offset || t->bytes ||
                t->block_count != blocks_of(t->full) || !span(t->block_first, t->block_count, h->words))
                return Q16_RANGE;
            for (uint32_t k = 0; k < t->block_count; k++) {
                uint32_t off = p->words[t->block_first + k];
                if (off != Q16P_NONE && !vtcm_block(off))
                    return Q16_RANGE;
            }
        }
    }
    for (uint32_t i = 0; i < h->ops; i++) {
        const Q16Op *o = &p->ops[i];
        if (!op_ok(p, o)) {
            *bad_op = i;
            return Q16_OPERATOR;
        }
        if (q16_op_scratch(p, o) > h->scratch_bytes -
            (h->version == Q16P_ARENA_VERSION ? h->reserved[0] : 0u)) {
            *bad_op = i;
            return Q16_SCRATCH;
        }
    }
    return Q16_OK;
}
