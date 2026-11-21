/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Q16P structural validation, shared by the host preflight and the DSP. */
#include "program.h"
#include <stddef.h>

static int span(uint64_t at, uint64_t n, uint64_t size) { return at <= size && n <= size - at; }
static uint32_t dim(Qnn16Shape s, uint32_t a) { return a == 0 ? s.b : a == 1 ? s.h : a == 2 ? s.w : s.c; }
static uint64_t elements64(Qnn16Shape s) { return (uint64_t)s.b * s.h * s.w * s.c; }
static int same(Qnn16Shape a, Qnn16Shape b) { return a.b == b.b && a.h == b.h && a.w == b.w && a.c == b.c; }
static uint32_t align128(uint64_t n) { return (uint32_t)((n + 127u) & ~(uint64_t)127u); }
static int u16_extent_ok(Qnn16Shape shape) {
    uint32_t remaining = Q16P_MAX_WORDS * 1024u;
    for (uint32_t axis = 0; axis < 4; axis++) {
        uint32_t d = dim(shape, axis);
        if (!d || d > remaining)
            return 0;
        remaining /= d;
    }
    return 1;
}
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
    case Q16_OUTPUT_U16:
        return logical(p, o->x);
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
/* Compact direct storage has no padding or row gaps. */
static int compact(const Q16Tensor *t) {
    return t->direct && same(t->dims, t->full) && !t->padding.b &&
           !t->padding.h && !t->padding.w && !t->padding.c;
}
static int is_f32(const Q16Program *p, uint32_t id) {
    return tensor_ok(p, id) && p->tensors[id].dtype == Q16_F32 && compact(&p->tensors[id]);
}
static int quantization_ok(const Q16Op *o) {
    /* Positive normal float32 scale, capped so any dequantized U16 fits F32. */
    return o->p[4] >= 0x00800000u && o->p[4] < 0x77800000u && o->p[5] <= UINT16_MAX;
}
static int io_tile_fits(Qnn16Shape tile, Qnn16Shape full, const Q16Op *o) {
    return o->p[3] <= full.b && tile.b <= full.b - o->p[3] && tile.c == full.c &&
           o->p[1] <= full.h && tile.h <= full.h - o->p[1] &&
           o->p[2] <= full.w && tile.w <= full.w - o->p[2];
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

/* A 32-token strip is eight consecutive width-4 block rows. SHARP's
 * partially populated final strip is part of the kernel's write contract. */
static int rearrange_16_ok(const Q16Program *p, const Q16Tensor *x, const Q16Tensor *out) {
    const Q16Tensor *narrow = x->dims.h == 1u ? x : out;
    const Q16Tensor *wide = narrow == x ? out : x;
    uint32_t channels = narrow->dims.c;
    Qnn16Shape narrow_dims = {1, 1, 577, channels}, narrow_full = {1, 8, 580, channels};
    Qnn16Shape wide_dims = {1, 8, 76, channels}, zero = {0, 0, 0, 0};
    if (x->dtype != Q16_U16 || out->dtype != Q16_U16 || x->direct || out->direct ||
        !channels || (channels % 32u) || !same(narrow->dims, narrow_dims) ||
        !same(narrow->full, narrow_full) || !same(wide->dims, wide_dims) ||
        !same(wide->full, wide_dims) || !same(wide->padding, zero) || narrow->padding.b ||
        narrow->padding.h >= 8u || narrow->padding.w || narrow->padding.c)
        return 0;
    uint32_t used[Q16P_VTCM_BYTES / (2048u * 32u)] = {0};
    const Q16Tensor *sides[] = {x, out};
    for (uint32_t side = 0; side < 2; side++)
        for (uint32_t i = 0; i < sides[side]->block_count; i++) {
            uint32_t offset = p->words[sides[side]->block_first + i];
            if (!vtcm_block(offset))
                return 0;
            uint32_t block = offset / 2048u, bit = 1u << (block % 32u);
            if (used[block / 32u] & bit)
                return 0;
            used[block / 32u] |= bit;
        }
    return 1;
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

static int pw_u16_placement(const Q16Program *p, const Q16Op *o) {
    const Q16Tensor *in = &p->tensors[o->x], *out = &p->tensors[o->out];
    const Q16Tensor *weight = &p->tensors[o->y], *fi = &p->tensors[o->z];
    uint32_t occupied[Q16P_COMPLETION_OFFSET / (2048u * 32u) + 1u] = {0};
    for (uint32_t i = 0; i < in->block_count; ++i) {
        uint32_t block = p->words[in->block_first + i];
        if (!vtcm_block(block))
            return 0;
        uint32_t tile = block / 2048u;
        occupied[tile / 32u] |= 1u << (tile % 32u);
    }
    for (uint32_t i = 0; i < out->block_count; ++i) {
        uint32_t block = p->words[out->block_first + i];
        if (!vtcm_block(block) ||
            (block < weight->offset + weight->bytes && weight->offset < block + 2048u) ||
            (block < fi->offset + fi->bytes && fi->offset < block + 2048u))
            return 0;
        uint32_t tile = block / 2048u, bit = 1u << (tile % 32u);
        if (occupied[tile / 32u] & bit)
            return 0;
        occupied[tile / 32u] |= bit;
    }
    return 1;
}

static int u16_same_tiles(const Q16Program *p, uint32_t x, uint32_t out) {
    if (!tensor_ok(p, x) || !tensor_ok(p, out))
        return 0;
    const Q16Tensor *a = &p->tensors[x], *b = &p->tensors[out];
    if (a->dtype != Q16_U16 || b->dtype != Q16_U16 || a->direct || b->direct ||
        !same(a->dims, b->dims) || !same(a->full, b->full) || !same(a->padding, b->padding))
        return 0;
    const uint32_t *in_blocks = p->words + a->block_first, *out_blocks = p->words + b->block_first;
    for (uint32_t i = 0; i < b->block_count; ++i) {
        if (!vtcm_block(in_blocks[i]) || !vtcm_block(out_blocks[i]))
            return 0;
        for (uint32_t j = 0; j < b->block_count; ++j)
            if (j != i && (out_blocks[i] == in_blocks[j] || out_blocks[i] == out_blocks[j]))
                return 0;
    }
    return 1;
}

static int positive_f32_bits(uint32_t bits) {
    return bits >= 0x00800000u && bits < 0x7f800000u;
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
    case Q16_MOMENTS_U16: {
        if (!tensor_ok(p, o->x) || !is_raw(p, o->out) ||
            TX.dtype != Q16_U16 || TX.direct || TO.direct != 1u || !compact(&TO) || (TO.offset & 3u))
            return 0;
        Qnn16Shape zero = {0, 0, 0, 0};
        Qnn16Shape moments = {1, 1, 3, TX.dims.w * 8u};
        if (TX.dims.b != 1u || TX.dims.h != 8u || TX.dims.c != 256u ||
            (TX.dims.w != 36u && TX.dims.w != 40u) || !same(TX.dims, TX.full) ||
            !same(TX.padding, zero) || !same(TO.dims, moments) ||
            TO.bytes != 3u * moments.c * sizeof(uint32_t))
            return 0;
        for (uint32_t i = 0; i < TX.block_count; i++) {
            uint32_t block = p->words[TX.block_first + i];
            if (!vtcm_block(block) || (block < TO.offset + TO.bytes && TO.offset < block + 2048u))
                return 0;
        }
        return 1;
    }
    case Q16_NORMVALS_U16: {
        if (!is_raw(p, o->out) || TO.direct != 1u || !compact(&TO) || (TO.offset & 3u) ||
            TO.dims.b != 1u || TO.dims.h != 1u || TO.dims.w != 3u ||
            (TO.dims.c != 288u && TO.dims.c != 320u) || TO.bytes != 12u * TO.dims.c)
            return 0;
        const uint32_t ids[] = {o->x, o->y, o->z, o->p[0]};
        for (uint32_t i = 0; i < 4u; ++i) {
            if (!is_raw(p, ids[i]))
                return 0;
            const Q16Tensor *in = &p->tensors[ids[i]];
            if (in->direct != 1u || !compact(in) || (in->offset & 3u) ||
                !same(in->dims, TO.dims) || in->bytes != TO.bytes ||
                (in->offset < TO.offset + TO.bytes && TO.offset < in->offset + in->bytes))
                return 0;
        }
        for (uint32_t i = 1; i <= 4u; ++i)
            if (o->p[i] < 0x00800000u || o->p[i] >= 0x7f800000u)
                return 0;
        return 1;
    }
    case Q16_NORMALIZE_U16: {
        if (!tensor_ok(p, o->x) || !tensor_ok(p, o->out) || !is_raw(p, o->y) ||
            !is_f16(p, o->z) || !is_raw(p, o->p[0]))
            return 0;
        Qnn16Shape zero = {0, 0, 0, 0};
        if (TX.dtype != Q16_U16 || TO.dtype != Q16_U16 || TX.direct || TO.direct ||
            TX.dims.b != 1u || TX.dims.h != 8u || TX.dims.c != 256u ||
            (TX.dims.w != 36u && TX.dims.w != 40u) || !same(TX.dims, TX.full) ||
            !same(TX.padding, zero) || !same(TX.dims, TO.dims) ||
            !same(TO.dims, TO.full) || !same(TO.padding, zero))
            return 0;
        const Q16Tensor *tables[] = {&p->tensors[o->y], &p->tensors[o->z], &p->tensors[o->p[0]]};
        const Qnn16Shape shapes[] = {{1, 1, 3, TX.dims.w * 8u}, {1, 1, 1, 512}, {1, 1, 1, 256}};
        const uint32_t sizes[] = {96u * TX.dims.w, 1024u, 1024u};
        for (uint32_t i = 0; i < 3u; ++i)
            if (tables[i]->direct != 1u || !compact(tables[i]) || (tables[i]->offset & 3u) ||
                !same(tables[i]->dims, shapes[i]) || tables[i]->bytes != sizes[i])
                return 0;
        for (uint32_t i = 0; i < TX.block_count; ++i)
            if (!vtcm_block(p->words[TX.block_first + i]))
                return 0;
        for (uint32_t i = 0; i < TO.block_count; ++i) {
            uint32_t block = p->words[TO.block_first + i];
            if (!vtcm_block(block))
                return 0;
            for (uint32_t j = 0; j < TX.block_count; ++j)
                if (block == p->words[TX.block_first + j])
                    return 0;
            for (uint32_t j = 0; j < i; ++j)
                if (block == p->words[TO.block_first + j])
                    return 0;
            for (uint32_t j = 0; j < 3u; ++j)
                if (block < tables[j]->offset + sizes[j] && tables[j]->offset < block + 2048u)
                    return 0;
        }
        return 1;
    }
    case Q16_PW_XSB_U16: {
        if (!tensor_ok(p, o->x) || !tensor_ok(p, o->out) || !is_raw(p, o->y) || !is_raw(p, o->z))
            return 0;
        const Q16Tensor *weight = &p->tensors[o->y], *fi = &p->tensors[o->z];
        uint32_t k = TX.dims.c, n = TO.dims.c;
        Qnn16Shape zero = {0, 0, 0, 0}, input = {1, 8, 76, k}, output = {1, 8, 76, n};
        Qnn16Shape weights = {1, 1, k, n}, epilogue = {1, n / 32u, 1, 128};
        if (TX.dtype != Q16_U16 || TO.dtype != Q16_U16 || TX.direct || TO.direct ||
            (k != 1024u && k != 4096u) || (n != 64u && n != 256u) ||
            !same(TX.dims, input) || !same(TX.full, input) || !same(TX.padding, zero) ||
            !same(TO.dims, output) || !same(TO.full, output) || !same(TO.padding, zero) ||
            weight->direct != 1u || !compact(weight) || !same(weight->dims, weights) || weight->bytes != k * n ||
            fi->direct != 1u || !compact(fi) || !same(fi->dims, epilogue) ||
            fi->bytes != n * 16u || (fi->offset & 3u) ||
            o->p[0] < 0x00800000u || o->p[0] >= 0x7f800000u || o->p[1] > UINT16_MAX || o->p[2] > 1u)
            return 0;
        return pw_u16_placement(p, o);
    }
    case Q16_PW_XPB_U16: {
        if (!tensor_ok(p, o->x) || !tensor_ok(p, o->out) || !is_raw(p, o->y) || !is_raw(p, o->z))
            return 0;
        const Q16Tensor *weight = &p->tensors[o->y], *fi = &p->tensors[o->z];
        uint32_t k = TX.dims.c, n = TO.dims.c, w = TX.dims.w;
        if (TX.dtype != Q16_U16 || TO.dtype != Q16_U16 || TX.direct || TO.direct ||
            !((k == 64u && (n == 65u || n == 256u)) || (k == 577u && n == 64u)) ||
            (w != 12u && w != 32u))
            return 0;
        uint32_t nb = (n + 31u) / 32u;
        Qnn16Shape input = {1, 8, w, k}, output = {1, 8, w, n}, zero = {0, 0, 0, 0};
        Qnn16Shape full_x = {1, 8, w, (k + 31u) & ~31u}, full_out = {1, 8, w, nb * 32u};
        Qnn16Shape packed = {1, 1, (k + 3u) & ~3u, n * 2u};
        Qnn16Shape full_packed = {1, 1, packed.w, (packed.c + 31u) & ~31u};
        Qnn16Shape epilogue = {1, nb, 1, 128};
        return same(TX.dims, input) && same(TX.full, full_x) && same(TX.padding, zero) &&
               same(TO.dims, output) && same(TO.full, full_out) && same(TO.padding, zero) &&
               weight->direct == 1u && same(weight->dims, packed) && same(weight->full, full_packed) &&
               same(weight->padding, zero) && weight->bytes >= full_packed.w * full_packed.c &&
               fi->direct == 1u && compact(fi) && same(fi->dims, epilogue) &&
               fi->bytes == nb * 512u && !(fi->offset & 3u) &&
               o->p[0] >= 0x00800000u && o->p[0] < 0x7f800000u && o->p[1] <= UINT16_MAX &&
               pw_u16_placement(p, o);
    }
    case Q16_REARRANGE_U16:
        return tensor_ok(p, o->x) && tensor_ok(p, o->out) && rearrange_16_ok(p, &TX, &TO);
    case Q16_SPILL_FILL: {
        uint32_t first = o->p[0], count = o->p[1];
        if (h->version != Q16P_SPILL_VERSION || o->p[2] > 1u ||
            !count || !span(first, (uint64_t)count * 3u, h->words))
            return 0;
        for (uint32_t k = 0; k < count; k++) {
            const uint32_t *copy = p->words + first + k * 3u;
            if (!copy[2] || ((copy[0] | copy[1] | copy[2]) & 2047u) ||
                !span(copy[0], copy[2], Q16P_COMPLETION_OFFSET) ||
                !span(copy[1], copy[2], h->reserved[1]))
                return 0;
        }
        return 1;
    }
    case Q16_SLICE_U16:
        if (!tensor_ok(p, o->x) || !tensor_ok(p, o->out) ||
            TX.dtype != Q16_U16 || TO.dtype != Q16_U16 || !compact(&TX) || !compact(&TO))
            return 0;
        /* Row copies assume disjoint storage, even when individual crops have gaps. */
        if (TX.direct == TO.direct && (uint64_t)TX.offset + TX.bytes > TO.offset &&
            (uint64_t)TO.offset + TO.bytes > TX.offset)
            return 0;
        for (uint32_t axis = 0; axis < 4; axis++)
            if (!span(o->p[axis], dim(TO.dims, axis), dim(TX.dims, axis)))
                return 0;
        return 1;
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
        return io_tile_fits(TX.dims, p->io[h->inputs + o->p[0]].dims, o);
    }
    case Q16_INPUT_U16:
        return o->p[0] < h->inputs && tensor_ok(p, o->out) && TO.dtype == Q16_U16 &&
               compact(&TO) && quantization_ok(o) && io_tile_fits(TO.dims, p->io[o->p[0]].dims, o);
    case Q16_OUTPUT_U16:
        return o->p[0] < h->outputs && tensor_ok(p, o->x) && TX.dtype == Q16_U16 &&
               quantization_ok(o) && io_tile_fits(TX.dims, p->io[h->inputs + o->p[0]].dims, o);
    case Q16_FORMAT:
        return tensor_ok(p, o->x) && tensor_ok(p, o->out) && same(TX.dims, TO.dims) &&
               TX.dtype == TO.dtype && (TX.dtype == Q16_F16 || TX.dtype == Q16_U16);
    case Q16_LINCLIP_U16: {
        if (!tensor_ok(p, o->x) || !tensor_ok(p, o->out) || TX.dtype != Q16_U16 || TO.dtype != Q16_U16 ||
            !same(TX.dims, TO.dims) || !o->p[0] || o->p[0] > (1u << 18) ||
            !o->p[1] || o->p[1] > 31u || o->p[2] > UINT16_MAX || o->p[3] > UINT16_MAX)
            return 0;
        if (TX.direct || TO.direct)
            return compact(&TX) && compact(&TO) &&
                   (TX.direct != TO.direct || TO.offset >= TX.offset + TX.bytes ||
                    TX.offset >= TO.offset + TO.bytes || TX.offset == TO.offset);
        if (!same(TX.full, TO.full) || !same(TX.padding, TO.padding))
            return 0;
        const uint32_t *in = p->words + TX.block_first, *out = p->words + TO.block_first;
        for (uint32_t i = 0; i < TO.block_count; ++i) {
            if (!vtcm_block(in[i]) || !vtcm_block(out[i]))
                return 0;
            for (uint32_t j = 0; j < TO.block_count; ++j)
                if (j != i && (out[i] == in[j] || out[i] == out[j]))
                    return 0;
        }
        return 1;
    }
    case Q16_PACK_XPB_U16: {
        if (!tensor_ok(p, o->x) || !is_raw(p, o->out) || TX.dtype != Q16_U16 ||
            TX.direct != 1u || TO.direct != 1u || !compact(&TX) || TX.dims.b != 1u || TX.dims.h != 1u)
            return 0;
        uint32_t k = TX.dims.w, n = TX.dims.c;
        if (!((k == 577u && n == 64u) || (k == 64u && (n == 65u || n == 256u))))
            return 0;
        Qnn16Shape dims = {1, 1, (k + 3u) & ~3u, n * 2u};
        Qnn16Shape full = {1, 1, dims.w, (dims.c + 31u) & ~31u}, zero = {0, 0, 0, 0};
        return same(TO.dims, dims) && same(TO.full, full) && same(TO.padding, zero) &&
               TO.bytes >= full.w * full.c &&
               (TX.offset >= TO.offset + TO.bytes || TO.offset >= TX.offset + TX.bytes);
    }
    case Q16_XPB_BIAS: {
        if (!is_raw(p, o->x) || !is_raw(p, o->out) || TX.direct != 1u || TO.direct != 1u ||
            !compact(&TO) || (TO.offset & 3u) || o->p[0] > UINT16_MAX)
            return 0;
        uint32_t n = TX.dims.c;
        if (n != 130u && n != 512u)
            return 0;
        uint32_t nb = n == 130u ? 96u : 256u;
        Qnn16Shape weights = {1, 1, 64, n}, full = {1, 1, 64, (n + 31u) & ~31u};
        Qnn16Shape bias = {1, 1, 1, nb}, zero = {0, 0, 0, 0};
        return same(TX.dims, weights) && same(TX.full, full) && same(TX.padding, zero) &&
               TX.bytes >= full.w * full.c && same(TO.dims, bias) && TO.bytes == nb * 4u &&
               (TX.offset >= TO.offset + TO.bytes || TO.offset >= TX.offset + TX.bytes);
    }
    case Q16_XPB_FI: {
        if (!is_raw(p, o->x) || !is_raw(p, o->out) || TX.direct != 1u || TO.direct != 1u ||
            !compact(&TX) || !compact(&TO) || ((TX.offset | TO.offset) & 3u))
            return 0;
        uint32_t n = TX.dims.c;
        Qnn16Shape bias = {1, 1, 1, n}, fi = {1, n / 32u, 1, 128};
        return (n == 96u || n == 256u) && same(TX.dims, bias) && same(TO.dims, fi) &&
               TX.bytes == n * 4u && TO.bytes == n * 16u && span(o->p[0], n * 2u, h->words) &&
               (TX.offset >= TO.offset + TO.bytes || TO.offset >= TX.offset + TX.bytes);
    }
    case Q16_SOFTMAX_U16: {
        if (!u16_same_tiles(p, o->x, o->out) || !positive_f32_bits(o->p[0]))
            return 0;
        Qnn16Shape dims = {1, 8, TX.dims.w, 577}, full = {1, 8, TX.dims.w, 608}, zero = {0, 0, 0, 0};
        return (TX.dims.w == 4u || TX.dims.w == 8u) && same(TX.dims, dims) &&
               same(TX.full, full) && same(TX.padding, zero);
    }
    case Q16_MUL_U16: {
        if (!u16_same_tiles(p, o->x, o->out) || !tensor_ok(p, o->y))
            return 0;
        const Q16Tensor *y = &p->tensors[o->y];
        Qnn16Shape scalar = {1, 1, 1, TX.dims.c}, full = {1, 8, 4, TX.dims.c}, zero = {0, 0, 0, 0};
        if (y->dtype != Q16_U16 || y->direct || !same(y->dims, scalar) || !same(y->full, full) ||
            !same(y->padding, zero) || (TX.dims.c != 32u && TX.dims.c != 64u) ||
            !same(TX.dims, TX.full) || !same(TX.padding, zero) ||
            !o->p[0] || o->p[0] > 0x01000000u || o->p[1] > UINT16_MAX || o->p[2] > UINT16_MAX)
            return 0;
        for (uint32_t i = 0; i < y->block_count; ++i)
            if (!vtcm_block(p->words[y->block_first + i]))
                return 0;
        return 1;
    }
    case Q16_ADD_U16:
        return u16_same_tiles(p, o->x, o->out) && u16_same_tiles(p, o->y, o->out) &&
               o->p[0] && o->p[0] <= 32768u && o->p[1] && o->p[1] <= 32768u &&
               o->p[2] && o->p[2] <= 30u;
    case Q16_GELU_U16:
        return u16_same_tiles(p, o->x, o->out) && positive_f32_bits(o->p[0]) &&
               positive_f32_bits(o->p[1]) && o->p[2] <= UINT16_MAX && o->p[3] <= UINT16_MAX;
    case Q16_MINMAX: case Q16_SIGMOID: case Q16_SWISH: case Q16_SOFTMAX:
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
        if (!tensor_ok(p, o->out) || (TO.dtype != Q16_F16 && TO.dtype != Q16_U16) ||
            axis > 3 || !count || !span(at, count, h->words))
            return 0;
        for (uint32_t i = 0; i < count; i++) {
            uint32_t id = p->words[at + i];
            if (!tensor_ok(p, id) || T[id].dtype != TO.dtype)
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
        if (!tensor_ok(p, o->x) || !tensor_ok(p, o->out) || TX.dtype != TO.dtype ||
            (TX.dtype != Q16_F16 && !(TX.dtype == Q16_U16 && compact(&TX) && compact(&TO))))
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
        (h->version < Q16P_VERSION || h->version > Q16P_SPILL_VERSION) || h->bytes != image_bytes || h->constant_bytes != constant_bytes ||
        h->vtcm_bytes != Q16P_VTCM_BYTES || !h->tensors || h->tensors > Q16P_MAX_TENSORS || !h->ops ||
        h->ops > Q16P_MAX_OPS || h->words > Q16P_MAX_WORDS || !h->inputs || h->inputs > Q16P_MAX_IO ||
        !h->outputs || h->outputs > Q16P_MAX_IO || !h->exchange_bytes || (h->scratch_bytes & 127u))
        return Q16_HEADER;
    if (h->version >= Q16P_ARENA_VERSION &&
        ((h->reserved[0] & 127u) || h->reserved[0] > h->scratch_bytes))
        return Q16_HEADER;
    if (h->version == Q16P_SPILL_VERSION &&
        (!h->reserved[1] || (h->reserved[1] & 2047u) || h->reserved[1] > Q16P_SPILL_MAX_BYTES))
        return Q16_HEADER;
    for (uint32_t i = h->version - 1u; i < 8; i++)
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
        if (t->dtype > Q16_U16 || t->reserved[0] || t->reserved[1] ||
            (t->dtype == Q16_U16 && !u16_extent_ok(t->full)))
            return Q16_RANGE;
        for (uint32_t a = 0; a < 4; a++)
            if (dim(t->dims, a) > dim(t->full, a) ||
                (uint64_t)dim(t->padding, a) + dim(t->dims, a) > dim(t->full, a))
                return Q16_SHAPE;
        if (t->direct) {
            uint32_t esize = t->dtype == Q16_RAW ? 1u : t->dtype == Q16_F32 ? 4u : 2u;
            if (t->block_first || t->block_count || !t->bytes ||
                (t->dtype == Q16_U16 && (t->offset & 1u)) ||
                (t->direct != 1u && (t->direct != 2u || h->version < Q16P_ARENA_VERSION)) ||
                !span(t->offset, t->bytes, t->direct == 2u ? h->reserved[0] : Q16P_COMPLETION_OFFSET) ||
                (t->dtype != Q16_RAW && t->bytes < elements64(t->full) * esize))
                return Q16_RANGE;
        } else {
            if ((t->dtype != Q16_F16 && t->dtype != Q16_U16) || t->offset || t->bytes ||
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
            (h->version >= Q16P_ARENA_VERSION ? h->reserved[0] : 0u)) {
            *bad_op = i;
            return Q16_SCRATCH;
        }
    }
    return Q16_OK;
}
