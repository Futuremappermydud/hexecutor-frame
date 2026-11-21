/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "graph.h"
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <float.h>
#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif
_Static_assert(sizeof(HxHeader) == 256, "HXFG header ABI");
_Static_assert(sizeof(HxTensor) == 32, "HXFG tensor ABI");
_Static_assert(sizeof(HxOp) == 64, "HXFG operator ABI");

uint32_t hx_tensor_bytes(const HxTensor *t) {
    uint64_t n = (uint64_t)t->h * t->w * t->stride * 4u;
    return n <= UINT32_MAX - 127u ? ((uint32_t)n + 127u) & ~127u : 0;
}
uint32_t hx_tensor_elements(const HxTensor *t) { return t->h * t->w * t->c; }
static uint64_t elements64(const HxTensor *t) { return (uint64_t)t->h * t->w * t->c; }
static int span(uint32_t at, uint64_t n, uint32_t bytes) { return at <= bytes && n <= bytes - at; }
static int overlap(uint32_t a, uint32_t an, uint32_t b, uint32_t bn) {
    return a < b + bn && b < a + an;
}
static int same(const HxTensor *a, const HxTensor *b) {
    return a->h == b->h && a->w == b->w && a->c == b->c;
}
static int constant(const HxGraph *g, uint32_t at, uint64_t n) {
    return !(at & 127u) && span(at, n, g->header->constant_bytes);
}
static int matmul_dims(const HxGraph *g, const HxOp *o,
                       uint32_t *B_out, uint32_t *M_out, uint32_t *K_out,
                       uint32_t *N_out, uint32_t *B_b_out) {
    if (o->y == HXFG_NONE || o->y >= g->header->tensors ||
        o->x >= g->header->tensors || o->out >= g->header->tensors)
        return 0;
    const HxTensor *a = &g->tensors[o->x];
    const HxTensor *b = &g->tensors[o->y];
    const HxTensor *out = &g->tensors[o->out];
    const uint32_t *p = o->p;
    uint32_t transB = p[0], transA = p[3];
    if (transB > 1 || transA > 1)
        return 0;

    uint32_t Ba = 1, Ma = 0, Ka = 0;
    if (a->w > 1) {
        Ba = a->h;
        Ma = transA ? a->c : a->w;
        Ka = transA ? a->w : a->c;
    } else {
        uint32_t b_param = p[2] > 1 ? p[2] : 1;
        if (a->h % b_param != 0)
            return 0;
        Ba = b_param;
        Ma = transA ? a->c : (a->h / b_param);
        Ka = transA ? (a->h / b_param) : a->c;
    }
    if (Ka > 4096 || Ma == 0 || Ka == 0)
        return 0;

    uint32_t Bb = 1, Kb = 0, Nb = 0;
    if (b->w > 1) {
        Bb = b->h;
        Kb = transB ? b->c : b->w;
        Nb = transB ? b->w : b->c;
    } else {
        if (Ba > 1 && b->h % Ba == 0) {
            Bb = Ba;
            Kb = transB ? b->c : (b->h / Ba);
            Nb = transB ? (b->h / Ba) : b->c;
        } else {
            Bb = 1;
            Kb = transB ? b->c : b->h;
            Nb = transB ? b->h : b->c;
        }
    }
    if (Kb != Ka || (Bb != Ba && Bb != 1) || Nb == 0)
        return 0;

    uint32_t Bout = 1, Mout = 0, Nout = 0;
    if (out->w > 1) {
        Bout = out->h;
        Mout = out->w;
        Nout = out->c;
    } else {
        if (Ba > 1) {
            if (out->h % Ba != 0)
                return 0;
            Bout = Ba;
            Mout = out->h / Ba;
            Nout = out->c;
        } else {
            Bout = 1;
            Mout = out->h;
            Nout = out->c;
        }
    }
    if (Bout != Ba || Mout != Ma || Nout != Nb)
        return 0;

    if (B_out) *B_out = Ba;
    if (M_out) *M_out = Ma;
    if (K_out) *K_out = Ka;
    if (N_out) *N_out = Nb;
    if (B_b_out) *B_b_out = Bb;
    return 1;
}

static int sdpa_dims(const HxGraph *g, const HxOp *o,
                     uint32_t *B_out, uint32_t *Lq_out, uint32_t *Lk_out,
                     uint32_t *Dk_out, uint32_t *Dv_out) {
    if (o->y == HXFG_NONE || o->z == HXFG_NONE ||
        o->x >= g->header->tensors || o->y >= g->header->tensors ||
        o->z >= g->header->tensors || o->out >= g->header->tensors)
        return 0;
    const HxTensor *q = &g->tensors[o->x];
    const HxTensor *k = &g->tensors[o->y];
    const HxTensor *v = &g->tensors[o->z];
    const HxTensor *out = &g->tensors[o->out];
    const uint32_t *p = o->p;

    uint32_t Bq = 1, Lq = 0, Dq = 0;
    if (q->w > 1) {
        Bq = q->h;
        Lq = q->w;
        Dq = q->c;
    } else {
        uint32_t b_param = p[3] > 1 ? p[3] : 1;
        if (q->h % b_param != 0)
            return 0;
        Bq = b_param;
        Lq = q->h / b_param;
        Dq = q->c;
    }
    if (Dq > 4096 || Lq == 0 || Dq == 0)
        return 0;

    uint32_t Bk = 1, Lk = 0, Dk = 0;
    if (k->w > 1) {
        Bk = k->h;
        Lk = k->w;
        Dk = k->c;
    } else {
        if (Bq > 1 && k->h % Bq == 0) {
            Bk = Bq;
            Lk = k->h / Bq;
            Dk = k->c;
        } else {
            Bk = 1;
            Lk = k->h;
            Dk = k->c;
        }
    }
    if (Dk != Dq || (Bk != Bq && Bk != 1) || Lk == 0)
        return 0;

    uint32_t Bv = 1, Lv = 0, Dv = 0;
    if (v->w > 1) {
        Bv = v->h;
        Lv = v->w;
        Dv = v->c;
    } else {
        if (Bq > 1 && v->h % Bq == 0) {
            Bv = Bq;
            Lv = v->h / Bq;
            Dv = v->c;
        } else {
            Bv = 1;
            Lv = v->h;
            Dv = v->c;
        }
    }
    if (Lv != Lk || (Bv != Bq && Bv != 1) || Dv == 0 || Dv > 4096)
        return 0;

    uint32_t Bout = 1, Lout = 0, Dout = 0;
    if (out->w > 1) {
        Bout = out->h;
        Lout = out->w;
        Dout = out->c;
    } else {
        if (Bq > 1) {
            if (out->h % Bq != 0)
                return 0;
            Bout = Bq;
            Lout = out->h / Bq;
            Dout = out->c;
        } else {
            Bout = 1;
            Lout = out->h;
            Dout = out->c;
        }
    }
    if (Bout != Bq || Lout != Lq || Dout != Dv)
        return 0;

    if (p[0] != HXFG_NONE) {
        if (p[0] >= g->header->tensors)
            return 0;
        const HxTensor *m = &g->tensors[p[0]];
        uint32_t mh = m->w > 1 ? m->h : (Bq > 1 && m->h % Bq == 0 ? Bq : 1);
        uint32_t mw = m->w > 1 ? m->w : (mh > 1 ? m->h / mh : m->h);
        uint32_t mc = m->c;
        if ((mh != Bq && mh != 1) || (mw != Lq && mw != 1) || (mc != Lk && mc != 1))
            return 0;
    }

    if (B_out) *B_out = Bq;
    if (Lq_out) *Lq_out = Lq;
    if (Lk_out) *Lk_out = Lk;
    if (Dk_out) *Dk_out = Dq;
    if (Dv_out) *Dv_out = Dv;
    return 1;
}
static float hx_float(uint32_t bits);
static int op_valid(const HxGraph *g, const HxOp *o) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p;
    uint32_t used = 0;
    switch (o->kind) {
    case HX_CONV:
        used = 11;
        break;
    case HX_DEPTHWISE:
        used = 9;
        break;
    case HX_MAXPOOL:
        used = 6;
        break;
    case HX_AVGPOOL:
        used = 7;
        break;
    case HX_ACT:
        used = 3;
        break;
    case HX_BINARY:
        used = 4;
        break;
    case HX_REDUCE:
        used = 2;
        break;
    case HX_NORM:
        used = 5;
        break;
    case HX_MATMUL:
        used = 4;
        break;
    case HX_RMSNORM:
        used = 3;
        break;
    case HX_ROPE:
        used = 7;
        break;
    case HX_SHORTCONV:
        used = 4;
        break;
    case HX_LINEAR_Q:
        used = 6;
        break;
    case HX_SDPA:
        used = 5;
        break;
    case HX_KV_WRITE:
        used = 3;
        break;
    case HX_ATTN_SOFTMAX:
        used = 4;
        break;
    case HX_ADD:
    case HX_PRELU:
    case HX_CONCAT:
        used = 1;
        break;
    case HX_PAD:
    case HX_SLICE:
    case HX_TRANSPOSE:
        used = 3;
        break;
    case HX_RESHAPE:
        used = 1;
        break;
    case HX_RESIZE:
        used = 6;
        break;
    case HX_ZEROINSERT:
        used = 4;
        break;
    case HX_GATHER_ELEMENTS:
        used = 4;
        break;
    case HX_GATHER_ND:
        used = 6;
        break;
    case HX_SCATTER_ND:
        used = 9;
        break;
    case HX_GATHER:
        used = 10;
        break;
    case HX_WHERE:
        used = 4;
        break;
    case HX_ARGMAX:
    case HX_ARGMIN:
        used = 2;
        break;
    case HX_TOPK:
        used = 4;
        break;
    case HX_TILE:
        used = 0;
        break;
    case HX_CAST:
        used = 1;
        break;
    default:
        break;
    }
    for (uint32_t i = used; i < 11; i++)
        if (p[i])
            return 0;
    if (o->kind == HX_CONV || o->kind == HX_DEPTHWISE || o->kind == HX_MAXPOOL ||
        o->kind == HX_AVGPOOL) {
        /* Effective extents (k - 1) * d + 1; dilation is 1 except for conv/depthwise. */
        const uint32_t eh = (p[0] - 1u) * hx_dilation_h(o) + 1u,
                       ew = (p[1] - 1u) * hx_dilation_w(o) + 1u;
        if (!p[0] || !p[1] || p[0] > 15 || p[1] > 15 || !p[2] || !p[3] || p[2] > 8 || p[3] > 8 ||
            p[4] >= eh || p[5] >= ew || p[8] > 3)
            return 0;
        /* Padding at the bottom/right may differ by one for SAME stride 2. */
        if ((b->h - 1u) * p[2] >= a->h + p[4] || (b->w - 1u) * p[3] >= a->w + p[5])
            return 0;
        if (o->kind == HX_MAXPOOL)
            return a->c == b->c && !p[6] && !p[7] && !p[8];
        if (o->kind == HX_AVGPOOL)
            return a->c == b->c && p[6] <= 1;
        if (o->kind == HX_DEPTHWISE && a->c != b->c)
            return 0;
        uint64_t weights =
            (uint64_t)p[0] * p[1] * b->stride * (o->kind == HX_CONV ? a->c : 1u) * 4u;
        if (!constant(g, p[6], weights) || !constant(g, p[7], b->stride * 4u))
            return 0;
        if (o->kind == HX_DEPTHWISE)
            return !p[9] && !p[10];
        if (!p[10])
            return !p[9];
        uint32_t kb = (p[0] * p[1] * a->c + 31u) / 32u;
        return p[10] == kb && kb <= 128u &&
               constant(g, p[9], (uint64_t)kb * 32u * b->stride * 2u) &&
               (uint64_t)kb * 2048u + (uint64_t)kb * 32u * b->stride * 2u + b->stride * 64u +
                       4096u + (uint64_t)kb * 256u <=
                   0x100000u;
    }
    switch (o->kind) {
    case HX_ADD:
        return same(a, b) && same(a, &g->tensors[o->y]) && p[0] <= 3;
    case HX_PRELU:
        return same(a, b) && constant(g, p[0], a->stride * 4u);
    case HX_PAD:
        return a->h + p[0] <= b->h && a->w + p[1] <= b->w && a->c + p[2] <= b->c;
    case HX_SLICE: /* start + output size must fit the input on each axis (overflow-safe) */
        return p[0] <= a->h && b->h <= a->h - p[0] && p[1] <= a->w && b->w <= a->w - p[1] &&
               p[2] <= a->c && b->c <= a->c - p[2];
    case HX_ZEROINSERT: /* stride 1..8; the last source row/column must land inside the output */
        return a->c == b->c && p[0] >= 1 && p[0] <= 8 && p[1] >= 1 && p[1] <= 8 &&
               p[2] < b->h && p[3] < b->w &&
               (uint64_t)p[2] + (uint64_t)(a->h - 1u) * p[0] + 1u <= b->h &&
               (uint64_t)p[3] + (uint64_t)(a->w - 1u) * p[1] + 1u <= b->w;
    case HX_RESIZE:
        if (a->c != b->c || p[2] > 1 || p[3] > 3 || p[4] > 3 || p[5] > 1)
            return 0;
        if (p[5] == 1)
            return p[2] == 0 && p[4] == 0;
        if (p[2] == 1)
            return !p[0] && !p[1];
        return p[4] == 0 && p[0] <= 1 && p[1] <= 1 && !(p[0] && p[1]);
    case HX_RESHAPE: /* p[0]: 0 HWC copy, 1 NCHW flatten to 1x1xN, 2 NCHW unflatten from 1x1xN */
        if (p[0] == 0)
            return hx_tensor_elements(a) == hx_tensor_elements(b);
        if (p[0] == 1)
            return b->h == 1 && b->w == 1 && elements64(a) == elements64(b);
        if (p[0] == 2)
            return a->h == 1 && a->w == 1 && elements64(a) == elements64(b);
        return 0;
    case HX_TRANSPOSE: { /* p[k] is the source axis of output axis k: a permutation of (h, w, c) */
        const uint32_t ad[3] = {a->h, a->w, a->c}, bd[3] = {b->h, b->w, b->c};
        uint32_t seen = 0;
        for (unsigned k = 0; k < 3; k++) {
            if (p[k] > 2 || (seen >> p[k] & 1u) || bd[k] != ad[p[k]])
                return 0;
            seen |= 1u << p[k];
        }
        return 1;
    }
    case HX_CONCAT: {
        const HxTensor *c = &g->tensors[o->y];
        uint32_t ad[3] = {a->h, a->w, a->c}, bd[3] = {b->h, b->w, b->c}, cd[3] = {c->h, c->w, c->c};
        if (p[0] > 2)
            return 0;
        for (unsigned i = 0; i < 3; i++)
            if (bd[i] != (i == p[0] ? ad[i] + cd[i] : ad[i]) || (i != p[0] && ad[i] != cd[i]))
                return 0;
        return 1;
    }
    case HX_MEAN:
        return b->h == 1 && b->w == 1 && a->c == b->c;
    case HX_LOGISTIC:
        return same(a, b);
    case HX_SOFTMAX:
        return same(a, b);
    case HX_ATTN_SOFTMAX: {
        if (!same(a, b))
            return 0;
        if (p[3]) { /* temperature: finite, positive, channel axis only */
            const float t = hx_float(p[3]);
            if (!(t > 0.0f && t < 3.0e38f) || p[0] != 2u)
                return 0;
        }
        if (o->y != HXFG_NONE) {
            if (o->y >= g->header->tensors)
                return 0;
            const HxTensor *yt = &g->tensors[o->y];
            if ((yt->h != b->h && yt->h != 1) ||
                (yt->w != b->w && yt->w != 1) ||
                (yt->c != b->c && yt->c != 1))
                return 0;
        }
        return p[0] <= 2 && p[1] <= 1;
    }
    case HX_MATMUL:
        return matmul_dims(g, o, NULL, NULL, NULL, NULL, NULL);
    case HX_SDPA:
        return sdpa_dims(g, o, NULL, NULL, NULL, NULL, NULL);
    case HX_NORM: /* unary; p[1] = groups (GROUP only, must divide C), p[3]/p[4] gamma/beta */
        return same(a, b) && p[0] < HX_NORM_COUNT &&
               (p[0] == HX_NORM_GROUP ? p[1] >= 1 && p[1] <= b->c && b->c % p[1] == 0 : !p[1]) &&
               constant(g, p[3], b->stride * 4u) && constant(g, p[4], b->stride * 4u);
    case HX_REDUCE: /* unary; reduced axes are size 1 in the output (keepdims), others kept */
        return p[0] < HX_RED_COUNT && p[1] && p[1] < 8u &&
               b->h == ((p[1] & 1u) ? 1u : a->h) && b->w == ((p[1] & 2u) ? 1u : a->w) &&
               b->c == ((p[1] & 4u) ? 1u : a->c);
    case HX_ACT:
        return same(a, b) && p[0] < HX_ACT_COUNT &&
               (((p[0] == HX_ACT_LEAKY || p[0] == HX_ACT_ELU || p[0] == HX_ACT_CELU) && !p[2]) || (p[0] == HX_ACT_HARDSIGMOID) ||
                (!p[1] && !p[2]));
    case HX_BINARY: {
        if (!same(a, b) || p[0] >= HX_BIN_COUNT || p[1] > 2 || p[3] > 1)
            return 0;
        if (p[1] == 1)
            return constant(g, p[2], b->stride * 4u);
        if (p[1] == 2)
            return constant(g, p[2], (uint64_t)b->h * b->w * b->stride * 4u);
        const HxTensor *c = &g->tensors[o->y];
        return !p[2] && (c->h == b->h || c->h == 1) && (c->w == b->w || c->w == 1) &&
               (c->c == b->c || c->c == 1);
    }
    case HX_RMSNORM:
        return same(a, b) && (p[1] == HXFG_NONE || constant(g, p[1], b->stride * 4u)) &&
               (p[2] == HXFG_NONE || constant(g, p[2], b->stride * 4u));
    case HX_ROPE: {
        if (!same(a, b) || !p[0] || (p[0] & 1u) || !p[1] || p[0] * p[1] > a->c)
            return 0;
        if (o->y != HXFG_NONE) {
            const HxTensor *yt = &g->tensors[o->y];
            if (yt->h != 1 || yt->w != 1 || yt->c < 1)
                return 0;
        }
        /* p[4] theta float bits (0: 10000), p[5] rotated width per head (0: head_dim; even, at most
         * head_dim; the rest of each head is copied), p[6] 1 adds the row index to the position. */
        const uint32_t rot = p[5] ? p[5] : p[0];
        if (rot > p[0] || (rot & 1u) || p[6] > 1u || (p[6] && a->w != 1))
            return 0;
        if (p[4] && !(hx_float(p[4]) > 1.0f && hx_float(p[4]) < 1e30f))
            return 0;
        return p[3] <= 1;
    }
    case HX_SHORTCONV: {
        const HxTensor *st = &g->tensors[o->y];
        const uint32_t k = p[0], s = k - 1u;
        if (!k || k < 2 || k > 15 || !same(a, b) || a->h != 1 || a->w != 1)
            return 0;
        if (st->c != a->c || (st->h * st->w != s))
            return 0;
        if (!constant(g, p[1], (uint64_t)k * b->stride * 4u))
            return 0;
        if (p[2] != HXFG_NONE && !constant(g, p[2], b->stride * 4u))
            return 0;
        if (o->z != HXFG_NONE && !same(a, &g->tensors[o->z]))
            return 0;
        return p[3] <= 3;
    }
    case HX_LINEAR_Q: {
        const uint32_t fmt = p[0], k = p[3], n = p[4];
        if (a->h != b->h || a->w != 1 || b->w != 1 || !k || !n || (k & 31u) || (n & 31u) ||
            a->c != k || b->c != n || fmt > 3)
            return 0;
        /* 2, 3: the 32-row grouped Q4 / Q8 layouts a host makes from 0, 1 in place (hx_lq_group_layout) */
        if (fmt >= 2 && ((k & 63u) || (p[1] & 127u)))
            return 0;
        const uint64_t block_bytes = (fmt & 1u) ? 34u : 18u;
        const uint64_t total_w_bytes = (uint64_t)(k / 32u) * n * block_bytes;
        if (!constant(g, p[1], total_w_bytes))
            return 0;
        if (p[2] != HXFG_NONE && !constant(g, p[2], b->stride * 4u))
            return 0;
        return p[5] <= 3;
    }
    case HX_KV_WRITE: {
        const HxTensor *outt = &g->tensors[o->out];
        if (a->c != outt->c || !(outt->reserved[0] & HX_TENSOR_PERSISTENT))
            return 0;
        if (o->y != HXFG_NONE) {
            const HxTensor *yt = &g->tensors[o->y];
            if (yt->h != 1 || yt->w != 1 || yt->c < 1)
                return 0;
        }
        return 1;
    }
    case HX_GATHER_ELEMENTS: {
        if (p[0] > 2 || p[1] > 1 || p[2] > 2)
            return 0;
        if (p[2] == 2) {
            if (!constant(g, p[3], hx_tensor_bytes(b)))
                return 0;
            if (p[0] == 0 && (b->w > a->w || b->c > a->c))
                return 0;
            if (p[0] == 1 && (b->h > a->h || b->c > a->c))
                return 0;
            if (p[0] == 2 && (b->h > a->h || b->w > a->w))
                return 0;
            return 1;
        }
        const HxTensor *yt = &g->tensors[o->y];
        if (yt->h != b->h || yt->w != b->w || yt->c != b->c)
            return 0;
        if (p[0] == 0 && (yt->w > a->w || yt->c > a->c))
            return 0;
        if (p[0] == 1 && (yt->h > a->h || yt->c > a->c))
            return 0;
        if (p[0] == 2 && (yt->h > a->h || yt->w > a->w))
            return 0;
        if (p[2] == 1)
            return constant(g, p[3], hx_tensor_bytes(a));
        return 1;
    }
    case HX_GATHER_ND: {
        if (p[1] > 1 || p[2] > 2)
            return 0;
        if (p[2] == 2) {
            uint32_t k = p[4] ? p[4] : 1;
            if (k < 1 || k > 3)
                return 0;
            uint32_t slice_size = p[5] ? p[5] : (k == 1 ? a->w * a->c : (k == 2 ? a->c : 1));
            uint32_t total_out = hx_tensor_elements(b);
            if (total_out % slice_size != 0)
                return 0;
            uint32_t M = total_out / slice_size;
            return constant(g, p[3], (uint64_t)M * k * 4u);
        }
        const HxTensor *yt = &g->tensors[o->y];
        uint32_t k = p[4] ? p[4] : yt->c;
        if (k < 1 || k > 3)
            return 0;
        uint32_t total_elements = hx_tensor_elements(yt);
        if (total_elements % k != 0)
            return 0;
        uint32_t M = total_elements / k;
        uint32_t slice_size = p[5] ? p[5] : (k == 1 ? a->w * a->c : (k == 2 ? a->c : 1));
        if (hx_tensor_elements(b) != M * slice_size)
            return 0;
        if (p[2] == 1)
            return constant(g, p[3], hx_tensor_bytes(a));
        return 1;
    }
    case HX_SCATTER_ND: {
        uint32_t k = p[2] ? p[2] : ((p[5] & 1) ? 1 : g->tensors[o->y].c);
        if (k < 1 || k > 3)
            return 0;
        uint32_t slice_size = p[3] ? p[3] : (k == 1 ? b->w * b->c : (k == 2 ? b->c : 1));
        if (a->h != b->h || a->w != b->w || a->c != b->c)
            return 0;
        uint32_t M = p[8];
        if (!M) {
            if (p[5] & 1) {
                if (!(p[5] & 2)) {
                    uint32_t total_z = hx_tensor_elements(&g->tensors[o->z]);
                    if (total_z % slice_size != 0)
                        return 0;
                    M = total_z / slice_size;
                }
            } else {
                const HxTensor *yt = &g->tensors[o->y];
                uint32_t total_elements = hx_tensor_elements(yt);
                if (total_elements % k != 0)
                    return 0;
                M = total_elements / k;
            }
        }
        if (p[5] & 1) {
            if (!constant(g, p[6], (uint64_t)M * k * 4u))
                return 0;
        }
        if (p[5] & 2) {
            if (!constant(g, p[7], (uint64_t)M * slice_size * 4u))
                return 0;
        } else {
            const HxTensor *zt = &g->tensors[o->z];
            if (hx_tensor_elements(zt) != M * slice_size)
                return 0;
        }
        return 1;
    }
    case HX_GATHER: {
        if (p[0] > 2 || p[3] > 1 || !p[2])
            return 0;
        uint32_t n_indices = (p[0] == 2) ? p[9] : hx_tensor_elements(a);
        if (p[0] == 2 && !constant(g, p[8], (uint64_t)n_indices * 4u))
            return 0;
        if (p[4] == 0 && (p[5] == 0 || p[5] == b->c)) {
            if ((uint64_t)b->h * b->w != n_indices)
                return 0;
            if (p[0] == 1)
                return constant(g, p[1], (uint64_t)p[2] * (p[7] ? p[7] : b->stride) * 4u);
            if (p[0] == 2)
                return a->c >= b->c && (uint64_t)a->h * a->w >= p[2];
            const HxTensor *tbl = &g->tensors[o->y];
            return tbl->c >= b->c && (uint64_t)tbl->h * tbl->w >= p[2];
        } else {
            uint32_t inner = p[5] ? p[5] : (p[4] == 2 ? 1 : b->c);
            uint32_t outer = p[6] ? p[6] : 1;
            if ((uint64_t)b->h * b->w * b->c < (uint64_t)outer * n_indices * inner)
                return 0;
            if (p[0] == 1)
                return constant(g, p[1], (uint64_t)outer * p[2] * (p[7] ? p[7] : inner) * 4u);
            if (p[0] == 2)
                return hx_tensor_elements(a) >= (uint64_t)outer * p[2] * inner;
            const HxTensor *tbl = &g->tensors[o->y];
            return hx_tensor_elements(tbl) >= (uint64_t)outer * p[2] * inner;
        }
    }
    case HX_WHERE: {
        if (p[0] > 3 || p[2] > 3)
            return 0;
        if (!((a->h == b->h || a->h == 1) && (a->w == b->w || a->w == 1) && (a->c == b->c || a->c == 1)))
            return 0;
        if (p[0] == 0) {
            const HxTensor *yt = &g->tensors[o->y];
            if (!((yt->h == b->h || yt->h == 1) && (yt->w == b->w || yt->w == 1) && (yt->c == b->c || yt->c == 1)))
                return 0;
        } else if (p[0] == 2 && !constant(g, p[1], b->stride * 4u))
            return 0;
        else if (p[0] == 3 && !constant(g, p[1], (uint64_t)b->h * b->w * b->stride * 4u))
            return 0;
        if (p[2] == 0) {
            const HxTensor *zt = &g->tensors[o->z];
            if (!((zt->h == b->h || zt->h == 1) && (zt->w == b->w || zt->w == 1) && (zt->c == b->c || zt->c == 1)))
                return 0;
        } else if (p[2] == 2 && !constant(g, p[3], b->stride * 4u))
            return 0;
        else if (p[2] == 3 && !constant(g, p[3], (uint64_t)b->h * b->w * b->stride * 4u))
            return 0;
        return 1;
    }
    case HX_ARGMAX:
    case HX_ARGMIN:
        return b->h == a->h && b->w == a->w && b->c == 1 && p[0] <= 1 && p[1] <= 1;
    case HX_TOPK: {
        if (b->h != a->h || b->w != a->w || !p[0] || p[0] > a->c || b->c != p[0] || p[1] > 1 || p[2] > 1 || p[3] > 1)
            return 0;
        if (o->z != HXFG_NONE) {
            const HxTensor *zt = &g->tensors[o->z];
            if (zt->h != b->h || zt->w != b->w || zt->c != b->c)
                return 0;
        }
        return 1;
    }
    case HX_TILE:
        return a->h && a->w && a->c && (b->h % a->h == 0) && (b->w % a->w == 0) && (b->c % a->c == 0);
    case HX_CAST:
        return same(a, b) && p[0] <= 6;
    default:
        return 0;
    }
}
uint32_t hx_graph_open(HxGraph *g, const void *image, uint32_t image_bytes, const void *constants,
                       uint32_t constant_bytes, void *arena, uint32_t arena_bytes,
                       uint32_t *bad_op) {
    if (bad_op)
        *bad_op = HXFG_NONE;
    if (!g || !image || ((uintptr_t)image & 3) || image_bytes < sizeof(HxHeader))
        return HX_HEADER;
    uint8_t *saved_state = g->state;
    uint32_t saved_state_bytes = g->state_bytes;
    memset(g, 0, sizeof *g);
    g->state = saved_state;
    g->state_bytes = saved_state_bytes;
    const HxHeader *h = (const HxHeader *)image;
    if (h->magic != HXFG_MAGIC || h->version != HXFG_VERSION || h->bytes != image_bytes ||
        !h->tensors || h->tensors > HXFG_MAX_TENSORS || !h->ops || h->ops > HXFG_MAX_OPS ||
        !h->inputs || h->inputs > HXFG_MAX_IO || !h->outputs || h->outputs > HXFG_MAX_IO ||
        !h->arena_bytes || h->arena_bytes > HXFG_MAX_ARENA_BYTES || h->arena_bytes > arena_bytes ||
        h->constant_bytes != constant_bytes || !constants || !constant_bytes ||
        constant_bytes > HXFG_MAX_CONSTANT_BYTES)
        return HX_HEADER;
    for (unsigned i = 0; i < 44; i++)
        if (h->reserved[i])
            return HX_HEADER;
    if (h->tensor_offset != sizeof *h ||
        h->op_offset != h->tensor_offset + h->tensors * sizeof(HxTensor) ||
        h->io_offset != h->op_offset + h->ops * sizeof(HxOp) ||
        !span(h->io_offset, (uint64_t)(h->inputs + h->outputs) * 4u, image_bytes) ||
        image_bytes != h->io_offset + (h->inputs + h->outputs) * 4u)
        return HX_RANGE;
    g->header = h;
    g->tensors = (const HxTensor *)((const uint8_t *)image + h->tensor_offset);
    g->ops = (const HxOp *)((const uint8_t *)image + h->op_offset);
    g->io = (const uint32_t *)((const uint8_t *)image + h->io_offset);
    g->constants = constants;
    g->arena = arena;
    for (uint32_t i = 0; i < h->tensors; i++) {
        const HxTensor *t = &g->tensors[i];
        uint32_t limit = ((t->flags & HX_TENSOR_PERSISTENT) && g->state) ? g->state_bytes : h->arena_bytes;
        if (!t->h || !t->w || !t->c || t->h > 4096 || t->w > 4096 || t->c > 4096 ||
            (t->stride != ((t->c + 31u) & ~31u) && !((t->c == 3 || t->c == 1) && t->stride == t->c)) ||
            (t->offset & 127u) || !hx_tensor_bytes(t) ||
            !span(t->offset, hx_tensor_bytes(t), limit) || (t->flags > 1u) ||
            (t->dtype >= HX_DTYPE_COUNT))
            return HX_SHAPE;
    }
    uint8_t live[HXFG_MAX_TENSORS] = {0}, produced[HXFG_MAX_TENSORS] = {0};
    for (uint32_t i = 0; i < h->tensors; i++) {
        if (g->tensors[i].flags & HX_TENSOR_PERSISTENT)
            live[i] = 1;
    }
    for (uint32_t i = 0; i < h->inputs + h->outputs; i++) {
        if (g->io[i] >= h->tensors)
            return HX_RANGE;
        for (uint32_t j = 0; j < i; j++)
            if (g->io[i] == g->io[j])
                return HX_ALIAS;
        if (i < h->inputs) {
            const HxTensor *a = &g->tensors[g->io[i]];
            for (uint32_t j = 0; j < i; j++) {
                const HxTensor *b = &g->tensors[g->io[j]];
                if (overlap(a->offset, hx_tensor_bytes(a), b->offset, hx_tensor_bytes(b)))
                    return HX_ALIAS;
            }
            live[g->io[i]] = produced[g->io[i]] = 1;
        }
    }
    for (uint32_t i = 0; i < h->ops; i++) {
        const HxOp *o = &g->ops[i];
        if (bad_op)
            *bad_op = i;
        int binary = o->kind == HX_ADD || o->kind == HX_CONCAT || o->kind == HX_MATMUL ||
                     o->kind == HX_SDPA || (o->kind == HX_BINARY && !o->p[1]) ||
                     (o->kind == HX_GATHER && !o->p[0]) || (o->kind == HX_WHERE && !o->p[0]) ||
                     ((o->kind == HX_GATHER_ELEMENTS || o->kind == HX_GATHER_ND) && o->p[2] != 2) ||
                     (o->kind == HX_SCATTER_ND && !(o->p[5] & 1));
        int has_y = binary || o->kind == HX_SHORTCONV || (o->kind == HX_ROPE && o->y != HXFG_NONE) ||
                    (o->kind == HX_KV_WRITE && o->y != HXFG_NONE) ||
                    (o->kind == HX_ATTN_SOFTMAX && o->y != HXFG_NONE);
        const int req_z = o->kind == HX_SDPA || (o->kind == HX_WHERE && !o->p[2]) ||
                          (o->kind == HX_SCATTER_ND && !(o->p[5] & 2));
        int has_z = req_z || (o->kind == HX_SHORTCONV && o->z != HXFG_NONE);
        const int produces_z = (o->kind == HX_TOPK && o->z != HXFG_NONE);
        const int has_mask = o->kind == HX_SDPA && o->p[0] != HXFG_NONE;
        /* z is NONE, or a dilation (dh << 8) | dw on conv/depthwise/pool only (1..8 each, not both 1). */
        const int conv_like = o->kind == HX_CONV || o->kind == HX_DEPTHWISE ||
                              o->kind == HX_MAXPOOL || o->kind == HX_AVGPOOL;
        const int dilated_z = o->z != HXFG_NONE && conv_like && (o->z >> 16) == 0 &&
                              hx_dilation_h(o) >= 1 && hx_dilation_h(o) <= 8 &&
                              hx_dilation_w(o) >= 1 && hx_dilation_w(o) <= 8 &&
                              (hx_dilation_h(o) > 1 || hx_dilation_w(o) > 1);
        if (o->x >= h->tensors || o->out >= h->tensors ||
            ((req_z || produces_z) && o->z >= h->tensors) ||
            (o->z != HXFG_NONE && !dilated_z && !has_z && !produces_z) ||
            (has_y ? o->y >= h->tensors : o->y != HXFG_NONE))
            return HX_OPERATOR;
        if (has_mask && (o->p[0] >= h->tensors || !live[o->p[0]]))
            return HX_OPERATOR;
        if (!live[o->x] || (has_y && !live[o->y]) || (has_z && !live[o->z]) ||
            (produced[o->out] && !(g->tensors[o->out].reserved[0] & HX_TENSOR_PERSISTENT)) ||
            (produces_z && produced[o->z]))
            return HX_UNINITIALIZED;
        const HxTensor *out = &g->tensors[o->out];
        const int out_pers = (out->reserved[0] & HX_TENSOR_PERSISTENT) != 0;
        for (uint32_t j = 0; j < h->tensors; j++)
            if (live[j]) {
                const HxTensor *t = &g->tensors[j];
                const int t_pers = (t->reserved[0] & HX_TENSOR_PERSISTENT) != 0;
                if (out_pers == t_pers &&
                    overlap(out->offset, hx_tensor_bytes(out), t->offset, hx_tensor_bytes(t))) {
                    if (j == o->x || (has_y && j == o->y) || (has_z && j == o->z)) {
                        if (!out_pers && !(o->kind == HX_SCATTER_ND && o->p[0] == 1 && j == o->x && o->out == o->x))
                            return HX_ALIAS;
                    } else
                        live[j] = 0;
                }
            }
        if (produces_z) {
            const HxTensor *zt = &g->tensors[o->z];
            if (overlap(out->offset, hx_tensor_bytes(out), zt->offset, hx_tensor_bytes(zt)))
                return HX_ALIAS;
            for (uint32_t j = 0; j < h->tensors; j++)
                if (live[j]) {
                    const HxTensor *t = &g->tensors[j];
                    if (overlap(zt->offset, hx_tensor_bytes(zt), t->offset, hx_tensor_bytes(t))) {
                        if (j == o->x)
                            return HX_ALIAS;
                        live[j] = 0;
                    }
                }
        }
        if (g->tensors[o->x].stride % 32u && o->kind != HX_CONV)
            return HX_SHAPE;
        if (has_y && g->tensors[o->y].stride % 32u)
            return HX_SHAPE;
        if (has_z && g->tensors[o->z].stride % 32u)
            return HX_SHAPE;
        if (has_mask && g->tensors[o->p[0]].stride % 32u)
            return HX_SHAPE;
        if (produces_z && g->tensors[o->z].stride % 32u)
            return HX_SHAPE;
        if (out->stride % 32u || !op_valid(g, o))
            return HX_SHAPE;
        live[o->out] = produced[o->out] = 1;
        if (produces_z)
            live[o->z] = produced[o->z] = 1;
    }
    for (uint32_t i = h->inputs; i < h->inputs + h->outputs; i++)
        if (!live[g->io[i]])
            return HX_UNINITIALIZED;
    if (bad_op)
        *bad_op = HXFG_NONE;
    return HX_OK;
}

static float hx_erf(float x) {
    /* Abramowitz-Stegun 7.1.26, |error| < 1.5e-7; avoids depending on a DSP libm erff. */
    float s = x < 0 ? -1.0f : 1.0f, a = fabsf(x), t = 1.0f / (1.0f + 0.3275911f * a);
    float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t +
                  0.254829592f) *
                 t;
    return s * (1.0f - poly * expf(-a * a));
}
static float hx_float(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v = {bits};
    return v.f;
}
float hx_activation(uint32_t function, uint32_t alpha_bits, uint32_t beta_bits, float x) {
    switch (function) {
    case HX_ACT_RELU:
        return x > 0 ? x : 0;
    case HX_ACT_RELU6:
        return x < 0 ? 0 : (x > 6 ? 6 : x);
    case HX_ACT_HARDSWISH: {
        float r = x + 3.0f;
        r = r < 0 ? 0 : (r > 6 ? 6 : r);
        return x * r * (1.0f / 6.0f);
    }
    case HX_ACT_HARDSIGMOID: {
        float r = hx_float(alpha_bits) * x + hx_float(beta_bits);
        return r < 0 ? 0 : (r > 1 ? 1 : r);
    }
    case HX_ACT_SILU:
        return x / (1.0f + expf(-x));
    case HX_ACT_TANH:
        return 1.0f - 2.0f / (expf(2.0f * x) + 1.0f);
    case HX_ACT_LEAKY:
        return x < 0 ? hx_float(alpha_bits) * x : x;
    case HX_ACT_GELU:
        return 0.5f * x * (1.0f + hx_erf(x * 0.70710678f));
    case HX_ACT_EXP:
        return expf(x);
    case HX_ACT_SQRT:
        return sqrtf(x);
    case HX_ACT_ABS:
        return fabsf(x);
    case HX_ACT_NEG:
        return -x;
    case HX_ACT_ERF:
        return hx_erf(x);
    case HX_ACT_LOG:
        return logf(fmaxf(x, 1e-20f));
    case HX_ACT_RECIPROCAL:
        return 1.0f / (fabsf(x) < 1e-20f ? (x < 0 ? -1e-20f : 1e-20f) : x);
    case HX_ACT_FLOOR:
        return floorf(x);
    case HX_ACT_CEIL:
        return ceilf(x);
    case HX_ACT_SIN:
        return sinf(x);
    case HX_ACT_COS:
        return cosf(x);
    case HX_ACT_NOT:
        return x == 0.0f ? 1.0f : 0.0f;
    case HX_ACT_SIGN:
        return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
    case HX_ACT_ROUND:
        return roundf(x);
    case HX_ACT_ELU:
        return x < 0 ? hx_float(alpha_bits) * (expf(x) - 1.0f) : x;
    case HX_ACT_SELU:
        return 1.05070102f * (x < 0 ? 1.67326319f * (expf(x) - 1.0f) : x);
    case HX_ACT_CELU: {
        float a = hx_float(alpha_bits);
        return x < 0 ? a * (expf(x / a) - 1.0f) : x;
    }
    case HX_ACT_SOFTPLUS:
        return x > 20.0f ? x : (x < -20.0f ? expf(x) : logf(1.0f + expf(x)));
    case HX_ACT_MISH: {
        float sp = x > 20.0f ? x : (x < -20.0f ? expf(x) : logf(1.0f + expf(x)));
        return x * (1.0f - 2.0f / (expf(2.0f * sp) + 1.0f));
    }
    case HX_ACT_SOFTSIGN:
        return x / (1.0f + fabsf(x));
    case HX_ACT_LOGSIGMOID:
        return x < -20.0f ? x : (x > 20.0f ? -expf(-x) : -logf(1.0f + expf(-x)));
    case HX_ACT_SINH:
        return 0.5f * (expf(x) - expf(-x));
    case HX_ACT_COSH:
        return 0.5f * (expf(x) + expf(-x));
    default:
        return x;
    }
}
#ifndef __hexagon__
static float activate(float x, uint32_t a) {
    if (a == 1 || a == 3)
        x = fmaxf(0, x);
    if (a == 2)
        x = fmaxf(-1, fminf(1, x));
    if (a == 3)
        x = fminf(6, x);
    return x;
}
#endif
#ifdef __hexagon__
static HVX_Vector splat(float f) {
    union {
        float f;
        uint32_t u;
    } v = {f};
    return Q6_V_vsplat_R(v.u);
}
static HVX_Vector mul(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}
static HVX_Vector add(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
/* exp and 1/d (d >= 1) on HVX, from the Qwen engine (examples/llm/qwen_dsp.c); SiLU checked in test_linear_q_hvx_sim. */
static HVX_Vector hx_vexpf(HVX_Vector v) {
    v = Q6_Vsf_vmax_VsfVsf(v, splat(-87.0f));
    v = Q6_Vsf_vmin_VsfVsf(v, splat(80.0f));
    const HVX_Vector scaled = mul(v, splat(1.4426950408889634f));
    const HVX_Vector magic = splat(12582912.0f); /* 1.5 * 2^23 */
    const HVX_Vector rounded = add(scaled, magic);
    const HVX_Vector n = Q6_Vw_vsub_VwVw(rounded, magic);
    const HVX_Vector r = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(scaled, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(rounded, magic))));
    /* Taylor series of 2^r to degree 9: the qf32 rounding above is not always nearest, so |r| can approach 1
     * (degree 6 is then off by 1.5e-5). */
    HVX_Vector q = splat(1.0178086e-07f);
    q = add(mul(q, r), splat(1.3215487e-06f));
    q = add(mul(q, r), splat(1.5252734e-05f));
    q = add(mul(q, r), splat(0.00015403530f));
    q = add(mul(q, r), splat(0.00133335581f));
    q = add(mul(q, r), splat(0.00961812911f));
    q = add(mul(q, r), splat(0.05550410866f));
    q = add(mul(q, r), splat(0.24022650695f));
    q = add(mul(q, r), splat(0.69314718056f));
    q = add(mul(q, r), splat(1.0f));
    return mul(Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(n, Q6_V_vsplat_R(127)), 23), q);
}
static HVX_Vector hx_vrecipf(HVX_Vector d) {
    HVX_Vector r = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7ef311c7), d);
    for (int i = 0; i < 3; i++)
        r = mul(r, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(splat(2.0f), mul(d, r))));
    return r;
}
static HVX_Vector hx_vsilu(HVX_Vector v) {
    const HVX_Vector e = hx_vexpf(Q6_V_vxor_VV(v, Q6_V_vsplat_R(0x80000000u)));
    return mul(v, hx_vrecipf(add(splat(1.0f), e)));
}
static HVX_Vector act(HVX_Vector v, uint32_t a) {
    if (a == 1 || a == 3)
        v = Q6_Vsf_vmax_VsfVsf(v, Q6_V_vzero());
    if (a == 2)
        v = Q6_Vsf_vmax_VsfVsf(splat(-1), Q6_Vsf_vmin_VsfVsf(splat(1), v));
    if (a == 3)
        v = Q6_Vsf_vmin_VsfVsf(splat(6), v);
    return v;
}
#endif
#ifdef __hexagon__
#include "depthwise_hvx.h"
#endif
static float element(const float *x, const HxTensor *t, uint32_t index) {
    return x[(index / t->c) * t->stride + index % t->c];
}
void hx_pack(float *dst, const float *src, const HxTensor *t) {
    if (t->stride == t->c) {
        memcpy(dst, src, hx_tensor_elements(t) * 4u);
        return;
    }
    for (uint32_t i = 0; i < t->h * t->w; i++) {
        memcpy(dst + i * t->stride, src + i * t->c, t->c * 4u);
        memset(dst + i * t->stride + t->c, 0, (t->stride - t->c) * 4u);
    }
}
void hx_unpack(float *dst, const float *src, const HxTensor *t) {
    if (t->stride == t->c) {
        memcpy(dst, src, hx_tensor_elements(t) * 4u);
        return;
    }
    for (uint32_t i = 0; i < t->h * t->w; i++)
        memcpy(dst + i * t->c, src + i * t->stride, t->c * 4u);
}
/* HX_NORM, plain C on both targets (reference speed). Each part recomputes the statistics it
 * needs over all pixels and writes only pixels lo..hi-1; padding lanes are written zero. */
static void normalise(const HxGraph *g, const HxOp *o, const float *x, float *out, uint32_t lo,
                      uint32_t hi) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p, c = b->c, pixels = a->h * a->w;
    const double eps = hx_float(p[2]);
    const float *gamma = (const float *)(g->constants + p[3]),
                *beta = (const float *)(g->constants + p[4]);
    for (uint32_t i = lo; i < hi; i++)
        for (uint32_t k = c; k < b->stride; k++)
            out[i * b->stride + k] = 0.0f;
    if (p[0] == HX_NORM_CHANNEL) {
        for (uint32_t i = lo; i < hi; i++) {
            const float *src = x + i * a->stride;
            float *dst = out + i * b->stride;
            double mean = 0, var = 0;
            for (uint32_t k = 0; k < c; k++)
                mean += src[k];
            mean /= c;
            for (uint32_t k = 0; k < c; k++)
                var += (src[k] - mean) * (src[k] - mean);
            var /= c;
            const double inv = 1.0 / sqrt(var + eps);
            for (uint32_t k = 0; k < c; k++)
                dst[k] = (float)((src[k] - mean) * inv * gamma[k] + beta[k]);
        }
        return;
    }
    /* GROUP: c % groups == 0 (op_valid), so every channel belongs to exactly one group. */
    const uint32_t groups = p[1], size = c / groups;
    for (uint32_t grp = 0; grp < groups; grp++) {
        const uint32_t k0 = grp * size, k1 = k0 + size;
        const double count = (double)pixels * size;
        double mean = 0, var = 0;
        for (uint32_t i = 0; i < pixels; i++)
            for (uint32_t k = k0; k < k1; k++)
                mean += x[i * a->stride + k];
        mean /= count;
        for (uint32_t i = 0; i < pixels; i++)
            for (uint32_t k = k0; k < k1; k++)
                var += (x[i * a->stride + k] - mean) * (x[i * a->stride + k] - mean);
        var /= count;
        const double inv = 1.0 / sqrt(var + eps);
        for (uint32_t i = lo; i < hi; i++)
            for (uint32_t k = k0; k < k1; k++)
                out[i * b->stride + k] = (float)((x[i * a->stride + k] - mean) * inv * gamma[k] + beta[k]);
    }
}
static inline float hx_half_to_float(uint16_t h) {
    union { uint32_t u; float f; } v;
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16, exp = (h >> 10) & 31u, mant = h & 1023u;
    if (!exp) { float x = (float)mant * 0x1p-24f; return sign ? -x : x; }
    v.u = sign | ((exp + 112u) << 23) | (mant << 13);
    return v.f;
}
static void hx_rmsnorm_op(const HxGraph *g, const HxOp *o, const float *x, float *out,
                          uint32_t lo, uint32_t hi) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p, c = b->c;
    const float eps_f = p[0] ? hx_float(p[0]) : 1e-5f;
    const double eps = (double)eps_f;
    const float *gamma = p[1] != HXFG_NONE ? (const float *)(g->constants + p[1]) : NULL;
    const float *beta = p[2] != HXFG_NONE ? (const float *)(g->constants + p[2]) : NULL;
    for (uint32_t i = lo; i < hi; i++) {
        const float *src = x + i * a->stride;
        float *dst = out + i * b->stride;
#ifdef __hexagon__
        /* Vector form (the double one is emulated, and gamma/beta are uncached: per-element round trips). */
        if (c % 32u == 0) {
            HVX_Vector acc = Q6_V_vzero();
            for (uint32_t k = 0; k < c; k += 32u) {
                const HVX_Vector v = *(const HVX_UVector *)(src + k);
                acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(v, v));
            }
            HVX_Vector s = Q6_Vsf_equals_Vqf32(acc);
            for (int shift = 64; shift >= 4; shift >>= 1)
                s = add(s, Q6_V_vror_VR(s, shift));
            union {
                HVX_Vector v;
                float f[32];
                uint32_t u[32];
            } r;
            r.v = s;
            r.f[1] = 1.0f / sqrtf(r.f[0] / (float)c + eps_f);
            const HVX_Vector inv = Q6_V_vsplat_R(r.u[1]);
            for (uint32_t k = 0; k < c; k += 32u) {
                HVX_Vector v = mul(*(const HVX_UVector *)(src + k), inv);
                if (gamma)
                    v = mul(v, *(const HVX_UVector *)(gamma + k));
                if (beta)
                    v = add(v, *(const HVX_UVector *)(beta + k));
                *(HVX_UVector *)(dst + k) = v;
            }
            for (uint32_t k = c; k < b->stride; k++)
                dst[k] = 0.0f;
            continue;
        }
#endif
        double sum_sq = 0.0;
        for (uint32_t k = 0; k < c; k++)
            sum_sq += (double)src[k] * (double)src[k];
        const double inv = 1.0 / sqrt(sum_sq / (double)c + eps);
        for (uint32_t k = 0; k < c; k++) {
            double v = (double)src[k] * inv;
            if (gamma)
                v *= (double)gamma[k];
            if (beta)
                v += (double)beta[k];
            dst[k] = (float)v;
        }
        for (uint32_t k = c; k < b->stride; k++)
            dst[k] = 0.0f;
    }
}
/* A position operand: float32 (the scalar written by a host that tracks it as float) or int32. */
static inline uint32_t hx_position(const HxGraph *g, uint32_t tensor, const float *y) {
    if (g->tensors[tensor].dtype == HX_DTYPE_INT32)
        return (uint32_t)*(const int32_t *)y;
    return (uint32_t)y[0];
}
static void hx_rope_op(const HxGraph *g, const HxOp *o, const float *x, const float *y,
                       float *out, uint32_t lo, uint32_t hi) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p;
    const uint32_t head_dim = p[0], num_heads = p[1];
    const uint32_t rot = p[5] ? p[5] : head_dim, half = rot / 2u;
    const uint32_t mode = p[3];
    const double base = p[4] ? (double)hx_float(p[4]) : 10000.0;
    /* The angles depend only on (pos, base, rot): compute them once per row for all heads, and in decode (one row,
     * so one worker runs this op) keep the last table, which every ROPE op of a token shares. Same double math. */
    static struct {
        uint32_t valid, pos, base_bits, rot;
        float cos_v[256], sin_v[256];
    } cache;
    float cos_t[256], sin_t[256];
#ifdef __hexagon__
    const int cacheable = b->h * b->w == 1u && half <= 256u;
#else
    const int cacheable = 0; /* host sessions may run graphs concurrently */
#endif
    for (uint32_t i = lo; i < hi; i++) {
        const float *src = x + i * a->stride;
        float *dst = out + i * b->stride;
        uint32_t pos = p[2];
        if (o->y != HXFG_NONE && y) {
            pos = hx_position(g, o->y, y);
        }
        if (p[6])
            pos += i;
        const float *cos_row = cos_t, *sin_row = sin_t;
        if (half <= 256u) {
            if (cacheable && cache.valid && cache.pos == pos && cache.base_bits == p[4] && cache.rot == rot) {
                cos_row = cache.cos_v;
                sin_row = cache.sin_v;
            } else {
                float *cw = cacheable ? cache.cos_v : cos_t, *sw = cacheable ? cache.sin_v : sin_t;
                cache.valid = 0;
                for (uint32_t j = 0; j < half; j++) {
                    const double theta = pow(base, -2.0 * (double)j / (double)rot);
                    const double angle = (double)pos * theta;
                    cw[j] = (float)cos(angle);
                    sw[j] = (float)sin(angle);
                }
                if (cacheable) {
                    cache.pos = pos;
                    cache.base_bits = p[4];
                    cache.rot = rot;
                    cache.valid = 1;
                    cos_row = cache.cos_v;
                    sin_row = cache.sin_v;
                }
            }
        }
        for (uint32_t h = 0; h < num_heads; h++) {
            const float *h_in = src + h * head_dim;
            float *h_out = dst + h * head_dim;
            for (uint32_t j = 0; j < half; j++) {
                float cos_v, sin_v;
                if (half <= 256u) {
                    cos_v = cos_row[j];
                    sin_v = sin_row[j];
                } else {
                    double theta = pow(base, -2.0 * (double)j / (double)rot);
                    double angle = (double)pos * theta;
                    cos_v = (float)cos(angle);
                    sin_v = (float)sin(angle);
                }
                if (mode == 0) {
                    float v0 = h_in[j], v1 = h_in[half + j];
                    h_out[j] = v0 * cos_v - v1 * sin_v;
                    h_out[half + j] = v0 * sin_v + v1 * cos_v;
                } else {
                    float v0 = h_in[2 * j], v1 = h_in[2 * j + 1];
                    h_out[2 * j] = v0 * cos_v - v1 * sin_v;
                    h_out[2 * j + 1] = v0 * sin_v + v1 * cos_v;
                }
            }
            for (uint32_t j = rot; j < head_dim; j++)
                h_out[j] = h_in[j];
        }
        for (uint32_t k = num_heads * head_dim; k < b->c; k++)
            dst[k] = src[k];
        for (uint32_t k = b->c; k < b->stride; k++)
            dst[k] = 0.0f;
    }
}
static inline uint8_t *hx_tensor_base(const HxGraph *g, const HxTensor *t) {
    if ((t->reserved[0] & HX_TENSOR_PERSISTENT) && g->state)
        return g->state;
    return g->arena;
}
static void hx_shortconv_op(const HxGraph *g, const HxOp *o, const float *x, float *out,
                            uint32_t part, uint32_t parts) {
    const HxTensor *b = &g->tensors[o->out];
    const HxTensor *st = &g->tensors[o->y];
    float *s_buf = (float *)(hx_tensor_base(g, st) + st->offset);
    const float *w = (const float *)(g->constants + o->p[1]);
    const float *bias = o->p[2] != HXFG_NONE ? (const float *)(g->constants + o->p[2]) : NULL;
    const float *gate = o->z != HXFG_NONE ? (const float *)(hx_tensor_base(g, &g->tensors[o->z]) + g->tensors[o->z].offset) : NULL;
    const uint32_t k_taps = o->p[0], s_len = k_taps - 1u, c = b->c;
    const uint32_t c0 = c * part / parts, c1 = c * (part + 1u) / parts;
#ifdef __hexagon__
    /* Vector form over channel vectors (the scalar one is emulated double over uncached state). */
    if (!(c & 31u) && k_taps >= 2u && o->p[3] != 2u && o->p[3] <= 3u) {
        const uint32_t vectors = c / 32u, v0 = vectors * part / parts, v1 = vectors * (part + 1u) / parts;
        for (uint32_t v = v0; v < v1; v++) {
            const uint32_t d = v * 32u;
            HVX_Vector acc = bias ? Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *)(bias + d), Q6_V_vzero()) : Q6_V_vzero();
            const HVX_Vector xv = *(const HVX_UVector *)(x + d);
            HVX_Vector prev = *(const HVX_UVector *)(s_buf + d);
            for (uint32_t j = 0; j < s_len; j++) {
                acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(prev, *(const HVX_UVector *)(w + j * b->stride + d)));
                const HVX_Vector next = j + 1u < s_len ? *(const HVX_UVector *)(s_buf + (j + 1u) * st->stride + d) : xv;
                *(HVX_UVector *)(s_buf + j * st->stride + d) = next;
                prev = next;
            }
            acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(xv, *(const HVX_UVector *)(w + s_len * b->stride + d)));
            HVX_Vector val = Q6_Vsf_equals_Vqf32(acc);
            if (o->p[3] == 1)
                val = Q6_Vsf_vmax_VsfVsf(val, Q6_V_vzero());
            else if (o->p[3] == 3)
                val = hx_vsilu(val);
            if (gate)
                val = mul(val, *(const HVX_UVector *)(gate + d));
            *(HVX_UVector *)(out + d) = val;
        }
        if (part == 0)
            for (uint32_t d = c; d < b->stride; d++)
                out[d] = 0.0f;
        return;
    }
#endif

    for (uint32_t d = c0; d < c1; d++) {
        double acc = bias ? (double)bias[d] : 0.0;
        for (uint32_t j = 0; j < s_len; j++) {
            acc += (double)s_buf[j * st->stride + d] * (double)w[j * b->stride + d];
        }
        acc += (double)x[d] * (double)w[s_len * b->stride + d];

        for (uint32_t j = 0; j < s_len - 1u; j++) {
            s_buf[j * st->stride + d] = s_buf[(j + 1u) * st->stride + d];
        }
        s_buf[(s_len - 1u) * st->stride + d] = x[d];

        float val = (float)acc;
        if (o->p[3] == 1) val = fmaxf(0.0f, val);
        else if (o->p[3] == 3) val = val / (1.0f + expf(-val));

        if (gate)
            val *= gate[d];
        out[d] = val;
    }
    if (part == 0) {
        for (uint32_t d = c; d < b->stride; d++)
            out[d] = 0.0f;
    }
}
#ifdef __hexagon__
/* One quantized row dot product on HVX: each 32-weight block becomes one float vector (32 lanes), multiplied
 * by the matching 32 activations and the block scale. The weights are uncached DDR (L2 is TCM), where every
 * access is a full round trip, so the row is read only through aligned vectors, one per 128 bytes, and each
 * block is cut out of a two-vector window with valign. The fp16 scale is also taken from the window (a scalar
 * read would be another round trip): moved to the low halfword, broadcast, and widened to fp32 bits in
 * vector registers (zero and subnormal scales read as 0). The integers convert exactly with vconv (v73+);
 * a magic-number subtraction in qf32 loses one unit at 2^23. The pool keeps padding after every constant, so
 * the window may read up to 256 bytes past the last block. */
static float hx_linear_q_row_hvx(const uint8_t *w_row, const float *x, uint32_t blocks, uint32_t fmt) {
    const HVX_Vector offset = Q6_V_vsplat_R(fmt == 0 ? 8u : 0u);
    const HVX_Vector low_nibble = Q6_Vh_vsplat_R(0x000f);
    const HVX_Vector abs_bits = Q6_V_vsplat_R(0x7fffu), sign_bit = Q6_V_vsplat_R(0x8000u);
    const HVX_Vector exp_bits = Q6_V_vsplat_R(0x7c00u), bias = Q6_V_vsplat_R(112u << 23);
    const HVX_VectorPred first_half = Q6_Q_vsetq_R(2);
    const uint32_t pitch = fmt == 0 ? 18u : 34u, scale_at = fmt == 0 ? 16u : 32u;
    const uint8_t *base = (const uint8_t *)((uintptr_t)w_row & ~(uintptr_t)127u);
    HVX_Vector lo_v = *(const HVX_Vector *)base, hi_v = *(const HVX_Vector *)(base + 128);
    uint32_t at = (uint32_t)((uintptr_t)w_row & 127u);
    HVX_Vector acc = Q6_V_vzero();
    for (uint32_t b_idx = 0; b_idx < blocks; b_idx++) {
        if (at >= 128u) {
            base += 128;
            at -= 128u;
            lo_v = hi_v;
            hi_v = *(const HVX_Vector *)(base + 128);
        }
        const HVX_Vector raw = Q6_V_valign_VVR(hi_v, lo_v, at);
        at += pitch;
        HVX_Vector words;
        if (fmt == 0) {
            /* byte j holds element 2j (low nibble) and 2j+1 (high nibble) */
            const HVX_Vector h = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(raw));
            const HVX_Vector lo = Q6_V_vand_VV(h, low_nibble), hi = Q6_Vuh_vlsr_VuhR(h, 4);
            words = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -2))));
        } else {
            words = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(raw))));
        }
        const HVX_Vector q = Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(words, offset));
        /* scale halfword -> every halfword -> fp32 bits in every word */
        HVX_Vector d = Q6_V_vmux_QVV(first_half, Q6_V_vror_VR(raw, scale_at), Q6_V_vzero());
        for (uint32_t sh = 2; sh < 128u; sh <<= 1)
            d = Q6_V_vor_VV(d, Q6_V_vror_VR(d, 128u - sh));
        const HVX_Vector magnitude = Q6_V_vand_VV(d, abs_bits);
        HVX_Vector scale = Q6_V_vor_VV(Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(magnitude, 13), bias),
                                       Q6_Vw_vasl_VwR(Q6_V_vand_VV(d, sign_bit), 16));
        scale = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(d, exp_bits), Q6_V_vzero()), Q6_V_vzero(), scale);
        const HVX_Vector xv = *(const HVX_UVector *)(x + b_idx * 32u);
        acc = add(acc, mul(mul(q, xv), scale));
    }
    for (int shift = 64; shift >= 4; shift >>= 1)
        acc = add(acc, Q6_V_vror_VR(acc, shift));
    union {
        HVX_Vector v;
        float f[32];
    } r;
    r.v = acc;
    return r.f[0];
}
/* Four rows at once (rows w0 + i * row_bytes): the activation load is shared and the four independent
 * accumulation chains hide each other's latency. The block scale is moved to halfword 0, broadcast with one
 * vdelta (control j & ~1 replicates halfword 0), and widened by an hf x hf -> qf32 multiply by 1.0 (even
 * lanes). Products accumulate in qf32. Rows may be unaligned; reads stay inside the 256-byte pool padding. */
static const uint8_t hx_lq_bcast[128] __attribute__((aligned(128))) = {
#define HX_B8(n) (n), (n), (n) + 2, (n) + 2, (n) + 4, (n) + 4, (n) + 6, (n) + 6
#define HX_B32(n) HX_B8(n), HX_B8((n) + 8), HX_B8((n) + 16), HX_B8((n) + 24)
    HX_B32(0), HX_B32(32), HX_B32(64), HX_B32(96)
#undef HX_B32
#undef HX_B8
};
static void hx_linear_q_rows4_hvx(const uint8_t *w0, uint32_t row_bytes, const float *x, uint32_t blocks,
                                  uint32_t fmt, float *out4) {
    const HVX_Vector offset = Q6_V_vsplat_R(8u);
    const HVX_Vector low_nibble = Q6_Vh_vsplat_R(0x000f);
    const HVX_Vector one_hf = Q6_V_vsplat_R(0x3c003c00u);
    const HVX_Vector bcast = *(const HVX_Vector *)hx_lq_bcast;
    const uint32_t pitch = fmt == 0 ? 18u : 34u, scale_at = fmt == 0 ? 16u : 32u;
    const uint8_t *base[4];
    HVX_Vector lo_v[4], hi_v[4], acc[4];
    uint32_t at[4];
#pragma clang loop unroll(full)
    for (uint32_t i = 0; i < 4u; i++) {
        const uint8_t *w = w0 + i * row_bytes;
        base[i] = (const uint8_t *)((uintptr_t)w & ~(uintptr_t)127u);
        at[i] = (uint32_t)((uintptr_t)w & 127u);
        lo_v[i] = *(const HVX_Vector *)base[i];
        hi_v[i] = *(const HVX_Vector *)(base[i] + 128);
        acc[i] = Q6_V_vzero();
    }
    for (uint32_t b_idx = 0; b_idx < blocks; b_idx++) {
        const HVX_Vector xv = *(const HVX_UVector *)(x + b_idx * 32u);
#pragma clang loop unroll(full)
        for (uint32_t i = 0; i < 4u; i++) {
            if (at[i] >= 128u) {
                base[i] += 128;
                at[i] -= 128u;
                lo_v[i] = hi_v[i];
                hi_v[i] = *(const HVX_Vector *)(base[i] + 128);
            }
            const HVX_Vector raw = Q6_V_valign_VVR(hi_v[i], lo_v[i], at[i]);
            at[i] += pitch;
            HVX_Vector words;
            if (fmt == 0) {
                const HVX_Vector h = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(raw));
                const HVX_Vector lo = Q6_V_vand_VV(h, low_nibble), hi = Q6_Vuh_vlsr_VuhR(h, 4);
                words = Q6_Vw_vsub_VwVw(Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -2)))), offset);
            } else {
                words = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(raw))));
            }
            const HVX_Vector d = Q6_V_vdelta_VV(Q6_V_vror_VR(raw, scale_at), bcast);
            const HVX_Vector scale = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(d, one_hf)));
            const HVX_Vector qx = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(words), xv));
            acc[i] = Q6_Vqf32_vadd_Vqf32Vqf32(acc[i], Q6_Vqf32_vmpy_VsfVsf(qx, scale));
        }
    }
#pragma clang loop unroll(full)
    for (uint32_t i = 0; i < 4u; i++) {
        HVX_Vector a = Q6_Vsf_equals_Vqf32(acc[i]);
        for (int shift = 64; shift >= 4; shift >>= 1)
            a = add(a, Q6_V_vror_VR(a, shift));
        union {
            HVX_Vector v;
            float f[32];
        } r;
        r.v = a;
        out4[i] = r.f[0];
    }
}
/* Activation row x[k] as int16 per 32-block for the grouped kernels: xq = round(x / sx[b]), sx[b] = max|x_b| / 32767
 * (error <= one unit, 3.1e-5 of the block maximum: the qf32 rounding is not exactly nearest), and corr[b] = -8 * sum(xq_b) for the Q4 offset. xq gets 64
 * bytes of slack: each block stores a whole vector whose upper half the next block overwrites. */
static void hx_lq_quantize_x(const float *x, uint32_t blocks, int16_t *xq, float *sx, int32_t *corr) {
    const HVX_Vector abs_mask = Q6_V_vsplat_R(0x7fffffffu);
    const HVX_Vector magic = Q6_V_vsplat_R(0x4b400000u); /* 1.5 * 2^23: adding it rounds to an integer */
    for (uint32_t b = 0; b < blocks; b++) {
        const HVX_Vector xv = *(const HVX_UVector *)(x + b * 32u);
        HVX_Vector m = Q6_V_vand_VV(xv, abs_mask);
        for (int shift = 64; shift >= 4; shift >>= 1)
            m = Q6_Vw_vmax_VwVw(m, Q6_V_vror_VR(m, shift)); /* non-negative floats order as integers */
        union {
            HVX_Vector v;
            float f[32];
            uint32_t u[32];
            int32_t i[32];
        } r;
        r.v = m;
        const float peak = r.f[0];
        if (!(peak > 0.0f) || !(peak < FLT_MAX)) {
            sx[b] = 0.0f;
            corr[b] = 0;
            *(HVX_UVector *)(xq + b * 32u) = Q6_V_vzero();
            continue;
        }
        sx[b] = peak / 32767.0f;
        r.f[0] = 32767.0f / peak;
        const HVX_Vector inv = Q6_V_vsplat_R(r.u[0]);
        const HVX_Vector y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(mul(xv, inv), magic));
        const HVX_Vector q = Q6_Vw_vsub_VwVw(y, magic); /* |q| <= 32767 (+1 at the rounding edge) */
        *(HVX_UVector *)(xq + b * 32u) = Q6_Vh_vpacke_VwVw(q, q);
        HVX_Vector t = q;
        for (int shift = 64; shift >= 4; shift >>= 1)
            t = Q6_Vw_vadd_VwVw(t, Q6_V_vror_VR(t, shift));
        r.v = t;
        corr[b] = -8 * r.i[0];
    }
}
/* One 32-row group of the grouped layouts (fmt 2 = Q4, 3 = Q8; see hx_lq_group_bytes): out[0..32) gets the 32
 * dot products. Per pair of 32-blocks: codes of block 2j, codes of block 2j+1 (Q8 1024 bytes each, Q4 512),
 * then 64 fp16 scales, halfword 2r = row r of block 2j and 2r+1 = row r of block 2j+1, so an hf x 1.0 multiply
 * yields both blocks' 32 row scales in order (even / odd lanes). A Q8 code vector v of a block is two k-pairs
 * (4v, 4v+1) and (4v+2, 4v+3): byte s*64 + 2r + j holds row r, k = 4v + 2s + j, so vunpack gives one word lane
 * per row holding its two halfwords, and vdmpy with the activation pair (xq[k], xq[k+1]) adds both products.
 * A Q4 code vector v holds that Q8 vector 2v in its low nibbles and 2v+1 in its high nibbles (codes + 8). */
static void hx_lq_group_hvx(const uint8_t *grp, const int16_t *xq, const float *sx, const int32_t *corr,
                            uint32_t blocks, uint32_t q4, float *out) {
    const HVX_Vector one_hf = Q6_V_vsplat_R(0x3c003c00u);
    const HVX_Vector nib = Q6_Vb_vsplat_R(0x0f);
    const uint32_t code_bytes = q4 ? 512u : 1024u;
    const uint32_t *xw = (const uint32_t *)xq;
    HVX_Vector acc = Q6_V_vzero();
    for (uint32_t bp = 0; bp < blocks; bp += 2u) {
        const uint8_t *pair = grp + (uint64_t)(bp / 2u) * (2u * code_bytes + 128u);
        const HVX_VectorPair sc = Q6_Wqf32_vmpy_VhfVhf(*(const HVX_Vector *)(pair + 2u * code_bytes), one_hf);
#pragma clang loop unroll(full)
        for (uint32_t half = 0; half < 2u; half++) {
            const uint32_t blk = bp + half;
            const HVX_Vector *cv = (const HVX_Vector *)(pair + half * code_bytes);
            const uint32_t *xp = xw + blk * 16u;
            HVX_Vector ia = q4 ? Q6_V_vsplat_R((uint32_t)corr[blk]) : Q6_V_vzero();
            HVX_Vector ib = Q6_V_vzero();
            if (q4) {
#pragma clang loop unroll(full)
                for (uint32_t v = 0; v < 4u; v++) {
                    const HVX_Vector c = cv[v];
                    const HVX_VectorPair lo = Q6_Wuh_vunpack_Vub(Q6_V_vand_VV(c, nib));
                    const HVX_VectorPair hi = Q6_Wuh_vunpack_Vub(Q6_Vub_vlsr_VubR(c, 4));
                    ia = Q6_Vw_vdmpyacc_VwVhRh_sat(ia, Q6_V_lo_W(lo), xp[4u * v + 0u]);
                    ib = Q6_Vw_vdmpyacc_VwVhRh_sat(ib, Q6_V_hi_W(lo), xp[4u * v + 1u]);
                    ia = Q6_Vw_vdmpyacc_VwVhRh_sat(ia, Q6_V_lo_W(hi), xp[4u * v + 2u]);
                    ib = Q6_Vw_vdmpyacc_VwVhRh_sat(ib, Q6_V_hi_W(hi), xp[4u * v + 3u]);
                }
            } else {
#pragma clang loop unroll(full)
                for (uint32_t v = 0; v < 8u; v++) {
                    const HVX_VectorPair h = Q6_Wh_vunpack_Vb(cv[v]);
                    ia = Q6_Vw_vdmpyacc_VwVhRh_sat(ia, Q6_V_lo_W(h), xp[2u * v + 0u]);
                    ib = Q6_Vw_vdmpyacc_VwVhRh_sat(ib, Q6_V_hi_W(h), xp[2u * v + 1u]);
                }
            }
            union {
                float f;
                uint32_t u;
            } s = {sx[blk]};
            const HVX_Vector row_scale = Q6_Vsf_equals_Vqf32(half ? Q6_V_hi_W(sc) : Q6_V_lo_W(sc));
            const HVX_Vector comb = mul(row_scale, Q6_V_vsplat_R(s.u));
            acc = Q6_Vqf32_vadd_Vqf32Vqf32(
                acc, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(Q6_Vw_vadd_VwVw(ia, ib)), comb));
        }
    }
    *(HVX_UVector *)out = Q6_Vsf_equals_Vqf32(acc);
}
#endif
#if defined(__hexagon__) && HX_LINEAR_Q_DMA
/* LINEAR_Q weight streaming. Uncached HVX loads are one DDR round trip each (~1 GB/s over four workers);
 * the user-DMA engine keeps many requests in flight. Every worker streams its own contiguous row range with
 * its own engine (the descriptor sequence is the Qwen engine's) through a ring of slots in its quarter of the
 * 1 MiB VTCM workspace (free outside HMX conv/staged ops), and computes from VTCM. A chunk is cut at row
 * boundaries; its DMA starts at the aligned-down first byte and covers 256 bytes past the last row for the
 * row kernel's window over-read (the constant pool keeps that padding). An engine that is busy or faults
 * leaves the op on the direct DDR path. */
#define HX_LQ_RING 0x31700000u
#define HX_LQ_PART_BYTES 0x40000u
#define HX_LQ_SLOTS 4u
#define HX_LQ_SLOT_BYTES (HX_LQ_PART_BYTES / HX_LQ_SLOTS)
#define HX_LQ_DESC (0xf1800000u + 0x7f0000u) /* uncached arena, 4 KiB per worker (Qwen's DESC_RING) */
#define HX_LQ_TIMEOUT 400000000ull
#ifndef HX_LQ_NOCOMPUTE
#define HX_LQ_NOCOMPUTE 0 /* measurement only: stream the weights, skip the dot products */
#endif
static uint64_t hx_lq_cycles(void) {
    uint64_t c;
    __asm__ volatile("%0=upcycle" : "=r"(c));
    return c;
}
static uint32_t hx_lq_dm_read(uint32_t index) {
    uint32_t value;
    __asm__ volatile("r1 = %1\n .word 0xa801c0a0\n %0 = r0" : "=r"(value) : "r"(index) : "r0", "r1", "memory");
    return value;
}
static void hx_lq_dm_write(uint32_t index, uint32_t value) {
    __asm__ volatile("r1 = %0\n r0 = %1\n .word 0xa801c0c0" ::"r"(index), "r"(value) : "r0", "r1", "memory");
}
static uint32_t hx_lq_poll(void) {
    uint32_t state;
    __asm__ volatile("%0 = dmpoll" : "=r"(state)::"memory");
    return state;
}
/* Counters for the host: [part*4 + 0] chunks, +1 wait cycles>>8, +2 fallbacks, +3 last error. */
#define HX_LQ_STATS ((volatile uint32_t *)(0xf1800000u + 0x7f4000u))
static volatile uint32_t *hx_lq_issue(uint32_t part, uint32_t *ring, uint32_t *tail, uint32_t src,
                                      uint32_t dst, uint32_t bytes) {
    const uint32_t addr = HX_LQ_DESC + part * 0x1000u + ((*ring)++ % 63u) * 32u;
    volatile uint32_t *d = (volatile uint32_t *)addr;
    d[0] = 0;
    d[1] = (bytes & 0xffffffu) | (1u << 24) | (1u << 28) | (1u << 29); /* 2D, bypass, !done */
    d[2] = src;
    d[3] = dst;
    d[4] = 9u; /* 2D, 24-bit fields */
    d[5] = (bytes & 0xffffffu) | (1u << 24);
    d[6] = (bytes & 0xffffffu) << 8;
    d[7] = 0;
    __asm__ volatile("barrier" ::: "memory");
    __asm__ volatile("release(%0):at" ::"r"(addr) : "memory");
    __asm__ volatile("dmlink(%0, %1)" ::"r"(*tail), "r"(addr) : "memory");
    *tail = addr;
    return d;
}
/* Rows [r0, r1) of every activation row; returns 1 when all were written. The row formats stream whole rows
 * (4 slots of 64 KiB); the grouped ones whole 32-row groups (2 slots of 128 KiB, the largest group is 127.5). */
static int hx_linear_q_dma(const HxGraph *g, const HxOp *o, const float *x, float *out, uint32_t part,
                           uint32_t r0, uint32_t r1) {
    const HxTensor *b = &g->tensors[o->out], *xt = &g->tensors[o->x];
    const uint32_t fmt = o->p[0], k = o->p[3], blocks = k / 32u, grouped = fmt >= 2u;
    const uint32_t row_bytes = blocks * ((fmt & 1u) ? 34u : 18u);
    const uint32_t unit_rows = grouped ? 32u : 1u, unit_bytes = row_bytes * unit_rows;
    const uint32_t slots = grouped ? 2u : HX_LQ_SLOTS, slot_bytes = HX_LQ_PART_BYTES / slots;
    const float *bias = o->p[2] != HXFG_NONE ? (const float *)(g->constants + o->p[2]) : NULL;
    volatile uint32_t *stats = HX_LQ_STATS + part * 4u;
    if (r0 >= r1 || part >= 4u || unit_bytes + 512u > slot_bytes || (grouped && (blocks > 128u || b->h > 1u)))
        return 0;
    const uint32_t state = hx_lq_poll();
    if ((state & 3u) == 1u) { /* still running: not ours to touch */
        stats[2]++;
        return 0;
    }
    if (state & 3u) {
        uint32_t ignored;
        __asm__ volatile("%0 = dmpause" : "=r"(ignored)::"memory");
    }
    int16_t xq[128 * 32 + 64] __attribute__((aligned(128)));
    float sx[128];
    int32_t corr[128];
    const uint32_t saved_dm2 = hx_lq_dm_read(2);
    hx_lq_dm_write(2, 0x2e3u);
    uint32_t tail = HX_LQ_DESC + part * 0x1000u + 63u * 32u, ring = 0;
    for (uint32_t i = 0; i < 8u; i++)
        ((volatile uint32_t *)tail)[i] = 0;
    const uint32_t u0 = r0 / unit_rows, u1 = r1 / unit_rows;
    uint32_t per = (slot_bytes - 512u) / unit_bytes;
    if (!grouped && per >= 4u)
        per &= ~3u; /* whole four-row groups per chunk */
    const uint32_t chunks = (u1 - u0 + per - 1u) / per;
    const uint8_t *w = (const uint8_t *)(g->constants + o->p[1]);
    uint8_t *ring_base = (uint8_t *)(HX_LQ_RING + part * HX_LQ_PART_BYTES);
    volatile uint32_t *desc[HX_LQ_SLOTS];
    uint32_t delta[HX_LQ_SLOTS];
    uint32_t issued = 0;
    int ok = 1;
#define HX_LQ_ISSUE_NEXT()                                                                          \
    do {                                                                                            \
        const uint32_t ca = u0 + issued * per, cb = ca + per < u1 ? ca + per : u1;                  \
        const uint32_t first = (uint32_t)(uintptr_t)(w + (uint64_t)ca * unit_bytes);                \
        const uint32_t start = first & ~127u;                                                       \
        const uint32_t end = ((uint32_t)(uintptr_t)(w + (uint64_t)cb * unit_bytes) + 256u + 127u) & \
                             ~127u;                                                                 \
        const uint32_t slot = issued % slots;                                                       \
        delta[slot] = first - start;                                                                \
        desc[slot] = hx_lq_issue(part, &ring, &tail, start,                                         \
                                 (uint32_t)(uintptr_t)(ring_base + slot * slot_bytes), end - start);  \
        issued++;                                                                                   \
    } while (0)
    while (issued < chunks && issued < slots)
        HX_LQ_ISSUE_NEXT();
    if (grouped) /* while the first chunks stream in */
        hx_lq_quantize_x(x, blocks, xq, sx, corr);
    for (uint32_t c = 0; c < chunks; c++) {
        const uint32_t slot = c % slots;
        const uint64_t began = hx_lq_cycles(), deadline = began + HX_LQ_TIMEOUT;
        while (!(desc[slot][1] >> 31)) {
            if ((hx_lq_poll() & 3u) == 2u || hx_lq_cycles() > deadline) {
                ok = 0;
                break;
            }
            const uint64_t until = hx_lq_cycles() + 600u;
            while (hx_lq_cycles() < until) {
            }
        }
        stats[1] += (uint32_t)((hx_lq_cycles() - began) >> 8);
        if (!ok) {
            stats[3] = hx_lq_poll() | 0x80000000u;
            break;
        }
        const uint32_t ca = u0 + c * per, cb = ca + per < u1 ? ca + per : u1;
        const uint8_t *units = ring_base + slot * slot_bytes + delta[slot];
        if (grouped) {
            for (uint32_t gi = ca; gi < (HX_LQ_NOCOMPUTE ? ca : cb); gi++) {
                float dots[32] __attribute__((aligned(128)));
                hx_lq_group_hvx(units + (uint64_t)(gi - ca) * unit_bytes, xq, sx, corr, blocks, fmt == 2u, dots);
                for (uint32_t i = 0; i < 32u; i++) {
                    const uint32_t r = gi * 32u + i;
                    float val = (bias ? bias[r] : 0.0f) + dots[i];
                    if (o->p[5] == 1) val = fmaxf(0.0f, val);
                    else if (o->p[5] == 3) val = val / (1.0f + expf(-val));
                    out[r] = val;
                }
            }
        } else
        for (uint32_t row = 0; row < (HX_LQ_NOCOMPUTE ? 0u : b->h); row++) {
            const float *xr = x + (uint64_t)row * xt->stride;
            float *orow = out + (uint64_t)row * b->stride;
            for (uint32_t r = ca; r < cb;) {
                float dots[4];
                uint32_t count = 1;
                if (r + 4u <= cb) {
                    hx_linear_q_rows4_hvx(units + (uint64_t)(r - ca) * row_bytes, row_bytes, xr, blocks, fmt, dots);
                    count = 4;
                } else
                    dots[0] = hx_linear_q_row_hvx(units + (uint64_t)(r - ca) * row_bytes, xr, blocks, fmt);
                for (uint32_t i = 0; i < count; i++, r++) {
                    float val = (bias ? bias[r] : 0.0f) + dots[i];
                    if (o->p[5] == 1) val = fmaxf(0.0f, val);
                    else if (o->p[5] == 3) val = val / (1.0f + expf(-val));
                    orow[r] = val;
                }
            }
        }
        if (issued < chunks)
            HX_LQ_ISSUE_NEXT();
    }
#undef HX_LQ_ISSUE_NEXT
    if (!ok) {
        /* An engine that did not finish may still write this ring; leave DM2 as is and stop using DMA
         * for this op. The caller recomputes every row from DDR. */
        stats[2]++;
        return 0;
    }
    hx_lq_dm_write(2, saved_dm2);
    stats[0] += chunks;
    return 1;
}
#endif
/* Grouped layouts (fmt 2 Q4, 3 Q8): byte offset of row r's code for input kk of 32-block bk, and of its fp16 scale,
 * inside the group that starts at the weight base + (r / 32) * 32 * row_bytes. See hx_lq_group_hvx. */
static inline uint32_t hx_lq_group_code_at(uint32_t q4, uint32_t rr, uint32_t bk, uint32_t kk) {
    const uint32_t code_bytes = q4 ? 512u : 1024u, pair = (bk / 2u) * (2u * code_bytes + 128u) + (bk & 1u) * code_bytes;
    const uint32_t v = kk / 4u, s = (kk / 2u) & 1u, j = kk & 1u;
    return pair + (q4 ? v / 2u : v) * 128u + s * 64u + 2u * rr + j;
}
static inline uint32_t hx_lq_group_scale_at(uint32_t q4, uint32_t rr, uint32_t bk) {
    const uint32_t code_bytes = q4 ? 512u : 1024u;
    return (bk / 2u) * (2u * code_bytes + 128u) + 2u * code_bytes + (2u * rr + (bk & 1u)) * 2u;
}
static inline int32_t hx_lq_group_code(const uint8_t *grp, uint32_t q4, uint32_t rr, uint32_t bk, uint32_t kk) {
    const uint8_t byte = grp[hx_lq_group_code_at(q4, rr, bk, kk)];
    if (!q4)
        return (int8_t)byte;
    return (int32_t)(((kk / 4u) & 1u) ? byte >> 4 : byte & 15u) - 8;
}
/* In place: rows of a fmt 0 (Q4_0) / 1 (Q8_0) weight [n][k/32 blocks] become the grouped layout of fmt 2 / 3,
 * which has the same size. Needs k % 64 == 0, n % 32 == 0 and scratch of 32 rows (32 * k / 32 * 34 bytes). */
void hx_lq_group_layout(uint8_t *w, uint32_t fmt, uint32_t k, uint32_t n, uint8_t *scratch) {
    const uint32_t q4 = fmt == 0u, blocks = k / 32u, pitch = q4 ? 18u : 34u, row_bytes = blocks * pitch;
    const uint32_t group_bytes = 32u * row_bytes;
    for (uint32_t gi = 0; gi < n / 32u; gi++) {
        uint8_t *grp = w + (uint64_t)gi * group_bytes;
        memcpy(scratch, grp, group_bytes);
        memset(grp, 0, group_bytes);
        for (uint32_t rr = 0; rr < 32u; rr++)
            for (uint32_t bk = 0; bk < blocks; bk++) {
                const uint8_t *src = scratch + rr * row_bytes + bk * pitch;
                const uint32_t sa = hx_lq_group_scale_at(q4, rr, bk);
                grp[sa] = src[q4 ? 16 : 32];
                grp[sa + 1] = src[q4 ? 17 : 33];
                for (uint32_t kk = 0; kk < 32u; kk++) {
                    const uint32_t at = hx_lq_group_code_at(q4, rr, bk, kk);
                    if (!q4)
                        grp[at] = src[kk];
                    else {
                        const uint8_t code = (kk & 1u) ? src[kk / 2u] >> 4 : src[kk / 2u] & 15u; /* code + 8 */
                        grp[at] |= ((kk / 4u) & 1u) ? (uint8_t)(code << 4) : code;
                    }
                }
            }
    }
}
static void hx_linear_q_grouped_op(const HxGraph *g, const HxOp *o, const float *x, float *out, uint32_t part,
                                   uint32_t parts) {
    const HxTensor *b = &g->tensors[o->out], *xt = &g->tensors[o->x];
    const uint32_t q4 = o->p[0] == 2u, k = o->p[3], n = o->p[4], blocks = k / 32u;
    const uint32_t row_bytes = blocks * (q4 ? 18u : 34u), groups = n / 32u;
    const uint32_t g0 = groups * part / parts, g1 = groups * (part + 1u) / parts;
    const float *bias = o->p[2] != HXFG_NONE ? (const float *)(g->constants + o->p[2]) : NULL;
    const uint8_t *w = (const uint8_t *)(g->constants + o->p[1]);
    if (part == 0)
        for (uint32_t row = 0; row < b->h; row++)
            for (uint32_t r = n; r < b->stride; r++)
                out[(uint64_t)row * b->stride + r] = 0.0f;
#ifdef __hexagon__
    if (b->h == 1u && blocks <= 128u && !((uintptr_t)w & 127u)) {
#if HX_LINEAR_Q_DMA
        if (parts == 4u && hx_linear_q_dma(g, o, x, out, part, g0 * 32u, g1 * 32u))
            return;
#endif
        int16_t xq[128 * 32 + 64] __attribute__((aligned(128)));
        float sx[128];
        int32_t corr[128];
        hx_lq_quantize_x(x, blocks, xq, sx, corr);
        for (uint32_t gi = g0; gi < g1; gi++) {
            float dots[32] __attribute__((aligned(128)));
            hx_lq_group_hvx(w + (uint64_t)gi * 32u * row_bytes, xq, sx, corr, blocks, q4, dots);
            for (uint32_t i = 0; i < 32u; i++) {
                const uint32_t r = gi * 32u + i;
                float val = (bias ? bias[r] : 0.0f) + dots[i];
                if (o->p[5] == 1) val = fmaxf(0.0f, val);
                else if (o->p[5] == 3) val = val / (1.0f + expf(-val));
                out[r] = val;
            }
        }
        return;
    }
#endif
    for (uint32_t row = 0; row < b->h; row++) {
        const float *xr = x + (uint64_t)row * xt->stride;
        float *orow = out + (uint64_t)row * b->stride;
        for (uint32_t r = g0 * 32u; r < g1 * 32u; r++) {
            const uint8_t *grp = w + (uint64_t)(r / 32u) * 32u * row_bytes;
            double sum = bias ? (double)bias[r] : 0.0;
            for (uint32_t bk = 0; bk < blocks; bk++) {
                const uint32_t at = hx_lq_group_scale_at(q4, r % 32u, bk);
                const double scale = hx_half_to_float((uint16_t)(grp[at] | (grp[at + 1] << 8)));
                double acc = 0.0;
                for (uint32_t kk = 0; kk < 32u; kk++)
                    acc += (double)xr[bk * 32u + kk] * (double)hx_lq_group_code(grp, q4, r % 32u, bk, kk);
                sum += acc * scale;
            }
            float val = (float)sum;
            if (o->p[5] == 1) val = fmaxf(0.0f, val);
            else if (o->p[5] == 3) val = val / (1.0f + expf(-val));
            orow[r] = val;
        }
    }
}
static void hx_linear_q_op(const HxGraph *g, const HxOp *o, const float *x, float *out,
                           uint32_t part, uint32_t parts) {
    const HxTensor *b = &g->tensors[o->out], *xt = &g->tensors[o->x];
    const uint32_t fmt = o->p[0], k = o->p[3], n = o->p[4];
    const float *bias = o->p[2] != HXFG_NONE ? (const float *)(g->constants + o->p[2]) : NULL;
    const uint32_t blocks = k / 32u;
    if (fmt >= 2u) {
        hx_linear_q_grouped_op(g, o, x, out, part, parts);
        return;
    }
    const uint32_t r0 = n * part / parts, r1 = n * (part + 1u) / parts;
#if defined(__hexagon__) && HX_LINEAR_Q_DMA
    if (k % 32u == 0 && parts == 4u && hx_linear_q_dma(g, o, x, out, part, r0, r1)) {
        if (part == 0)
            for (uint32_t row = 0; row < b->h; row++)
                for (uint32_t r = n; r < b->stride; r++)
                    out[(uint64_t)row * b->stride + r] = 0.0f;
        return;
    }
#endif

    for (uint32_t row = 0; row < b->h; row++) {
    const float *xr = x + (uint64_t)row * xt->stride;
    float *orow = out + (uint64_t)row * b->stride;
    for (uint32_t r = r0; r < r1; r++) {
        double sum = bias ? (double)bias[r] : 0.0;
#ifdef __hexagon__
        if (k % 32u == 0) {
            const uint8_t *w_row = (const uint8_t *)(g->constants + o->p[1]) + (uint64_t)r * blocks * (fmt == 0 ? 18u : 34u);
            float val = (bias ? bias[r] : 0.0f) + hx_linear_q_row_hvx(w_row, xr, blocks, fmt);
            if (o->p[5] == 1) val = fmaxf(0.0f, val);
            else if (o->p[5] == 3) val = val / (1.0f + expf(-val));
            orow[r] = val;
            continue;
        }
#endif
        if (fmt == 0) {
            const uint8_t *w_row = (const uint8_t *)(g->constants + o->p[1]) + (uint64_t)r * blocks * 18u;
            for (uint32_t b_idx = 0; b_idx < blocks; b_idx++) {
                const uint8_t *blk = w_row + b_idx * 18u;
                const uint16_t d16 = (uint16_t)blk[16] | ((uint16_t)blk[17] << 8);
                const float scale = hx_half_to_float(d16);
                const float *x_blk = xr + b_idx * 32u;
                double blk_acc = 0.0;
                for (uint32_t j = 0; j < 16u; j++) {
                    const uint8_t byte = blk[j];
                    const int32_t q0 = (int32_t)(byte & 0x0fu) - 8;
                    const int32_t q1 = (int32_t)(byte >> 4u) - 8;
                    blk_acc += (double)x_blk[2u * j] * (double)q0 + (double)x_blk[2u * j + 1u] * (double)q1;
                }
                sum += blk_acc * (double)scale;
            }
        } else {
            const uint8_t *w_row = (const uint8_t *)(g->constants + o->p[1]) + (uint64_t)r * blocks * 34u;
            for (uint32_t b_idx = 0; b_idx < blocks; b_idx++) {
                const uint8_t *blk = w_row + b_idx * 34u;
                const uint16_t d16 = (uint16_t)blk[32] | ((uint16_t)blk[33] << 8);
                const float scale = hx_half_to_float(d16);
                const float *x_blk = xr + b_idx * 32u;
                double blk_acc = 0.0;
                for (uint32_t j = 0; j < 32u; j++) {
                    const int8_t q = (int8_t)blk[j];
                    blk_acc += (double)x_blk[j] * (double)q;
                }
                sum += blk_acc * (double)scale;
            }
        }
        float val = (float)sum;
        if (o->p[5] == 1) val = fmaxf(0.0f, val);
        else if (o->p[5] == 3) val = val / (1.0f + expf(-val));
        orow[r] = val;
    }
    if (part == 0) {
        for (uint32_t r = n; r < b->stride; r++)
            orow[r] = 0.0f;
    }
    }
}
static void hx_kv_write_op(const HxGraph *g, const HxOp *o, const float *x, const float *y,
                           float *out, uint32_t part, uint32_t parts) {
    const HxTensor *xt = &g->tensors[o->x];
    const HxTensor *outt = &g->tensors[o->out];
    uint32_t pos = o->p[0];
    if (o->y != HXFG_NONE && y) {
        pos = hx_position(g, o->y, y);
    }
    uint32_t tokens = xt->h * xt->w;
    uint32_t max_tokens = outt->h * outt->w;
    if (pos >= max_tokens)
        return;
    if (pos + tokens > max_tokens)
        tokens = max_tokens - pos;
    uint32_t c = xt->c;
    uint32_t c0 = c * part / parts, c1 = c * (part + 1u) / parts;
    for (uint32_t t = 0; t < tokens; t++) {
        const float *src = x + t * xt->stride;
        float *dst = out + (pos + t) * outt->stride;
        for (uint32_t ch = c0; ch < c1; ch++)
            dst[ch] = src[ch];
    }
    if (part == 0 && outt->stride > c) {
        for (uint32_t t = 0; t < tokens; t++) {
            float *dst = out + (pos + t) * outt->stride;
            for (uint32_t ch = c; ch < outt->stride; ch++)
                dst[ch] = 0.0f;
        }
    }
}
static void hx_gather_op(const HxGraph *g, const HxOp *o, const float *x_float, float *out,
                         uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x]; // indices or data
    const HxTensor *b = &g->tensors[o->out]; // out
    const uint32_t *p = o->p;
    const uint32_t vocab = p[2];
    const uint32_t axis = p[4];
    const int int_indices = (p[3] == 0) || (p[0] == 2) || (a->dtype == HX_DTYPE_INT32);
    const int32_t *x_int = (p[0] == 2) ? (const int32_t *)(g->constants + p[8]) : (const int32_t *)x_float;
    const uint32_t n_indices = (p[0] == 2) ? p[9] : hx_tensor_elements(a);

    if (axis == 0 && (p[5] == 0 || p[5] == b->c)) {
        const uint32_t stride_tbl = p[7] ? p[7] : (p[0] == 1 ? b->stride : (p[0] == 2 ? a->stride : g->tensors[o->y].stride));
        const float *tbl_base = (p[0] == 1)
                                    ? ((const float *)(g->constants + p[1]))
                                    : ((p[0] == 2) ? x_float : ((const float *)(hx_tensor_base(g, &g->tensors[o->y]) + g->tensors[o->y].offset)));
        const uint32_t pixels = b->h * b->w;
        const uint32_t lo = pixels * part / parts, hi = pixels * (part + 1u) / parts;
        for (uint32_t i = lo; i < hi; i++) {
            int32_t idx = 0;
            if (p[0] == 2) {
                idx = x_int[i % n_indices];
            } else {
                uint32_t a_pixel = a->c ? i / a->c : 0, a_c = a->c ? i % a->c : 0;
                uint32_t idx_pos = a_pixel * a->stride + a_c;
                idx = int_indices ? x_int[idx_pos] : (int32_t)lrintf(x_float[idx_pos]);
            }
            if (idx < 0) idx += (int32_t)vocab;
            idx = idx < 0 ? 0 : (idx >= (int32_t)vocab ? (int32_t)vocab - 1 : idx);
            const float *row = tbl_base + (uint32_t)idx * stride_tbl;
            float *dst = out + i * b->stride;
            for (uint32_t c = 0; c < b->stride; c++)
                dst[c] = c < b->c ? row[c] : 0.0f;
        }
    } else {
        const uint32_t inner = p[5] ? p[5] : (axis == 2 ? 1 : b->c);
        const uint32_t outer = p[6] ? p[6] : 1;
        const uint32_t stride_tbl = p[7] ? p[7] : (p[0] == 1 ? inner : (p[0] == 2 ? a->stride : g->tensors[o->y].stride));
        const float *tbl_base = (p[0] == 1)
                                    ? ((const float *)(g->constants + p[1]))
                                    : ((p[0] == 2) ? x_float : ((const float *)(hx_tensor_base(g, &g->tensors[o->y]) + g->tensors[o->y].offset)));
        const uint32_t total_tasks = outer * n_indices;
        const uint32_t lo = total_tasks * part / parts, hi = total_tasks * (part + 1u) / parts;

        for (uint32_t task = lo; task < hi; task++) {
            uint32_t o_idx = task / n_indices;
            uint32_t j = task % n_indices;
            int32_t idx = 0;
            if (p[0] == 2) {
                idx = x_int[j];
            } else {
                uint32_t a_pixel = a->c ? j / a->c : 0, a_c = a->c ? j % a->c : 0;
                uint32_t idx_pos = a_pixel * a->stride + a_c;
                idx = int_indices ? x_int[idx_pos] : (int32_t)lrintf(x_float[idx_pos]);
            }
            if (idx < 0) idx += (int32_t)vocab;
            idx = idx < 0 ? 0 : (idx >= (int32_t)vocab ? (int32_t)vocab - 1 : idx);

            if (axis == 2 && inner == 1) {
                uint32_t src_pos = o_idx * stride_tbl + (uint32_t)idx;
                uint32_t dst_pos = o_idx * b->stride + j;
                out[dst_pos] = tbl_base[src_pos];
            } else {
                const float *src_row = tbl_base + (o_idx * vocab + (uint32_t)idx) * stride_tbl;
                float *dst_row = out + (o_idx * n_indices + j) * b->stride;
                for (uint32_t k = 0; k < inner; k++)
                    dst_row[k] = src_row[k];
                for (uint32_t k = inner; k < b->stride; k++)
                    dst_row[k] = 0.0f;
            }
        }
    }
}

static void hx_gather_elements_op(const HxGraph *g, const HxOp *o, const float *data, const float *indices_raw,
                                  float *out, uint32_t part, uint32_t parts) {
    const HxTensor *dt = &g->tensors[o->x]; // data
    const HxTensor *ot = &g->tensors[o->out]; // out
    const uint32_t *p = o->p;
    const uint32_t axis = p[0];
    const int const_indices = (p[2] == 2);
    const int int_indices = (p[1] == 0) || const_indices ||
                            (!const_indices && (g->tensors[o->y].dtype == HX_DTYPE_INT32));
    const int32_t *idx_int = const_indices ? (const int32_t *)(g->constants + p[3]) : (const int32_t *)indices_raw;
    const float *data_base = (p[2] == 1) ? (const float *)(g->constants + p[3]) : data;

    const uint32_t pixels = ot->h * ot->w;
    const uint32_t lo = pixels * part / parts, hi = pixels * (part + 1u) / parts;

    for (uint32_t px = lo; px < hi; px++) {
        uint32_t oy = px / ot->w, ox = px % ot->w;
        float *dst = out + px * ot->stride;
        for (uint32_t c = 0; c < ot->stride; c++) {
            if (c >= ot->c) {
                dst[c] = 0.0f;
                continue;
            }
            uint32_t idx_loc = const_indices ? (px * ot->c + c) : ((oy * g->tensors[o->y].w + ox) * g->tensors[o->y].stride + c);
            int32_t k = int_indices ? idx_int[idx_loc] : (int32_t)lrintf(indices_raw[idx_loc]);
            uint32_t sy = oy, sx = ox, sc = c;
            if (axis == 0) {
                if (k < 0) k += (int32_t)dt->h;
                sy = k < 0 ? 0 : (k >= (int32_t)dt->h ? dt->h - 1 : (uint32_t)k);
            } else if (axis == 1) {
                if (k < 0) k += (int32_t)dt->w;
                sx = k < 0 ? 0 : (k >= (int32_t)dt->w ? dt->w - 1 : (uint32_t)k);
            } else {
                if (k < 0) k += (int32_t)dt->c;
                sc = k < 0 ? 0 : (k >= (int32_t)dt->c ? dt->c - 1 : (uint32_t)k);
            }
            uint32_t src_loc = (sy * dt->w + sx) * dt->stride + sc;
            dst[c] = data_base[src_loc];
        }
    }
}

static void hx_gather_nd_op(const HxGraph *g, const HxOp *o, const float *data, const float *indices_raw,
                            float *out, uint32_t part, uint32_t parts) {
    const HxTensor *dt = &g->tensors[o->x]; // data
    const HxTensor *ot = &g->tensors[o->out]; // out
    const uint32_t *p = o->p;
    const int const_indices = (p[2] == 2);
    const int int_indices = (p[1] == 0) || const_indices ||
                            (!const_indices && (g->tensors[o->y].dtype == HX_DTYPE_INT32));
    const int32_t *idx_int = const_indices ? (const int32_t *)(g->constants + p[3]) : (const int32_t *)indices_raw;
    const float *data_base = (p[2] == 1) ? (const float *)(g->constants + p[3]) : data;
    const uint32_t k = p[4] ? p[4] : (const_indices ? 1 : g->tensors[o->y].c);
    const uint32_t slice_size = p[5] ? p[5] : (k == 1 ? dt->w * dt->c : (k == 2 ? dt->c : 1));
    const uint32_t total_elements = const_indices ? (hx_tensor_elements(ot) / slice_size * k) : hx_tensor_elements(&g->tensors[o->y]);
    const uint32_t M = k ? total_elements / k : 0;

    const uint32_t lo = M * part / parts, hi = M * (part + 1u) / parts;

    for (uint32_t m = lo; m < hi; m++) {
        int32_t coords[3] = {0, 0, 0};
        for (uint32_t d = 0; d < k && d < 3; d++) {
            uint32_t elem_idx = m * k + d;
            int32_t val = 0;
            if (const_indices) {
                val = idx_int[elem_idx];
            } else {
                const HxTensor *it = &g->tensors[o->y];
                uint32_t px = it->c ? elem_idx / it->c : 0, ch = it->c ? elem_idx % it->c : 0;
                uint32_t loc = px * it->stride + ch;
                val = int_indices ? idx_int[loc] : (int32_t)lrintf(indices_raw[loc]);
            }
            coords[d] = val;
        }
        int32_t sy = 0, sx = 0, sc = 0;
        if (k == 1) {
            sy = coords[0];
        } else if (k == 2) {
            if (dt->w == 1 && coords[0] == 0) {
                sy = coords[1];
                sx = 0;
            } else {
                sy = coords[0];
                sx = coords[1];
            }
        } else {
            if (dt->w == 1 && coords[0] == 0) {
                sy = coords[1];
                sx = 0;
                sc = coords[2];
            } else {
                sy = coords[0];
                sx = coords[1];
                sc = coords[2];
            }
        }
        if (sy < 0) sy += (int32_t)dt->h;
        if (sx < 0) sx += (int32_t)dt->w;
        if (sc < 0) sc += (int32_t)dt->c;

        sy = sy < 0 ? 0 : (sy >= (int32_t)dt->h ? (int32_t)dt->h - 1 : sy);
        sx = sx < 0 ? 0 : (sx >= (int32_t)dt->w ? (int32_t)dt->w - 1 : sx);
        sc = sc < 0 ? 0 : (sc >= (int32_t)dt->c ? (int32_t)dt->c - 1 : sc);

        if (k == 1) {
            uint32_t y = (uint32_t)sy;
            for (uint32_t x = 0; x < dt->w; x++) {
                const float *src = data_base + (y * dt->w + x) * dt->stride;
                float *dst = out + (m * dt->w + x) * ot->stride;
                for (uint32_t c = 0; c < ot->stride; c++)
                    dst[c] = c < dt->c ? src[c] : 0.0f;
            }
        } else if (k == 2) {
            const float *src = data_base + ((uint32_t)sy * dt->w + (uint32_t)sx) * dt->stride;
            float *dst = out + m * ot->stride;
            for (uint32_t c = 0; c < ot->stride; c++)
                dst[c] = c < ot->c ? src[c] : 0.0f;
        } else {
            float val = data_base[((uint32_t)sy * dt->w + (uint32_t)sx) * dt->stride + (uint32_t)sc];
            uint32_t dst_px = ot->c ? m / ot->c : 0, dst_c = ot->c ? m % ot->c : 0;
            out[dst_px * ot->stride + dst_c] = val;
        }
    }
}

static void hx_scatter_nd_op(const HxGraph *g, const HxOp *o, const float *data, const float *indices_raw,
                             const float *updates, float *out, uint32_t part, uint32_t parts) {
    const HxTensor *dt = &g->tensors[o->x]; // data
    const HxTensor *ot = &g->tensors[o->out]; // out
    const uint32_t *p = o->p;
    const int in_place = (p[0] == 1) || (o->out == o->x);
    const int const_indices = (p[5] & 1) != 0;
    const int const_updates = (p[5] & 2) != 0;
    const int int_indices = (p[1] == 0) || const_indices ||
                            (!const_indices && (g->tensors[o->y].dtype == HX_DTYPE_INT32));
    const int32_t *idx_int = const_indices ? (const int32_t *)(g->constants + p[6]) : (const int32_t *)indices_raw;
    const float *upd_base = const_updates ? (const float *)(g->constants + p[7]) : updates;
    const uint32_t k = p[2] ? p[2] : (const_indices ? 1 : g->tensors[o->y].c);
    const uint32_t slice_size = p[3] ? p[3] : (k == 1 ? ot->w * ot->c : (k == 2 ? ot->c : 1));
    const uint32_t reduction = p[4];

    if (!in_place) {
        const uint32_t pixels = ot->h * ot->w;
        const uint32_t lo = pixels * part / parts, hi = pixels * (part + 1u) / parts;
        for (uint32_t px = lo; px < hi; px++) {
            const float *src = data + px * dt->stride;
            float *dst = out + px * ot->stride;
            for (uint32_t c = 0; c < ot->stride; c++)
                dst[c] = c < ot->c ? src[c] : 0.0f;
        }
    }

    if (part + 1 == parts) {
        uint32_t M = p[8];
        if (!M) {
            if (const_indices) {
                uint32_t total_u = const_updates ? (p[3] ? slice_size * 2 : slice_size) : hx_tensor_elements(&g->tensors[o->z]);
                M = slice_size ? total_u / slice_size : 0;
            } else {
                const uint32_t total_elements = hx_tensor_elements(&g->tensors[o->y]);
                M = k ? total_elements / k : 0;
            }
        }
        for (uint32_t m = 0; m < M; m++) {
            int32_t coords[3] = {0, 0, 0};
            for (uint32_t d = 0; d < k && d < 3; d++) {
                uint32_t elem_idx = m * k + d;
                int32_t val = 0;
                if (const_indices) {
                    val = idx_int[elem_idx];
                } else {
                    const HxTensor *it = &g->tensors[o->y];
                    uint32_t px = it->c ? elem_idx / it->c : 0, ch = it->c ? elem_idx % it->c : 0;
                    uint32_t loc = px * it->stride + ch;
                    val = int_indices ? idx_int[loc] : (int32_t)lrintf(indices_raw[loc]);
                }
                coords[d] = val;
            }
            if (coords[0] < 0) coords[0] += (int32_t)ot->h;
            int32_t dy = 0, dx = 0, dc = 0;
            if (k == 1) {
                dy = coords[0];
            } else if (k == 2) {
                if (ot->w == 1 && coords[0] == 0) {
                    dy = coords[1];
                    dx = 0;
                } else {
                    dy = coords[0];
                    dx = coords[1];
                }
            } else {
                if (ot->w == 1 && coords[0] == 0) {
                    dy = coords[1];
                    dx = 0;
                    dc = coords[2];
                } else {
                    dy = coords[0];
                    dx = coords[1];
                    dc = coords[2];
                }
            }
            if (dy < 0) dy += (int32_t)ot->h;
            if (dx < 0) dx += (int32_t)ot->w;
            if (dc < 0) dc += (int32_t)ot->c;

            dy = dy < 0 ? 0 : (dy >= (int32_t)ot->h ? (int32_t)ot->h - 1 : dy);
            dx = dx < 0 ? 0 : (dx >= (int32_t)ot->w ? (int32_t)ot->w - 1 : dx);
            dc = dc < 0 ? 0 : (dc >= (int32_t)ot->c ? (int32_t)ot->c - 1 : dc);

            if (k == 1) {
                uint32_t y = (uint32_t)dy;
                for (uint32_t x = 0; x < ot->w; x++) {
                    const float *u = const_updates ? (upd_base + (m * ot->w + x) * ot->c) :
                                     (upd_base + (m * ot->w + x) * g->tensors[o->z].stride);
                    float *dst = out + (y * ot->w + x) * ot->stride;
                    for (uint32_t c = 0; c < ot->c; c++) {
                        float uv = u[c];
                        if (reduction == 1) dst[c] += uv;
                        else if (reduction == 2) dst[c] *= uv;
                        else if (reduction == 3) dst[c] = fmaxf(dst[c], uv);
                        else if (reduction == 4) dst[c] = fminf(dst[c], uv);
                        else dst[c] = uv;
                    }
                }
            } else if (k == 2) {
                const float *u = const_updates ? (upd_base + m * ot->c) : (upd_base + m * g->tensors[o->z].stride);
                float *dst = out + ((uint32_t)dy * ot->w + (uint32_t)dx) * ot->stride;
                for (uint32_t c = 0; c < ot->c; c++) {
                    float uv = u[c];
                    if (reduction == 1) dst[c] += uv;
                    else if (reduction == 2) dst[c] *= uv;
                    else if (reduction == 3) dst[c] = fmaxf(dst[c], uv);
                    else if (reduction == 4) dst[c] = fminf(dst[c], uv);
                    else dst[c] = uv;
                }
            } else {
                float uv = 0.0f;
                if (const_updates) {
                    uv = upd_base[m];
                } else {
                    const HxTensor *ut = &g->tensors[o->z];
                    uint32_t up_px = ut->c ? m / ut->c : 0, up_c = ut->c ? m % ut->c : 0;
                    uv = upd_base[up_px * ut->stride + up_c];
                }
                float *dst = out + ((uint32_t)dy * ot->w + (uint32_t)dx) * ot->stride + (uint32_t)dc;
                if (reduction == 1) *dst += uv;
                else if (reduction == 2) *dst *= uv;
                else if (reduction == 3) *dst = fmaxf(*dst, uv);
                else if (reduction == 4) *dst = fminf(*dst, uv);
                else *dst = uv;
            }
        }
    }
}

static void hx_cubic_weights(float t, float w[4]) {
    const float A = -0.75f;
    for (int i = 0; i < 4; i++) {
        float d = fabsf(t - (float)(i - 1));
        if (d <= 1.0f)
            w[i] = ((A + 2.0f) * d - (A + 3.0f)) * d * d + 1.0f;
        else if (d < 2.0f)
            w[i] = ((A * d - 5.0f * A) * d + 8.0f * A) * d - 4.0f * A;
        else
            w[i] = 0.0f;
    }
}
#ifdef __hexagon__
/* One SDPA query row on HVX. Keys after `last` (causal) and keys whose additive mask is <= -1e8 are skipped:
 * with any unmasked key their weight exp(score - peak) is exactly 0 in fp32. If every key is masked the
 * caller's scalar path runs instead (return 0). Dots and the V sum accumulate in qf32 vectors; K/V rows are
 * read as whole vectors (uncached persistent state: one round trip per vector instead of per element). */
static __attribute__((noinline)) int hx_sdpa_row_hvx(const float *q_row, const float *kb, uint32_t ks, const float *vb, uint32_t vs,
                           uint32_t lk, uint32_t dk, uint32_t dv, float scale, uint32_t last, const float *m_row,
                           uint32_t mask_scalar, float *dst) {
    float scores[2048]; /* worker stacks are 64 KiB and the caller already holds 16 KiB of scores */
    uint16_t keep[2048];
    HVX_Vector qv[16], ov[16];
    uint32_t kept = 0;
    for (uint32_t j = 0; j < dk / 32u; j++)
        qv[j] = *(const HVX_UVector *)(q_row + j * 32u);
    float peak = -FLT_MAX;
    const uint32_t end = last < lk ? last + 1u : lk;
    for (uint32_t k = 0; k < end; k++) {
        const float m = m_row ? (mask_scalar ? m_row[0] : m_row[k]) : 0.0f;
        if (m <= -1e8f)
            continue;
        const float *k_row = kb + (uint64_t)k * ks;
        HVX_Vector acc = Q6_V_vzero();
        for (uint32_t j = 0; j < dk / 32u; j++)
            acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(qv[j], *(const HVX_UVector *)(k_row + j * 32u)));
        HVX_Vector a = Q6_Vsf_equals_Vqf32(acc);
        for (int shift = 64; shift >= 4; shift >>= 1)
            a = add(a, Q6_V_vror_VR(a, shift));
        union {
            HVX_Vector v;
            float f[32];
        } r;
        r.v = a;
        const float dot = r.f[0] * scale + m;
        scores[kept] = dot;
        keep[kept++] = (uint16_t)k;
        if (dot > peak)
            peak = dot;
    }
    if (!kept)
        return 0;
    float sum = 0.0f;
    for (uint32_t i = 0; i < kept; i++) {
        scores[i] = expf(scores[i] - peak);
        sum += scores[i];
    }
    const float inv_sum = 1.0f / sum;
    for (uint32_t j = 0; j < dv / 32u; j++)
        ov[j] = Q6_V_vzero();
    for (uint32_t i = 0; i < kept; i++) {
        const float w = scores[i] * inv_sum;
        if (!(w > 0.0f))
            continue;
        union {
            float f;
            uint32_t u;
        } wb = {w};
        const HVX_Vector wv = Q6_V_vsplat_R(wb.u);
        const float *v_row = vb + (uint64_t)keep[i] * vs;
        for (uint32_t j = 0; j < dv / 32u; j++)
            ov[j] = Q6_Vqf32_vadd_Vqf32Vqf32(ov[j], Q6_Vqf32_vmpy_VsfVsf(wv, *(const HVX_UVector *)(v_row + j * 32u)));
    }
    for (uint32_t j = 0; j < dv / 32u; j++)
        *(HVX_UVector *)(dst + j * 32u) = Q6_Vsf_equals_Vqf32(ov[j]);
    return 1;
}
#endif
#ifdef __hexagon__
/* One pixel, many channels (decode): the pixel split would leave the whole op on one worker, so each worker takes
 * a quarter of the channel vectors. Float ADD/SUB/MUL and SiLU only; returns 0 for anything else. */
static int hx_single_pixel_hvx(const HxGraph *g, const HxOp *o, const float *x, const float *y, float *out,
                               uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p, c = b->c;
    if (b->h * b->w != 1u || (c & 31u) || b->stride != c || a->stride < c || c < 256u ||
        a->dtype != HX_DTYPE_FLOAT32 || b->dtype != HX_DTYPE_FLOAT32)
        return 0;
    const uint32_t vectors = c / 32u, v0 = vectors * part / parts, v1 = vectors * (part + 1u) / parts;
    if (o->kind == HX_ACT) {
        if (p[0] != HX_ACT_SILU)
            return 0;
        for (uint32_t v = v0; v < v1; v++)
            *(HVX_UVector *)(out + v * 32u) = hx_vsilu(*(const HVX_UVector *)(x + v * 32u));
        return 1;
    }
    if (o->kind != HX_BINARY || p[0] > HX_BIN_MUL)
        return 0;
    const HxTensor *t = o->y == HXFG_NONE ? NULL : &g->tensors[o->y];
    const float *other;
    uint32_t broadcast = 0;
    if (p[1] == 1)
        other = (const float *)(g->constants + p[2]); /* the constant wins over y, as in the generic path */
    else if (t) {
        if (t->dtype != HX_DTYPE_FLOAT32 || t->h * t->w != 1u || (t->c != c && t->c != 1u))
            return 0;
        other = y;
        broadcast = t->c == 1u;
    } else if (p[1] == 2)
        other = (const float *)(g->constants + p[2]);
    else
        return 0;
    const HVX_Vector one = broadcast ? splat(other[0]) : Q6_V_vzero();
    for (uint32_t v = v0; v < v1; v++) {
        HVX_Vector l = *(const HVX_UVector *)(x + v * 32u);
        HVX_Vector u = broadcast ? one : *(const HVX_UVector *)(other + v * 32u);
        if (p[3]) {
            const HVX_Vector s = l;
            l = u;
            u = s;
        }
        HVX_Vector r;
        if (p[0] == HX_BIN_ADD)
            r = add(l, u);
        else if (p[0] == HX_BIN_SUB)
            r = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(l, u));
        else
            r = mul(l, u);
        *(HVX_UVector *)(out + v * 32u) = r;
    }
    return 1;
}
#endif
void hx_execute_op(const HxGraph *g, uint32_t index, uint32_t part, uint32_t parts) {
    const HxOp *o = &g->ops[index];
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
#ifdef __hexagon__
    if (o->kind == HX_DEPTHWISE && o->z == HXFG_NONE && o->p[0] == o->p[1]) {
        if (o->p[0] == 3) {
            depthwise3(g, o, part, parts);
            return;
        }
        if (o->p[0] == 5) {
            depthwise5(g, o, part, parts);
            return;
        }
    }
#endif
    const float *x = (const float *)(hx_tensor_base(g, a) + a->offset);
    const float *y = o->y == HXFG_NONE ? NULL : (const float *)(hx_tensor_base(g, &g->tensors[o->y]) + g->tensors[o->y].offset);
    float *out = (float *)(hx_tensor_base(g, b) + b->offset);
    const uint32_t *p = o->p;
    const uint32_t dil_h = hx_dilation_h(o), dil_w = hx_dilation_w(o);
    uint32_t pixels = b->h * b->w, lo = pixels * part / parts, hi = pixels * (part + 1u) / parts;
#ifdef __hexagon__
    if ((o->kind == HX_ACT || o->kind == HX_BINARY) && pixels == 1u && hx_single_pixel_hvx(g, o, x, y, out, part, parts))
        return;
#endif
    /* Row-major reshape of unpadded tensors is a flat copy; split it by element ranges, not output pixels. */
    if (o->kind == HX_RESHAPE && !p[0] && a->stride == a->c && b->stride == b->c &&
        a->dtype == b->dtype && hx_tensor_elements(a) == hx_tensor_elements(b)) {
        const uint32_t n = hx_tensor_elements(b), e0 = (n * part / parts) & ~31u;
        const uint32_t e1 = part + 1u == parts ? n : ((n * (part + 1u) / parts) & ~31u);
        if (e1 > e0 && out != x)
            memcpy(out + e0, x + e0, (e1 - e0) * 4u);
        return;
    }
    /* Channel concat of one pixel: two copies. */
    if (o->kind == HX_CONCAT && p[0] == 2u && pixels == 1u && o->y != HXFG_NONE && y &&
        g->tensors[o->y].h * g->tensors[o->y].w == 1u && a->h * a->w == 1u &&
        a->c + g->tensors[o->y].c == b->c) {
        if (part == 0) {
            memcpy(out, x, a->c * 4u);
            memcpy(out + a->c, y, g->tensors[o->y].c * 4u);
            for (uint32_t c = b->c; c < b->stride; c++)
                out[c] = 0.0f;
        }
        return;
    }
    if (o->kind == HX_MEAN) {
        for (uint32_t c = part * 32u; c < b->stride; c += parts * 32u) {
#ifdef __hexagon__
            HVX_Vector sum = Q6_V_vzero();
            for (uint32_t i = 0; i < a->h * a->w; i++)
                sum = add(sum, *(const HVX_Vector *)(x + i * a->stride + c));
            *(HVX_Vector *)(out + c) = mul(sum, splat(1.0f / (float)(a->h * a->w)));
#else
            for (uint32_t k = c; k < c + 32u; k++) {
                float s = 0;
                for (uint32_t i = 0; i < a->h * a->w; i++)
                    s += x[i * a->stride + k];
                out[k] = s / (float)(a->h * a->w);
            }
#endif
        }
        return;
    }
    if (o->kind == HX_NORM) {
        normalise(g, o, x, out, lo, hi);
        return;
    }
    if (o->kind == HX_RMSNORM) {
        hx_rmsnorm_op(g, o, x, out, lo, hi);
        return;
    }
    if (o->kind == HX_ROPE) {
        hx_rope_op(g, o, x, y, out, lo, hi);
        return;
    }
    if (o->kind == HX_SHORTCONV) {
        hx_shortconv_op(g, o, x, out, part, parts);
        return;
    }
    if (o->kind == HX_LINEAR_Q) {
        hx_linear_q_op(g, o, x, out, part, parts);
        return;
    }
    if (o->kind == HX_KV_WRITE) {
        hx_kv_write_op(g, o, x, y, out, part, parts);
        return;
    }
    if (o->kind == HX_GATHER) {
        hx_gather_op(g, o, x, out, part, parts);
        return;
    }
    if (o->kind == HX_GATHER_ELEMENTS) {
        hx_gather_elements_op(g, o, x, y, out, part, parts);
        return;
    }
    if (o->kind == HX_GATHER_ND) {
        hx_gather_nd_op(g, o, x, y, out, part, parts);
        return;
    }
    if (o->kind == HX_SCATTER_ND) {
        const float *z = (o->z != HXFG_NONE) ? (const float *)(hx_tensor_base(g, &g->tensors[o->z]) + g->tensors[o->z].offset) : NULL;
        hx_scatter_nd_op(g, o, x, y, z, out, part, parts);
        return;
    }
    for (uint32_t i = lo; i < hi; i++) {
        uint32_t oy = i / b->w, ox = i % b->w;
        float *dst = out + i * b->stride;
        if (o->kind == HX_ACT) {
            if (a->dtype == HX_DTYPE_BOOL || a->dtype == HX_DTYPE_INT32 || b->dtype == HX_DTYPE_BOOL || b->dtype == HX_DTYPE_INT32) {
                const int32_t *src = (const int32_t *)x + i * a->stride;
                int32_t *idst = (int32_t *)dst;
                for (uint32_t c = 0; c < b->stride; c++) {
                    if (c >= b->c) { idst[c] = 0; continue; }
                    if (p[0] == HX_ACT_NOT) {
                        idst[c] = (src[c] == 0) ? 1 : 0;
                    } else if (p[0] == HX_ACT_NEG) {
                        idst[c] = -src[c];
                    } else if (p[0] == HX_ACT_ABS) {
                        idst[c] = src[c] < 0 ? -src[c] : src[c];
                    } else if (p[0] == HX_ACT_SIGN) {
                        idst[c] = src[c] > 0 ? 1 : (src[c] < 0 ? -1 : 0);
                    } else {
                        idst[c] = (int32_t)hx_activation(p[0], p[1], p[2], (float)src[c]);
                    }
                }
            } else {
                const float *src = x + i * a->stride;
                for (uint32_t c = 0; c < b->stride; c++)
                    dst[c] = c < b->c ? hx_activation(p[0], p[1], p[2], src[c]) : 0.0f;
            }
            continue;
        }
        if (o->kind == HX_ZEROINSERT) {
            /* Source pixel on the stride grid below/right of the pad; every other output is zero. */
            const uint32_t ry = oy - p[2], rx = ox - p[3];
            const int on_grid = oy >= p[2] && ox >= p[3] && ry % p[0] == 0 && rx % p[1] == 0 &&
                                 ry / p[0] < a->h && rx / p[1] < a->w;
            const float *src = on_grid ? x + ((ry / p[0]) * a->w + rx / p[1]) * a->stride : NULL;
            for (uint32_t c = 0; c < b->stride; c++)
                dst[c] = src && c < b->c ? src[c] : 0.0f;
            continue;
        }
        if (o->kind == HX_SLICE) {
            const float *src = x + ((oy + p[0]) * a->w + ox + p[1]) * a->stride + p[2];
            for (uint32_t c = 0; c < b->stride; c++)
                dst[c] = c < b->c ? src[c] : 0.0f;
            continue;
        }
        if (o->kind == HX_TRANSPOSE) {
            /* Output axis k carries source axis p[k]: the output index on axis k is the source
             * index on axis p[k]. */
            for (uint32_t c = 0; c < b->stride; c++) {
                float v = 0;
                if (c < b->c) {
                    const uint32_t at3[3] = {oy, ox, c};
                    uint32_t s[3];
                    for (unsigned k = 0; k < 3; k++)
                        s[p[k]] = at3[k];
                    v = x[(s[0] * a->w + s[1]) * a->stride + s[2]];
                }
                dst[c] = v;
            }
            continue;
        }
        if (o->kind == HX_RESHAPE && p[0]) {
            /* NCHW order: flat index n = c*H*W + h*W + w, with (H, W) the spatial extent of the
             * HxWxC tensor. Flatten (1) writes n = c of the 1x1xN destination from (h, w, c) of
             * the source; unflatten (2) writes (h, w, c) of the destination from n of the source. */
            for (uint32_t c = 0; c < b->stride; c++) {
                float v = 0;
                if (c < b->c) {
                    if (p[0] == 1) {
                        const uint32_t hw = a->h * a->w, ch = c / hw, pixel = c % hw;
                        v = x[pixel * a->stride + ch];
                    } else {
                        v = x[c * b->h * b->w + oy * b->w + ox];
                    }
                }
                dst[c] = v;
            }
            continue;
        }
        if (o->kind == HX_BINARY) {
            const HxTensor *t = o->y == HXFG_NONE ? NULL : &g->tensors[o->y];
            /* The operand types pick the arithmetic; the output type only picks the store. A float
             * comparison into a bool output compares floats (the float path stores 0/1 ints). */
            const int is_int = (a->dtype == HX_DTYPE_INT32 || a->dtype == HX_DTYPE_BOOL ||
                                (t && (t->dtype == HX_DTYPE_INT32 || t->dtype == HX_DTYPE_BOOL)));
            if (is_int) {
                const int32_t *src = (const int32_t *)x + i * a->stride;
                const int32_t *k = (const int32_t *)(g->constants + p[2]);
                const int32_t *other = NULL;
                if (t)
                    other = (const int32_t *)y + ((t->h == 1 ? 0 : oy) * t->w + (t->w == 1 ? 0 : ox)) * t->stride;
                else if (p[1] == 2)
                    other = k + i * b->stride;
                for (uint32_t c = 0; c < b->stride; c++) {
                    int32_t iv = 0;
                    if (c < b->c) {
                        int32_t u = (p[1] == 1) ? k[c] : (other ? other[t && t->c == 1 ? 0 : c] : 0),
                                l = src[c];
                        if (p[3]) {
                            int32_t s = l;
                            l = u;
                            u = s;
                        }
                        switch (p[0]) {
                        case HX_BIN_ADD: iv = (int32_t)((uint32_t)l + (uint32_t)u); break;
                        case HX_BIN_SUB: iv = (int32_t)((uint32_t)l - (uint32_t)u); break;
                        case HX_BIN_MUL: iv = (int32_t)((uint32_t)l * (uint32_t)u); break;
                        case HX_BIN_DIV: iv = u == 0 ? 0 : l / u; break;
                        case HX_BIN_MIN: iv = l < u ? l : u; break;
                        case HX_BIN_MAX: iv = l > u ? l : u; break;
                        case HX_BIN_POW: {
                            int32_t r = 1;
                            for (int32_t exp = 0; exp < u; exp++) r *= l;
                            iv = r;
                            break;
                        }
                        case HX_BIN_EQUAL: iv = (l == u) ? 1 : 0; break;
                        case HX_BIN_GREATER: iv = (l > u) ? 1 : 0; break;
                        case HX_BIN_LESS: iv = (l < u) ? 1 : 0; break;
                        case HX_BIN_GE: iv = (l >= u) ? 1 : 0; break;
                        case HX_BIN_LE: iv = (l <= u) ? 1 : 0; break;
                        case HX_BIN_AND: iv = (l != 0 && u != 0) ? 1 : 0; break;
                        case HX_BIN_OR: iv = (l != 0 || u != 0) ? 1 : 0; break;
                        default: iv = 0; break;
                        }
                    }
                    if (b->dtype == HX_DTYPE_FLOAT32)
                        dst[c] = (float)iv;
                    else
                        ((int32_t *)dst)[c] = iv;
                }
                continue;
            }
            const float *src = x + i * a->stride, *k = (const float *)(g->constants + p[2]);
            const float *other = NULL;
            if (t)
                other = y + ((t->h == 1 ? 0 : oy) * t->w + (t->w == 1 ? 0 : ox)) * t->stride;
            else if (p[1] == 2)
                other = k + i * b->stride;
            for (uint32_t c = 0; c < b->stride; c++) {
                float v = 0.0f;
                if (c < b->c) {
                    float u = (p[1] == 1) ? k[c] : other[t && t->c == 1 ? 0 : c], l = src[c];
                    if (p[3]) {
                        float s = l;
                        l = u;
                        u = s;
                    }
                    switch (p[0]) {
                    case HX_BIN_ADD: v = l + u; break;
                    case HX_BIN_SUB: v = l - u; break;
                    case HX_BIN_MUL: v = l * u; break;
                    case HX_BIN_DIV: v = u == 0.0f ? 0.0f : l / u; break;
                    case HX_BIN_MIN: v = fminf(l, u); break;
                    case HX_BIN_MAX: v = fmaxf(l, u); break;
                    case HX_BIN_POW: v = powf(l, u); break;
                    case HX_BIN_EQUAL: v = (l == u) ? 1.0f : 0.0f; break;
                    case HX_BIN_GREATER: v = (l > u) ? 1.0f : 0.0f; break;
                    case HX_BIN_LESS: v = (l < u) ? 1.0f : 0.0f; break;
                    case HX_BIN_GE: v = (l >= u) ? 1.0f : 0.0f; break;
                    case HX_BIN_LE: v = (l <= u) ? 1.0f : 0.0f; break;
                    case HX_BIN_AND: v = (l != 0.0f && u != 0.0f) ? 1.0f : 0.0f; break;
                    case HX_BIN_OR: v = (l != 0.0f || u != 0.0f) ? 1.0f : 0.0f; break;
                    default: v = 0.0f; break;
                    }
                }
                if (b->dtype == HX_DTYPE_BOOL || b->dtype == HX_DTYPE_INT32)
                    ((int32_t *)dst)[c] = (int32_t)v;
                else
                    dst[c] = v;
            }
            continue;
        }

        if (o->kind == HX_WHERE) {
            const HxTensor *cond_t = a;
            uint32_t cy = (cond_t->h == 1) ? 0 : oy, cx = (cond_t->w == 1) ? 0 : ox;
            const uint32_t *cond_p = (const uint32_t *)x + (cy * cond_t->w + cx) * cond_t->stride;
            const uint32_t *yp = NULL, *zp = NULL;
            const HxTensor *yt = o->y != HXFG_NONE ? &g->tensors[o->y] : NULL;
            const HxTensor *zt = o->z != HXFG_NONE ? &g->tensors[o->z] : NULL;
            if (p[0] == 0 && yt)
                yp = (const uint32_t *)y + ((yt->h == 1 ? 0 : oy) * yt->w + (yt->w == 1 ? 0 : ox)) * yt->stride;
            else if (p[0] == 2)
                yp = (const uint32_t *)(g->constants + p[1]);
            else if (p[0] == 3)
                yp = ((const uint32_t *)(g->constants + p[1])) + i * b->stride;
            if (p[2] == 0 && zt) {
                const uint32_t *z_arena = (const uint32_t *)(g->arena + zt->offset);
                zp = z_arena + ((zt->h == 1 ? 0 : oy) * zt->w + (zt->w == 1 ? 0 : ox)) * zt->stride;
            } else if (p[2] == 2)
                zp = (const uint32_t *)(g->constants + p[3]);
            else if (p[2] == 3)
                zp = ((const uint32_t *)(g->constants + p[3])) + i * b->stride;

            uint32_t *udst = (uint32_t *)dst;
            for (uint32_t c = 0; c < b->stride; c++) {
                uint32_t v = 0;
                if (c < b->c) {
                    uint32_t cond_raw = cond_p[cond_t->c == 1 ? 0 : c];
                    int is_true;
                    if (cond_t->dtype == HX_DTYPE_FLOAT32) {
                        float fcond;
                        memcpy(&fcond, &cond_raw, sizeof(float));
                        is_true = (fcond != 0.0f);
                    } else {
                        is_true = (cond_raw != 0);
                    }
                    uint32_t xv = (p[0] == 1) ? p[1] : (yp ? yp[yt && yt->c == 1 ? 0 : c] : 0);
                    uint32_t zv = (p[2] == 1) ? p[3] : (zp ? zp[zt && zt->c == 1 ? 0 : c] : 0);
                    v = is_true ? xv : zv;
                }
                udst[c] = v;
            }
            continue;
        }
        if (o->kind == HX_ARGMAX || o->kind == HX_ARGMIN) {
            const int is_min = (o->kind == HX_ARGMIN);
            int32_t best_c = 0;
            if (a->dtype == HX_DTYPE_INT32) {
                const int32_t *src = (const int32_t *)x + i * a->stride;
                int32_t best_val = is_min ? INT32_MAX : INT32_MIN;
                for (uint32_t c = 0; c < a->c; c++) {
                    int better;
                    if (is_min)
                        better = p[0] ? (src[c] <= best_val) : (src[c] < best_val);
                    else
                        better = p[0] ? (src[c] >= best_val) : (src[c] > best_val);
                    if (better) {
                        best_val = src[c];
                        best_c = (int32_t)c;
                    }
                }
            } else {
                const float *src = x + i * a->stride;
                float best_val = is_min ? FLT_MAX : -FLT_MAX;
                for (uint32_t c = 0; c < a->c; c++) {
                    int better;
                    if (is_min)
                        better = p[0] ? (src[c] <= best_val) : (src[c] < best_val);
                    else
                        better = p[0] ? (src[c] >= best_val) : (src[c] > best_val);
                    if (better) {
                        best_val = src[c];
                        best_c = (int32_t)c;
                    }
                }
            }
            if (b->dtype == HX_DTYPE_INT32 || p[1] == 1)
                ((int32_t *)dst)[0] = best_c;
            else
                dst[0] = (float)best_c;
            for (uint32_t c = 1; c < b->stride; c++)
                ((uint32_t *)dst)[c] = 0;
            continue;
        }
        if (o->kind == HX_TOPK) {
            const float *src = x + i * a->stride;
            const uint32_t K = p[0], largest = p[1], sorted = p[2], dtype = p[3];
            float *out_val = dst;
            int32_t *out_idx = (o->z != HXFG_NONE) ? (int32_t *)(g->arena + g->tensors[o->z].offset + i * g->tensors[o->z].stride * 4u) : NULL;
            struct { float val; int32_t idx; } top[128];
            uint32_t nk = 0;
            for (uint32_t c = 0; c < a->c; c++) {
                float v = src[c];
                int32_t idx = (int32_t)c;
                if (nk < K) {
                    top[nk].val = v; top[nk].idx = idx;
                    nk++;
                    if (nk == K && sorted) {
                        for (uint32_t m = 0; m < K; m++)
                            for (uint32_t n = m + 1; n < K; n++)
                                if (largest ? (top[n].val > top[m].val) : (top[n].val < top[m].val)) {
                                    float tv = top[m].val; int32_t ti = top[m].idx;
                                    top[m].val = top[n].val; top[m].idx = top[n].idx;
                                    top[n].val = tv; top[n].idx = ti;
                                }
                    }
                } else {
                    uint32_t worst = 0;
                    for (uint32_t k = 1; k < K; k++)
                        if (largest ? (top[k].val < top[worst].val) : (top[k].val > top[worst].val))
                            worst = k;
                    if (largest ? (v > top[worst].val) : (v < top[worst].val)) {
                        top[worst].val = v; top[worst].idx = idx;
                    }
                }
            }
            if (sorted) {
                for (uint32_t m = 0; m < K; m++)
                    for (uint32_t n = m + 1; n < K; n++)
                        if (largest ? (top[n].val > top[m].val) : (top[n].val < top[m].val)) {
                            float tv = top[m].val; int32_t ti = top[m].idx;
                            top[m].val = top[n].val; top[m].idx = top[n].idx;
                            top[n].val = tv; top[n].idx = ti;
                        }
            }
            for (uint32_t k = 0; k < K; k++) {
                out_val[k] = top[k].val;
                if (out_idx) {
                    if (dtype == 1)
                        out_idx[k] = top[k].idx;
                    else
                        ((float *)out_idx)[k] = (float)top[k].idx;
                }
            }
            for (uint32_t k = K; k < b->stride; k++) {
                out_val[k] = 0.0f;
                if (out_idx) out_idx[k] = 0;
            }
            continue;
        }
        if (o->kind == HX_TILE) {
            uint32_t sy = oy % a->h, sx = ox % a->w;
            const float *src = x + (sy * a->w + sx) * a->stride;
            for (uint32_t c = 0; c < b->stride; c++)
                dst[c] = c < b->c ? src[c % a->c] : 0.0f;
            continue;
        }
        if (o->kind == HX_CAST) {
            for (uint32_t c = 0; c < b->stride; c++) {
                if (c >= b->c) {
                    ((uint32_t *)dst)[c] = 0;
                    continue;
                }
                uint32_t mode = p[0];
                if (mode == 0) {
                    const float *src = x + i * a->stride;
                    ((int32_t *)dst)[c] = (int32_t)src[c];
                } else if (mode == 1) {
                    const int32_t *src = (const int32_t *)x + i * a->stride;
                    dst[c] = (float)src[c];
                } else if (mode == 2) {
                    const int32_t *src = (const int32_t *)x + i * a->stride;
                    dst[c] = (src[c] != 0) ? 1.0f : 0.0f;
                } else if (mode == 3) {
                    const float *src = x + i * a->stride;
                    ((int32_t *)dst)[c] = (src[c] != 0.0f) ? 1 : 0;
                } else if (mode == 4) {
                    const int32_t *src = (const int32_t *)x + i * a->stride;
                    ((int32_t *)dst)[c] = (src[c] != 0) ? 1 : 0;
                } else if (mode == 5) {
                    const int32_t *src = (const int32_t *)x + i * a->stride;
                    ((int32_t *)dst)[c] = (src[c] != 0) ? 1 : 0;
                } else {
                    const uint32_t *src = (const uint32_t *)x + i * a->stride;
                    ((uint32_t *)dst)[c] = src[c];
                }
            }
            continue;
        }
        if (o->kind == HX_REDUCE) {
            /* Reduces the input over the axes in p[1] (bit0 h, bit1 w, bit2 c) into this output
             * pixel; only real output channels are written, padding lanes are zero. */
            const uint32_t h0 = p[1] & 1u ? 0 : oy, h1 = p[1] & 1u ? a->h : oy + 1u;
            const uint32_t w0 = p[1] & 2u ? 0 : ox, w1 = p[1] & 2u ? a->w : ox + 1u;
            for (uint32_t c = 0; c < b->stride; c++) {
                float acc = p[0] == HX_RED_MAX ? -FLT_MAX : (p[0] == HX_RED_MIN ? FLT_MAX : 0.0f);
                uint32_t count = 0;
                if (c < b->c) {
                    const uint32_t c0 = p[1] & 4u ? 0 : c, c1 = p[1] & 4u ? a->c : c + 1u;
                    for (uint32_t sy = h0; sy < h1; sy++)
                        for (uint32_t sx = w0; sx < w1; sx++)
                            for (uint32_t sc = c0; sc < c1; sc++) {
                                const float v = x[(sy * a->w + sx) * a->stride + sc];
                                if (p[0] == HX_RED_MAX)
                                    acc = fmaxf(acc, v);
                                else if (p[0] == HX_RED_MIN)
                                    acc = fminf(acc, v);
                                else
                                    acc += v;
                                count++;
                            }
                }
                dst[c] = c < b->c && p[0] == HX_RED_MEAN ? acc / (float)count
                                                          : (c < b->c ? acc : 0.0f);
            }
            continue;
        }
        if (o->kind == HX_MATMUL) {
            uint32_t Ba = 1, Ma = 0, Ka = 0, Nout = 0, Bb = 1;
            matmul_dims(g, o, &Ba, &Ma, &Ka, &Nout, &Bb);
            const float scale = p[1] ? hx_float(p[1]) : 1.0f;
            const uint32_t transB = p[0], transA = p[3];
            const HxTensor *yt = &g->tensors[o->y];

            uint32_t batch_idx = (b->w > 1) ? (i / b->w) : (Ba > 1 ? (i / Ma) : 0);
            uint32_t m_idx = (b->w > 1) ? (i % b->w) : (Ba > 1 ? (i % Ma) : i);
            uint32_t b_batch = (Bb == 1) ? 0 : batch_idx;

            uint32_t a_row_base = (a->w > 1) ? (batch_idx * a->w) : (Ba > 1 ? (batch_idx * Ma) : 0);

            for (uint32_t n = 0; n < b->stride; n++) {
                if (n >= Nout) {
                    dst[n] = 0.0f;
                    continue;
                }
                float acc = 0.0f;
                if (!transA && !transB) {
                    const float *src_a = x + (a_row_base + m_idx) * a->stride;
                    uint32_t b_row_base = (yt->w > 1) ? (b_batch * yt->w) : (Bb > 1 ? (b_batch * Ka) : 0);
                    for (uint32_t k = 0; k < Ka; k++)
                        acc += src_a[k] * y[(b_row_base + k) * yt->stride + n];
                } else if (!transA && transB) {
                    const float *src_a = x + (a_row_base + m_idx) * a->stride;
                    uint32_t b_row_base = (yt->w > 1) ? (b_batch * yt->w) : (Bb > 1 ? (b_batch * Nout) : 0);
                    const float *src_b = y + (b_row_base + n) * yt->stride;
                    for (uint32_t k = 0; k < Ka; k++)
                        acc += src_a[k] * src_b[k];
                } else if (transA && !transB) {
                    uint32_t b_row_base = (yt->w > 1) ? (b_batch * yt->w) : (Bb > 1 ? (b_batch * Ka) : 0);
                    for (uint32_t k = 0; k < Ka; k++)
                        acc += x[(a_row_base + k) * a->stride + m_idx] * y[(b_row_base + k) * yt->stride + n];
                } else {
                    uint32_t b_row_base = (yt->w > 1) ? (b_batch * yt->w) : (Bb > 1 ? (b_batch * Nout) : 0);
                    const float *src_b = y + (b_row_base + n) * yt->stride;
                    for (uint32_t k = 0; k < Ka; k++)
                        acc += x[(a_row_base + k) * a->stride + m_idx] * src_b[k];
                }
                dst[n] = acc * scale;
            }
            continue;
        }
        if (o->kind == HX_SDPA) {
            uint32_t Bq = 1, Lq = 0, Lk = 0, Dk = 0, Dv = 0;
            sdpa_dims(g, o, &Bq, &Lq, &Lk, &Dk, &Dv);

            const HxTensor *q_t = a, *k_t = &g->tensors[o->y], *v_t = &g->tensors[o->z];
            const float *qv = x;
            const float *kv = (const float *)(hx_tensor_base(g, k_t) + k_t->offset);
            const float *vv = (const float *)(hx_tensor_base(g, v_t) + v_t->offset);
            const HxTensor *mask_t = (p[0] != HXFG_NONE) ? &g->tensors[p[0]] : NULL;
            const float *mask_data = mask_t ? (const float *)(hx_tensor_base(g, mask_t) + mask_t->offset) : NULL;

            uint32_t Bk = (k_t->w > 1) ? k_t->h : ((Bq > 1 && k_t->h % Bq == 0) ? Bq : 1);
            uint32_t Bv = (v_t->w > 1) ? v_t->h : ((Bq > 1 && v_t->h % Bq == 0) ? Bq : 1);

            float scale = p[1] ? hx_float(p[1]) : (1.0f / sqrtf((float)Dk));
            uint32_t causal = p[2];

            uint32_t b_idx = (b->w > 1) ? (i / b->w) : (Bq > 1 ? (i / Lq) : 0);
            uint32_t q = (b->w > 1) ? (i % b->w) : (Bq > 1 ? (i % Lq) : i);
            uint32_t k_batch = (Bk == 1) ? 0 : b_idx;
            uint32_t v_batch = (Bv == 1) ? 0 : b_idx;

            const float *q_row = qv + ((q_t->w > 1 ? b_idx * q_t->w : b_idx * Lq) + q) * q_t->stride;

            const float *m_row = NULL;
            if (mask_data) {
                uint32_t mb = (mask_t->h == 1 ? 0 : b_idx);
                uint32_t mq = (mask_t->w == 1 ? 0 : q);
                m_row = mask_data + (mb * mask_t->w + mq) * mask_t->stride;
            }

#ifdef __hexagon__
            if (Dk % 32u == 0 && Dv % 32u == 0 && Dk <= 512u && Dv <= 512u && Lk <= 2048u &&
                hx_sdpa_row_hvx(q_row, kv + (uint64_t)(k_t->w > 1 ? k_batch * k_t->w : k_batch * Lk) * k_t->stride,
                                k_t->stride,
                                vv + (uint64_t)(v_t->w > 1 ? v_batch * v_t->w : v_batch * Lk) * v_t->stride,
                                v_t->stride, Lk, Dk, Dv, scale, causal ? q : 0xffffffffu, m_row,
                                mask_t && mask_t->c == 1, dst)) {
                for (uint32_t d = Dv; d < b->stride; d++)
                    dst[d] = 0.0f;
                continue;
            }
#endif
            float scores[4096];
            float peak = -FLT_MAX, sum = 0.0f;
            for (uint32_t k = 0; k < Lk; k++) {
                if (causal && k > q) {
                    scores[k] = -FLT_MAX;
                    continue;
                }
                const float *k_row = kv + ((k_t->w > 1 ? k_batch * k_t->w : k_batch * Lk) + k) * k_t->stride;
                float dot = 0.0f;
                for (uint32_t d = 0; d < Dk; d++)
                    dot += q_row[d] * k_row[d];
                dot *= scale;
                if (m_row) dot += (mask_t->c == 1 ? m_row[0] : m_row[k]);
                scores[k] = dot;
                if (dot > peak) peak = dot;
            }
            if (peak == -FLT_MAX) peak = 0.0f;

            for (uint32_t k = 0; k < Lk; k++) {
                if (causal && k > q) {
                    scores[k] = 0.0f;
                } else {
                    scores[k] = expf(scores[k] - peak);
                    sum += scores[k];
                }
            }
            float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (uint32_t k = 0; k < Lk; k++)
                scores[k] *= inv_sum;

            for (uint32_t d = 0; d < Dv; d++) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < Lk; k++) {
                    if (scores[k] > 0.0f) {
                        const float *v_row = vv + ((v_t->w > 1 ? v_batch * v_t->w : v_batch * Lk) + k) * v_t->stride;
                        acc += scores[k] * v_row[d];
                    }
                }
                dst[d] = acc;
            }
            for (uint32_t d = Dv; d < b->stride; d++)
                dst[d] = 0.0f;
            continue;
        }
        if (o->kind == HX_SOFTMAX || o->kind == HX_ATTN_SOFTMAX) {
            uint32_t axis = (o->kind == HX_ATTN_SOFTMAX) ? p[0] : 2;
            if (axis == 2) {
                const float *src = x + i * a->stride;
                const uint32_t causal = p[1];
                const uint32_t M = b->w > 1 ? b->w : (p[2] > 0 ? p[2] : b->h);
                const uint32_t q = M > 0 ? (i % M) : 0;

                const HxTensor *mask_t = (o->y != HXFG_NONE) ? &g->tensors[o->y] : NULL;
                const float *mask_data = mask_t ? (const float *)(hx_tensor_base(g, mask_t) + mask_t->offset) : NULL;
                const float *m_row = NULL;
                if (mask_data) {
                    uint32_t my = mask_t->h == 1 ? 0 : oy;
                    uint32_t mx = mask_t->w == 1 ? 0 : ox;
                    m_row = mask_data + (my * mask_t->w + mx) * mask_t->stride;
                }

                const float temp = p[3] ? hx_float(p[3]) : 1.0f;
                float peak = -FLT_MAX, sum = 0.0f;
                for (uint32_t c = 0; c < b->c; c++) {
                    if (causal && c > q)
                        continue;
                    float v = src[c] * temp;
                    if (m_row)
                        v += (mask_t->c == 1 ? m_row[0] : m_row[c]);
                    if (v > peak)
                        peak = v;
                }
                if (peak == -FLT_MAX)
                    peak = 0.0f;

                for (uint32_t c = 0; c < b->c; c++) {
                    if (causal && c > q) {
                        dst[c] = 0.0f;
                    } else {
                        float v = src[c] * temp;
                        if (m_row)
                            v += (mask_t->c == 1 ? m_row[0] : m_row[c]);
                        dst[c] = expf(v - peak);
                        sum += dst[c];
                    }
                }
                float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (uint32_t c = 0; c < b->c; c++)
                    dst[c] *= inv_sum;
                for (uint32_t c = b->c; c < b->stride; c++)
                    dst[c] = 0.0f;
            } else if (axis == 1) {
                if (ox == 0) {
                    for (uint32_t c = 0; c < b->c; c++) {
                        float peak = -FLT_MAX, sum = 0.0f;
                        for (uint32_t w = 0; w < b->w; w++) {
                            float v = x[(oy * a->w + w) * a->stride + c];
                            if (v > peak) peak = v;
                        }
                        for (uint32_t w = 0; w < b->w; w++) {
                            float v = expf(x[(oy * a->w + w) * a->stride + c] - peak);
                            out[(oy * b->w + w) * b->stride + c] = v;
                            sum += v;
                        }
                        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                        for (uint32_t w = 0; w < b->w; w++)
                            out[(oy * b->w + w) * b->stride + c] *= inv;
                    }
                    for (uint32_t w = 0; w < b->w; w++)
                        for (uint32_t c = b->c; c < b->stride; c++)
                            out[(oy * b->w + w) * b->stride + c] = 0.0f;
                }
            } else {
                if (oy == 0) {
                    for (uint32_t c = 0; c < b->c; c++) {
                        float peak = -FLT_MAX, sum = 0.0f;
                        for (uint32_t h = 0; h < b->h; h++) {
                            float v = x[(h * a->w + ox) * a->stride + c];
                            if (v > peak) peak = v;
                        }
                        for (uint32_t h = 0; h < b->h; h++) {
                            float v = expf(x[(h * a->w + ox) * a->stride + c] - peak);
                            out[(h * b->w + ox) * b->stride + c] = v;
                            sum += v;
                        }
                        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                        for (uint32_t h = 0; h < b->h; h++)
                            out[(h * b->w + ox) * b->stride + c] *= inv;
                    }
                    for (uint32_t h = 0; h < b->h; h++)
                        for (uint32_t c = b->c; c < b->stride; c++)
                            out[(h * b->w + ox) * b->stride + c] = 0.0f;
                }
            }
            continue;
        }
        if (o->kind == HX_AVGPOOL) {
            for (uint32_t c = 0; c < b->stride; c++) {
                float sum = 0;
                uint32_t taps = 0;
                if (c < b->c)
                    for (uint32_t ky = 0; ky < p[0]; ky++) {
                        int32_t iy = (int32_t)(oy * p[2] + ky * dil_h) - (int32_t)p[4];
                        if (iy < 0 || iy >= (int32_t)a->h)
                            continue;
                        for (uint32_t kx = 0; kx < p[1]; kx++) {
                            int32_t ix = (int32_t)(ox * p[3] + kx * dil_w) - (int32_t)p[5];
                            if (ix < 0 || ix >= (int32_t)a->w)
                                continue;
                            sum += x[((uint32_t)iy * a->w + (uint32_t)ix) * a->stride + c];
                            taps++;
                        }
                    }
                if (p[6])
                    taps = p[0] * p[1];
                dst[c] = c < b->c && taps ? sum / (float)taps : 0.0f;
            }
            continue;
        }
        if (o->kind == HX_CONV || o->kind == HX_DEPTHWISE || o->kind == HX_MAXPOOL) {
            const float *w = (const float *)(g->constants + p[6]),
                        *bias = (const float *)(g->constants + p[7]);
            for (uint32_t c = 0; c < b->stride; c += 32u) {
#ifdef __hexagon__
                HVX_Vector acc =
                    o->kind == HX_MAXPOOL ? splat(-FLT_MAX) : *(const HVX_Vector *)(bias + c);
                if (o->kind == HX_DEPTHWISE)
                    acc = Q6_Vqf32_vadd_VsfVsf(acc, Q6_V_vzero());
#else
                float acc[32];
                for (unsigned n = 0; n < 32; n++)
                    acc[n] = o->kind == HX_MAXPOOL ? -FLT_MAX : bias[c + n];
#endif
                for (uint32_t ky = 0; ky < p[0]; ky++) {
                    int32_t iy = (int32_t)(oy * p[2] + ky * dil_h) - (int32_t)p[4];
                    if (iy < 0 || iy >= (int32_t)a->h)
                        continue;
                    for (uint32_t kx = 0; kx < p[1]; kx++) {
                        int32_t ix = (int32_t)(ox * p[3] + kx * dil_w) - (int32_t)p[5];
                        if (ix < 0 || ix >= (int32_t)a->w)
                            continue;
                        const float *src = x + ((uint32_t)iy * a->w + (uint32_t)ix) * a->stride;
                        if (o->kind == HX_CONV) {
                            const float *ww = w + ((ky * p[1] + kx) * a->c) * b->stride + c;
                            for (uint32_t k = 0; k < a->c; k++) {
#ifdef __hexagon__
                                acc = add(acc, mul(splat(src[k]),
                                                   *(const HVX_Vector *)(ww + k * b->stride)));
#else
                                for (unsigned n = 0; n < 32; n++)
                                    acc[n] += src[k] * ww[k * b->stride + n];
#endif
                            }
                        } else {
#ifdef __hexagon__
                            HVX_Vector v = *(const HVX_Vector *)(src + c);
                            if (o->kind == HX_MAXPOOL)
                                acc = Q6_Vsf_vmax_VsfVsf(acc, v);
                            else
                                acc = Q6_Vqf32_vadd_Vqf32Vqf32(
                                    acc,
                                    Q6_Vqf32_vmpy_VsfVsf(
                                        v, *(const HVX_Vector *)(w + (ky * p[1] + kx) * b->stride +
                                                                 c)));
#else
                            for (unsigned n = 0; n < 32; n++) {
                                if (o->kind == HX_MAXPOOL)
                                    acc[n] = fmaxf(acc[n], src[c + n]);
                                else
                                    acc[n] += src[c + n] * w[(ky * p[1] + kx) * b->stride + c + n];
                            }
#endif
                        }
                    }
                }
#ifdef __hexagon__
                if (o->kind == HX_DEPTHWISE)
                    acc = Q6_Vsf_equals_Vqf32(acc);
                *(HVX_Vector *)(dst + c) = act(acc, p[8]);
#else
                for (unsigned n = 0; n < 32; n++)
                    dst[c + n] = activate(acc[n], p[8]);
#endif
            }
        } else if (o->kind == HX_ADD || o->kind == HX_PRELU) {
            if (o->kind == HX_ADD && (a->dtype == HX_DTYPE_INT32 || b->dtype == HX_DTYPE_INT32)) {
                const int32_t *src = (const int32_t *)x + i * a->stride;
                const int32_t *other = (const int32_t *)y + i * b->stride;
                int32_t *idst = (int32_t *)dst;
                for (uint32_t c = 0; c < b->stride; c++)
                    idst[c] = (c < b->c) ? (int32_t)((uint32_t)src[c] + (uint32_t)other[c]) : 0;
            } else {
                const float *src = x + i * a->stride;
                const float *other =
                    o->kind == HX_ADD ? y + i * b->stride : (const float *)(g->constants + p[0]);
                for (uint32_t c = 0; c < b->stride; c += 32) {
#ifdef __hexagon__
                    HVX_Vector v = *(const HVX_Vector *)(src + c),
                               v2 = *(const HVX_Vector *)(other + c);
                    HVX_Vector vout =
                        o->kind == HX_ADD
                            ? act(add(v, v2), p[0])
                            : Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(Q6_V_vzero(), v), mul(v, v2), v);
                    *(HVX_Vector *)(dst + c) = vout;
#else
                    for (unsigned n = 0; n < 32; n++) {
                        float v = src[c + n];
                        dst[c + n] = o->kind == HX_ADD ? activate(v + other[c + n], p[0])
                                                       : (v < 0 ? v * other[c + n] : v);
                    }
#endif
                }
            }
        } else if (o->kind == HX_RESIZE && p[2] == 1) {
            /* Nearest: coordinate transformation mode in p[3], nearest mode in p[4] */
            const uint32_t coord_mode = p[3], near_mode = p[4];
            uint32_t sy, sx;
            if (coord_mode == 0 && near_mode == 0) {
                /* Asymmetric floor (backward compatible fast path) */
                sy = (uint32_t)((uint64_t)oy * a->h / b->h);
                sx = (uint32_t)((uint64_t)ox * a->w / b->w);
            } else {
                float fy = 0.0f, fx = 0.0f;
                if (coord_mode == 0) { /* asymmetric */
                    fy = (float)oy * (float)a->h / (float)b->h;
                    fx = (float)ox * (float)a->w / (float)b->w;
                } else if (coord_mode == 1) { /* align_corners */
                    fy = b->h > 1 ? (float)oy * (float)(a->h - 1) / (float)(b->h - 1) : 0.0f;
                    fx = b->w > 1 ? (float)ox * (float)(a->w - 1) / (float)(b->w - 1) : 0.0f;
                } else if (coord_mode == 3) { /* pytorch_half_pixel */
                    fy = b->h > 1 ? ((float)oy + 0.5f) * (float)a->h / (float)b->h - 0.5f : 0.0f;
                    fx = b->w > 1 ? ((float)ox + 0.5f) * (float)a->w / (float)b->w - 0.5f : 0.0f;
                } else { /* half_pixel (2) */
                    fy = ((float)oy + 0.5f) * (float)a->h / (float)b->h - 0.5f;
                    fx = ((float)ox + 0.5f) * (float)a->w / (float)b->w - 0.5f;
                }
                int32_t iy, ix;
                if (near_mode == 1) { /* round_prefer_floor */
                    iy = (int32_t)ceilf(fy - 0.5f);
                    ix = (int32_t)ceilf(fx - 0.5f);
                } else if (near_mode == 2) { /* round_prefer_ceil */
                    iy = (int32_t)floorf(fy + 0.5f);
                    ix = (int32_t)floorf(fx + 0.5f);
                } else if (near_mode == 3) { /* ceil */
                    iy = (int32_t)ceilf(fy);
                    ix = (int32_t)ceilf(fx);
                } else { /* floor */
                    iy = (int32_t)floorf(fy);
                    ix = (int32_t)floorf(fx);
                }
                sy = (uint32_t)(iy < 0 ? 0 : (iy >= (int32_t)a->h ? (int32_t)a->h - 1 : iy));
                sx = (uint32_t)(ix < 0 ? 0 : (ix >= (int32_t)a->w ? (int32_t)a->w - 1 : ix));
            }
            sy = sy < a->h ? sy : a->h - 1u;
            sx = sx < a->w ? sx : a->w - 1u;
            const float *src = x + (sy * a->w + sx) * a->stride;
            for (uint32_t c = 0; c < b->stride; c++)
                dst[c] = src[c];
        } else if (o->kind == HX_RESIZE && p[5] == 1) {
            /* Bicubic Catmull-Rom A = -0.75f */
            const uint32_t coord_mode = p[3];
            float fy = 0.0f, fx = 0.0f;
            if (coord_mode == 1 && b->h > 1 && b->w > 1) {
                fy = (float)oy * (float)(a->h - 1) / (float)(b->h - 1);
                fx = (float)ox * (float)(a->w - 1) / (float)(b->w - 1);
            } else if (coord_mode == 0) { /* asymmetric */
                fy = (float)oy * (float)a->h / (float)b->h;
                fx = (float)ox * (float)a->w / (float)b->w;
            } else if (coord_mode == 3) { /* pytorch_half_pixel */
                fy = b->h > 1 ? ((float)oy + 0.5f) * (float)a->h / (float)b->h - 0.5f : 0.0f;
                fx = b->w > 1 ? ((float)ox + 0.5f) * (float)a->w / (float)b->w - 0.5f : 0.0f;
            } else { /* half_pixel (2) */
                fy = ((float)oy + 0.5f) * (float)a->h / (float)b->h - 0.5f;
                fx = ((float)ox + 0.5f) * (float)a->w / (float)b->w - 0.5f;
            }
            int32_t y0 = (int32_t)floorf(fy), x0 = (int32_t)floorf(fx);
            float wy[4], wx[4];
            hx_cubic_weights(fy - (float)y0, wy);
            hx_cubic_weights(fx - (float)x0, wx);
            for (uint32_t c = 0; c < b->stride; c++) {
                float acc = 0.0f;
                if (c < b->c) {
                    for (int j = 0; j < 4; j++) {
                        int32_t sy = y0 + j - 1;
                        sy = sy < 0 ? 0 : (sy >= (int32_t)a->h ? (int32_t)a->h - 1 : sy);
                        for (int k = 0; k < 4; k++) {
                            int32_t sx = x0 + k - 1;
                            sx = sx < 0 ? 0 : (sx >= (int32_t)a->w ? (int32_t)a->w - 1 : sx);
                            acc += wy[j] * wx[k] * x[(sy * a->w + sx) * a->stride + c];
                        }
                    }
                }
                dst[c] = acc;
            }
        } else if (o->kind == HX_RESIZE) {
            const uint32_t coord_mode = p[3];
            float fy, fx;
            if (coord_mode == 1 || (coord_mode == 0 && p[0])) { /* align_corners */
                fy = b->h > 1 ? (float)oy * (float)(a->h - 1) / (float)(b->h - 1) : 0.0f;
                fx = b->w > 1 ? (float)ox * (float)(a->w - 1) / (float)(b->w - 1) : 0.0f;
            } else if (coord_mode == 3) { /* pytorch_half_pixel */
                fy = b->h > 1 ? ((float)oy + 0.5f) * (float)a->h / (float)b->h - 0.5f : 0.0f;
                fx = b->w > 1 ? ((float)ox + 0.5f) * (float)a->w / (float)b->w - 0.5f : 0.0f;
            } else if (coord_mode == 2 || (coord_mode == 0 && p[1])) { /* half_pixel */
                fy = ((float)oy + 0.5f) * (float)a->h / (float)b->h - 0.5f;
                fx = ((float)ox + 0.5f) * (float)a->w / (float)b->w - 0.5f;
            } else { /* asymmetric */
                fy = (float)oy * (float)a->h / (float)b->h;
                fx = (float)ox * (float)a->w / (float)b->w;
            }
            fy = fmaxf(0.0f, fy);
            fx = fmaxf(0.0f, fx);
            uint32_t y0 = (uint32_t)fy, x0 = (uint32_t)fx, y1 = y0 + 1 < a->h ? y0 + 1 : y0,
                     x1 = x0 + 1 < a->w ? x0 + 1 : x0;
            float dy = fy - (float)y0, dx = fx - (float)x0;
            for (uint32_t c = 0; c < b->stride; c += 32u) {
#ifdef __hexagon__
                HVX_Vector v0 = *(const HVX_Vector *)(x + (y0 * a->w + x0) * a->stride + c),
                           v1 = *(const HVX_Vector *)(x + (y0 * a->w + x1) * a->stride + c);
                HVX_Vector v2 = *(const HVX_Vector *)(x + (y1 * a->w + x0) * a->stride + c),
                           v3 = *(const HVX_Vector *)(x + (y1 * a->w + x1) * a->stride + c);
                HVX_Vector top =
                    add(v0, mul(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v1, v0)), splat(dx)));
                HVX_Vector bot =
                    add(v2, mul(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v2)), splat(dx)));
                *(HVX_Vector *)(dst + c) =
                    add(top, mul(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(bot, top)), splat(dy)));
#else
                for (uint32_t k = c; k < c + 32; k++) {
                    float v0 = x[(y0 * a->w + x0) * a->stride + k],
                          v1 = x[(y0 * a->w + x1) * a->stride + k];
                    float v2 = x[(y1 * a->w + x0) * a->stride + k],
                          v3 = x[(y1 * a->w + x1) * a->stride + k];
                    float top = v0 + (v1 - v0) * dx, bot = v2 + (v3 - v2) * dx;
                    dst[k] = top + (bot - top) * dy;
                }
#endif
            }
        } else if (o->kind == HX_PAD) {
            memset(dst, 0, b->stride * 4u);
            if (oy >= p[0] && oy - p[0] < a->h && ox >= p[1] && ox - p[1] < a->w)
                memcpy(dst + p[2], x + ((oy - p[0]) * a->w + ox - p[1]) * a->stride, a->c * 4u);
        } else {
            for (uint32_t c = 0; c < b->stride; c++) {
                float v = 0;
                if (c < b->c) {
                    if (o->kind == HX_RESHAPE)
                        v = element(x, a, i * b->c + c);
                    else if (o->kind == HX_LOGISTIC)
                        v = 1.0f / (1.0f + expf(-x[i * a->stride + c]));
                    else if (o->kind == HX_PAD) {
                        if (oy >= p[0] && oy - p[0] < a->h && ox >= p[1] && ox - p[1] < a->w &&
                            c >= p[2] && c - p[2] < a->c)
                            v = x[((oy - p[0]) * a->w + ox - p[1]) * a->stride + c - p[2]];
                    } else if (o->kind == HX_CONCAT) {
                        const HxTensor *t = a;
                        const float *src = x;
                        uint32_t yy = oy, xx = ox, cc = c;
                        uint32_t coord = p[0] == 0 ? oy : (p[0] == 1 ? ox : c),
                                 size = p[0] == 0 ? a->h : (p[0] == 1 ? a->w : a->c);
                        if (coord >= size) {
                            t = &g->tensors[o->y];
                            src = y;
                            if (p[0] == 0)
                                yy -= size;
                            else if (p[0] == 1)
                                xx -= size;
                            else
                                cc -= size;
                        }
                        v = src[(yy * t->w + xx) * t->stride + cc];
                    }
                }
                dst[c] = v;
            }
        }
    }
}
void hx_graph_set_state(HxGraph *g, void *state, uint32_t state_bytes) {
    if (g) {
        g->state = (uint8_t *)state;
        g->state_bytes = state_bytes;
    }
}

