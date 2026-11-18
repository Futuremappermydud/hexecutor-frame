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
        used = 4;
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
    case HX_ATTN_SOFTMAX:
        used = 3;
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
    case HX_GATHER:
        used = 4;
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
        return !p[3] && !p[4] && p[0] <= 1 && p[1] <= 1 && !(p[0] && p[1]);
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
               ((p[0] == HX_ACT_LEAKY && !p[2]) || (p[0] == HX_ACT_HARDSIGMOID) ||
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
        return same(a, b) && constant(g, p[1], b->stride * 4u) &&
               (p[2] == HXFG_NONE || constant(g, p[2], b->stride * 4u));
    case HX_ROPE: {
        if (!same(a, b) || !p[0] || (p[0] & 1u) || !p[1] || p[0] * p[1] > a->c)
            return 0;
        if (o->y != HXFG_NONE) {
            const HxTensor *yt = &g->tensors[o->y];
            if (yt->h != 1 || yt->w != 1 || yt->c < 1)
                return 0;
        }
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
            a->c != k || b->c != n || fmt > 1)
            return 0;
        const uint64_t block_bytes = fmt == 0 ? 18u : 34u;
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
    case HX_GATHER: {
        if (p[0] > 1 || p[3] > 1 || !p[2])
            return 0;
        if ((uint64_t)b->h * b->w != hx_tensor_elements(a))
            return 0;
        if (p[0] == 1)
            return constant(g, p[1], (uint64_t)p[2] * b->stride * 4u);
        const HxTensor *tbl = &g->tensors[o->y];
        return tbl->c >= b->c && (uint64_t)tbl->h * tbl->w >= p[2];
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
        !h->arena_bytes || h->arena_bytes > 0x1000000u || h->arena_bytes > arena_bytes ||
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
            (t->stride != ((t->c + 31u) & ~31u) && !(t->c == 3 && t->stride == 3)) ||
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
                     (o->kind == HX_GATHER && !o->p[0]) || (o->kind == HX_WHERE && !o->p[0]);
        int has_y = binary || o->kind == HX_SHORTCONV || (o->kind == HX_ROPE && o->y != HXFG_NONE) ||
                    (o->kind == HX_KV_WRITE && o->y != HXFG_NONE) ||
                    (o->kind == HX_ATTN_SOFTMAX && o->y != HXFG_NONE);
        const int req_z = o->kind == HX_SDPA || (o->kind == HX_WHERE && !o->p[2]);
        int has_z = req_z || (o->kind == HX_SHORTCONV && o->z != HXFG_NONE);
        const int produces_z = (o->kind == HX_TOPK && o->z != HXFG_NONE);
        const int has_mask = o->kind == HX_SDPA && o->p[0] != HXFG_NONE;
        /* z is NONE, or a dilation (dh << 8) | dw on conv/depthwise only (1..8 each, not both 1). */
        const int conv_like = o->kind == HX_CONV || o->kind == HX_DEPTHWISE;
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
                        if (!out_pers)
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
    const double eps = (double)hx_float(p[0]);
    const float *gamma = (const float *)(g->constants + p[1]);
    const float *beta = p[2] != HXFG_NONE ? (const float *)(g->constants + p[2]) : NULL;
    for (uint32_t i = lo; i < hi; i++) {
        const float *src = x + i * a->stride;
        float *dst = out + i * b->stride;
        double sum_sq = 0.0;
        for (uint32_t k = 0; k < c; k++)
            sum_sq += (double)src[k] * (double)src[k];
        const double inv = 1.0 / sqrt(sum_sq / (double)c + eps);
        for (uint32_t k = 0; k < c; k++)
            dst[k] = (float)((double)src[k] * inv * (double)gamma[k] + (beta ? (double)beta[k] : 0.0));
        for (uint32_t k = c; k < b->stride; k++)
            dst[k] = 0.0f;
    }
}
static void hx_rope_op(const HxGraph *g, const HxOp *o, const float *x, const float *y,
                       float *out, uint32_t lo, uint32_t hi) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p;
    const uint32_t head_dim = p[0], num_heads = p[1], half = head_dim / 2u;
    const uint32_t mode = p[3];
    for (uint32_t i = lo; i < hi; i++) {
        const float *src = x + i * a->stride;
        float *dst = out + i * b->stride;
        uint32_t pos = p[2];
        if (o->y != HXFG_NONE && y) {
            pos = (uint32_t)y[0];
        }
        for (uint32_t h = 0; h < num_heads; h++) {
            const float *h_in = src + h * head_dim;
            float *h_out = dst + h * head_dim;
            for (uint32_t j = 0; j < half; j++) {
                double theta = pow(10000.0, -2.0 * (double)j / (double)head_dim);
                double angle = (double)pos * theta;
                float cos_v = (float)cos(angle);
                float sin_v = (float)sin(angle);
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
static void hx_linear_q_op(const HxGraph *g, const HxOp *o, const float *x, float *out,
                           uint32_t part, uint32_t parts) {
    const HxTensor *b = &g->tensors[o->out];
    const uint32_t fmt = o->p[0], k = o->p[3], n = o->p[4];
    const float *bias = o->p[2] != HXFG_NONE ? (const float *)(g->constants + o->p[2]) : NULL;
    const uint32_t blocks = k / 32u;
    const uint32_t r0 = n * part / parts, r1 = n * (part + 1u) / parts;

    for (uint32_t r = r0; r < r1; r++) {
        double sum = bias ? (double)bias[r] : 0.0;
        if (fmt == 0) {
            const uint8_t *w_row = (const uint8_t *)(g->constants + o->p[1]) + (uint64_t)r * blocks * 18u;
            for (uint32_t b_idx = 0; b_idx < blocks; b_idx++) {
                const uint8_t *blk = w_row + b_idx * 18u;
                const uint16_t d16 = (uint16_t)blk[16] | ((uint16_t)blk[17] << 8);
                const float scale = hx_half_to_float(d16);
                const float *x_blk = x + b_idx * 32u;
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
                const float *x_blk = x + b_idx * 32u;
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
        out[r] = val;
    }
    if (part == 0) {
        for (uint32_t r = n; r < b->stride; r++)
            out[r] = 0.0f;
    }
}
static void hx_kv_write_op(const HxGraph *g, const HxOp *o, const float *x, const float *y,
                           float *out, uint32_t part, uint32_t parts) {
    const HxTensor *xt = &g->tensors[o->x];
    const HxTensor *outt = &g->tensors[o->out];
    uint32_t pos = o->p[0];
    if (o->y != HXFG_NONE && y) {
        pos = (uint32_t)(*y);
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
            const int is_int = (a->dtype == HX_DTYPE_INT32 || a->dtype == HX_DTYPE_BOOL ||
                                (t && (t->dtype == HX_DTYPE_INT32 || t->dtype == HX_DTYPE_BOOL)) ||
                                b->dtype == HX_DTYPE_INT32 || b->dtype == HX_DTYPE_BOOL);
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
        if (o->kind == HX_GATHER) {
            uint32_t a_pixel = i / a->c, a_c = i % a->c;
            int32_t idx = (p[3] == 0)
                              ? ((const int32_t *)x)[a_pixel * a->stride + a_c]
                              : (int32_t)lrintf(x[a_pixel * a->stride + a_c]);
            const uint32_t vocab = p[2];
            if (idx < 0)
                idx += (int32_t)vocab;
            idx = idx < 0 ? 0 : (idx >= (int32_t)vocab ? (int32_t)vocab - 1 : idx);
            const float *row = (p[0] == 1)
                                   ? ((const float *)(g->constants + p[1])) + (uint32_t)idx * b->stride
                                   : ((const float *)(g->arena + g->tensors[o->y].offset)) +
                                         (uint32_t)idx * g->tensors[o->y].stride;
            for (uint32_t c = 0; c < b->stride; c++)
                dst[c] = c < b->c ? row[c] : 0.0f;
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
            const float *kv = (const float *)(g->arena + k_t->offset);
            const float *vv = (const float *)(g->arena + v_t->offset);
            const HxTensor *mask_t = (p[0] != HXFG_NONE) ? &g->tensors[p[0]] : NULL;
            const float *mask_data = mask_t ? (const float *)(g->arena + mask_t->offset) : NULL;

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
                const float *mask_data = mask_t ? (const float *)(g->arena + mask_t->offset) : NULL;
                const float *m_row = NULL;
                if (mask_data) {
                    uint32_t my = mask_t->h == 1 ? 0 : oy;
                    uint32_t mx = mask_t->w == 1 ? 0 : ox;
                    m_row = mask_data + (my * mask_t->w + mx) * mask_t->stride;
                }

                float peak = -FLT_MAX, sum = 0.0f;
                for (uint32_t c = 0; c < b->c; c++) {
                    if (causal && c > q)
                        continue;
                    float v = src[c];
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
                        float v = src[c];
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
                        int32_t iy = (int32_t)(oy * p[2] + ky) - (int32_t)p[4];
                        if (iy < 0 || iy >= (int32_t)a->h)
                            continue;
                        for (uint32_t kx = 0; kx < p[1]; kx++) {
                            int32_t ix = (int32_t)(ox * p[3] + kx) - (int32_t)p[5];
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
            float sy = p[0] && b->h > 1 ? (float)(a->h - 1) / (float)(b->h - 1)
                                        : (float)a->h / (float)b->h;
            float sx = p[0] && b->w > 1 ? (float)(a->w - 1) / (float)(b->w - 1)
                                        : (float)a->w / (float)b->w;
            float fy = fmaxf(0, ((float)oy + (p[1] ? 0.5f : 0)) * sy - (p[1] ? 0.5f : 0));
            float fx = fmaxf(0, ((float)ox + (p[1] ? 0.5f : 0)) * sx - (p[1] ? 0.5f : 0));
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

