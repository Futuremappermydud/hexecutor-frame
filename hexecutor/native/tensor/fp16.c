#include "fp16.h"
#include "hmx.h"
#include "fp16_hvx.h"
#include "fp16_pack.h"
#include <string.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
typedef struct {
    uint32_t m, n, k, a_kstep, a_mstep, c_plane;
    const void *bias;
} HxHmxStrip;
extern void hx_hmx_init(void *scale);
extern void hx_hmx_strip(const void *a, const void *b, void *c, const HxHmxStrip *strip);
extern void hx_hmx_finish(void *c);

static inline float hx_float(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } pun;
    pun.u = bits;
    return pun.f;
}
static inline uint32_t physical_width(const HxTensor *t) { return (t->w + 1u) & ~1u; }
static inline uint32_t tiles(const HxTensor *t) { return (t->h * physical_width(t) + 31u) / 32u; }
static inline uint32_t plane_bytes(const HxTensor *t) { return tiles(t) * 2048u; }
static inline uint8_t *tensor_data(const HxGraph *g, uint32_t tensor) {
    if (g->tensors[tensor].reserved[0] & HX_TENSOR_PERSISTENT)
        return (g->state ? g->state : g->arena) + g->tensors[tensor].offset;
    return g->arena + g->fp16->offset[tensor];
}
/* Halfword index of a physical pixel and channel. */
static inline uint32_t at(const HxTensor *t, uint32_t pixel, uint32_t c) {
    return (c >> 5) * (plane_bytes(t) / 2u) + (pixel >> 1) * 64u + (c & 31u) * 2u + (pixel & 1u);
}
static inline HVX_Vector load(const uint16_t *x, const HxTensor *t, uint32_t pixel, uint32_t c) {
    return *(const HVX_Vector *)(x + at(t, pixel & ~1u, c));
}
static HVX_Vector rows(const uint16_t *x, const HxTensor *t, int32_t p0, int32_t p1, uint32_t c) {
    if (p0 >= 0 && !(p0 & 1) && p1 == p0 + 1)
        return load(x, t, (uint32_t)p0, c);
    HVX_Vector v0 = p0 < 0 ? Q6_V_vzero() : load(x, t, (uint32_t)p0, c);
    HVX_Vector v1 = p1 < 0 ? Q6_V_vzero() : load(x, t, (uint32_t)p1, c);
    if (p0 >= 0 && (p0 & 1))
        v0 = Q6_Vuw_vlsr_VuwR(v0, 16);
    if (p1 >= 0 && (p1 & 1))
        v1 = Q6_Vuw_vlsr_VuwR(v1, 16);
    return Q6_Vh_vshuffe_VhVh(v1, v0);
}
static HVX_VectorPair floats(HVX_Vector h) {
    return Q6_Wqf32_vmpy_VhfVhf(h, Q6_Vh_vsplat_R(0x3c00));
}
static inline HVX_Vector half_key(HVX_Vector v) {
    return Q6_V_vxor_VV(v, Q6_V_vand_VV(Q6_Vh_vasr_VhR(v, 15), Q6_Vh_vsplat_R(0x7fff)));
}
static inline HVX_Vector max_half(HVX_Vector a, HVX_Vector b) {
    return half_key(Q6_Vh_vmax_VhVh(half_key(a), half_key(b)));
}
static inline HVX_Vector activate_half(HVX_Vector v, uint32_t a) {
    if (a == 1 || a == 3)
        v = Q6_Vh_vmax_VhVh(v, Q6_V_vzero());
    if (a == 2)
        v = half_key(
            Q6_Vh_vmax_VhVh(half_key(Q6_Vh_vsplat_R(0xbc00)),
                            Q6_Vh_vmin_VhVh(half_key(Q6_Vh_vsplat_R(0x3c00)), half_key(v))));
    if (a == 3)
        v = Q6_Vh_vmin_VhVh(Q6_Vh_vsplat_R(0x4600), v);
    return v;
}
/* Internal arithmetic follows native QF32->HF conversion. External import
 * uses exact IEEE round-to-nearest-even, including subnormals. */
static inline HVX_Vector pack(HVX_VectorPair q, uint32_t activation) {
    return activate_half(Q6_Vhf_equals_Wqf32(q), activation);
}
static inline HVX_VectorPair plus(HVX_VectorPair a, HVX_VectorPair b) {
    return Q6_W_vcombine_VV(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(a), Q6_V_hi_W(b)),
                            Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(a), Q6_V_lo_W(b)));
}
static float scalar(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
    union {
        uint32_t u;
        float f;
    } v;
    if (!e) {
        if (!m) {
            v.u = sign;
            return v.f;
        }
        return (h & 0x8000u ? -1.0f : 1.0f) * (float)m * 0x1p-24f;
    }
    v.u = sign | (e == 31 ? 0x7f800000u : (e + 112u) << 23) | (m << 13);
    return v.f;
}
static uint16_t half(float f) {
    float values[32] __attribute__((aligned(128)));
    uint32_t result[32] __attribute__((aligned(128)));
    *(HVX_Vector *)values = splat(f);
    *(HVX_Vector *)result = half_words(*(HVX_Vector *)values);
    return (uint16_t)result[0];
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
#include "fp16_depthwise_hvx.h"

static void run(const HxFp16Job *job, HxFp16Parallel parallel) {
    if (parallel)
        parallel(job);
    else
        hx_fp16_job(job, 0, 1);
}

/* ------------------------------ planning ------------------------------ */
uint32_t hx_fp16_table_words(const HxGraph *g) { return g->header->tensors + 2u * g->header->ops; }
static int is_input(const HxGraph *g, uint32_t tensor) {
    for (uint32_t j = 0; j < g->header->inputs; j++)
        if (g->io[j] == tensor)
            return 1;
    return 0;
}
static int is_output(const HxGraph *g, uint32_t tensor) {
    for (uint32_t j = 0; j < g->header->outputs; j++)
        if (g->io[g->header->inputs + j] == tensor)
            return 1;
    return 0;
}
/* A pointwise convolution reads the blocked activations in place. Every other dense
 * convolution (any kernel, stride or padding, and the compact RGB input) packs an im2col
 * operand into the workspace first. */
static int pointwise(const HxOp *o, const HxTensor *a, const HxTensor *b) {
    return o->p[0] == 1 && o->p[1] == 1 && o->p[2] == 1 && o->p[3] == 1 && !o->p[4] && !o->p[5] &&
           a->h == b->h && a->w == b->w;
}
static int packed_conv(const HxOp *o, const HxTensor *a, const HxTensor *b) {
    return a->stride == 3 || !pointwise(o, a, b);
}
/* Depthwise shapes with a hand-scheduled HVX kernel; every other shape (including any dilated
 * one) runs the scalar-lane path. */
static int dw_fast(const HxOp *o) {
    return o->z == HXFG_NONE && o->p[0] == o->p[1] && (o->p[0] == 3 || o->p[0] == 5) &&
           o->p[2] == o->p[3] && (o->p[2] == 1 || o->p[2] == 2);
}
static float scalar_activation(float x, uint32_t a) {
    if (a == 1 || a == 3)
        x = fmaxf(0.0f, x);
    if (a == 2)
        x = fmaxf(-1.0f, fminf(1.0f, x));
    if (a == 3)
        x = fminf(6.0f, x);
    return x;
}
static int supported(const HxGraph *g) {
    uint32_t compact_consumer = HXFG_NONE;
    for (uint32_t i = 0; i < g->header->tensors; i++)
        if (g->tensors[i].stride == 3 && (!is_input(g, i) || is_output(g, i)))
            return 0;
    for (uint32_t i = 0; i < g->header->ops; i++) {
        const HxOp *o = &g->ops[i];
        if (o->kind < 1 || o->kind > 38)
            return 0;
        const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
        if (a->stride == 3 && o->kind != HX_CONV)
            return 0;
        if (o->kind == HX_CONV) {
            const uint32_t kb = o->p[10];
            if (!kb)
                return 0;
            if (a->stride == 3) {
                if (o->z != HXFG_NONE || o->p[0] != o->p[1] || (o->p[0] != 3 && o->p[0] != 5) ||
                    o->p[2] != o->p[3] || o->p[2] > 2u || kb * 2048u * 4u > HX_FP16_MIN_WORKSPACE)
                    return 0;
                if (compact_consumer != HXFG_NONE)
                    return 0;
                compact_consumer = i;
            } else if (pointwise(o, a, b)) {
                if (kb * 32u > a->stride)
                    return 0;
            } else if (kb * 2048u + 4096u > HX_FP16_MIN_WORKSPACE)
                return 0; /* one im2col tile plus the completion tile must fit the minimum workspace */
        }
    }
    return 1;
}
static uint32_t aux_bytes(const HxGraph *g, const HxOp *o) {
    const uint32_t blocks = g->tensors[o->out].stride / 32u;
    if (o->kind == HX_CONV)
        return blocks * 256u;
    if (o->kind == HX_DEPTHWISE)
        return dw_fast(o) ? blocks * (o->p[0] * o->p[1] + 1u) * 128u : 0u;
    if (o->kind == HX_PRELU)
        return blocks * 128u;
    return 0;
}
int hx_fp16_plan(const HxGraph *g, struct HxFp16Plan *plan, uint32_t *tables, uint8_t *vtcm_end) {
    const uint32_t tensors = g->header->tensors, ops = g->header->ops;
    if (!supported(g))
        return 0;
    plan->offset = tables;
    plan->weights = tables + tensors;
    plan->aux = plan->weights + ops;
    plan->peak = 0;
    plan->image_rows = plan->image_row_words = plan->image_top = 0;
    for (uint32_t i = 0; i < ops; i++) {
        const HxOp *o = &g->ops[i];
        const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
        if (o->kind == HX_CONV && a->stride == 3) {
            /* One vector of zeros either side covers any window offset. */
            const int32_t below = (int32_t)((b->h - 1u) * o->p[2]) - (int32_t)o->p[4] +
                                  (int32_t)o->p[0] - (int32_t)a->h;
            plan->image_top = o->p[4];
            plan->image_rows = o->p[4] + a->h + (below > 0 ? (uint32_t)below : 0u);
            plan->image_row_words = ((a->w * 3u + 31u) & ~31u) + 64u;
        }
    }
    /* First-fit over liveness with whole tiles. An output is placed before its
     * inputs are released, so a kernel never aliases its own source. */
    enum { LIVE = 64 };
    struct {
        uint32_t tensor, begin, end;
    } live[LIVE];
    uint32_t count = 0;
    const uint32_t limit = (uint32_t)(vtcm_end - g->arena) - HX_FP16_MIN_WORKSPACE;
#define HX_PLACE(id)                                                                               \
    do {                                                                                           \
        const HxTensor *t_ = &g->tensors[id];                                                      \
        if (t_->reserved[0] & HX_TENSOR_PERSISTENT) {                                              \
            plan->offset[id] = t_->offset;                                                         \
            break;                                                                                 \
        }                                                                                          \
        uint32_t need = t_->stride == 3                                                            \
                            ? (plan->image_rows * plan->image_row_words * 4u + 2047u) & ~2047u     \
                                        : plane_bytes(t_) * (t_->stride / 32u),                    \
                 at_ = 0;                                                                          \
        for (int moved = 1; moved;) {                                                              \
            moved = 0;                                                                             \
            for (uint32_t l = 0; l < count; l++)                                                   \
                if (at_ < live[l].end && live[l].begin < at_ + need) {                             \
                    at_ = live[l].end;                                                             \
                    moved = 1;                                                                     \
                }                                                                                  \
        }                                                                                          \
        if (at_ + need > limit || count == LIVE)                                                   \
            return 0;                                                                              \
        plan->offset[id] = at_;                                                                    \
        live[count].tensor = id;                                                                   \
        live[count].begin = at_;                                                                   \
        live[count++].end = at_ + need;                                                            \
        if (at_ + need > plan->peak)                                                               \
            plan->peak = at_ + need;                                                               \
    } while (0)
    for (uint32_t i = 0; i < g->header->inputs; i++)
        HX_PLACE(g->io[i]);
    for (uint32_t i = 0; i < ops; i++) {
        const HxOp *o = &g->ops[i];
        HX_PLACE(o->out);
        for (uint32_t l = 0; l < count;) {
            uint32_t id = live[l].tensor, used = is_output(g, id);
            for (uint32_t j = i + 1u; j < ops && !used; j++)
                used = g->ops[j].x == id || g->ops[j].y == id || g->ops[j].z == id;
            if (!used)
                live[l] = live[--count];
            else
                l++;
        }
    }
#undef HX_PLACE
    /* Filter, bias and slope records must be resident. HMX weights fill what
     * remains, largest first: each byte kept saves one DDR copy per inference. */
    uint32_t next = (plan->peak + 255u) & ~255u;
    for (uint32_t i = 0; i < ops; i++) {
        plan->weights[i] = 0;
        plan->aux[i] = 0;
        uint32_t n = aux_bytes(g, &g->ops[i]);
        if (n) {
            if (next + n > limit)
                return 0;
            plan->aux[i] = (uint32_t)(g->arena + next);
            next = (next + n + 255u) & ~255u;
        }
    }
    next = (next + 2047u) & ~2047u;
    for (;;) {
        uint32_t best = HXFG_NONE, best_bytes = 0;
        for (uint32_t i = 0; i < ops; i++) {
            const HxOp *o = &g->ops[i];
            if (o->kind != HX_CONV || plan->weights[i])
                continue;
            uint32_t n = o->p[10] * (g->tensors[o->out].stride / 32u) * 2048u;
            if (n > best_bytes && next + n <= limit) {
                best = i;
                best_bytes = n;
            }
        }
        if (best == HXFG_NONE)
            break;
        plan->weights[best] = (uint32_t)(g->arena + next);
        next += best_bytes;
    }
    next = (next + 2047u) & ~2047u;
    plan->workspace = g->arena + next;
    plan->workspace_bytes = (uint32_t)(vtcm_end - plan->workspace);
    for (uint32_t i = 0; i < ops; i++) {
        const HxOp *o = &g->ops[i];
        if (o->kind == HX_CONV && !plan->weights[i] &&
            o->p[10] * (g->tensors[o->out].stride / 32u) * 2048u + 4096u +
                    (packed_conv(o, &g->tensors[o->x], &g->tensors[o->out]) ? o->p[10] * 2048u : 0u) >
                plan->workspace_bytes)
            return 0;
    }
    return 1;
}
/* Several graphs in one session: re-place every graph's resident records so
 * that no two graphs' records overlap, letting a graph switch skip re-staging.
 * Each graph keeps its standalone activation offsets; activations share
 * [arena, arena + max peak) and kernels share one workspace at the top of VTCM,
 * sized for the largest weight copy any non-resident convolution can need.
 * Filter/bias/slope records are mandatory; HMX weights then fill what remains,
 * largest first across all graphs. Returns 0 and changes nothing when the
 * mandatory records do not fit; the standalone plans stay valid. */
int hx_fp16_share(HxGraph *const *graphs, uint32_t count, uint8_t *vtcm_end) {
    if (count < 2u)
        return 0;
    uint8_t *const base = graphs[0]->arena;
    uint32_t peak = 0, workspace = HX_FP16_MIN_WORKSPACE, records = 0, packed = 0;
    for (uint32_t k = 0; k < count; k++) {
        const HxGraph *g = graphs[k];
        if (!g->fp16 || g->arena != base)
            return 0;
        if (g->fp16->peak > peak)
            peak = g->fp16->peak;
        for (uint32_t i = 0; i < g->header->ops; i++) {
            const HxOp *o = &g->ops[i];
            const uint32_t n = aux_bytes(g, o);
            if (n)
                records = ((records + n + 255u) & ~255u);
            if (o->kind == HX_CONV) {
                uint32_t need = o->p[10] * (g->tensors[o->out].stride / 32u) * 2048u + 4096u +
                                (packed_conv(o, &g->tensors[o->x], &g->tensors[o->out])
                                     ? o->p[10] * 2048u
                                     : 0u);
                if (need > workspace)
                    workspace = need;
                /* A compact-RGB convolution packs as many output tiles per HMX
                 * batch as the workspace holds; a small workspace means many
                 * batches (palm +7.7 % at 274 KB in the simulator). */
                if (packed_conv(o, &g->tensors[o->x], &g->tensors[o->out])) {
                    uint32_t whole = need + tiles(&g->tensors[o->out]) * o->p[10] * 2048u;
                    if (whole > packed)
                        packed = whole;
                }
            }
        }
    }
    workspace = (workspace + 2047u) & ~2047u;
    const uint32_t first = (peak + 255u) & ~255u, total = (uint32_t)(vtcm_end - base);
    if (workspace > total || first + records > total - workspace)
        return 0;
    /* Prefer one RGB batch (capped at 1 MiB) when the mandatory records still fit. */
    packed = ((packed < 0x100000u ? packed : 0x100000u) + 2047u) & ~2047u;
    if (packed > workspace && first + records <= total - packed)
        workspace = packed;
    const uint32_t limit = total - workspace;
    uint32_t next = first;
    for (uint32_t k = 0; k < count; k++) {
        const HxGraph *g = graphs[k];
        for (uint32_t i = 0; i < g->header->ops; i++) {
            const uint32_t n = aux_bytes(g, &g->ops[i]);
            g->fp16->weights[i] = 0;
            g->fp16->aux[i] = 0;
            if (n) {
                g->fp16->aux[i] = (uint32_t)(base + next);
                next = (next + n + 255u) & ~255u;
            }
        }
    }
    next = (next + 2047u) & ~2047u;
    for (;;) {
        uint32_t best_graph = 0, best = HXFG_NONE, best_bytes = 0;
        for (uint32_t k = 0; k < count; k++) {
            const HxGraph *g = graphs[k];
            for (uint32_t i = 0; i < g->header->ops; i++) {
                const HxOp *o = &g->ops[i];
                if (o->kind != HX_CONV || g->fp16->weights[i])
                    continue;
                uint32_t n = o->p[10] * (g->tensors[o->out].stride / 32u) * 2048u;
                if (n > best_bytes && next + n <= limit) {
                    best_graph = k;
                    best = i;
                    best_bytes = n;
                }
            }
        }
        if (best == HXFG_NONE)
            break;
        graphs[best_graph]->fp16->weights[best] = (uint32_t)(base + next);
        next += best_bytes;
    }
    for (uint32_t k = 0; k < count; k++) {
        /* The plan structure is session metadata the engine owns. */
        struct HxFp16Plan *plan = (struct HxFp16Plan *)graphs[k]->fp16;
        plan->workspace = base + limit;
        plan->workspace_bytes = workspace;
    }
    return 1;
}

void hx_fp16_stage(const HxGraph *g, HxFp16Parallel parallel) {
    for (uint32_t i = 0; i < g->header->ops; i++) {
        const HxOp *o = &g->ops[i];
        const uint32_t blocks = g->tensors[o->out].stride / 32u;
        if (g->fp16->weights[i]) {
            HxFp16Job job = {.kind = HX_JOB_COPY};
            job.src = g->constants + o->p[9];
            job.dst = (void *)g->fp16->weights[i];
            job.bytes = o->p[10] * blocks * 2048u;
            run(&job, parallel);
        }
        if (o->kind == HX_CONV) {
            const HVX_Vector *bias = (const HVX_Vector *)(g->constants + o->p[7]);
            uint32_t *table = (uint32_t *)g->fp16->aux[i];
            for (uint32_t n = 0; n < blocks; n++) {
                HVX_Vector h = half_words(bias[n]);
                *(HVX_Vector *)(table + n * 64u) =
                    Q6_V_vor_VV(Q6_Vw_vasl_VwR(h, 16), Q6_V_vsplat_R(0x3c00));
                *(HVX_Vector *)(table + n * 64u + 32u) = Q6_V_vzero();
            }
        } else if (o->kind == HX_DEPTHWISE && dw_fast(o)) {
            const uint32_t taps = o->p[0] * o->p[1];
            const HxTensor *b = &g->tensors[o->out];
            const float *w = (const float *)(g->constants + o->p[6]);
            const float *bias = (const float *)(g->constants + o->p[7]);
            HVX_Vector *record = (HVX_Vector *)g->fp16->aux[i];
            for (uint32_t c = 0; c < blocks; c++) {
                for (uint32_t k = 0; k < taps; k++) {
                    HVX_Vector h = half_words(*(const HVX_Vector *)(w + k * b->stride + c * 32u));
                    *record++ = Q6_Vh_vshuffe_VhVh(h, h);
                }
                *record++ =
                    Q6_Vqf32_vadd_VsfVsf(*(const HVX_Vector *)(bias + c * 32u), Q6_V_vzero());
            }
        } else if (o->kind == HX_PRELU) {
            const float *w = (const float *)(g->constants + o->p[0]);
            HVX_Vector *record = (HVX_Vector *)g->fp16->aux[i];
            for (uint32_t c = 0; c < blocks; c++) {
                HVX_Vector h = half_words(*(const HVX_Vector *)(w + c * 32u));
                record[c] = Q6_Vh_vshuffe_VhVh(h, h);
            }
        }
    }
}

/* ------------------------------- external ------------------------------ */
/* Mask selecting word lanes [dst, dst + count). vsetq(128) is empty on this
 * ISA, so the top of a full-width range is explicit. */
static inline HVX_Vector rgb_mask(uint32_t dst, uint32_t count) {
    HVX_VectorPred q = Q6_Q_not_Q(Q6_Q_vsetq_R(dst * 4u));
    if (dst + count < 32u)
        q = Q6_Q_and_QQ(q, Q6_Q_vsetq_R((dst + count) * 4u));
    return Q6_V_vand_QR(q, 0xffffffffu);
}
void hx_fp16_import(const HxGraph *g, uint32_t tensor, const float *src, HxFp16Parallel parallel) {
    const HxTensor *t = &g->tensors[tensor];
    if (t->stride == 3) {
        /* One thread reads DDR and lays out the padded image; all convert. */
        const struct HxFp16Plan *plan = g->fp16;
        const uint32_t row_vectors = plan->image_row_words / 32u, data = row_vectors - 2u,
                       tail = (t->w * 3u) & 31u;
        const HVX_Vector zero = Q6_V_vzero(),
                         last = tail ? rgb_mask(0, tail) : Q6_V_vnot_V(zero);
        HVX_Vector *d = (HVX_Vector *)tensor_data(g, tensor);
        const uint8_t *row = (const uint8_t *)src;
        const uint32_t row_bytes = t->w * 12u;
        const int whole = !((uintptr_t)src & 127u) && !(row_bytes & 127u);
        for (uint32_t y = 0; y < plan->image_rows; y++) {
            const int inside = y >= plan->image_top && y < plan->image_top + t->h;
            *d++ = zero;
            if (inside && whole) {
                const HVX_Vector *v = (const HVX_Vector *)row;
                for (uint32_t i = 0; i < data; i++)
                    *d++ = v[i];
            } else if (inside) {
                const uint8_t *u = row;
                for (uint32_t i = 0; i < data; i++, u += 128) {
                    HVX_Vector v = *(const HVX_UVector *)u;
                    *d++ = i + 1u == data ? Q6_V_vand_VV(v, last) : v;
                }
            } else
                for (uint32_t i = 0; i < data; i++)
                    *d++ = zero;
            *d++ = zero;
            if (inside)
                row += row_bytes;
        }
        HxFp16Job job = {.kind = HX_JOB_IMAGE_HALF};
        job.g = g;
        job.t = t;
        job.dst = tensor_data(g, tensor);
        run(&job, parallel);
        return;
    }
    uint16_t *d = (uint16_t *)tensor_data(g, tensor);
    for (uint32_t y = 0; y < t->h; y++)
        for (uint32_t x = 0; x < t->w; x += 2)
            for (uint32_t c = 0; c < t->stride; c += 32) {
                const float *s = src + (y * t->w + x) * t->stride + c;
                HVX_Vector a = *(const HVX_Vector *)s;
                HVX_Vector b = x + 1u < t->w ? *(const HVX_Vector *)(s + t->stride) : Q6_V_vzero();
                *(HVX_Vector *)(d + at(t, y * physical_width(t) + x, c)) =
                    Q6_Vh_vshuffe_VhVh(half_words(b), half_words(a));
            }
}
static void export_part(const HxTensor *t, const uint16_t *s, float *dst, uint32_t part,
                        uint32_t parts) {
    const uint32_t pw = physical_width(t);
    uint32_t y0 = t->h * part / parts, y1 = t->h * (part + 1u) / parts;
    uint32_t x0 = 0, x1 = t->w;
    if (t->h < parts) {
        /* A single-row output splits by pair. */
        y0 = 0;
        y1 = t->h;
        x0 = (t->w + 1u) / 2u * part / parts * 2u;
        x1 = (t->w + 1u) / 2u * (part + 1u) / parts * 2u;
        if (x1 > t->w)
            x1 = t->w;
    }
    for (uint32_t y = y0; y < y1; y++)
        for (uint32_t x = x0; x < x1; x += 2)
            for (uint32_t c = 0; c < t->stride; c += 32) {
                HVX_Vector h = load(s, t, y * pw + x, c);
                float *d = dst + (y * t->w + x) * t->stride + c;
                *(HVX_Vector *)d = float_words(h);
                if (x + 1u < t->w)
                    *(HVX_Vector *)(d + t->stride) = float_words(Q6_Vuw_vlsr_VuwR(h, 16));
            }
}
void hx_fp16_export(const HxGraph *g, uint32_t tensor, float *dst, HxFp16Parallel parallel) {
    HxFp16Job job = {.kind = HX_JOB_EXPORT};
    job.g = g;
    job.t = &g->tensors[tensor];
    job.src = tensor_data(g, tensor);
    job.dst = dst;
    run(&job, parallel);
}

/* ----------------------------- matrix unit ----------------------------- */
typedef struct {
    HVX_Vector m[7];
} RgbMasks;
static void rgb_masks(RgbMasks *r, uint32_t K) {
    static const uint8_t k3[3][2] = {{0, 9}, {9, 9}, {18, 9}};
    static const uint8_t k5[7][2] = {{0, 15}, {15, 15}, {30, 2}, {0, 13}, {13, 15}, {28, 4}, {0, 11}};
    for (uint32_t i = 0; i < (K == 3 ? 3u : 7u); i++)
        r->m[i] = K == 3 ? rgb_mask(k3[i][0], k3[i][1]) : rgb_mask(k5[i][0], k5[i][1]);
}
/* The image holds exact half values in the low half of each RGB word, padded so
 * that every window of every output pixel is contiguous. The stride-shifted
 * twin of a window always fits the same unaligned load: at most 21 of 32 lanes. */
static void rgb_pack_tile(const HxGraph *g, const HxOp *o, const uint32_t *image, uint16_t *ap,
                          uint32_t tile, const RgbMasks *masks) {
    const HxTensor *b = &g->tensors[o->out];
    const struct HxFp16Plan *plan = g->fp16;
    const uint32_t *p = o->p, K = p[0], pw = physical_width(b), rw = plan->image_row_words;
    const uint32_t kb = p[10], stride = p[2], step = p[3];
    const int32_t shift = (int32_t)(step * 12u);
    const HVX_Vector *m = masks->m;
    uint32_t pixel = tile * 32u, oy = pixel / pw, ox = pixel % pw;
    HVX_Vector *out = (HVX_Vector *)ap;
#define HX_RGB(row_, src_, dst_, mask_)                                                            \
    Q6_V_vand_VV(Q6_V_vror_VR(row_, (int32_t)((src_) * 4u) - (int32_t)((dst_) * 4u)), m[mask_])
#define HX_ROW(n_)                                                                                 \
    Q6_Vh_vshuffe_VhVh(Q6_V_vror_VR(*(const HVX_UVector *)(src + (n_) * rw), shift),               \
                       *(const HVX_UVector *)(src + (n_) * rw))
    for (uint32_t r = 0; r < 16u; r++) {
        if (oy >= b->h) {
            for (uint32_t k = 0; k < kb; k++)
                out[k * 16u + r] = Q6_V_vzero();
        } else {
            const uint32_t *src = image + (oy * stride - p[4] + plan->image_top) * rw + 32u +
                                  (int32_t)(ox * step - p[5]) * 3;
            if (K == 3) {
                const HVX_Vector r0 = HX_ROW(0), r1 = HX_ROW(1), r2 = HX_ROW(2);
                out[r] = Q6_V_vor_VV(HX_RGB(r0, 0, 0, 0),
                                     Q6_V_vor_VV(HX_RGB(r1, 0, 9, 1), HX_RGB(r2, 0, 18, 2)));
            } else {
                const HVX_Vector r0 = HX_ROW(0), r1 = HX_ROW(1), r2 = HX_ROW(2), r3 = HX_ROW(3),
                                 r4 = HX_ROW(4);
                out[r] = Q6_V_vor_VV(HX_RGB(r0, 0, 0, 0),
                                     Q6_V_vor_VV(HX_RGB(r1, 0, 15, 1), HX_RGB(r2, 0, 30, 2)));
                out[16u + r] = Q6_V_vor_VV(HX_RGB(r2, 2, 0, 3),
                                           Q6_V_vor_VV(HX_RGB(r3, 0, 13, 4), HX_RGB(r4, 0, 28, 5)));
                out[32u + r] = HX_RGB(r4, 4, 0, 6);
            }
        }
        ox += 2u;
        if (ox >= pw) {
            ox = 0;
            oy++;
        }
    }
#undef HX_ROW
#undef HX_RGB
}
static void activate_tensor(const HxGraph *g, uint32_t index, HxFp16Parallel parallel) {
    const HxOp *o = &g->ops[index];
    const HxTensor *b = &g->tensors[o->out];
    if (!o->p[8])
        return;
    HxFp16Job job = {.kind = HX_JOB_ACTIVATE};
    job.g = g;
    job.t = b;
    job.dst = tensor_data(g, o->out);
    job.index = o->p[8];
    job.bytes = plane_bytes(b) * (b->stride / 32u);
    if (job.bytes >= 32768u)
        run(&job, parallel);
    else
        hx_fp16_job(&job, 0, 1);
}
void hx_fp16_conv(const HxGraph *g, uint32_t index, HxFp16Parallel parallel) {
    const HxOp *o = &g->ops[index];
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t kb = o->p[10], nt = b->stride / 32u, weight_bytes = kb * nt * 2048u;
    uint8_t *const workspace = g->fp16->workspace;
    uint8_t *cursor = workspace;
    const void *weights = (const void *)g->fp16->weights[index];
    if (!weights) {
        HxFp16Job job = {.kind = HX_JOB_COPY};
        job.src = g->constants + o->p[9];
        job.dst = cursor;
        job.bytes = weight_bytes;
        run(&job, parallel);
        weights = cursor;
        cursor += weight_bytes;
    }
    void *bias = (void *)g->fp16->aux[index];
    uint8_t *done = cursor;
    cursor += 2048u;
    HxHmxStrip strip = {tiles(b), nt, kb, 2048u, kb * 2048u, plane_bytes(b), bias};
    uint8_t *out = tensor_data(g, o->out);
    hx_hmx_init(bias);
    if (!packed_conv(o, a, b)) {
        strip.a_kstep = plane_bytes(a);
        strip.a_mstep = 2048u;
        hx_hmx_strip(tensor_data(g, o->x), weights, out, &strip);
        hx_hmx_finish(done);
    } else {
        /* Workers pack a run of activation tiles (the compact RGB image or an im2col of the
         * blocked input), then the matrix unit consumes it. The completion before reuse
         * drains its loads and stores. */
        const uint32_t total = tiles(b),
                       space = (uint32_t)(workspace + g->fp16->workspace_bytes - cursor);
        const uint32_t batch = space / (kb * 2048u);
        for (uint32_t first = 0; first < total; first += batch) {
            HxFp16Job job = {.kind = a->stride == 3 ? HX_JOB_RGB_PACK : HX_JOB_IM2COL};
            job.g = g;
            job.src = tensor_data(g, o->x);
            job.dst = cursor;
            job.index = index;
            job.first = first;
            job.count = total - first < batch ? total - first : batch;
            run(&job, parallel);
            strip.m = job.count;
            hx_hmx_init(bias);
            hx_hmx_strip(cursor, weights, out + first * 2048u, &strip);
            hx_hmx_finish(done);
        }
    }
    activate_tensor(g, index, parallel);
}
void hx_fp16_complete(const HxGraph *g, uint32_t index) {
    const HxOp *o = &g->ops[index];
    const HxTensor *b = &g->tensors[o->out];
    if ((b->reserved[0] & HX_TENSOR_PERSISTENT) || !(b->w & 1u))
        return;
    /* Odd width: the last pair of each row carries a pad pixel in its high half. */
    const HVX_Vector keep = Q6_V_vsplat_R(0xffff);
    HVX_Vector *out = (HVX_Vector *)tensor_data(g, o->out);
    const uint32_t pairs = physical_width(b) / 2u, plane = plane_bytes(b) / 128u;
    for (uint32_t c = 0; c < b->stride / 32u; c++)
        for (uint32_t y = 0; y < b->h; y++) {
            HVX_Vector *v = out + c * plane + y * pairs + pairs - 1u;
            *v = Q6_V_vand_VV(*v, keep);
        }
}
void hx_fp16_job(const HxFp16Job *job, uint32_t part, uint32_t parts) {
    switch (job->kind) {
    case HX_JOB_COPY: {
        const uint32_t words = job->bytes / 128u, v0 = words * part / parts,
                       v1 = words * (part + 1u) / parts;
        const uint8_t *s = (const uint8_t *)job->src + v0 * 128u;
        HVX_Vector *d = (HVX_Vector *)job->dst + v0;
        /* An unaligned form makes two DDR reads per vector. */
        if ((uintptr_t)job->src & 127u)
            for (uint32_t v = v0; v < v1; v++, s += 128)
                *d++ = *(const HVX_UVector *)s;
        else
            for (uint32_t v = v0; v < v1; v++, s += 128)
                *d++ = *(const HVX_Vector *)s;
        break;
    }
    case HX_JOB_IMAGE_HALF: {
        const struct HxFp16Plan *plan = job->g->fp16;
        const uint32_t row_vectors = plan->image_row_words / 32u, h = job->t->h;
        HVX_Vector *image = (HVX_Vector *)job->dst + plan->image_top * row_vectors;
        for (uint32_t y = h * part / parts; y < h * (part + 1u) / parts; y++) {
            HVX_Vector *d = image + y * row_vectors + 1u;
            for (uint32_t i = 0; i + 2u < row_vectors; i++)
                d[i] = half_words(d[i]);
        }
        break;
    }
    case HX_JOB_RGB_PACK: {
        const HxOp *o = &job->g->ops[job->index];
        const uint32_t kb = o->p[10], t0 = job->count * part / parts,
                       t1 = job->count * (part + 1u) / parts;
        RgbMasks masks;
        rgb_masks(&masks, o->p[0]);
        for (uint32_t t = t0; t < t1; t++)
            rgb_pack_tile(job->g, o, (const uint32_t *)job->src,
                          (uint16_t *)((uint8_t *)job->dst + t * kb * 2048u), job->first + t,
                          &masks);
        break;
    }
    case HX_JOB_IM2COL: {
        const HxOp *o = &job->g->ops[job->index];
        const HxTensor *a = &job->g->tensors[o->x], *b = &job->g->tensors[o->out];
        const uint32_t kb = o->p[10], t0 = job->count * part / parts,
                       t1 = job->count * (part + 1u) / parts;
        for (uint32_t t = t0; t < t1; t++)
            hx_im2col_tile((uint16_t *)((uint8_t *)job->dst + t * kb * 2048u),
                           (const uint16_t *)job->src, a->h, a->w, a->c, plane_bytes(a) / 2u, b->h,
                           b->w, o->p, kb, job->first + t, hx_dilation_h(o), hx_dilation_w(o));
        break;
    }
    case HX_JOB_ACTIVATE: {
        const uint32_t words = job->bytes / 128u, v0 = words * part / parts,
                       v1 = words * (part + 1u) / parts;
        HVX_Vector *d = (HVX_Vector *)job->dst;
        if (job->index == 3) {
            const HVX_Vector ceiling = Q6_Vh_vsplat_R(0x4600), floor_ = Q6_V_vzero();
            for (uint32_t v = v0; v < v1; v++)
                d[v] = Q6_Vh_vmin_VhVh(ceiling, Q6_Vh_vmax_VhVh(d[v], floor_));
        } else
            for (uint32_t v = v0; v < v1; v++)
                d[v] = activate_half(d[v], job->index);
        break;
    }
    case HX_JOB_EXPORT:
        export_part(job->t, (const uint16_t *)job->src, (float *)job->dst, part, parts);
        break;
    }
}

/* ---------------------------- HVX operators ---------------------------- */
static void elementwise(const HxGraph *g, const HxOp *o, uint32_t index, uint32_t part,
                        uint32_t parts) {
    const HxTensor *b = &g->tensors[o->out];
    const uint32_t plane = plane_bytes(b) / 128u, pairs = tiles(b) * 16u, blocks = b->stride / 32u;
    const uint32_t lo = pairs * part / parts, hi = pairs * (part + 1u) / parts;
    const HVX_Vector *x = (const HVX_Vector *)tensor_data(g, o->x);
    HVX_Vector *out = (HVX_Vector *)tensor_data(g, o->out);
    if (o->kind == HX_ADD) {
        const HVX_Vector *y = (const HVX_Vector *)tensor_data(g, o->y);
        const uint32_t activation = o->p[0];
        for (uint32_t c = 0; c < blocks; c++) {
            const HVX_Vector *a0 = x + c * plane, *a1 = y + c * plane;
            HVX_Vector *d = out + c * plane;
            for (uint32_t i = lo; i < hi; i++)
                d[i] = pack(plus(floats(a0[i]), floats(a1[i])), activation);
        }
    } else {
        const HVX_Vector *slopes = (const HVX_Vector *)g->fp16->aux[index];
        for (uint32_t c = 0; c < blocks; c++) {
            const HVX_Vector *a0 = x + c * plane, slope = slopes[c];
            HVX_Vector *d = out + c * plane;
            for (uint32_t i = lo; i < hi; i++) {
                HVX_Vector v = a0[i];
                d[i] = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), v),
                                     pack(Q6_Wqf32_vmpy_VhfVhf(v, slope), 0), v);
            }
        }
    }
}
static void maxpool(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p, apw = physical_width(a), obw = physical_width(b);
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);
    if (p[0] == 2 && p[1] == 2 && p[2] == 2 && p[3] == 2 && !p[4] && !p[5] && a->h == b->h * 2u &&
        a->w == b->w * 2u && !(b->w & 1u)) {
        /* Each output pair spans two input pairs on two rows. */
        const uint32_t irow = apw / 2u, iplane = plane_bytes(a) / 128u, oplane = plane_bytes(b) / 128u,
                       orow = obw / 2u, blocks = b->stride / 32u;
        const HVX_Vector *in = (const HVX_Vector *)x;
        HVX_Vector *dst = (HVX_Vector *)out;
        for (uint32_t c = 0; c < blocks; c++)
            for (uint32_t y = b->h * part / parts; y < b->h * (part + 1u) / parts; y++) {
                const HVX_Vector *r0 = in + c * iplane + y * 2u * irow, *r1 = r0 + irow;
                HVX_Vector *d = dst + c * oplane + y * orow;
                for (uint32_t j = 0; j < orow; j++, r0 += 2, r1 += 2) {
                    HVX_Vector a0 = r0[0], a1 = r0[1], b0 = r1[0], b1 = r1[1];
                    HVX_Vector m = max_half(Q6_Vh_vshuffe_VhVh(a1, a0), Q6_Vh_vshuffo_VhVh(a1, a0));
                    m = max_half(m, Q6_Vh_vshuffe_VhVh(b1, b0));
                    d[j] = max_half(m, Q6_Vh_vshuffo_VhVh(b1, b0));
                }
            }
        return;
    }
    for (uint32_t y = b->h * part / parts; y < b->h * (part + 1u) / parts; y++)
        for (uint32_t ox = 0; ox < obw; ox += 2)
            for (uint32_t c = 0; c < b->stride; c += 32) {
                HVX_Vector maximum = Q6_Vh_vsplat_R(0xfbff);
                for (uint32_t ky = 0; ky < p[0]; ky++)
                    for (uint32_t kx = 0; kx < p[1]; kx++) {
                        int32_t pos[2];
                        for (uint32_t q = 0; q < 2; q++) {
                            int32_t iy = (int32_t)(y * p[2] + ky) - (int32_t)p[4],
                                    ix = (int32_t)((ox + q) * p[3] + kx) - (int32_t)p[5];
                            pos[q] = ox + q >= b->w || iy < 0 || ix < 0 || iy >= (int32_t)a->h ||
                                             ix >= (int32_t)a->w
                                         ? -1
                                         : iy * (int32_t)apw + ix;
                        }
                        HVX_Vector v = rows(x, a, pos[0], pos[1], c);
                        if (pos[0] < 0)
                            v = Q6_V_vor_VV(v, Q6_V_vsplat_R(0x0000fbff));
                        if (pos[1] < 0)
                            v = Q6_V_vor_VV(v, Q6_V_vsplat_R(0xfbff0000));
                        maximum = max_half(maximum, v);
                    }
                *(HVX_Vector *)(out + at(b, y * obw + ox, c)) = maximum;
            }
}
static void mean(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t apw = physical_width(a);
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);
    const HVX_Vector scale = splat(1.0f / (float)(a->h * a->w));
    for (uint32_t c = 0; c < b->stride; c += 32) {
        if ((c / 32u) % parts != part)
            continue;
        HVX_Vector sum = Q6_V_vzero();
        for (uint32_t y = 0; y < a->h; y++)
            for (uint32_t ix = 0; ix < a->w; ix++) {
                HVX_VectorPair v = floats(rows(x, a, (int32_t)(y * apw + ix), -1, c));
                sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_V_lo_W(v));
            }
        HVX_Vector v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(sum), scale));
        *(HVX_Vector *)(out + at(b, 0, c)) = Q6_Vh_vshuffe_VhVh(Q6_V_vzero(), half_words(v));
    }
}
static void resize(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p, apw = physical_width(a), obw = physical_width(b);
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);
    const float sy = p[0] && b->h > 1 ? (float)(a->h - 1) / (b->h - 1) : (float)a->h / b->h;
    const float sx = p[0] && b->w > 1 ? (float)(a->w - 1) / (b->w - 1) : (float)a->w / b->w;
    for (uint32_t y = b->h * part / parts; y < b->h * (part + 1u) / parts; y++)
        for (uint32_t ox = 0; ox < obw; ox += 2) {
            int valid[2];
            uint32_t pos[2][4];
            HVX_Vector ty[2], tx[2];
            for (uint32_t q = 0; q < 2; q++) {
                valid[q] = ox + q < b->w;
                if (!valid[q])
                    continue;
                float fy = fmaxf(0, ((float)y + (p[1] ? .5f : 0)) * sy - (p[1] ? .5f : 0));
                float fx = fmaxf(0, ((float)(ox + q) + (p[1] ? .5f : 0)) * sx - (p[1] ? .5f : 0));
                uint32_t y0 = fy, x0 = fx, y1 = y0 + 1 < a->h ? y0 + 1 : y0,
                         x1 = x0 + 1 < a->w ? x0 + 1 : x0;
                pos[q][0] = y0 * apw + x0;
                pos[q][1] = y0 * apw + x1;
                pos[q][2] = y1 * apw + x0;
                pos[q][3] = y1 * apw + x1;
                ty[q] = splat(fy - y0);
                tx[q] = splat(fx - x0);
            }
            for (uint32_t c = 0; c < b->stride; c += 32) {
                HVX_Vector result[2] = {Q6_V_vzero(), Q6_V_vzero()};
                for (uint32_t q = 0; q < 2; q++) {
                    if (!valid[q])
                        continue;
                    HVX_Vector v[4];
                    for (uint32_t n = 0; n < 4; n++)
                        v[n] = Q6_Vsf_equals_Vqf32(
                            Q6_V_lo_W(floats(rows(x, a, (int32_t)pos[q][n], -1, c))));
                    HVX_Vector top = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(
                        v[0], Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(
                                  Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v[1], v[0])), tx[q]))));
                    HVX_Vector bot = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(
                        v[2], Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(
                                  Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v[3], v[2])), tx[q]))));
                    HVX_Vector sum = Q6_Vqf32_vadd_VsfVsf(
                        top, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(
                                 Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(bot, top)), ty[q])));
                    /* Native conversion; each word's half is its even lane. */
                    result[q] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(sum, sum));
                }
                *(HVX_Vector *)(out + at(b, y * obw + ox, c)) =
                    Q6_Vh_vshuffe_VhVh(result[1], result[0]);
            }
        }
}
/* Row-major copy with running source coordinates. Each run stays inside one
 * source and one destination channel block, so it is a strided halfword copy. */
static void reshape(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t obw = physical_width(b), apw = physical_width(a);
    const uint32_t opairs = b->h * obw / 2u, lo = opairs * part / parts,
                   hi = opairs * (part + 1u) / parts;
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);
    const uint32_t iplane = plane_bytes(a) / 2u, oplane = plane_bytes(b) / 2u;
    const uint32_t ochannels = b->c, oblocks = b->stride / 32u;
    uint32_t y = lo * 2u / obw, ox = lo * 2u % obw;
    /* Divide once per part; the source then advances with the output. */
    const uint32_t start = (y * b->w + ox) * ochannels;
    uint32_t ic = start % a->c, pixel = start / a->c, iy = pixel / a->w, ix = pixel % a->w;
    for (uint32_t pair = lo; pair < hi; pair++) {
        for (uint32_t q = 0; q < 2u; q++) {
            uint16_t *d = out + (size_t)pair * 64u + q;
            uint32_t c = 0;
            if (ox + q < b->w)
                while (c < ochannels) {
                    const uint32_t ip = iy * apw + ix;
                    const uint16_t *s = x + (ic >> 5) * iplane + (ip >> 1) * 64u + (ic & 31u) * 2u +
                                        (ip & 1u);
                    uint32_t run = 32u - (ic & 31u), room = 32u - (c & 31u);
                    if (run > room)
                        run = room;
                    if (run > ochannels - c)
                        run = ochannels - c;
                    if (run > a->c - ic)
                        run = a->c - ic;
                    uint16_t *e = d + (c >> 5) * oplane + (c & 31u) * 2u;
                    for (uint32_t n = 0; n < run; n++)
                        e[n * 2u] = s[n * 2u];
                    c += run;
                    ic += run;
                    if (ic == a->c) {
                        ic = 0;
                        if (++ix == a->w) {
                            ix = 0;
                            iy++;
                        }
                    }
                }
            for (; c < oblocks * 32u; c++)
                d[(c >> 5) * oplane + (c & 31u) * 2u] = 0;
        }
        ox += 2u;
        if (ox >= obw) {
            ox = 0;
            y++;
        }
    }
}
static int32_t place(const HxOp *o, const HxTensor *a, const HxGraph *g, uint32_t yy, uint32_t xx,
                     const uint16_t **src, const HxTensor **t) {
    const uint32_t *p = o->p;
    *t = a;
    *src = (const uint16_t *)tensor_data(g, o->x);
    if (o->kind == HX_PAD) {
        if (yy >= p[0] && yy - p[0] < a->h && xx >= p[1] && xx - p[1] < a->w)
            return (int32_t)((yy - p[0]) * physical_width(a) + xx - p[1]);
        return -1;
    }
    const uint32_t coord = p[0] == 0 ? yy : xx, size = p[0] == 0 ? a->h : a->w;
    if (coord >= size) {
        *t = &g->tensors[o->y];
        *src = (const uint16_t *)tensor_data(g, o->y);
        if (!p[0])
            yy -= size;
        else
            xx -= size;
    }
    return (int32_t)(yy * physical_width(*t) + xx);
}
/* HX_NORM on the scalar lanes (reference speed). Statistics are float (two-pass mean, then
 * variance). Every part recomputes the statistics it needs over all pixels and writes only its
 * own pixel pairs; pad columns and pad channel lanes are written zero. */
static void norm_scalar(const HxGraph *g, const HxOp *o, uint32_t lo, uint32_t hi) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);
    const uint32_t *p = o->p;
    const uint32_t pw = physical_width(a), obw = physical_width(b), c = b->c, pixels = a->h * a->w;
    float eps;
    memcpy(&eps, &p[2], sizeof eps);
    const float *gamma = (const float *)(g->constants + p[3]),
                *beta = (const float *)(g->constants + p[4]);
    for (uint32_t pair = lo; pair < hi; pair++)
        for (uint32_t q = 0; q < 2; q++) {
            const uint32_t px = pair * 2u + q, ox = pair * 2u % obw + q;
            for (uint32_t k = ox >= b->w ? 0u : c; k < b->stride; k++)
                out[at(b, px, k)] = 0;
        }
    if (p[0] == HX_NORM_CHANNEL) {
        for (uint32_t pair = lo; pair < hi; pair++)
            for (uint32_t q = 0; q < 2; q++) {
                const uint32_t px = pair * 2u + q, ox = pair * 2u % obw + q;
                if (ox >= b->w)
                    continue;
                float mean = 0, var = 0;
                for (uint32_t k = 0; k < c; k++)
                    mean += scalar(x[at(a, px, k)]);
                mean /= (float)c;
                for (uint32_t k = 0; k < c; k++) {
                    const float d = scalar(x[at(a, px, k)]) - mean;
                    var += d * d;
                }
                var /= (float)c;
                const float inv = 1.0f / sqrtf(var + eps);
                for (uint32_t k = 0; k < c; k++)
                    out[at(b, px, k)] =
                        half((scalar(x[at(a, px, k)]) - mean) * inv * gamma[k] + beta[k]);
            }
        return;
    }
    /* GROUP: c % groups == 0 (op_valid), so each channel belongs to one group. */
    const uint32_t groups = p[1], size = c / groups;
    for (uint32_t grp = 0; grp < groups; grp++) {
        const uint32_t k0 = grp * size, k1 = k0 + size;
        const float count = (float)(pixels * size);
        float mean = 0, var = 0;
        for (uint32_t iy = 0; iy < a->h; iy++)
            for (uint32_t ix = 0; ix < a->w; ix++)
                for (uint32_t k = k0; k < k1; k++)
                    mean += scalar(x[at(a, iy * pw + ix, k)]);
        mean /= count;
        for (uint32_t iy = 0; iy < a->h; iy++)
            for (uint32_t ix = 0; ix < a->w; ix++)
                for (uint32_t k = k0; k < k1; k++) {
                    const float d = scalar(x[at(a, iy * pw + ix, k)]) - mean;
                    var += d * d;
                }
        var /= count;
        const float inv = 1.0f / sqrtf(var + eps);
        for (uint32_t pair = lo; pair < hi; pair++)
            for (uint32_t q = 0; q < 2; q++) {
                const uint32_t px = pair * 2u + q, ox = pair * 2u % obw + q;
                if (ox >= b->w)
                    continue;
                for (uint32_t k = k0; k < k1; k++)
                    out[at(b, px, k)] =
                        half((scalar(x[at(a, px, k)]) - mean) * inv * gamma[k] + beta[k]);
            }
    }
}
/* HX_MATMUL in vector lanes, same scratch slice as matmul_scalar. B is kept as float blocks
 * [K][NBP] (32 lanes each, zero beyond N and beyond NB blocks, NBP = NB rounded up to 4). Each A
 * row is converted once by HVX, and each output row accumulates four 32-channel blocks at a time
 * in qf32 registers. Returns 0 when the slice is too small. */
static int matmul_vector(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out], *t = &g->tensors[o->y];
    const uint32_t K = a->c, N = b->c, NB = (N + 31u) / 32u, NBP = (NB + 3u) & ~3u;
    const uint32_t KB = (K + 31u) / 32u, bytes_b = K * NBP * 128u, bytes_a = KB * 128u;
    /* transB 1 converts B rows (N x KB vectors) into a float temporary before the transpose. */
    const uint32_t bytes_t = o->p[0] ? N * KB * 128u : 0u;
    const uint32_t slice = (g->fp16->workspace_bytes / parts) & ~2047u;
    if (bytes_b + bytes_a + bytes_t > slice)
        return 0;
    const uint32_t lo = b->h * part / parts, hi = b->h * (part + 1u) / parts;
    if (lo == hi)
        return 1;
    uint8_t *ws = g->fp16->workspace + part * slice;
    HVX_Vector *bf = (HVX_Vector *)ws, *arow = (HVX_Vector *)(ws + bytes_b);
    const HVX_Vector *xv = (const HVX_Vector *)tensor_data(g, o->x);
    HVX_Vector *out = (HVX_Vector *)tensor_data(g, o->out);
    float scale = 1.0f;
    if (o->p[1])
        memcpy(&scale, &o->p[1], sizeof scale);
    const uint32_t plane_a = plane_bytes(a) / 128u, plane_b = plane_bytes(b) / 128u;
    if (!o->p[0]) {
        /* B is [K,N]: block nb of row k is one vector of the blocked operand. */
        const HVX_Vector *yvec = (const HVX_Vector *)tensor_data(g, o->y);
        const uint32_t plane_t = plane_bytes(t) / 128u;
        for (uint32_t k = 0; k < K; k++) {
            for (uint32_t nb = 0; nb < NB; nb++) {
                HVX_Vector v = float_words(yvec[nb * plane_t + k]);
                if (nb == NB - 1u && (N & 31u))
                    v = Q6_V_vand_VV(v, rgb_mask(0, N & 31u));
                bf[k * NBP + nb] = v;
            }
            for (uint32_t nb = NB; nb < NBP; nb++)
                bf[k * NBP + nb] = Q6_V_vzero();
        }
    } else {
        /* B is [N,K]: HVX converts each row to float (row n at tmp + n * KB * 32), then the
         * transpose is a scalar copy into the same layout as above. */
        const HVX_Vector *yvec = (const HVX_Vector *)tensor_data(g, o->y);
        const uint32_t plane_t = plane_bytes(t) / 128u, row_stride = KB * 32u;
        float *tmp = (float *)(ws + bytes_b + bytes_a), *bfs = (float *)bf;
        for (uint32_t n = 0; n < N; n++)
            for (uint32_t kb = 0; kb < KB; kb++)
                ((HVX_Vector *)(tmp + n * row_stride))[kb] = float_words(yvec[kb * plane_t + n]);
        for (uint32_t k = 0; k < K; k++) {
            float *dst = bfs + k * NBP * 32u;
            for (uint32_t n = 0; n < N; n++)
                dst[n] = tmp[n * row_stride + k];
            for (uint32_t n = N; n < NBP * 32u; n++)
                dst[n] = 0.0f;
        }
    }
    for (uint32_t i = lo; i < hi; i++) {
        for (uint32_t kb = 0; kb < KB; kb++) {
            HVX_Vector v = float_words(xv[kb * plane_a + i]);
            if (scale != 1.0f)
                v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v, splat(scale)));
            arow[kb] = v;
        }
        const float *ar = (const float *)arow;
        for (uint32_t nb0 = 0; nb0 < NBP; nb0 += 4u) {
            HVX_Vector c0 = Q6_V_vzero(), c1 = Q6_V_vzero(), c2 = Q6_V_vzero(), c3 = Q6_V_vzero();
            for (uint32_t k = 0; k < K; k++) {
                const HVX_Vector s = splat(ar[k]), *r = bf + k * NBP + nb0;
                c0 = Q6_Vqf32_vadd_Vqf32Vqf32(c0, Q6_Vqf32_vmpy_VsfVsf(s, r[0]));
                c1 = Q6_Vqf32_vadd_Vqf32Vqf32(c1, Q6_Vqf32_vmpy_VsfVsf(s, r[1]));
                c2 = Q6_Vqf32_vadd_Vqf32Vqf32(c2, Q6_Vqf32_vmpy_VsfVsf(s, r[2]));
                c3 = Q6_Vqf32_vadd_Vqf32Vqf32(c3, Q6_Vqf32_vmpy_VsfVsf(s, r[3]));
            }
            const HVX_Vector res[4] = {c0, c1, c2, c3};
            for (uint32_t j = 0; j < 4u && nb0 + j < NB; j++)
                out[(nb0 + j) * plane_b + i] = half_words(Q6_Vsf_equals_Vqf32(res[j]));
        }
    }
    return 1;
}

/* HX_MATMUL in the scalar lanes, one row at a time. Each part keeps B as float in its own
 * workspace slice ([K][NP], NP = 32 * ceil(N / 32), zero beyond N), converts each A row once, and
 * accumulates a whole output row in float. Returns 0 without writing when the slice cannot hold
 * these operands; the caller then tries the scalar-lane version or the per-element loop. */
static int matmul_scalar(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out], *t = &g->tensors[o->y];
    const uint32_t K = a->c, N = b->c, NP = (N + 31u) & ~31u;
    const uint32_t bytes_b = (K * NP * 4u + 127u) & ~127u, bytes_a = (K * 4u + 127u) & ~127u;
    const uint32_t slice = (g->fp16->workspace_bytes / parts) & ~2047u;
    if (bytes_b + bytes_a + NP * 4u > slice)
        return 0;
    const uint32_t lo = b->h * part / parts, hi = b->h * (part + 1u) / parts;
    if (lo == hi)
        return 1;
    uint8_t *ws = g->fp16->workspace + part * slice;
    float *bf = (float *)ws, *arow = (float *)(ws + bytes_b), *acc = (float *)(ws + bytes_b + bytes_a);
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    const uint16_t *yv = (const uint16_t *)tensor_data(g, o->y);
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);
    float scale = 1.0f;
    if (o->p[1])
        memcpy(&scale, &o->p[1], sizeof scale);
    /* B row-major: bf[k * NP + n] = B[k][n]. With transB 1, B is stored [N,K] and element (n, k)
     * is read instead. */
    for (uint32_t k = 0; k < K; k++)
        for (uint32_t n = 0; n < NP; n++) {
            float v = 0.0f;
            if (n < N)
                v = scalar(o->p[0] ? yv[at(t, n * physical_width(t), k)]
                                   : yv[at(t, k * physical_width(t), n)]);
            bf[k * NP + n] = v;
        }
    const uint32_t plane = plane_bytes(b) / 128u;
    for (uint32_t i = lo; i < hi; i++) {
        for (uint32_t k = 0; k < K; k++)
            arow[k] = scalar(x[at(a, i * physical_width(a), k)]) * scale;
        for (uint32_t n = 0; n < NP; n++)
            acc[n] = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
            const float l = arow[k];
            const float *row = bf + k * NP;
            for (uint32_t n = 0; n < NP; n++)
                acc[n] += l * row[n];
        }
        /* Blocked output: one vector per 32 channels, lanes = halves at even positions. */
        for (uint32_t nb = 0; nb < NP / 32u; nb++)
            ((HVX_Vector *)out)[nb * plane + i] = half_words(*(const HVX_Vector *)(acc + nb * 32u));
    }
    return 1;
}

static inline HVX_Vector vsplatf(float f) {
    union {
        float f;
        uint32_t u;
    } v;
    v.f = f;
    return Q6_V_vsplat_R(v.u);
}
static inline float vlane0f(HVX_Vector v) {
    union {
        uint32_t u;
        float f;
    } r;
    r.u = Q6_R_vextract_VR(v, 0);
    return r.f;
}
static inline HVX_Vector vaddf(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
static inline HVX_Vector vmulf(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}
static inline HVX_Vector vhsumf(HVX_Vector v) {
    for (int s = 64; s >= 4; s >>= 1)
        v = vaddf(v, Q6_V_vror_VR(v, s));
    return v;
}

static void rmsnorm_fp16(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p, c = b->c, stride = b->stride;
    const float eps_f = p[0] ? hx_float(p[0]) : 1e-5f;
    const double eps = (double)eps_f;
    const float *gamma = p[1] != HXFG_NONE ? (const float *)(g->constants + p[1]) : NULL;
    const float *beta = p[2] != HXFG_NONE ? (const float *)(g->constants + p[2]) : NULL;

    const uint32_t obw = physical_width(b), opairs = b->h * obw / 2u;
    const uint32_t lo = opairs * part / parts, hi = opairs * (part + 1u) / parts;
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);

    if (b->w == 1u && (c % 32u == 0) && c >= 32u) {
        const uint32_t blocks = c / 32u;
        for (uint32_t pair = lo; pair < hi; pair++) {
            const uint32_t px = pair * 2u;
            HVX_Vector acc = Q6_V_vzero();
            for (uint32_t b_idx = 0; b_idx < blocks; b_idx++) {
                HVX_Vector h = load(x, a, px, b_idx * 32u);
                HVX_Vector xv = float_words(h);
                acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(xv, xv));
            }
            float sum = vlane0f(vhsumf(Q6_Vsf_equals_Vqf32(acc)));
            float inv = 1.0f / sqrtf(sum / (float)c + (float)eps);
            HVX_Vector inv_v = vsplatf(inv);

            for (uint32_t b_idx = 0; b_idx < blocks; b_idx++) {
                HVX_Vector h = load(x, a, px, b_idx * 32u);
                HVX_Vector xv = float_words(h);
                HVX_Vector res = vmulf(xv, inv_v);
                if (gamma) {
                    HVX_Vector gv = *(const HVX_UVector *)(gamma + b_idx * 32u);
                    res = vmulf(res, gv);
                }
                if (beta) {
                    HVX_Vector bv = *(const HVX_UVector *)(beta + b_idx * 32u);
                    res = vaddf(res, bv);
                }
                *(HVX_Vector *)(out + at(b, px, b_idx * 32u)) = half_words(res);
            }
            for (uint32_t b_idx = blocks; b_idx < stride / 32u; b_idx++) {
                *(HVX_Vector *)(out + at(b, px, b_idx * 32u)) = Q6_V_vzero();
            }
        }
        return;
    }

    for (uint32_t pair = lo; pair < hi; pair++) {
        for (uint32_t q = 0; q < 2u; q++) {
            const uint32_t px = pair * 2u + q, ox = px % obw;
            if (ox >= b->w) {
                for (uint32_t ch = 0; ch < stride; ch++)
                    out[at(b, px, ch)] = 0;
                continue;
            }
            double sum_sq = 0.0;
            for (uint32_t ch = 0; ch < c; ch++) {
                float val = scalar(x[at(a, px, ch)]);
                sum_sq += (double)val * (double)val;
            }
            const float inv = (float)(1.0 / sqrt(sum_sq / (double)c + eps));
            for (uint32_t ch = 0; ch < c; ch++) {
                float val = scalar(x[at(a, px, ch)]) * inv;
                if (gamma)
                    val *= gamma[ch];
                if (beta)
                    val += beta[ch];
                out[at(b, px, ch)] = half(val);
            }
            for (uint32_t ch = c; ch < stride; ch++) {
                out[at(b, px, ch)] = 0;
            }
        }
    }
}

static void rope_fp16(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p;
    const uint32_t head_dim = p[0], num_heads = p[1], half_dim = head_dim / 2u;
    const uint32_t mode = p[3];

    uint32_t pos = p[2];
    if (o->y != HXFG_NONE) {
        const HxTensor *yt = &g->tensors[o->y];
        if (yt->reserved[0] & HX_TENSOR_PERSISTENT) {
            pos = (uint32_t)(*(const float *)((g->state ? g->state : g->arena) + yt->offset));
        } else {
            const uint16_t *yv = (const uint16_t *)tensor_data(g, o->y);
            pos = (uint32_t)scalar(yv[at(yt, 0, 0)]);
        }
    }

    const uint32_t obw = physical_width(b), opairs = b->h * obw / 2u;
    const uint32_t lo = opairs * part / parts, hi = opairs * (part + 1u) / parts;
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);

    if (b->w == 1u && head_dim == 64u && mode == 0 && (b->stride % 32u == 0)) {
        float cos_tab[32] __attribute__((aligned(128)));
        float sin_tab[32] __attribute__((aligned(128)));
        for (uint32_t j = 0; j < 32u; j++) {
            double theta = pow(10000.0, -2.0 * (double)j / 64.0);
            double angle = (double)pos * theta;
            cos_tab[j] = (float)cos(angle);
            sin_tab[j] = (float)sin(angle);
        }
        HVX_Vector cos_v = *(const HVX_Vector *)cos_tab;
        HVX_Vector sin_v = *(const HVX_Vector *)sin_tab;

        for (uint32_t pair = lo; pair < hi; pair++) {
            const uint32_t px = pair * 2u;
            for (uint32_t h = 0; h < num_heads; h++) {
                uint32_t ch0 = h * 64u;
                uint32_t ch1 = ch0 + 32u;
                HVX_Vector v0 = float_words(load(x, a, px, ch0));
                HVX_Vector v1 = float_words(load(x, a, px, ch1));

                HVX_Vector out0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(
                    vmulf(v0, cos_v), vmulf(v1, sin_v)));
                HVX_Vector out1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(
                    vmulf(v0, sin_v), vmulf(v1, cos_v)));

                *(HVX_Vector *)(out + at(b, px, ch0)) = half_words(out0);
                *(HVX_Vector *)(out + at(b, px, ch1)) = half_words(out1);
            }
            for (uint32_t ch = num_heads * head_dim; ch < b->c; ch++) {
                out[at(b, px, ch)] = x[at(a, px, ch)];
            }
            for (uint32_t ch = b->c; ch < b->stride; ch++) {
                out[at(b, px, ch)] = 0;
            }
        }
        return;
    }

    for (uint32_t pair = lo; pair < hi; pair++) {
        for (uint32_t q = 0; q < 2u; q++) {
            const uint32_t px = pair * 2u + q, ox = px % obw;
            if (ox >= b->w) {
                for (uint32_t ch = 0; ch < b->stride; ch++)
                    out[at(b, px, ch)] = 0;
                continue;
            }
            for (uint32_t h = 0; h < num_heads; h++) {
                uint32_t h_base = h * head_dim;
                for (uint32_t j = 0; j < half_dim; j++) {
                    double theta = pow(10000.0, -2.0 * (double)j / (double)head_dim);
                    double angle = (double)pos * theta;
                    float cos_v = (float)cos(angle);
                    float sin_v = (float)sin(angle);
                    if (mode == 0) {
                        float v0 = scalar(x[at(a, px, h_base + j)]);
                        float v1 = scalar(x[at(a, px, h_base + half_dim + j)]);
                        out[at(b, px, h_base + j)] = half(v0 * cos_v - v1 * sin_v);
                        out[at(b, px, h_base + half_dim + j)] = half(v0 * sin_v + v1 * cos_v);
                    } else {
                        float v0 = scalar(x[at(a, px, h_base + 2u * j)]);
                        float v1 = scalar(x[at(a, px, h_base + 2u * j + 1u)]);
                        out[at(b, px, h_base + 2u * j)] = half(v0 * cos_v - v1 * sin_v);
                        out[at(b, px, h_base + 2u * j + 1u)] = half(v0 * sin_v + v1 * cos_v);
                    }
                }
            }
            for (uint32_t ch = num_heads * head_dim; ch < b->c; ch++) {
                out[at(b, px, ch)] = x[at(a, px, ch)];
            }
            for (uint32_t ch = b->c; ch < b->stride; ch++) {
                out[at(b, px, ch)] = 0;
            }
        }
    }
}

static void shortconv_fp16(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *b = &g->tensors[o->out];
    const HxTensor *st = &g->tensors[o->y];
    float *s_buf = (float *)((g->state ? g->state : g->arena) + st->offset);
    const float *w = (const float *)(g->constants + o->p[1]);
    const float *bias = o->p[2] != HXFG_NONE ? (const float *)(g->constants + o->p[2]) : NULL;
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    const uint16_t *gate = o->z != HXFG_NONE ? (const uint16_t *)tensor_data(g, o->z) : NULL;
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);

    const uint32_t k_taps = o->p[0], s_len = k_taps - 1u, c = b->c;
    const uint32_t c0 = c * part / parts, c1 = c * (part + 1u) / parts;

    if ((c0 % 32u == 0) && (c1 % 32u == 0)) {
        for (uint32_t ch = c0; ch < c1; ch += 32u) {
            /* The bias is IEEE float; the accumulator is qf32, so convert it (a raw sf reads as ~0). */
            HVX_Vector acc = bias ? Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *)(bias + ch), Q6_V_vzero())
                                  : Q6_V_vzero();
            for (uint32_t j = 0; j < s_len; j++) {
                HVX_Vector sv = *(const HVX_UVector *)(s_buf + j * st->stride + ch);
                HVX_Vector wv = *(const HVX_UVector *)(w + j * b->stride + ch);
                acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(sv, wv));
            }
            HVX_Vector xv = float_words(load(x, &g->tensors[o->x], 0, ch));
            HVX_Vector w_last = *(const HVX_UVector *)(w + s_len * b->stride + ch);
            acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(xv, w_last));

            if (s_len > 1u) {
                for (uint32_t j = 0; j < s_len - 1u; j++) {
                    *(HVX_UVector *)(s_buf + j * st->stride + ch) =
                        *(const HVX_UVector *)(s_buf + (j + 1u) * st->stride + ch);
                }
            }
            if (s_len > 0u) {
                *(HVX_UVector *)(s_buf + (s_len - 1u) * st->stride + ch) = xv;
            }

            float acc_buf[32] __attribute__((aligned(128)));
            *(HVX_Vector *)acc_buf = Q6_Vsf_equals_Vqf32(acc);
            for (uint32_t i = 0; i < 32u; i++) {
                float val = acc_buf[i];
                if (o->p[3] == 1) val = fmaxf(0.0f, val);
                else if (o->p[3] == 2) val = fmaxf(-1.0f, fminf(1.0f, val));
                else if (o->p[3] == 3) val = val / (1.0f + expf(-val));
                if (gate)
                    val *= scalar(gate[at(&g->tensors[o->z], 0, ch + i)]);
                acc_buf[i] = val;
            }
            *(HVX_Vector *)(out + at(b, 0, ch)) = half_words(*(const HVX_Vector *)acc_buf);
        }
    } else {
        for (uint32_t d = c0; d < c1; d++) {
            double acc = bias ? (double)bias[d] : 0.0;
            for (uint32_t j = 0; j < s_len; j++) {
                acc += (double)s_buf[j * st->stride + d] * (double)w[j * b->stride + d];
            }
            float x_val = scalar(x[at(&g->tensors[o->x], 0, d)]);
            acc += (double)x_val * (double)w[s_len * b->stride + d];

            if (s_len > 1u) {
                for (uint32_t j = 0; j < s_len - 1u; j++) {
                    s_buf[j * st->stride + d] = s_buf[(j + 1u) * st->stride + d];
                }
            }
            if (s_len > 0u) {
                s_buf[(s_len - 1u) * st->stride + d] = x_val;
            }

            float val = (float)acc;
            if (o->p[3] == 1) val = fmaxf(0.0f, val);
            else if (o->p[3] == 2) val = fmaxf(-1.0f, fminf(1.0f, val));
            else if (o->p[3] == 3) val = val / (1.0f + expf(-val));

            if (gate)
                val *= scalar(gate[at(&g->tensors[o->z], 0, d)]);
            out[at(b, 0, d)] = half(val);
        }
    }
    if (part == 0) {
        for (uint32_t d = c; d < b->stride; d++)
            out[at(b, 0, d)] = 0;
        for (uint32_t d = 0; d < b->stride; d++)
            out[at(b, 1, d)] = 0;
    }
}

static void linear_q_fp16(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *b = &g->tensors[o->out];
    const uint32_t fmt = o->p[0], k = o->p[3], n = o->p[4];
    const float *bias = o->p[2] != HXFG_NONE ? (const float *)(g->constants + o->p[2]) : NULL;
    const uint32_t blocks = k / 32u;
    const uint32_t r0 = n * part / parts, r1 = n * (part + 1u) / parts;

    const uint32_t slice = (g->fp16->workspace_bytes / parts) & ~2047u;
    uint8_t *ws = g->fp16->workspace + part * slice;
    float *xf = (float *)ws;

    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);

    if (k * 4u <= slice) {
        for (uint32_t b_idx = 0; b_idx < blocks; b_idx++) {
            HVX_Vector in_h = load(x, &g->tensors[o->x], 0, b_idx * 32u);
            *(HVX_Vector *)(xf + b_idx * 32u) = float_words(in_h);
        }
        for (uint32_t rem = blocks * 32u; rem < k; rem++) {
            xf[rem] = scalar(x[at(&g->tensors[o->x], 0, rem)]);
        }
    }

    for (uint32_t r = r0; r < r1; r++) {
        float sum = bias ? bias[r] : 0.0f;
        if (fmt == 0) {
            const uint8_t *w_row = (const uint8_t *)(g->constants + o->p[1]) + (uint64_t)r * blocks * 18u;
            for (uint32_t b_idx = 0; b_idx < blocks; b_idx++) {
                const uint8_t *blk = w_row + b_idx * 18u;
                const uint16_t d16 = (uint16_t)blk[16] | ((uint16_t)blk[17] << 8);
                const float scale = scalar(d16);
                float blk_acc = 0.0f;
                if (k * 4u <= slice) {
                    const float *x_blk = xf + b_idx * 32u;
                    for (uint32_t j = 0; j < 16u; j++) {
                        const uint8_t byte = blk[j];
                        const int32_t q0 = (int32_t)(byte & 0x0fu) - 8;
                        const int32_t q1 = (int32_t)(byte >> 4u) - 8;
                        blk_acc += x_blk[2u * j] * (float)q0 + x_blk[2u * j + 1u] * (float)q1;
                    }
                } else {
                    for (uint32_t j = 0; j < 16u; j++) {
                        const uint8_t byte = blk[j];
                        const int32_t q0 = (int32_t)(byte & 0x0fu) - 8;
                        const int32_t q1 = (int32_t)(byte >> 4u) - 8;
                        float x0 = scalar(x[at(&g->tensors[o->x], 0, b_idx * 32u + 2u * j)]);
                        float x1 = scalar(x[at(&g->tensors[o->x], 0, b_idx * 32u + 2u * j + 1u)]);
                        blk_acc += x0 * (float)q0 + x1 * (float)q1;
                    }
                }
                sum += blk_acc * scale;
            }
        } else {
            const uint8_t *w_row = (const uint8_t *)(g->constants + o->p[1]) + (uint64_t)r * blocks * 34u;
            for (uint32_t b_idx = 0; b_idx < blocks; b_idx++) {
                const uint8_t *blk = w_row + b_idx * 34u;
                const uint16_t d16 = (uint16_t)blk[32] | ((uint16_t)blk[33] << 8);
                const float scale = scalar(d16);
                float blk_acc = 0.0f;
                if (k * 4u <= slice) {
                    const float *x_blk = xf + b_idx * 32u;
                    for (uint32_t j = 0; j < 32u; j++) {
                        const int8_t q = (int8_t)blk[j];
                        blk_acc += x_blk[j] * (float)q;
                    }
                } else {
                    for (uint32_t j = 0; j < 32u; j++) {
                        const int8_t q = (int8_t)blk[j];
                        float xj = scalar(x[at(&g->tensors[o->x], 0, b_idx * 32u + j)]);
                        blk_acc += xj * (float)q;
                    }
                }
                sum += blk_acc * scale;
            }
        }
        float val = sum;
        if (o->p[5] == 1) val = fmaxf(0.0f, val);
        else if (o->p[5] == 2) val = fmaxf(-1.0f, fminf(1.0f, val));
        else if (o->p[5] == 3) val = val / (1.0f + expf(-val));
        out[at(b, 0, r)] = half(val);
    }
    if (part == 0) {
        for (uint32_t r = n; r < b->stride; r++)
            out[at(b, 0, r)] = 0;
        for (uint32_t r = 0; r < b->stride; r++)
            out[at(b, 1, r)] = 0;
    }
}

static void kv_write_fp16(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *xt = &g->tensors[o->x];
    const HxTensor *outt = &g->tensors[o->out];
    uint32_t pos = o->p[0];
    if (o->y != HXFG_NONE) {
        const HxTensor *yt = &g->tensors[o->y];
        if (yt->reserved[0] & HX_TENSOR_PERSISTENT) {
            pos = (uint32_t)(*(const float *)((g->state ? g->state : g->arena) + yt->offset));
        } else {
            const uint16_t *yv = (const uint16_t *)tensor_data(g, o->y);
            pos = (uint32_t)scalar(yv[at(yt, 0, 0)]);
        }
    }
    uint32_t tokens = xt->h * xt->w;
    uint32_t max_tokens = outt->h * outt->w;
    if (pos >= max_tokens)
        return;
    if (pos + tokens > max_tokens)
        tokens = max_tokens - pos;
    uint32_t c = xt->c;
    uint32_t c0 = c * part / parts, c1 = c * (part + 1u) / parts;

    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    float *out = (float *)((g->state ? g->state : g->arena) + outt->offset);

    for (uint32_t t = 0; t < tokens; t++) {
        uint32_t ty = t / xt->w, tx = t % xt->w;
        uint32_t px = ty * physical_width(xt) + tx;
        float *dst = out + (pos + t) * outt->stride;

        if (!(px & 1u) && (c0 % 32u == 0) && (c1 % 32u == 0)) {
            for (uint32_t ch = c0; ch < c1; ch += 32u) {
                HVX_Vector v = float_words(load(x, xt, px, ch));
                *(HVX_UVector *)(dst + ch) = v;
            }
        } else {
            for (uint32_t ch = c0; ch < c1; ch++) {
                dst[ch] = scalar(x[at(xt, px, ch)]);
            }
        }
    }
    if (part == 0 && outt->stride > c) {
        for (uint32_t t = 0; t < tokens; t++) {
            float *dst = out + (pos + t) * outt->stride;
            for (uint32_t ch = c; ch < outt->stride; ch++)
                dst[ch] = 0.0f;
        }
    }
}

/* HX_MATMUL in scalar lanes. Accumulates in float32 and converts to half. */
static void matmul_fp16_scalar(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out], *yt = &g->tensors[o->y];
    const uint16_t *xv = (const uint16_t *)tensor_data(g, o->x);
    const uint16_t *yv = (const uint16_t *)tensor_data(g, o->y);
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);

    uint32_t Ba = 1, Ma = 0, Ka = 0;
    if (a->w > 1) {
        Ba = a->h;
        Ma = o->p[3] ? a->c : a->w;
        Ka = o->p[3] ? a->w : a->c;
    } else {
        uint32_t b_param = o->p[2] > 1 ? o->p[2] : 1;
        Ba = b_param;
        Ma = o->p[3] ? a->c : (a->h / b_param);
        Ka = o->p[3] ? (a->h / b_param) : a->c;
    }

    uint32_t Bb = 1, Kb = 0, Nb = 0;
    if (yt->w > 1) {
        Bb = yt->h;
        Kb = o->p[0] ? yt->c : yt->w;
        Nb = o->p[0] ? yt->w : yt->c;
    } else {
        if (Ba > 1 && yt->h % Ba == 0) {
            Bb = Ba;
            Kb = o->p[0] ? yt->c : (yt->h / Ba);
            Nb = o->p[0] ? (yt->h / Ba) : yt->c;
        } else {
            Bb = 1;
            Kb = o->p[0] ? yt->c : yt->h;
            Nb = o->p[0] ? yt->h : yt->c;
        }
    }

    float scale = 1.0f;
    if (o->p[1])
        memcpy(&scale, &o->p[1], sizeof scale);

    const uint32_t transB = o->p[0], transA = o->p[3];
    const uint32_t pw_a = physical_width(a), pw_b = physical_width(b), pw_y = physical_width(yt);
    const uint32_t total_rows = Ba * Ma;
    const uint32_t r0 = total_rows * part / parts, r1 = total_rows * (part + 1u) / parts;

    for (uint32_t r = r0; r < r1; r++) {
        uint32_t batch_idx = r / Ma;
        uint32_t m_idx = r % Ma;
        uint32_t b_batch = (Bb == 1) ? 0 : batch_idx;

        uint32_t out_y = (b->w > 1) ? batch_idx : r;
        uint32_t out_x = (b->w > 1) ? m_idx : 0;
        uint32_t out_px = out_y * pw_b + out_x;

        for (uint32_t n = 0; n < b->stride; n++) {
            if (n >= Nb) {
                out[at(b, out_px, n)] = 0;
                continue;
            }
            float acc = 0.0f;
            if (!transA && !transB) {
                uint32_t a_y = (a->w > 1) ? batch_idx : (batch_idx * Ma + m_idx);
                uint32_t a_x = (a->w > 1) ? m_idx : 0;
                uint32_t a_px = a_y * pw_a + a_x;
                for (uint32_t k = 0; k < Ka; k++) {
                    uint32_t y_y = (yt->w > 1) ? b_batch : (b_batch * Kb + k);
                    uint32_t y_x = (yt->w > 1) ? k : 0;
                    acc += scalar(xv[at(a, a_px, k)]) * scalar(yv[at(yt, y_y * pw_y + y_x, n)]);
                }
            } else if (!transA && transB) {
                uint32_t a_y = (a->w > 1) ? batch_idx : (batch_idx * Ma + m_idx);
                uint32_t a_x = (a->w > 1) ? m_idx : 0;
                uint32_t a_px = a_y * pw_a + a_x;
                uint32_t y_y = (yt->w > 1) ? b_batch : (b_batch * Nb + n);
                uint32_t y_x = (yt->w > 1) ? n : 0;
                uint32_t y_px = y_y * pw_y + y_x;
                for (uint32_t k = 0; k < Ka; k++)
                    acc += scalar(xv[at(a, a_px, k)]) * scalar(yv[at(yt, y_px, k)]);
            } else if (transA && !transB) {
                for (uint32_t k = 0; k < Ka; k++) {
                    uint32_t a_y = (a->w > 1) ? batch_idx : (batch_idx * Ka + k);
                    uint32_t a_x = (a->w > 1) ? k : 0;
                    uint32_t y_y = (yt->w > 1) ? b_batch : (b_batch * Kb + k);
                    uint32_t y_x = (yt->w > 1) ? k : 0;
                    acc += scalar(xv[at(a, a_y * pw_a + a_x, m_idx)]) *
                           scalar(yv[at(yt, y_y * pw_y + y_x, n)]);
                }
            } else {
                uint32_t y_y = (yt->w > 1) ? b_batch : (b_batch * Nb + n);
                uint32_t y_x = (yt->w > 1) ? n : 0;
                uint32_t y_px = y_y * pw_y + y_x;
                for (uint32_t k = 0; k < Ka; k++) {
                    uint32_t a_y = (a->w > 1) ? batch_idx : (batch_idx * Ka + k);
                    uint32_t a_x = (a->w > 1) ? k : 0;
                    acc += scalar(xv[at(a, a_y * pw_a + a_x, m_idx)]) * scalar(yv[at(yt, y_px, k)]);
                }
            }
            out[at(b, out_px, n)] = half(acc * scale);
        }
    }
}

/* HX_SOFTMAX / HX_ATTN_SOFTMAX in scalar lanes. */
static void softmax_fp16_scalar(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);
    const HxTensor *mask_t = (o->y != HXFG_NONE) ? &g->tensors[o->y] : NULL;
    const uint16_t *mask_v = mask_t ? (const uint16_t *)tensor_data(g, o->y) : NULL;

    const uint32_t pw_a = physical_width(a), pw_b = physical_width(b);
    const uint32_t pw_m = mask_t ? physical_width(mask_t) : 0;
    const uint32_t causal = o->p[1];
    const uint32_t M = b->w > 1 ? b->w : (o->p[2] > 0 ? o->p[2] : b->h);
    const uint32_t axis = (o->kind == HX_ATTN_SOFTMAX) ? o->p[0] : 2;

    if (axis == 1) {
        if (part == 0) {
            for (uint32_t oy = 0; oy < b->h; oy++) {
                for (uint32_t c = 0; c < b->c; c++) {
                    float peak = -FLT_MAX, sum = 0.0f;
                    for (uint32_t ox = 0; ox < b->w; ox++) {
                        float v = scalar(x[at(a, oy * pw_a + ox, c)]);
                        if (v > peak) peak = v;
                    }
                    for (uint32_t ox = 0; ox < b->w; ox++) {
                        float v = expf(scalar(x[at(a, oy * pw_a + ox, c)]) - peak);
                        sum += v;
                    }
                    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                    for (uint32_t ox = 0; ox < b->w; ox++) {
                        float v = expf(scalar(x[at(a, oy * pw_a + ox, c)]) - peak);
                        out[at(b, oy * pw_b + ox, c)] = half(v * inv);
                    }
                }
                for (uint32_t ox = 0; ox < b->w; ox++) {
                    for (uint32_t c = b->c; c < b->stride; c++)
                        out[at(b, oy * pw_b + ox, c)] = 0;
                }
            }
        }
        return;
    } else if (axis == 0) {
        if (part == 0) {
            for (uint32_t ox = 0; ox < b->w; ox++) {
                for (uint32_t c = 0; c < b->c; c++) {
                    float peak = -FLT_MAX, sum = 0.0f;
                    for (uint32_t oy = 0; oy < b->h; oy++) {
                        float v = scalar(x[at(a, oy * pw_a + ox, c)]);
                        if (v > peak) peak = v;
                    }
                    for (uint32_t oy = 0; oy < b->h; oy++) {
                        float v = expf(scalar(x[at(a, oy * pw_a + ox, c)]) - peak);
                        sum += v;
                    }
                    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                    for (uint32_t oy = 0; oy < b->h; oy++) {
                        float v = expf(scalar(x[at(a, oy * pw_a + ox, c)]) - peak);
                        out[at(b, oy * pw_b + ox, c)] = half(v * inv);
                    }
                }
            }
            for (uint32_t oy = 0; oy < b->h; oy++) {
                for (uint32_t ox = 0; ox < b->w; ox++) {
                    for (uint32_t c = b->c; c < b->stride; c++)
                        out[at(b, oy * pw_b + ox, c)] = 0;
                }
            }
        }
        return;
    }

    const uint32_t pixels = b->h * b->w;
    const uint32_t lo = pixels * part / parts, hi = pixels * (part + 1u) / parts;

    for (uint32_t i = lo; i < hi; i++) {
        uint32_t oy = i / b->w, ox = i % b->w;
        uint32_t px = oy * pw_b + ox;
        uint32_t q = M > 0 ? (i % M) : 0;

        uint32_t my = mask_t ? (mask_t->h == 1 ? 0 : oy) : 0;
        uint32_t mx = mask_t ? (mask_t->w == 1 ? 0 : ox) : 0;
        uint32_t m_px = mask_t ? (my * pw_m + mx) : 0;

        float peak = -FLT_MAX, sum = 0.0f;
        for (uint32_t c = 0; c < b->c; c++) {
            if (causal && c > q)
                continue;
            float v = scalar(x[at(a, px, c)]);
            if (mask_v)
                v += scalar(mask_v[at(mask_t, m_px, mask_t->c == 1 ? 0 : c)]);
            if (v > peak)
                peak = v;
        }
        if (peak == -FLT_MAX)
            peak = 0.0f;

        for (uint32_t c = 0; c < b->c; c++) {
            if (causal && c > q) {
                out[at(b, px, c)] = 0;
            } else {
                float v = scalar(x[at(a, px, c)]);
                if (mask_v)
                    v += scalar(mask_v[at(mask_t, m_px, mask_t->c == 1 ? 0 : c)]);
                sum += expf(v - peak);
            }
        }
        float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (uint32_t c = 0; c < b->c; c++) {
            if (causal && c > q) {
                out[at(b, px, c)] = 0;
            } else {
                float v = scalar(x[at(a, px, c)]);
                if (mask_v)
                    v += scalar(mask_v[at(mask_t, m_px, mask_t->c == 1 ? 0 : c)]);
                out[at(b, px, c)] = half(expf(v - peak) * inv_sum);
            }
        }
        for (uint32_t c = b->c; c < b->stride; c++)
            out[at(b, px, c)] = 0;
    }
}

/* HX_SDPA in scalar lanes. */
static void sdpa_fp16_scalar(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *q_t = &g->tensors[o->x], *k_t = &g->tensors[o->y], *v_t = &g->tensors[o->z], *b = &g->tensors[o->out];
    const uint16_t *qv = (const uint16_t *)tensor_data(g, o->x);
    const uint16_t *kv = (const uint16_t *)tensor_data(g, o->y);
    const uint16_t *vv = (const uint16_t *)tensor_data(g, o->z);
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);

    const HxTensor *mask_t = (o->p[0] != HXFG_NONE) ? &g->tensors[o->p[0]] : NULL;
    const uint16_t *mask_v = mask_t ? (const uint16_t *)tensor_data(g, o->p[0]) : NULL;

    uint32_t Bq = 1, Lq = 0, Dq = q_t->c;
    if (q_t->w > 1) { Bq = q_t->h; Lq = q_t->w; }
    else { uint32_t b_param = o->p[3] > 1 ? o->p[3] : 1; Bq = b_param; Lq = q_t->h / b_param; }

    uint32_t Bk = 1, Lk = 0;
    if (k_t->w > 1) { Bk = k_t->h; Lk = k_t->w; }
    else { Bk = (Bq > 1 && k_t->h % Bq == 0) ? Bq : 1; Lk = (Bk > 1) ? (k_t->h / Bk) : k_t->h; }

    uint32_t Bv = (v_t->w > 1) ? v_t->h : ((Bq > 1 && v_t->h % Bq == 0) ? Bq : 1);
    uint32_t Dv = b->c;

    float scale = 1.0f / sqrtf((float)Dq);
    if (o->p[1])
        memcpy(&scale, &o->p[1], sizeof scale);
    const uint32_t causal = o->p[2];

    const uint32_t pw_q = physical_width(q_t), pw_k = physical_width(k_t),
                   pw_v = physical_width(v_t), pw_b = physical_width(b);
    const uint32_t pw_m = mask_t ? physical_width(mask_t) : 0;

    const uint32_t pixels = b->h * b->w;
    const uint32_t lo = pixels * part / parts, hi = pixels * (part + 1u) / parts;

    float scores[4096];
    for (uint32_t i = lo; i < hi; i++) {
        uint32_t b_idx = (b->w > 1) ? (i / b->w) : (Bq > 1 ? (i / Lq) : 0);
        uint32_t q = (b->w > 1) ? (i % b->w) : (Bq > 1 ? (i % Lq) : i);
        uint32_t k_batch = (Bk == 1) ? 0 : b_idx;
        uint32_t v_batch = (Bv == 1) ? 0 : b_idx;

        uint32_t q_y = (q_t->w > 1) ? b_idx : (b_idx * Lq + q);
        uint32_t q_x = (q_t->w > 1) ? q : 0;
        uint32_t q_px = q_y * pw_q + q_x;

        uint32_t out_y = i / b->w, out_x = i % b->w;
        uint32_t out_px = out_y * pw_b + out_x;

        uint32_t mb = mask_t ? (mask_t->h == 1 ? 0 : b_idx) : 0;
        uint32_t mq = mask_t ? (mask_t->w == 1 ? 0 : q) : 0;
        uint32_t m_px = mask_t ? (mb * pw_m + mq) : 0;

        float peak = -FLT_MAX, sum = 0.0f;
        for (uint32_t k = 0; k < Lk; k++) {
            if (causal && k > q) {
                scores[k] = -FLT_MAX;
                continue;
            }
            uint32_t k_y = (k_t->w > 1) ? k_batch : (k_batch * Lk + k);
            uint32_t k_x = (k_t->w > 1) ? k : 0;
            uint32_t k_px = k_y * pw_k + k_x;

            float dot = 0.0f;
            for (uint32_t d = 0; d < Dq; d++)
                dot += scalar(qv[at(q_t, q_px, d)]) * scalar(kv[at(k_t, k_px, d)]);
            dot *= scale;
            if (mask_v)
                dot += scalar(mask_v[at(mask_t, m_px, mask_t->c == 1 ? 0 : k)]);
            scores[k] = dot;
            if (dot > peak)
                peak = dot;
        }
        if (peak == -FLT_MAX)
            peak = 0.0f;

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
                    uint32_t v_y = (v_t->w > 1) ? v_batch : (v_batch * Lk + k);
                    uint32_t v_x = (v_t->w > 1) ? k : 0;
                    uint32_t v_px = v_y * pw_v + v_x;
                    acc += scores[k] * scalar(vv[at(v_t, v_px, d)]);
                }
            }
            out[at(b, out_px, d)] = half(acc);
        }
        for (uint32_t d = Dv; d < b->stride; d++)
            out[at(b, out_px, d)] = 0;
    }
}

static inline int32_t get_index_val(const HxGraph *g, const HxTensor *t, const uint16_t *t_fp16,
                                    uint32_t elem_idx, int int_dtype) {
    if (t->dtype == HX_DTYPE_INT32) {
        const int32_t *t_int = (const int32_t *)tensor_data(g, (uint32_t)(t - g->tensors));
        return t_int[elem_idx];
    }
    const uint32_t tc = t->c ? t->c : 1;
    const uint32_t px = elem_idx / tc, ch = elem_idx % tc;
    const uint32_t pw = physical_width(t);
    const uint32_t py = t->w ? px / t->w : 0, pxx = t->w ? px % t->w : 0;
    uint16_t raw = t_fp16[at(t, py * pw + pxx, ch)];
    if (int_dtype == 0) {
        float f = scalar(raw);
        int32_t idx = (int32_t)lrintf(f);
        if (idx == 0 && raw != 0 && raw < 0x3c00)
            idx = (int32_t)(int16_t)raw;
        return idx;
    } else {
        return (int32_t)lrintf(scalar(raw));
    }
}

static void gather_fp16_scalar(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *a = &g->tensors[o->x]; // indices or data
    const HxTensor *b = &g->tensors[o->out]; // out
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    const uint32_t *p = o->p;
    const uint32_t vocab = p[2];
    const int int_indices = (p[3] == 0) || (p[0] == 2) || (a->dtype == HX_DTYPE_INT32);
    const uint32_t n_indices = (p[0] == 2) ? p[9] : a->h * a->w * a->c;
    const uint32_t axis = p[4];
    const uint32_t obw = physical_width(b);
    const int32_t *const_idx = (p[0] == 2) ? (const int32_t *)(g->constants + p[8]) : NULL;

    if (axis == 0 && (p[5] == 0 || p[5] == b->c)) {
        const uint32_t stride_tbl = p[7] ? p[7] : (p[0] == 1 ? b->stride : (p[0] == 2 ? a->stride : g->tensors[o->y].stride));
        const float *const_tbl = (p[0] == 1) ? ((const float *)(g->constants + p[1])) : NULL;
        const HxTensor *yt = (p[0] == 1) ? NULL : ((p[0] == 2) ? a : &g->tensors[o->y]);
        const uint16_t *ytbl = (p[0] == 1) ? NULL : ((p[0] == 2) ? x : (const uint16_t *)tensor_data(g, o->y));
        const uint32_t y_pw = yt ? physical_width(yt) : 0;

        const uint32_t pixels = b->h * b->w;
        const uint32_t lo = pixels * part / parts, hi = pixels * (part + 1u) / parts;

        for (uint32_t i = lo; i < hi; i++) {
            uint32_t oy = i / b->w, ox = i % b->w;
            uint32_t out_px = oy * obw + ox;
            int32_t idx = const_idx ? const_idx[i % n_indices] : get_index_val(g, a, x, i, int_indices ? 0 : 1);
            if (idx < 0) idx += (int32_t)vocab;
            idx = idx < 0 ? 0 : (idx >= (int32_t)vocab ? (int32_t)vocab - 1 : idx);

            if (p[0] == 1) {
                const float *row = const_tbl + (uint32_t)idx * stride_tbl;
                for (uint32_t c = 0; c < b->stride; c++)
                    out[at(b, out_px, c)] = c < b->c ? half(row[c]) : 0;
            } else {
                uint32_t y_py = yt->w ? (uint32_t)idx / yt->w : 0;
                uint32_t y_px = yt->w ? (uint32_t)idx % yt->w : 0;
                for (uint32_t c = 0; c < b->stride; c++)
                    out[at(b, out_px, c)] = c < b->c ? ytbl[at(yt, y_py * y_pw + y_px, c)] : 0;
            }
        }
    } else {
        const uint32_t inner = p[5] ? p[5] : (axis == 2 ? 1 : b->c);
        const uint32_t outer = p[6] ? p[6] : 1;
        const uint32_t stride_tbl = p[7] ? p[7] : (p[0] == 1 ? inner : (p[0] == 2 ? a->stride : g->tensors[o->y].stride));
        const float *const_tbl = (p[0] == 1) ? ((const float *)(g->constants + p[1])) : NULL;
        const HxTensor *yt = (p[0] == 1) ? NULL : ((p[0] == 2) ? a : &g->tensors[o->y]);
        const uint16_t *ytbl = (p[0] == 1) ? NULL : ((p[0] == 2) ? x : (const uint16_t *)tensor_data(g, o->y));
        const uint32_t y_pw = yt ? physical_width(yt) : 0;

        const uint32_t total_tasks = outer * n_indices;
        const uint32_t lo = total_tasks * part / parts, hi = total_tasks * (part + 1u) / parts;

        for (uint32_t task = lo; task < hi; task++) {
            uint32_t o_idx = task / n_indices;
            uint32_t j = task % n_indices;
            int32_t idx = const_idx ? const_idx[j] : get_index_val(g, a, x, j, int_indices ? 0 : 1);
            if (idx < 0) idx += (int32_t)vocab;
            idx = idx < 0 ? 0 : (idx >= (int32_t)vocab ? (int32_t)vocab - 1 : idx);

            if (axis == 2 && inner == 1) {
                uint32_t out_px = o_idx * obw;
                uint16_t v = 0;
                if (p[0] == 1) {
                    uint32_t src_pos = o_idx * stride_tbl + (uint32_t)idx;
                    v = half(const_tbl[src_pos]);
                } else {
                    uint32_t in_px = o_idx * y_pw;
                    v = ytbl[at(yt, in_px, (uint32_t)idx)];
                }
                out[at(b, out_px, j)] = v;
            } else {
                uint32_t out_pixel = o_idx * n_indices + j;
                uint32_t dst_py = b->w ? out_pixel / b->w : 0, dst_ox = b->w ? out_pixel % b->w : 0;
                uint32_t out_px = dst_py * obw + dst_ox;
                for (uint32_t k = 0; k < b->stride; k++) {
                    uint16_t v = 0;
                    if (k < inner) {
                        if (p[0] == 1) {
                            const float *src_row = const_tbl + (o_idx * vocab + (uint32_t)idx) * stride_tbl;
                            v = half(src_row[k]);
                        } else {
                            uint32_t src_elem = (o_idx * vocab + (uint32_t)idx) * yt->stride + k;
                            uint32_t s_px = yt->c ? src_elem / yt->c : 0, s_c = yt->c ? src_elem % yt->c : 0;
                            uint32_t s_py = yt->w ? s_px / yt->w : 0, s_ox = yt->w ? s_px % yt->w : 0;
                            v = ytbl[at(yt, s_py * y_pw + s_ox, s_c)];
                        }
                    }
                    out[at(b, out_px, k)] = v;
                }
            }
        }
    }
}

static void gather_elements_fp16_scalar(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *dt = &g->tensors[o->x]; // data
    const HxTensor *ot = &g->tensors[o->out]; // out
    const uint16_t *data = (const uint16_t *)tensor_data(g, o->x);
    const int const_indices = (o->p[2] == 2);
    const HxTensor *it = const_indices ? NULL : &g->tensors[o->y];
    const uint16_t *indices_fp16 = const_indices ? NULL : (const uint16_t *)tensor_data(g, o->y);
    const int32_t *const_idx = const_indices ? (const int32_t *)(g->constants + o->p[3]) : NULL;
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);
    const uint32_t *p = o->p;
    const uint32_t axis = p[0];
    const int int_indices = (p[1] == 0) || const_indices ||
                            (!const_indices && (it->dtype == HX_DTYPE_INT32));
    const float *data_const = (p[2] == 1) ? (const float *)(g->constants + p[3]) : NULL;

    const uint32_t obw = physical_width(ot);
    const uint32_t dpw = physical_width(dt);
    const uint32_t pixels = ot->h * ot->w;
    const uint32_t lo = pixels * part / parts, hi = pixels * (part + 1u) / parts;

    for (uint32_t px = lo; px < hi; px++) {
        uint32_t oy = px / ot->w, ox = px % ot->w;
        uint32_t out_px = oy * obw + ox;
        for (uint32_t c = 0; c < ot->stride; c++) {
            if (c >= ot->c) {
                out[at(ot, out_px, c)] = 0;
                continue;
            }
            int32_t k = 0;
            if (const_indices) {
                k = const_idx[px * ot->c + c];
            } else {
                uint32_t elem_idx = (oy * it->w + ox) * it->c + c;
                k = get_index_val(g, it, indices_fp16, elem_idx, int_indices ? 0 : 1);
            }
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
            uint16_t v = 0;
            if (data_const) {
                uint32_t src_loc = (sy * dt->w + sx) * dt->stride + sc;
                v = half(data_const[src_loc]);
            } else {
                v = data[at(dt, sy * dpw + sx, sc)];
            }
            out[at(ot, out_px, c)] = v;
        }
    }
}

static void gather_nd_fp16_scalar(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *dt = &g->tensors[o->x]; // data
    const HxTensor *ot = &g->tensors[o->out]; // out
    const uint16_t *data = (const uint16_t *)tensor_data(g, o->x);
    const int const_indices = (o->p[2] == 2);
    const HxTensor *it = const_indices ? NULL : &g->tensors[o->y];
    const uint16_t *indices_fp16 = const_indices ? NULL : (const uint16_t *)tensor_data(g, o->y);
    const int32_t *const_idx = const_indices ? (const int32_t *)(g->constants + o->p[3]) : NULL;
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);
    const uint32_t *p = o->p;
    const int int_indices = (p[1] == 0) || const_indices ||
                            (!const_indices && (it->dtype == HX_DTYPE_INT32));
    const float *data_const = (p[2] == 1) ? (const float *)(g->constants + p[3]) : NULL;
    const uint32_t k = p[4] ? p[4] : (const_indices ? 1 : it->c);
    const uint32_t slice_size = p[5] ? p[5] : (k == 1 ? dt->w * dt->c : (k == 2 ? dt->c : 1));
    const uint32_t total_elements = const_indices ? (ot->h * ot->w * ot->c / slice_size * k) : (it->h * it->w * it->c);
    const uint32_t M = k ? total_elements / k : 0;
    const uint32_t dpw = physical_width(dt);
    const uint32_t opw = physical_width(ot);

    const uint32_t lo = M * part / parts, hi = M * (part + 1u) / parts;

    for (uint32_t m = lo; m < hi; m++) {
        int32_t coords[3] = {0, 0, 0};
        for (uint32_t d = 0; d < k && d < 3; d++) {
            uint32_t elem_idx = m * k + d;
            coords[d] = const_indices ? const_idx[elem_idx] : get_index_val(g, it, indices_fp16, elem_idx, int_indices ? 0 : 1);
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
                uint32_t dst_pixel = m * dt->w + x;
                uint32_t dy = ot->w ? dst_pixel / ot->w : 0, dx = ot->w ? dst_pixel % ot->w : 0;
                for (uint32_t c = 0; c < ot->stride; c++) {
                    uint16_t v = 0;
                    if (c < dt->c) {
                        if (data_const)
                            v = half(data_const[(y * dt->w + x) * dt->stride + c]);
                        else
                            v = data[at(dt, y * dpw + x, c)];
                    }
                    out[at(ot, dy * opw + dx, c)] = v;
                }
            }
        } else if (k == 2) {
            uint32_t dy = ot->w ? m / ot->w : 0, dx = ot->w ? m % ot->w : 0;
            for (uint32_t c = 0; c < ot->stride; c++) {
                uint16_t v = 0;
                if (c < ot->c) {
                    if (data_const)
                        v = half(data_const[((uint32_t)sy * dt->w + (uint32_t)sx) * dt->stride + c]);
                    else
                        v = data[at(dt, (uint32_t)sy * dpw + (uint32_t)sx, c)];
                }
                out[at(ot, dy * opw + dx, c)] = v;
            }
        } else {
            uint16_t val = 0;
            if (data_const)
                val = half(data_const[((uint32_t)sy * dt->w + (uint32_t)sx) * dt->stride + (uint32_t)sc]);
            else
                val = data[at(dt, (uint32_t)sy * dpw + (uint32_t)sx, (uint32_t)sc)];
            uint32_t dst_px = ot->c ? m / ot->c : 0, dst_c = ot->c ? m % ot->c : 0;
            uint32_t dy = ot->w ? dst_px / ot->w : 0, dx = ot->w ? dst_px % ot->w : 0;
            out[at(ot, dy * opw + dx, dst_c)] = val;
        }
    }
}

static void scatter_nd_fp16_scalar(const HxGraph *g, const HxOp *o, uint32_t part, uint32_t parts) {
    const HxTensor *dt = &g->tensors[o->x]; // data
    const HxTensor *ot = &g->tensors[o->out]; // out
    const uint16_t *data = (const uint16_t *)tensor_data(g, o->x);
    const uint32_t *p = o->p;
    const int in_place = (p[0] == 1) || (o->out == o->x);
    const int const_indices = (p[5] & 1) != 0;
    const int const_updates = (p[5] & 2) != 0;
    const HxTensor *it = const_indices ? NULL : &g->tensors[o->y];
    const uint16_t *indices_fp16 = const_indices ? NULL : (const uint16_t *)tensor_data(g, o->y);
    const int32_t *const_idx = const_indices ? (const int32_t *)(g->constants + p[6]) : NULL;
    const HxTensor *ut = const_updates ? NULL : &g->tensors[o->z];
    const uint16_t *updates = const_updates ? NULL : (const uint16_t *)tensor_data(g, o->z);
    const float *upd_const = const_updates ? (const float *)(g->constants + p[7]) : NULL;
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);
    const int int_indices = (p[1] == 0) || const_indices ||
                            (!const_indices && (it->dtype == HX_DTYPE_INT32));
    const uint32_t k = p[2] ? p[2] : (const_indices ? 1 : it->c);
    const uint32_t slice_size = p[3] ? p[3] : (k == 1 ? ot->w * ot->c : (k == 2 ? ot->c : 1));
    const uint32_t reduction = p[4];
    const uint32_t obw = physical_width(ot);
    const uint32_t dpw = physical_width(dt);
    const uint32_t upw = ut ? physical_width(ut) : 0;

    if (!in_place) {
        const uint32_t pixels = ot->h * ot->w;
        const uint32_t lo = pixels * part / parts, hi = pixels * (part + 1u) / parts;
        for (uint32_t px = lo; px < hi; px++) {
            uint32_t oy = px / ot->w, ox = px % ot->w;
            uint32_t out_px = oy * obw + ox;
            uint32_t in_px = oy * dpw + ox;
            for (uint32_t c = 0; c < ot->stride; c++)
                out[at(ot, out_px, c)] = c < ot->c ? data[at(dt, in_px, c)] : 0;
        }
    }

    if (part + 1 == parts) {
        uint32_t M = p[8];
        if (!M) {
            if (const_indices) {
                uint32_t total_u = const_updates ? (p[3] ? slice_size * 2 : slice_size) : (ut->h * ut->w * ut->c);
                M = slice_size ? total_u / slice_size : 0;
            } else {
                const uint32_t total_elements = it->h * it->w * it->c;
                M = k ? total_elements / k : 0;
            }
        }
        for (uint32_t m = 0; m < M; m++) {
            int32_t coords[3] = {0, 0, 0};
            for (uint32_t d = 0; d < k && d < 3; d++) {
                uint32_t elem_idx = m * k + d;
                coords[d] = const_indices ? const_idx[elem_idx] : get_index_val(g, it, indices_fp16, elem_idx, int_indices ? 0 : 1);
            }
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
                    for (uint32_t c = 0; c < ot->c; c++) {
                        float u = 0.0f;
                        if (const_updates) {
                            u = upd_const[(m * ot->w + x) * ot->c + c];
                        } else {
                            uint32_t up_pixel = m * ot->w + x;
                            uint32_t uy = ut->w ? up_pixel / ut->w : 0, ux = ut->w ? up_pixel % ut->w : 0;
                            u = scalar(updates[at(ut, uy * upw + ux, c)]);
                        }
                        float cur = scalar(out[at(ot, y * obw + x, c)]);
                        float res = u;
                        if (reduction == 1) res = cur + u;
                        else if (reduction == 2) res = cur * u;
                        else if (reduction == 3) res = fmaxf(cur, u);
                        else if (reduction == 4) res = fminf(cur, u);
                        out[at(ot, y * obw + x, c)] = half(res);
                    }
                }
            } else if (k == 2) {
                for (uint32_t c = 0; c < ot->c; c++) {
                    float u = 0.0f;
                    if (const_updates) {
                        u = upd_const[m * ot->c + c];
                    } else {
                        uint32_t uy = ut->w ? m / ut->w : 0, ux = ut->w ? m % ut->w : 0;
                        u = scalar(updates[at(ut, uy * upw + ux, c)]);
                    }
                    float cur = scalar(out[at(ot, (uint32_t)dy * obw + (uint32_t)dx, c)]);
                    float res = u;
                    if (reduction == 1) res = cur + u;
                    else if (reduction == 2) res = cur * u;
                    else if (reduction == 3) res = fmaxf(cur, u);
                    else if (reduction == 4) res = fminf(cur, u);
                    out[at(ot, (uint32_t)dy * obw + (uint32_t)dx, c)] = half(res);
                }
            } else {
                float u = 0.0f;
                if (const_updates) {
                    u = upd_const[m];
                } else {
                    uint32_t up_px = ut->c ? m / ut->c : 0, up_c = ut->c ? m % ut->c : 0;
                    uint32_t uy = ut->w ? up_px / ut->w : 0, ux = ut->w ? up_px % ut->w : 0;
                    u = scalar(updates[at(ut, uy * upw + ux, up_c)]);
                }
                float cur = scalar(out[at(ot, (uint32_t)dy * obw + (uint32_t)dx, (uint32_t)dc)]);
                float res = u;
                if (reduction == 1) res = cur + u;
                else if (reduction == 2) res = cur * u;
                else if (reduction == 3) res = fmaxf(cur, u);
                else if (reduction == 4) res = fminf(cur, u);
                out[at(ot, (uint32_t)dy * obw + (uint32_t)dx, (uint32_t)dc)] = half(res);
            }
        }
    }
}

void hx_fp16_op(const HxGraph *g, uint32_t index, uint32_t part, uint32_t parts) {
    const HxOp *o = &g->ops[index];
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p;
    switch (o->kind) {
    case HX_RMSNORM:
        rmsnorm_fp16(g, o, part, parts);
        return;
    case HX_ROPE:
        rope_fp16(g, o, part, parts);
        return;
    case HX_SHORTCONV:
        shortconv_fp16(g, o, part, parts);
        return;
    case HX_LINEAR_Q:
        linear_q_fp16(g, o, part, parts);
        return;
    case HX_KV_WRITE:
        kv_write_fp16(g, o, part, parts);
        return;
    case HX_MATMUL:
        /* Vector lanes if the slice holds B as float blocks, else scalar lanes, else the
         * per-element loop below. */
        if (o->p[2] <= 1u && !o->p[3] && a->w == 1u &&
            (matmul_vector(g, o, part, parts) || matmul_scalar(g, o, part, parts)))
            return;
        if (o->p[2] > 1u || o->p[3] || a->w > 1u) {
            matmul_fp16_scalar(g, o, part, parts);
            return;
        }
        break;
    case HX_SDPA:
        sdpa_fp16_scalar(g, o, part, parts);
        return;
    case HX_GATHER:
        gather_fp16_scalar(g, o, part, parts);
        return;
    case HX_GATHER_ELEMENTS:
        gather_elements_fp16_scalar(g, o, part, parts);
        return;
    case HX_GATHER_ND:
        gather_nd_fp16_scalar(g, o, part, parts);
        return;
    case HX_SCATTER_ND:
        scatter_nd_fp16_scalar(g, o, part, parts);
        return;
    case HX_SOFTMAX:
    case HX_ATTN_SOFTMAX:
        softmax_fp16_scalar(g, o, part, parts);
        return;
    case HX_DEPTHWISE:
        if (dw_fast(o)) {
            blocked_depthwise(g, index, part, parts);
            return;
        }
        break; /* other shapes: scalar lanes below */
    case HX_ADD:
    case HX_PRELU:
        elementwise(g, o, index, part, parts);
        return;
    case HX_MAXPOOL:
        maxpool(g, o, part, parts);
        return;
    case HX_MEAN:
        mean(g, o, part, parts);
        return;
    case HX_RESIZE:
        if (!p[2]) {
            resize(g, o, part, parts);
            return;
        }
        break; /* nearest: scalar lanes below */
    case HX_RESHAPE:
        if (!p[0]) { /* NCHW flatten/unflatten (p[0] 1 or 2) run in the scalar lanes below */
            reshape(g, o, part, parts);
            return;
        }
        break;
    default:
        break;
    }
    const uint32_t obw = physical_width(b), opairs = b->h * obw / 2u, lo = opairs * part / parts,
                   hi = opairs * (part + 1u) / parts;
    uint16_t *out = (uint16_t *)tensor_data(g, o->out);
    if (o->kind == HX_PAD && !p[0] && !p[1] && !p[2] && a->h == b->h && a->w == b->w) {
        /* Only trailing channel blocks appear: copy planes, zero the rest. */
        const uint32_t plane = plane_bytes(b) / 128u, pairs = tiles(b) * 16u,
                       v0 = pairs * part / parts, v1 = pairs * (part + 1u) / parts;
        const HVX_Vector *in = (const HVX_Vector *)tensor_data(g, o->x);
        HVX_Vector *d = (HVX_Vector *)out;
        for (uint32_t c = 0; c < b->stride / 32u; c++)
            for (uint32_t v = v0; v < v1; v++)
                d[c * plane + v] = c < a->stride / 32u ? in[c * plane + v] : Q6_V_vzero();
        return;
    }
    if (o->kind == HX_CONCAT && p[0] == 1 && b->h == 1u && a->h == 1u && !(a->w & 1u)) {
        /* One row: the first tensor's pairs, then the second's. */
        const HxTensor *t1 = &g->tensors[o->y];
        const uint32_t first = a->w / 2u, plane = plane_bytes(b) / 128u, ip0 = plane_bytes(a) / 128u,
                       ip1 = plane_bytes(t1) / 128u;
        const HVX_Vector *s0 = (const HVX_Vector *)tensor_data(g, o->x),
                         *s1 = (const HVX_Vector *)tensor_data(g, o->y);
        HVX_Vector *d = (HVX_Vector *)out;
        for (uint32_t c = 0; c < b->stride / 32u; c++)
            for (uint32_t v = lo; v < hi; v++)
                d[c * plane + v] = v < first ? s0[c * ip0 + v] : s1[c * ip1 + v - first];
        return;
    }
    if ((o->kind == HX_PAD && !p[2]) || (o->kind == HX_CONCAT && p[0] != 2)) {
        for (uint32_t pair = lo; pair < hi; pair++) {
            const uint32_t yy = pair * 2u / obw, xx = pair * 2u % obw;
            int32_t pos[2] = {-1, -1};
            const uint16_t *src[2] = {NULL, NULL};
            const HxTensor *ts[2] = {a, a};
            for (uint32_t q = 0; q < 2; q++)
                if (xx + q < b->w)
                    pos[q] = place(o, a, g, yy, xx + q, &src[q], &ts[q]);
            if (!src[1])
                src[1] = src[0], ts[1] = ts[0];
            for (uint32_t c = 0; c < b->stride; c += 32) {
                HVX_Vector v;
                if (c >= ts[0]->stride)
                    v = Q6_V_vzero();
                else if (src[0] == src[1])
                    v = rows(src[0], ts[0], pos[0], pos[1], c);
                else
                    v = Q6_V_vor_VV(rows(src[0], ts[0], pos[0], -1, c),
                                    rows(src[1], ts[1], -1, pos[1], c));
                *(HVX_Vector *)(out + at(b, pair * 2u, c)) = v;
            }
        }
        return;
    }
    if (o->kind == HX_REDUCE) {
        /* Reduction over the p[1] axes, scalar lanes as in graph.c: accumulate in float per output
         * element; padding lanes and the pad column are written as zero. */
        const uint16_t *src_tensor = (const uint16_t *)tensor_data(g, o->x);
        for (uint32_t pair = lo; pair < hi; pair++) {
            for (uint32_t q = 0; q < 2; q++) {
                const uint32_t px = pair * 2u + q, oy = px / obw, ox = px % obw;
                if (ox >= b->w) {
                    for (uint32_t c = 0; c < b->stride; c++)
                        out[at(b, px, c)] = 0;
                    continue;
                }
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
                                    const float v = scalar(src_tensor[at(a, sy * physical_width(a) + sx, sc)]);
                                    if (p[0] == HX_RED_MAX)
                                        acc = fmaxf(acc, v);
                                    else if (p[0] == HX_RED_MIN)
                                        acc = fminf(acc, v);
                                    else
                                        acc += v;
                                    count++;
                                }
                    }
                    uint16_t v = 0;
                    if (c < b->c)
                        v = half(p[0] == HX_RED_MEAN ? acc / (float)count : acc);
                    out[at(b, px, c)] = v;
                }
            }
        }
        return;
    }
    if (o->kind == HX_NORM) {
        norm_scalar(g, o, lo, hi);
        return;
    }
    if (o->kind == HX_SOFTMAX) {
        /* Softmax over channels, one pixel at a time: the max and sum are computed once per pixel.
         * Input and output have the same shape, so physical pixel indices coincide. */
        const uint16_t *src_tensor = (const uint16_t *)tensor_data(g, o->x);
        for (uint32_t pair = lo; pair < hi; pair++) {
            for (uint32_t q = 0; q < 2; q++) {
                const uint32_t px = pair * 2u + q, ox = pair * 2u % obw + q;
                if (ox >= b->w) {
                    for (uint32_t c = 0; c < b->stride; c++)
                        out[at(b, px, c)] = 0;
                    continue;
                }
                float peak = -FLT_MAX, sum = 0.0f;
                for (uint32_t c = 0; c < b->c; c++) {
                    const float v = scalar(src_tensor[at(a, px, c)]);
                    peak = fmaxf(peak, v);
                }
                for (uint32_t c = 0; c < b->c; c++)
                    sum += expf(scalar(src_tensor[at(a, px, c)]) - peak);
                for (uint32_t c = 0; c < b->stride; c++) {
                    uint16_t v = 0;
                    if (c < b->c)
                        v = half(expf(scalar(src_tensor[at(a, px, c)]) - peak) / sum);
                    out[at(b, px, c)] = v;
                }
            }
        }
        return;
    }
    /* Channel-wise pad/concat and the logistic: small tensors, scalar lanes. */
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    for (uint32_t pair = lo; pair < hi; pair++) {
        const uint32_t oy = pair * 2u / obw;
        for (uint32_t q = 0; q < 2; q++) {
            const uint32_t ox = pair * 2u % obw + q;
            for (uint32_t c = 0; c < b->stride; c++) {
                uint16_t v = 0;
                if (ox < b->w && c < b->c) {
                    if (o->kind == HX_DEPTHWISE) {
                        const float *w = (const float *)(g->constants + p[6]);
                        float acc = ((const float *)(g->constants + p[7]))[c];
                        for (uint32_t ky = 0; ky < p[0]; ky++) {
                            const int32_t iy =
                                (int32_t)(oy * p[2] + ky * hx_dilation_h(o)) - (int32_t)p[4];
                            if (iy < 0 || iy >= (int32_t)a->h)
                                continue;
                            for (uint32_t kx = 0; kx < p[1]; kx++) {
                                const int32_t ix =
                                    (int32_t)(ox * p[3] + kx * hx_dilation_w(o)) - (int32_t)p[5];
                                if (ix < 0 || ix >= (int32_t)a->w)
                                    continue;
                                acc += scalar(x[at(a, (uint32_t)iy * physical_width(a) + (uint32_t)ix,
                                                   c)]) *
                                       w[(ky * p[1] + kx) * b->stride + c];
                            }
                        }
                        v = half(scalar_activation(acc, p[8]));
                    } else if (o->kind == HX_PAD) {
                        if (oy >= p[0] && oy - p[0] < a->h && ox >= p[1] && ox - p[1] < a->w &&
                            c >= p[2] && c - p[2] < a->c)
                            v = x[at(a, (oy - p[0]) * physical_width(a) + ox - p[1], c - p[2])];
                    } else if (o->kind == HX_ACT) {
                        v = half(hx_activation(p[0], p[1], p[2],
                                               scalar(x[at(a, oy * physical_width(a) + ox, c)])));
                    } else if (o->kind == HX_BINARY) {
                        /* Scalar lanes, like the logistic: broadcast by clamping coordinates. */
                        float l = scalar(x[at(a, oy * physical_width(a) + ox, c)]), u = 0.0f;
                        if (p[1] == 1) {
                            u = ((const float *)(g->constants + p[2]))[c];
                        } else if (p[1] == 2) {
                            u = ((const float *)(g->constants + p[2]))[(oy * b->w + ox) * b->stride + c];
                        } else {
                            const HxTensor *t = &g->tensors[o->y];
                            const uint16_t *yv = (const uint16_t *)tensor_data(g, o->y);
                            u = scalar(yv[at(t,
                                             (t->h == 1 ? 0 : oy) * physical_width(t) +
                                                 (t->w == 1 ? 0 : ox),
                                             t->c == 1 ? 0 : c)]);
                        }
                        if (p[3]) {
                            float s = l;
                            l = u;
                            u = s;
                        }
                        float res = 0.0f;
                        switch (p[0]) {
                        case HX_BIN_ADD: res = l + u; break;
                        case HX_BIN_SUB: res = l - u; break;
                        case HX_BIN_MUL: res = l * u; break;
                        case HX_BIN_DIV: res = u == 0.0f ? 0.0f : l / u; break;
                        case HX_BIN_MIN: res = fminf(l, u); break;
                        case HX_BIN_MAX: res = fmaxf(l, u); break;
                        case HX_BIN_POW: res = powf(l, u); break;
                        case HX_BIN_EQUAL: res = (l == u) ? 1.0f : 0.0f; break;
                        case HX_BIN_GREATER: res = (l > u) ? 1.0f : 0.0f; break;
                        case HX_BIN_LESS: res = (l < u) ? 1.0f : 0.0f; break;
                        case HX_BIN_GE: res = (l >= u) ? 1.0f : 0.0f; break;
                        case HX_BIN_LE: res = (l <= u) ? 1.0f : 0.0f; break;
                        case HX_BIN_AND: res = (l != 0.0f && u != 0.0f) ? 1.0f : 0.0f; break;
                        case HX_BIN_OR: res = (l != 0.0f || u != 0.0f) ? 1.0f : 0.0f; break;
                        default: res = 0.0f; break;
                        }
                        v = half(res);
                    } else if (o->kind == HX_MATMUL) {
                        /* Row oy of A against B in scalar lanes, float accumulate. transB 0 reads
                         * B element (k, c) at pixel k; transB 1 reads (c, k) at pixel c. */
                        const HxTensor *t = &g->tensors[o->y];
                        const uint16_t *yv = (const uint16_t *)tensor_data(g, o->y);
                        float scale = 1.0f, acc = 0.0f;
                        if (p[1])
                            memcpy(&scale, &p[1], sizeof scale);
                        for (uint32_t k = 0; k < a->c; k++) {
                            const float l = scalar(x[at(a, oy * physical_width(a), k)]);
                            const float u = p[0] ? scalar(yv[at(t, c * physical_width(t), k)])
                                                 : scalar(yv[at(t, k * physical_width(t), c)]);
                            acc += l * u;
                        }
                        v = half(acc * scale);

                    } else if (o->kind == HX_WHERE) {
                        const HxTensor *cond_t = a;
                        uint32_t cy = (cond_t->h == 1) ? 0 : oy, cx = (cond_t->w == 1) ? 0 : ox;
                        float cond_v = scalar(x[at(cond_t, cy * physical_width(cond_t) + cx, cond_t->c == 1 ? 0 : c)]);
                        float val = 0.0f;
                        if (cond_v != 0.0f) {
                            if (p[0] == 1) val = hx_float(p[1]);
                            else if (p[0] == 2) val = ((const float *)(g->constants + p[1]))[c];
                            else if (p[0] == 3) val = ((const float *)(g->constants + p[1]))[(oy * b->w + ox) * b->stride + c];
                            else {
                                const HxTensor *yt = &g->tensors[o->y];
                                const uint16_t *yv = (const uint16_t *)tensor_data(g, o->y);
                                val = scalar(yv[at(yt, (yt->h == 1 ? 0 : oy) * physical_width(yt) + (yt->w == 1 ? 0 : ox), yt->c == 1 ? 0 : c)]);
                            }
                        } else {
                            if (p[2] == 1) val = hx_float(p[3]);
                            else if (p[2] == 2) val = ((const float *)(g->constants + p[3]))[c];
                            else if (p[2] == 3) val = ((const float *)(g->constants + p[3]))[(oy * b->w + ox) * b->stride + c];
                            else {
                                const HxTensor *zt = &g->tensors[o->z];
                                const uint16_t *zv = (const uint16_t *)tensor_data(g, o->z);
                                val = scalar(zv[at(zt, (zt->h == 1 ? 0 : oy) * physical_width(zt) + (zt->w == 1 ? 0 : ox), zt->c == 1 ? 0 : c)]);
                            }
                        }
                        v = half(val);
                    } else if (o->kind == HX_ARGMAX || o->kind == HX_ARGMIN) {
                        const int is_min = (o->kind == HX_ARGMIN);
                        float best_val = is_min ? FLT_MAX : -FLT_MAX;
                        int32_t best_c = 0;
                        for (uint32_t ch = 0; ch < a->c; ch++) {
                            float fv = scalar(x[at(a, oy * physical_width(a) + ox, ch)]);
                            int better;
                            if (is_min)
                                better = p[0] ? (fv <= best_val) : (fv < best_val);
                            else
                                better = p[0] ? (fv >= best_val) : (fv > best_val);
                            if (better) {
                                best_val = fv;
                                best_c = (int32_t)ch;
                            }
                        }
                        v = (c == 0) ? (p[1] == 1 ? (uint16_t)best_c : half((float)best_c)) : 0;
                    } else if (o->kind == HX_TOPK) {
                        const uint32_t K = p[0], largest = p[1], sorted = p[2];
                        struct { float val; int32_t idx; } top[128];
                        uint32_t nk = 0;
                        for (uint32_t ch = 0; ch < a->c; ch++) {
                            float fv = scalar(x[at(a, oy * physical_width(a) + ox, ch)]);
                            int32_t idx = (int32_t)ch;
                            if (nk < K) {
                                top[nk].val = fv; top[nk].idx = idx; nk++;
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
                                if (largest ? (fv > top[worst].val) : (fv < top[worst].val)) {
                                    top[worst].val = fv; top[worst].idx = idx;
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
                        v = (c < K) ? half(top[c].val) : 0;
                        if (o->z != HXFG_NONE && c < K) {
                            uint16_t *zout = (uint16_t *)tensor_data(g, o->z);
                            zout[at(&g->tensors[o->z], pair * 2u + q, c)] = half((float)top[c].idx);
                        }
                    } else if (o->kind == HX_TILE) {
                        uint32_t sy = oy % a->h, sx = ox % a->w;
                        v = x[at(a, sy * physical_width(a) + sx, c % a->c)];
                    } else if (o->kind == HX_CAST) {
                        float fv = scalar(x[at(a, oy * physical_width(a) + ox, c)]);
                        if (p[0] == 0) {
                            int32_t iv = (int32_t)fv;
                            v = (uint16_t)iv;
                        } else if (p[0] == 1) {
                            int32_t iv = (int32_t)x[at(a, oy * physical_width(a) + ox, c)];
                            v = half((float)iv);
                        } else if (p[0] == 2) {
                            int32_t iv = (int32_t)x[at(a, oy * physical_width(a) + ox, c)];
                            v = half(iv != 0 ? 1.0f : 0.0f);
                        } else if (p[0] == 3) {
                            v = (fv != 0.0f) ? 1 : 0;
                        } else if (p[0] == 4) {
                            int32_t iv = (int32_t)x[at(a, oy * physical_width(a) + ox, c)];
                            v = (iv != 0) ? 1 : 0;
                        } else if (p[0] == 5) {
                            int32_t iv = (int32_t)x[at(a, oy * physical_width(a) + ox, c)];
                            v = (iv != 0) ? 1 : 0;
                        } else {
                            v = x[at(a, oy * physical_width(a) + ox, c)];
                        }
                    } else if (o->kind == HX_AVGPOOL) {
                        float sum = 0;
                        uint32_t taps = 0;
                        for (uint32_t ky = 0; ky < p[0]; ky++) {
                            int32_t iy = (int32_t)(oy * p[2] + ky) - (int32_t)p[4];
                            if (iy < 0 || iy >= (int32_t)a->h)
                                continue;
                            for (uint32_t kx = 0; kx < p[1]; kx++) {
                                int32_t ix = (int32_t)(ox * p[3] + kx) - (int32_t)p[5];
                                if (ix < 0 || ix >= (int32_t)a->w)
                                    continue;
                                sum += scalar(x[at(a, (uint32_t)iy * physical_width(a) + (uint32_t)ix, c)]);
                                taps++;
                            }
                        }
                        if (p[6])
                            taps = p[0] * p[1];
                        v = half(taps ? sum / (float)taps : 0.0f);
                    } else if (o->kind == HX_RESIZE && p[2] == 1) {
                        /* Nearest: coordinate mode in p[3], nearest mode in p[4] */
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
                        v = x[at(a, sy * physical_width(a) + sx, c)];
                    } else if (o->kind == HX_RESIZE && p[5] == 1) {
                        /* Bicubic */
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
                        float acc = 0.0f;
                        for (int j = 0; j < 4; j++) {
                            int32_t sy = y0 + j - 1;
                            sy = sy < 0 ? 0 : (sy >= (int32_t)a->h ? (int32_t)a->h - 1 : sy);
                            for (int k = 0; k < 4; k++) {
                                int32_t sx = x0 + k - 1;
                                sx = sx < 0 ? 0 : (sx >= (int32_t)a->w ? (int32_t)a->w - 1 : sx);
                                acc += wy[j] * wx[k] * scalar(x[at(a, (uint32_t)sy * physical_width(a) + (uint32_t)sx, c)]);
                            }
                        }
                        v = half(acc);
                    } else if (o->kind == HX_LOGISTIC) {
                        v = half(1.0f /
                                 (1.0f + expf(-scalar(x[at(a, oy * physical_width(a) + ox, c)]))));
                    } else if (o->kind == HX_ZEROINSERT) {
                        /* Same grid as graph.c: non-source pixels keep the zero halfword. */
                        const uint32_t ry = oy - p[2], rx = ox - p[3];
                        if (oy >= p[2] && ox >= p[3] && ry % p[0] == 0 && rx % p[1] == 0 &&
                            ry / p[0] < a->h && rx / p[1] < a->w)
                            v = x[at(a, (ry / p[0]) * physical_width(a) + rx / p[1], c)];
                    } else if (o->kind == HX_SLICE) {
                        v = x[at(a, (oy + p[0]) * physical_width(a) + ox + p[1], c + p[2])];
                    } else if (o->kind == HX_TRANSPOSE) {
                        const uint32_t at3[3] = {oy, ox, c};
                        uint32_t s[3];
                        for (uint32_t k = 0; k < 3; k++)
                            s[p[k]] = at3[k];
                        v = x[at(a, s[0] * physical_width(a) + s[1], s[2])];
                    } else if (o->kind == HX_RESHAPE) { /* NCHW flatten (1) or unflatten (2) */
                        if (p[0] == 1) {
                            const uint32_t hw = a->h * a->w, rem = c % hw, ch = c / hw;
                            v = x[at(a, (rem / a->w) * physical_width(a) + rem % a->w, ch)];
                        } else {
                            v = x[at(a, 0, c * b->h * b->w + oy * b->w + ox)];
                        }
                    } else { /* HX_CONCAT on channels */
                        const HxTensor *t = c >= a->c ? &g->tensors[o->y] : a;
                        const uint16_t *src = c >= a->c ? (const uint16_t *)tensor_data(g, o->y) : x;
                        v = src[at(t, oy * physical_width(t) + ox, c >= a->c ? c - a->c : c)];
                    }
                }
                out[at(b, pair * 2u + q, c)] = v;
            }
        }
    }
}
