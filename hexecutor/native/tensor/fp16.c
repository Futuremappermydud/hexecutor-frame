#include "fp16.h"
#include "hmx.h"
#include "fp16_hvx.h"
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

static inline uint32_t physical_width(const HxTensor *t) { return (t->w + 1u) & ~1u; }
static inline uint32_t tiles(const HxTensor *t) { return (t->h * physical_width(t) + 31u) / 32u; }
static inline uint32_t plane_bytes(const HxTensor *t) { return tiles(t) * 2048u; }
static inline uint8_t *tensor_data(const HxGraph *g, uint32_t tensor) {
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
static int supported(const HxGraph *g) {
    uint32_t compact_consumer = HXFG_NONE;
    for (uint32_t i = 0; i < g->header->tensors; i++)
        if (g->tensors[i].stride == 3 && (!is_input(g, i) || is_output(g, i)))
            return 0;
    for (uint32_t i = 0; i < g->header->ops; i++) {
        const HxOp *o = &g->ops[i];
        const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
        if (a->stride == 3 && o->kind != HX_CONV)
            return 0;
        if (o->kind == HX_CONV) {
            const uint32_t kb = o->p[10];
            if (!kb)
                return 0;
            if (a->stride == 3) {
                if (o->p[0] != o->p[1] || (o->p[0] != 3 && o->p[0] != 5) || o->p[2] != o->p[3] ||
                    o->p[2] > 2u || kb * 2048u * 4u > HX_FP16_MIN_WORKSPACE)
                    return 0;
                if (compact_consumer != HXFG_NONE)
                    return 0;
                compact_consumer = i;
            } else if (o->p[0] != 1 || o->p[1] != 1 || o->p[2] != 1 || o->p[3] != 1 || o->p[4] ||
                       o->p[5] || a->h != b->h || a->w != b->w || kb * 32u > a->stride)
                return 0;
        }
        if (o->kind == HX_DEPTHWISE &&
            (o->p[0] != o->p[1] || (o->p[0] != 3 && o->p[0] != 5) || o->p[2] > 2 || o->p[3] > 2 ||
             !o->p[2] || !o->p[3] || o->p[2] != o->p[3]))
            return 0;
    }
    return 1;
}
static uint32_t aux_bytes(const HxGraph *g, const HxOp *o) {
    const uint32_t blocks = g->tensors[o->out].stride / 32u;
    if (o->kind == HX_CONV)
        return blocks * 256u;
    if (o->kind == HX_DEPTHWISE)
        return blocks * (o->p[0] * o->p[1] + 1u) * 128u;
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
                used = g->ops[j].x == id || g->ops[j].y == id;
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
                    (g->tensors[o->x].stride == 3 ? o->p[10] * 2048u : 0u) >
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
                                (g->tensors[o->x].stride == 3 ? o->p[10] * 2048u : 0u);
                if (need > workspace)
                    workspace = need;
                /* A compact-RGB convolution packs as many output tiles per HMX
                 * batch as the workspace holds; a small workspace means many
                 * batches (palm +7.7 % at 274 KB in the simulator). */
                if (g->tensors[o->x].stride == 3) {
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
        } else if (o->kind == HX_DEPTHWISE) {
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
    if (a->stride != 3) {
        strip.a_kstep = plane_bytes(a);
        strip.a_mstep = 2048u;
        hx_hmx_strip(tensor_data(g, o->x), weights, out, &strip);
        hx_hmx_finish(done);
    } else {
        /* Workers pack a run of activation tiles, then the matrix unit consumes
         * it. The completion before reuse drains its loads and stores. */
        const uint32_t total = tiles(b),
                       space = (uint32_t)(workspace + g->fp16->workspace_bytes - cursor);
        const uint32_t batch = space / (kb * 2048u);
        for (uint32_t first = 0; first < total; first += batch) {
            HxFp16Job job = {.kind = HX_JOB_RGB_PACK};
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
    if (!(b->w & 1u))
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
void hx_fp16_op(const HxGraph *g, uint32_t index, uint32_t part, uint32_t parts) {
    const HxOp *o = &g->ops[index];
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p;
    switch (o->kind) {
    case HX_DEPTHWISE:
        blocked_depthwise(g, index, part, parts);
        return;
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
        resize(g, o, part, parts);
        return;
    case HX_RESHAPE:
        reshape(g, o, part, parts);
        return;
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
    /* Channel-wise pad/concat and the logistic: small tensors, scalar lanes. */
    const uint16_t *x = (const uint16_t *)tensor_data(g, o->x);
    for (uint32_t pair = lo; pair < hi; pair++) {
        const uint32_t oy = pair * 2u / obw;
        for (uint32_t q = 0; q < 2; q++) {
            const uint32_t ox = pair * 2u % obw + q;
            for (uint32_t c = 0; c < b->stride; c++) {
                uint16_t v = 0;
                if (ox < b->w && c < b->c) {
                    if (o->kind == HX_PAD) {
                        if (oy >= p[0] && oy - p[0] < a->h && ox >= p[1] && ox - p[1] < a->w &&
                            c >= p[2] && c - p[2] < a->c)
                            v = x[at(a, (oy - p[0]) * physical_width(a) + ox - p[1], c - p[2])];
                    } else if (o->kind == HX_ACT) {
                        v = half(hx_activation(p[0], p[1], p[2],
                                               scalar(x[at(a, oy * physical_width(a) + ox, c)])));
                    } else if (o->kind == HX_BINARY) {
                        /* Scalar lanes, like the logistic: broadcast by clamping coordinates. */
                        float l = scalar(x[at(a, oy * physical_width(a) + ox, c)]), u;
                        if (o->y == HXFG_NONE) {
                            u = ((const float *)(g->constants + p[2]))[c];
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
                        v = half(p[0] == HX_BIN_ADD ? l + u : (p[0] == HX_BIN_SUB ? l - u : l * u));
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
                    } else if (o->kind == HX_LOGISTIC) {
                        v = half(1.0f /
                                 (1.0f + expf(-scalar(x[at(a, oy * physical_width(a) + ox, c)]))));
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
