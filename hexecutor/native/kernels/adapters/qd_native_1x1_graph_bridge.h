#ifndef QD_NATIVE_1X1_GRAPH_BRIDGE_H
#define QD_NATIVE_1X1_GRAPH_BRIDGE_H

/* Source-included bridge for the opt-in QDRX ConvLayer_s1 HMX path.
 * Include from qd_graph_dsp.c after dsp_tensor(), dsp_clean_range(),
 * dsp_packet_copy(), send_work(), and now_ns() are defined. qd_graph.c owns
 * the static prototype and invokes this before its FI/weight conversion path.
 *
 * Diagnostics: [0] prepared commands, [1] prepare cycles, [2] leaf cycles,
 * [3] finish/scatter cycles, [4] declined HMX commands, [5] hard failures,
 * [6] staged workspace bytes, [7] reserved. All cycle values are upcycle
 * deltas. The bridge itself allocates no heap memory.
 */

#if !defined(QDG_DSP) || !defined(QDS_NATIVE_1X1) || !QDS_NATIVE_1X1
#error "qd_native_1x1_graph_bridge.h requires QDG_DSP and QDS_NATIVE_1X1=1"
#endif
#if !defined(QDS_PRESERVE_VTCM) || !QDS_PRESERVE_VTCM
#error "native 1x1 graph dispatch requires QDS_PRESERVE_VTCM=1"
#endif

#include "qd_native_1x1_adapter.h"
#include "qd_native_1x1_graph_eligibility.h"
#include "../../qd_vtcm_lease.h"

#define QD_N1X1_LEASE_STATE_OFFSET 3328u
#define QD_N1X1_WORKSPACE_OFFSET  0x700000u
#define QD_N1X1_DONE_OFFSET       0x58000u
#define QD_N1X1_NATIVE_FAILURE    0x4e310001u
#define QD_N1X1_HMX_STATE_OFFSET  512u
#define QD_N1X1_HMX_ACTIVE        0x484d5831u

/* Layout-compatible read-only prefix of QdHmxState from qd_hmx_power.h. Keep
 * this local view so the source-included DSP file does not import its clock
 * helpers, which use a different cycles() provider. */
typedef struct {
    uint32_t active, rsc[17], trigger, address, ssr, powered, clocked;
    uint32_t clocks[18], error, restored;
} QdN1x1HmxState;

extern void qd_hmx_finish(void *, void *);
/* Implemented by the resident worker module. HMX is global state shared with
 * the legacy worker, so both its per-worker resident keys and graph key must
 * be invalidated around this leaf. */
extern void qdrc_invalidate_native(void);

static int native_1x1_defer(QdgGraph *g) {
    if (g) g->dsp_native_1x1_diag[4]++;
    return QDG_DSP_DEFER;
}

static int native_1x1_pool_span(QdgGraph *g, const T *t, uint32_t bytes) {
    if (t->pool >= QDG_POOLS || !g->pool[t->pool] ||
        t->offset > g->pool_bytes[t->pool] ||
        (uint64_t)bytes > g->pool_bytes[t->pool] - t->offset)
        return 0;
    return 1;
}

static int native_1x1_lease_valid(void) {
    QdVtcmLease *lease = (QdVtcmLease *)(SERVICE + QD_N1X1_LEASE_STATE_OFFSET);
    if (lease->magic != QDVT_MAGIC || lease->phase != 1 || lease->error ||
        lease->base != QDVT_BASE || lease->bytes != QDVT_BYTES ||
        lease->backup_base != QDVT_BACKUP_BASE ||
        (uintptr_t)lease->backup != QDVT_BACKUP_BASE ||
        lease->tlb8 != QDVT_TLB8_VALUE || lease->tlb9 != QDVT_TLB9_VALUE)
        return 0;
    return qdvt_read_tlb(QDVT_TLB8) == QDVT_TLB8_VALUE &&
           qdvt_read_tlb(QDVT_TLB9) == QDVT_TLB9_VALUE;
}

static int native_1x1_hmx_owned(void) {
    QdN1x1HmxState *state = (QdN1x1HmxState *)(SERVICE + QD_N1X1_HMX_STATE_OFFSET);
    return state->active == QD_N1X1_HMX_ACTIVE && state->powered && state->clocked &&
           !state->error && !state->restored;
}

static int native_1x1_workers_idle(void) {
    uint32_t generation = work->generation;
    __asm__ volatile("barrier" ::: "memory");
    int idle=1;
    for(uint32_t t=1;t<QDS_WORKER_SLOTS;t++)idle &= work->done[t]==generation;
    __asm__ volatile("barrier" ::: "memory");
    return work->generation==generation && idle;
}

static int native_1x1_try(QdgGraph *g, const C *c, uint32_t index) {
    (void)index;
    if (!g || !c || c->op != QDR_HMX_CONV || c->nin < 3 || c->nout != 1 ||
        c->variant == 36)
        return native_1x1_defer(g);

    uint32_t input_id = c->in[0], weight_id = c->in[1], fi_id = c->in[2];
    uint32_t output_id = c->out[0];
    if (input_id >= g->ntensors || weight_id >= g->ntensors ||
        fi_id >= g->ntensors || output_id >= g->ntensors)
        return native_1x1_defer(g);

    const T *x = dsp_tensor(g, input_id);
    const T *y = dsp_tensor(g, output_id);
    const T *fi = dsp_tensor(g, fi_id);
    uint32_t weight_source = g->static_source[weight_id];
    if (weight_source == QDR_NONE) weight_source = weight_id;
    if (weight_source >= g->ntensors) return native_1x1_defer(g);
    const T *wt = dsp_tensor(g, weight_source);
    if (!x || !y || !fi || !wt) return native_1x1_defer(g);

    QdN1x1GraphTensorMeta xm, wm, fim, ym;
    memset(&xm, 0, sizeof(xm)); memset(&wm, 0, sizeof(wm));
    memset(&fim, 0, sizeof(fim)); memset(&ym, 0, sizeof(ym));
    const T *source[4] = {x, wt, fi, y};
    QdN1x1GraphTensorMeta *meta[4] = {&xm, &wm, &fim, &ym};
    for (uint32_t k = 0; k < 4; ++k) {
        meta[k]->flags = source[k]->flags;
        meta[k]->dtype = source[k]->dtype;
        meta[k]->esize = source[k]->esize;
        meta[k]->crouton = source[k]->crouton;
        memcpy(meta[k]->d, source[k]->d, sizeof(meta[k]->d));
        memcpy(meta[k]->m, source[k]->m, sizeof(meta[k]->m));
        memcpy(meta[k]->p, source[k]->p, sizeof(meta[k]->p));
    }
    xm.is_cb = is_class(x, "CB", "cb");
    ym.is_cb = is_class(y, "CB", "cb");
    fim.is_fi = is_class(fi, "Fi", "fi");
    wm.is_xpb = is_class(wt, "Xpb", "xpb");
    wm.is_xwb = is_class(wt, "Xwb", "xwb");
    QdN1x1GraphCandidate candidate;
    if (!qd_native_1x1_graph_candidate(&xm, &wm, &fim, &ym, &candidate) ||
        fi->offset & 3u)
        return native_1x1_defer(g);
    int xpb = (int)candidate.xpb;
    uint32_t raw_ci = candidate.raw_ci;
    uint32_t native_ci = candidate.native_ci;

    uint64_t weight_bytes64 = (uint64_t)(uint32_t)wt->m[2] * (uint32_t)wt->m[3];
    uint64_t fi_bytes64 = (uint64_t)(uint32_t)wt->m[3] * 8u;
    if (weight_bytes64 > UINT32_MAX || fi_bytes64 > UINT32_MAX ||
        !native_1x1_pool_span(g, wt, (uint32_t)weight_bytes64) ||
        !native_1x1_pool_span(g, fi, (uint32_t)fi_bytes64) ||
        (x->nblocks && !x->blocks) || (y->nblocks && !y->blocks) ||
        !x->blocks || !y->blocks)
        return native_1x1_defer(g);

    /* Once metadata proves this is in the native-supported set, a missing
     * lease/power/quiescence contract is a failed opt-in engine invariant.
     * Falling back here would hide an invalid native-engine bootstrap. */
    if (!g->use_hmx || !native_1x1_lease_valid() || !native_1x1_hmx_owned() ||
        !native_1x1_workers_idle()) {
        g->error_detail = QD_N1X1_NATIVE_FAILURE | 3u;
        g->dsp_native_1x1_diag[5]++;
        return QDG_ERR_DSP;
    }

    QdN1x1Request request;
    memset(&request, 0, sizeof(request));
    uint32_t pool_bytes[QDG_POOLS];
    for (uint32_t p = 0; p < QDG_POOLS; ++p) {
        if (g->pool_bytes[p] > UINT32_MAX) return native_1x1_defer(g);
        pool_bytes[p] = (uint32_t)g->pool_bytes[p];
        request.pool[p] = g->pool[p];
    }
    request.pool_count = QDG_POOLS;
    request.pool_bytes = pool_bytes;
    for (uint32_t d = 0; d < 4; ++d) {
        request.input.dims[d] = (uint32_t)x->d[d];
        request.input.max_dims[d] = (uint32_t)x->m[d];
        request.input.padding[d] = (uint32_t)x->p[d];
        request.output.dims[d] = (uint32_t)y->d[d];
        request.output.max_dims[d] = (uint32_t)y->m[d];
        request.output.padding[d] = (uint32_t)y->p[d];
    }
    request.input.slice_offset = x->slice;
    request.input.block_count = x->nblocks;
    request.input.blocks = x->blocks;
    request.output.slice_offset = y->slice;
    request.output.block_count = y->nblocks;
    request.output.blocks = y->blocks;
    request.weights = g->pool[wt->pool] + wt->offset;
    request.weights_bytes = (uint32_t)weight_bytes64;
    request.weight_ci = raw_ci;
    request.weight_co = (uint32_t)wt->m[3];
#if QDS_NATIVE_PADDED_1X1
    request.weight_native_ci = native_ci;
#endif
    request.weight_kernel_h = 1;
    request.weight_kernel_w = 1;
    request.input_dtype = QD_N1X1_DTYPE_U8;
    request.output_dtype = QD_N1X1_DTYPE_U8;
    request.weight_dtype = QD_N1X1_DTYPE_I8;
    request.fi_dtype = QD_N1X1_DTYPE_I32;
    request.weight_format = xpb ? QD_N1X1_WEIGHT_XPB : QD_N1X1_WEIGHT_XWB;
    request.fi = (const int32_t *)(g->pool[fi->pool] + fi->offset);
    request.fi_bytes = (uint32_t)fi_bytes64;
    request.workspace = (void *)(QDVT_BASE + QD_N1X1_WORKSPACE_OFFSET);
    request.workspace_bytes = QD_N1X1_WORKSPACE_MAX;

    /* Direct constants can be scalar-cached. Publish their contents before
     * the adapter's aligned HVX copies read them. The pool metadata and raw
     * bytes are validated above before either cache operation or workspace
     * write occurs. */
    if (((uintptr_t)request.weights & 127u) || ((uintptr_t)request.fi & 127u))
        return native_1x1_defer(g);
    dsp_clean_range(request.weights, request.weights_bytes);
    dsp_clean_range(request.fi, request.fi_bytes);

    QdN1x1Prepared prepared;
    uint64_t started = now_ns();
    int status = qd_native_1x1_prepare(&request, &prepared);
    uint64_t prep_cycles = now_ns() - started;
    if (status) return native_1x1_defer(g);
    g->dsp_native_1x1_diag[0]++;
    g->dsp_native_1x1_diag[1] += prep_cycles;
    g->dsp_native_1x1_diag[6] += prepared.workspace_used;

    /* send_work() is synchronous, but verify its generation barrier again
     * before entering global HMX state. A failure here is a hard error because
     * preparation has already staged into VTCM. */
    if (!native_1x1_workers_idle() || !native_1x1_lease_valid() ||
        !native_1x1_hmx_owned()) {
        g->error_detail = QD_N1X1_NATIVE_FAILURE;
        g->dsp_native_1x1_diag[5]++;
        return QDG_ERR_DSP;
    }

    volatile uint32_t *done = (volatile uint32_t *)(QDVT_BASE +
        QD_N1X1_WORKSPACE_OFFSET + QD_N1X1_DONE_OFFSET);
    *done = 0xffffffffu;
    g->resident_key = 0;
    qdrc_invalidate_native();
    __asm__ volatile("mxclracc\n barrier" ::: "memory");

    started = now_ns();
    qd_qnn_hmx_1x1(prepared.output_desc, prepared.input_desc, prepared.weights,
                   prepared.fi, prepared.cfg, prepared.mode);
    uint64_t leaf_cycles = now_ns() - started;
    g->dsp_native_1x1_diag[2] += leaf_cycles;

    started = now_ns();
    qd_hmx_finish(prepared.zero_words, (void *)done);
    uint64_t deadline = now_ns() + 100000000ull;
    while (*done != 0 && now_ns() <= deadline) __asm__ volatile("nop");
    if (*done != 0) {
        g->resident_key = 0;
        qdrc_invalidate_native();
        g->error_detail = QD_N1X1_NATIVE_FAILURE | 1u;
        g->dsp_native_1x1_diag[5]++;
        return QDG_ERR_DSP;
    }
    __asm__ volatile("barrier" ::: "memory");
    if (!native_1x1_workers_idle() || !native_1x1_lease_valid()) {
        g->resident_key = 0;
        qdrc_invalidate_native();
        g->error_detail = QD_N1X1_NATIVE_FAILURE | 2u;
        g->dsp_native_1x1_diag[5]++;
        return QDG_ERR_DSP;
    }
    status = qd_native_1x1_finish(&prepared);
    g->dsp_native_1x1_diag[3] += now_ns() - started;
    g->resident_key = 0;
    qdrc_invalidate_native();
    if (status) {
        g->error_detail = QD_N1X1_NATIVE_FAILURE | ((uint32_t)status << 8);
        g->dsp_native_1x1_diag[5]++;
        return QDG_ERR_DSP;
    }
    return 0;
}

#undef QD_N1X1_LEASE_STATE_OFFSET
#undef QD_N1X1_WORKSPACE_OFFSET
#undef QD_N1X1_DONE_OFFSET
#undef QD_N1X1_NATIVE_FAILURE
#undef QD_N1X1_HMX_STATE_OFFSET
#undef QD_N1X1_HMX_ACTIVE

#endif
