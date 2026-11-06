#ifndef QD_NATIVE_1X1_GRAPH_ELIGIBILITY_H
#define QD_NATIVE_1X1_GRAPH_ELIGIBILITY_H

/* Pure metadata classifier shared by the source-included graph bridge and
 * host tests over real compiled-QDRX tensor records. */
#include <stdint.h>

#ifndef QDS_NATIVE_PADDED_1X1
#define QDS_NATIVE_PADDED_1X1 0
#endif

enum {
    QD_N1X1_G_BLOCKED = 2u,
    QD_N1X1_G_DIRECT = 4u,
    QD_N1X1_G_U8 = 1u,
    QD_N1X1_G_I8 = 2u,
    QD_N1X1_G_I32 = 5u
};

typedef struct {
    uint32_t flags, dtype, esize, crouton;
    uint32_t is_cb, is_fi, is_xpb, is_xwb;
    int32_t d[4], m[4], p[4];
} QdN1x1GraphTensorMeta;

typedef struct {
    uint32_t raw_ci, native_ci, co, padded_ci, xpb;
} QdN1x1GraphCandidate;

static int qd_native_1x1_graph_candidate(
    const QdN1x1GraphTensorMeta *x,
    const QdN1x1GraphTensorMeta *w,
    const QdN1x1GraphTensorMeta *fi,
    const QdN1x1GraphTensorMeta *y,
    QdN1x1GraphCandidate *out) {
    if (!x || !w || !fi || !y || !out) return 0;
    if (!(x->flags & QD_N1X1_G_BLOCKED) || (x->flags & QD_N1X1_G_DIRECT) ||
        !(y->flags & QD_N1X1_G_BLOCKED) || (y->flags & QD_N1X1_G_DIRECT) ||
        !x->crouton || !y->crouton || !x->is_cb || !y->is_cb ||
        x->dtype != QD_N1X1_G_U8 || y->dtype != QD_N1X1_G_U8 ||
        x->esize != 1 || y->esize != 1 ||
        !(w->flags & QD_N1X1_G_DIRECT) || (w->flags & QD_N1X1_G_BLOCKED) ||
        w->dtype != QD_N1X1_G_I8 || w->esize != 1 ||
        !(w->is_xpb || w->is_xwb) ||
        !(fi->flags & QD_N1X1_G_DIRECT) || (fi->flags & QD_N1X1_G_BLOCKED) ||
        fi->dtype != QD_N1X1_G_I32 || fi->esize != 4 || !fi->is_fi)
        return 0;

    if (w->m[0] != 1 || w->m[1] != 1 || w->m[2] <= 0 || w->m[3] <= 0 ||
        w->d[0] != 1 || w->d[1] != 1 || w->d[2] != w->m[2] ||
        w->d[3] != w->m[3] || x->d[0] != 1 || y->d[0] != 1 ||
        y->m[3] != w->m[3] || y->d[3] != w->m[3] || y->p[3] != 0 ||
        (w->m[2] & 3) || (w->m[3] & 31) || x->p[3] != 0)
        return 0;
    uint64_t fi_elements = 1;
    for (uint32_t d = 0; d < 4; ++d) {
        if (fi->m[d] <= 0 || fi->d[d] != fi->m[d] || fi->p[d] != 0)
            return 0;
        if (fi_elements > UINT64_MAX / (uint32_t)fi->m[d]) return 0;
        fi_elements *= (uint32_t)fi->m[d];
    }
    if (fi_elements != (uint64_t)(uint32_t)w->m[3] * 2u) return 0;

    uint32_t raw_ci = (uint32_t)w->m[2];
    uint32_t native_ci = raw_ci;
    uint32_t padded = 0;
#if QDS_NATIVE_PADDED_1X1
    if (w->is_xpb && x->m[3] == 32 &&
        ((raw_ci == 12u && x->d[3] == 9) ||
         (raw_ci == 20u && x->d[3] == 18))) {
        native_ci = 32;
        padded = 1;
    }
#endif
    if (x->m[3] != (int32_t)native_ci ||
        (!padded && x->d[3] != (int32_t)raw_ci) ||
        (native_ci & 31u))
        return 0;

    for (uint32_t d = 0; d < 4; ++d) {
        if (x->d[d] <= 0 || x->m[d] <= 0 || x->p[d] < 0 ||
            y->d[d] <= 0 || y->m[d] <= 0 || y->p[d] < 0 ||
            x->p[d] > x->m[d] || y->p[d] > y->m[d] ||
            (uint32_t)x->d[d] > (uint32_t)x->m[d] - (uint32_t)x->p[d] ||
            (uint32_t)y->d[d] > (uint32_t)y->m[d] - (uint32_t)y->p[d])
            return 0;
    }
    out->raw_ci = raw_ci;
    out->native_ci = native_ci;
    out->co = (uint32_t)w->m[3];
    out->padded_ci = padded;
    out->xpb = w->is_xpb;
    return 1;
}

#endif
