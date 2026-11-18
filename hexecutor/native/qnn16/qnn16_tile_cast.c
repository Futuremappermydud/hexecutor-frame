/* HVX tile form (qnn16_tile.h): cast. */
#include "qnn16_tile_priv.h"

/* Direct tensor without padding whose full shape equals its dims. */
static int cast_plain(const Qnn16Tensor *t) {
    const Qnn16Shape d = t->dims, f = t->full, p = t->padding;
    return f.b == d.b && f.h == d.h && f.w == d.w && f.c == d.c && !p.b && !p.h && !p.w && !p.c;
}

int qnn16_tile_cast_f16(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts) {
    const Qnn16Shape d = x->dims, e = out->dims;
    if (d.b != e.b || d.h != e.h || d.w != e.w || d.c != e.c)
        return 0;
    if (!cast_plain(x) || !cast_plain(out) || !direct_ok(x, 4u) || !direct_ok(out, 2u))
        return 0;
    /* Contiguous element index n; chunk i covers elements [64 i, 64 i + 64). */
    const uint32_t n = d.b * d.h * d.w * d.c;
    const uint8_t *src_base = x->base + x->direct_offset;
    uint8_t *dst_base = out->base + out->direct_offset;
    const uint32_t chunks = (n + 63u) / 64u;
    for (uint32_t i = part; i < chunks; i += parts) {
        const uint32_t first = i * 64u;
        const uint32_t r = n - first < 64u ? n - first : 64u;
        const float *src = (const float *)(src_base + 4u * first);
        /* Each load is 128 bytes, within the window direct_ok guarantees; the second is read only for tails over 32 elements. */
        const HVX_Vector w0 = f2h_words(*(const HVX_UVector *)src);
        HVX_Vector w1 = w0;
        if (r > 32u)
            w1 = f2h_words(*(const HVX_UVector *)(src + 32));
        const HVX_Vector value = Q6_Vh_vpacke_VwVw(w1, w0);
        store_bytes(dst_base + 2u * first, value, 2u * r);
    }
    return 1;
}
/* Blocked -> blocked, same logical shape, any w/h/b padding on each side: logical pixel (b, y, x) of x is
 * logical pixel (b, y, x) of out. Out pair pp holds full columns 2pp, 2pp + 1, which are x full columns
 * 2pp + s, 2pp + s + 1 with s = x pad w - out pad w; tap() yields those two pixels for any column. */
static int format_blocked(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts) {
    const Qnn16Shape d = x->dims;
    const Map mx = map_of(x), mo = map_of(out);
    const int32_t s = (int32_t)x->padding.w - (int32_t)out->padding.w;
    const uint32_t p0 = out->padding.w / 2u, p1 = (out->padding.w + d.w + 1u) / 2u;
    const uint32_t groups = (d.c + 31u) / 32u;
    uint32_t row = 0; /* (group, batch, row) triples; each output tile is written by exactly one part */
    for (uint32_t g = 0; g < groups; g++)
        for (uint32_t b = 0; b < d.b; b++)
            for (uint32_t y = 0; y < d.h; y++, row++) {
                if (row % parts != part)
                    continue;
                const HVX_VectorPred qc = channel_mask(out, g);
                const uint32_t bi = b + x->padding.b, yi = y + x->padding.h;
                const uint32_t bo = b + out->padding.b, yo = y + out->padding.h;
                for (uint32_t pp = p0; pp < p1; pp++)
                    store_live(vec(&mo, bo, yo, (int32_t)pp, g), tap(&mx, bi, yi, 2 * (int32_t)pp + s, g),
                               lanes_of(out, pp), qc);
            }
    return 1;
}

/* Direct FP16 single-channel rows -> tiles. Source word q of a row (halfwords
 * 2q, 2q+1 = pixels 2q, 2q+1) lands in bytes 0-3 of output pair q + pw/2. */
int qnn16_tile_format_fast(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts) {
    const Qnn16Shape d = x->dims, e = out->dims;
    if (d.b != e.b || d.h != e.h || d.w != e.w || d.c != e.c)
        return 0;
    /* Distinct tiles only: an in-place shift would read pairs already overwritten. */
    if (blocked_any(x) && blocked_any(out) && x->block_offsets != out->block_offsets)
        return format_blocked(x, out, part, parts);
    if (d.c != 1u)
        return 0;
    if (!cast_plain(x) || !direct_ok(x, 2u) || !blocked_any(out) || out->padding.w & 1u || out->padding.c)
        return 0;
    const Map mo = map_of(out);
    const uint32_t pw = out->padding.w / 2u;
    const uint32_t pairs = (d.w + 1u) / 2u;
    const HVX_VectorPred q4 = Q6_Q_vsetq_R(4), q2 = Q6_Q_vsetq_R(2); /* bytes 0-3 / 0-1: channel 0 only */
    uint32_t slot = 0;
    for (uint32_t b = 0; b < d.b; b++)
        for (uint32_t y = 0; y < d.h; y++) {
            PART_ROW(slot);
            const uint8_t *row = direct_at(x, b, y, 0, 0, 2u);
            for (uint32_t k = 0; k < pairs; k += 32u) {
                /* 128 bytes from pair k: pairs k .. k + 31; the tail may read past the row, not the tensor. */
                const HVX_Vector s = *(const HVX_UVector *)(row + 4u * k);
                const uint32_t n = pairs - k < 32u ? pairs - k : 32u;
                const uint32_t fy = y + out->padding.h, fb = b + out->padding.b;
                HVX_Vector *ptr[32];
                const int have = row_ptrs(&mo, fb, fy, (int32_t)(k + pw), n, 0, ptr);
                for (uint32_t j = 0; j < n; j++) {
                    const uint32_t q = k + j;
                    HVX_Vector *p = have ? ptr[j] : vec(&mo, fb, fy, (int32_t)(q + pw), 0);
                    if (!p)
                        continue;
                    const HVX_VectorPred qm = (2u * q + 1u < d.w) ? q4 : q2;
                    Q6_vmem_QRIV(qm, p, Q6_V_vror_VR(s, (int)(4u * j)));
                }
            }
        }
    return 1;
}
