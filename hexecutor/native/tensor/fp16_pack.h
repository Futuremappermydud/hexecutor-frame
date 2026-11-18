#ifndef HEXECUTOR_TENSOR_FP16_PACK_H
#define HEXECUTOR_TENSOR_FP16_PACK_H
/* Plain C (no intrinsics) so the host can test the matrix-unit operand layout.
 *
 * Dense convolutions other than the in-place 1x1 case are run as a matrix product over an
 * im2col matrix: one row per output pixel, K = kh * kw * input channels columns in the order
 * the packed weights use (tap-major, channel-minor). A batch of 32-pixel output tiles is
 * packed into the workspace, each tile as `kb` consecutive 2 KiB K-tiles. Inside a K-tile the
 * layout is the blocked activation layout: vector r holds pixels 2r and 2r+1 of the tile,
 * halfword (k * 2 + q) is column k of pixel 2r + q. Taps outside the input read as zero, as do
 * the pad column, rows past the output and the K padding. Tap (ky, kx) reads input row
 * oy * sh + ky * dh - pad_top and column ox * sw + kx * dw - pad_left (dh = dw = 1 when not dilated). */
#include <stdint.h>

static inline void hx_im2col_tile(uint16_t *dst, const uint16_t *src, uint32_t ah, uint32_t aw,
                                  uint32_t ac, uint32_t a_plane, uint32_t oh, uint32_t ow,
                                  const uint32_t *p, uint32_t kb, uint32_t tile, uint32_t dh,
                                  uint32_t dw) {
    const uint32_t opw = (ow + 1u) & ~1u, apw = (aw + 1u) & ~1u, kh = p[0], kw = p[1], sh = p[2],
                   sw = p[3];
    const int32_t pt = (int32_t)p[4], pl = (int32_t)p[5];
    for (uint32_t m = 0; m < 32u; m++) {
        const uint32_t pixel = tile * 32u + m, oy = pixel / opw, ox = pixel % opw;
        const int valid = oy < oh && ox < ow;
        uint16_t *d = dst + (m >> 1) * 64u + (m & 1u);
        uint32_t i = 0;
        for (uint32_t ky = 0; ky < kh; ky++)
            for (uint32_t kx = 0; kx < kw; kx++) {
                const int32_t iy = (int32_t)(oy * sh + ky * dh) - pt,
                              ix = (int32_t)(ox * sw + kx * dw) - pl;
                if (valid && iy >= 0 && iy < (int32_t)ah && ix >= 0 && ix < (int32_t)aw) {
                    const uint32_t at = (uint32_t)iy * apw + (uint32_t)ix;
                    const uint16_t *s = src + (at >> 1) * 64u + (at & 1u);
                    for (uint32_t c = 0; c < ac; c++, i++)
                        d[(i >> 5) * 1024u + (i & 31u) * 2u] = s[(c >> 5) * a_plane + (c & 31u) * 2u];
                } else
                    for (uint32_t c = 0; c < ac; c++, i++)
                        d[(i >> 5) * 1024u + (i & 31u) * 2u] = 0;
            }
        for (; i < kb * 32u; i++)
            d[(i >> 5) * 1024u + (i & 31u) * 2u] = 0;
    }
}
#endif
