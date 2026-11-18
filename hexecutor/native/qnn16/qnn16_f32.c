/* Float32 execution forms of the convolution-family leaves (qnn16_f32.h). */
#include "qnn16_f32.h"
#include <math.h>
#include "qnn16_half.h"

#define BIG 3.0e38f

static uint16_t fi_word_half(const uint8_t *fi, uint32_t word, uint32_t high) {
    uint32_t v = *(const uint32_t *)(fi + word * 4u);
    return (uint16_t)(high ? v >> 16 : v);
}
static float scale_of(const uint8_t *fi, uint32_t tables, uint32_t c) {
    return hf(fi_word_half(fi + (c / 32u) * tables * 256u, c % 32u, 0));
}

void qnn16_prepare_epilogue(Qnn16Epilogue e, const uint8_t *fi, uint32_t tables, uint32_t channels) {
    for (uint32_t c = 0; c < channels; c++) {
        const uint8_t *table = fi + (c / 32u) * tables * 256u;
        uint32_t lane = c % 32u;
        e.bias[c] = hf(fi_word_half(table, 32u + lane, 1));
        /* A second table clamps at zero and its words' high halves (6.0). */
        e.lo[c] = tables == 2 ? 0.0f : -BIG;
        e.hi[c] = tables == 2 ? hf(fi_word_half(table + 256u, lane, 1)) : BIG;
    }
}
void qnn16_prepare_dw3x3(float *w, const uint16_t *weights, const uint8_t *fi, uint32_t tables,
                         uint32_t channels) {
    for (uint32_t c = 0; c < channels; c++) {
        uint32_t base = (c / 32u) * 576u + ((c % 32u) / 2u) * 4u + (c % 2u) * 3u;
        float s = scale_of(fi, tables, c);
        for (uint32_t ky = 0; ky < 3; ky++)
            for (uint32_t kx = 0; kx < 3; kx++)
                w[c * 9u + ky * 3u + kx] = hf(weights[base + (ky * 3u + 2u - kx) * 64u]) * s;
    }
}
void qnn16_prepare_dw3x3_s2(float *w, const uint16_t *weights, const uint8_t *fi, uint32_t tables,
                            uint32_t channels, uint32_t compact) {
    static const uint8_t tile_of_tap[9] = {1, 4, 0, 7, 8, 6, 3, 5, 2};
    for (uint32_t c = 0; c < channels; c++) {
        uint32_t g = c / 32u, lane = c % 32u;
        float s = scale_of(fi, tables, c);
        for (uint32_t k = 0; k < 9; k++)
            w[c * 9u + k] = hf(compact ? weights[(g * 9u + k) * 64u + (lane / 2u) * 4u + (lane % 2u) * 3u]
                                       : weights[(g * 9u + tile_of_tap[k]) * 1024u + (lane / 2u) * 64u +
                                                 lane * 2u + lane % 2u]) * s;
    }
}
void qnn16_prepare_dw1x1(float *w, const uint16_t *weights, const uint8_t *fi, uint32_t tables,
                         uint32_t channels) {
    for (uint32_t c = 0; c < channels; c++) {
        float s = scale_of(fi, tables, c);
        for (uint32_t j = 0; j < 2; j++)
            w[c * 2u + j] = hf(weights[(c / 32u) * 1024u + ((c % 32u) / 2u) * 4u + j * 3u]) * s;
    }
}
void qnn16_prepare_conv3x3(float *w, const uint16_t *weights, const uint8_t *fi, uint32_t tables,
                           uint32_t ic, uint32_t logical_ic, uint32_t oc) {
    uint32_t kg = (ic + 31u) / 32u;
    for (uint32_t co = 0; co < oc; co++) {
        uint32_t g = co / 32u, lane = co % 32u;
        float s = scale_of(fi, tables, co);
        for (uint32_t ky = 0; ky < 3; ky++)
            for (uint32_t kx = 0; kx < 3; kx++)
                for (uint32_t ci = 0; ci < logical_ic; ci++) {
                    uint32_t q = ky * 3u + 2u - kx;
                    uint32_t wi = (((g * kg + ci / 32u) * 9u + q) * 16u + (ci % 32u) / 2u) * 64u + lane * 2u + ci % 2u;
                    w[(co * 9u + ky * 3u + kx) * logical_ic + ci] = hf(weights[wi]) * s;
                }
    }
}
/* ic is the packed input-channel count; the logical tensor may have fewer (logical_ic). */
void qnn16_prepare_pw_swish(float *w, Qnn16Epilogue e, const uint16_t *weights, const uint8_t *fi, uint32_t ic, uint32_t logical_ic, uint32_t oc) {
    uint32_t pairs = (ic + 1u) / 2u;
    for (uint32_t co = 0; co < oc; co++) {
        const uint8_t *table = fi + ((co / 32u) * 5u + 4u) * 256u;
        uint32_t lane = co % 32u;
        float scale = hf(fi_word_half(table, lane, 0));
        e.bias[co] = hf(fi_word_half(table, 32u + lane, 1));
        for (uint32_t ci = 0; ci < logical_ic; ci++)
            w[co * logical_ic + ci] = hf(weights[((co / 32u) * pairs + ci / 2u) * 64u + lane * 2u + ci % 2u]) * scale;
    }
}
void qnn16_pw_swish_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oc, const float *w, Qnn16Epilogue e, uint32_t part, uint32_t parts) {
    uint32_t pixels = xs.b * xs.h * xs.w;
    for (uint32_t p = part; p < pixels; p += parts)
        for (uint32_t co = 0; co < oc; co++) {
            float v = e.bias[co];
            for (uint32_t ci = 0; ci < xs.c; ci++) v += x[p * xs.c + ci] * w[co * xs.c + ci];
            y[p * oc + co] = v / (1.f + expf(-v));
        }
}
/* Six-table Fi: a pointwise convolution fused with QAIRT's piecewise tanh. Table 3 holds the slope of the
 * central segment times the convolution scale in its low halves, so scale = that half / unit_slope. Every
 * table carries the bias. The activation here is the mathematical tanh. */
void qnn16_prepare_pw_tanh(float *w, Qnn16Epilogue e, const uint16_t *weights, const uint8_t *fi, uint32_t ic,
                           uint32_t logical_ic, uint32_t oc, float unit_slope) {
    uint32_t pairs = (ic + 1u) / 2u;
    for (uint32_t co = 0; co < oc; co++) {
        const uint8_t *table = fi + ((co / 32u) * 6u + 3u) * 256u;
        uint32_t lane = co % 32u;
        float scale = hf(fi_word_half(table, lane, 0)) / unit_slope;
        e.bias[co] = hf(fi_word_half(table, 32u + lane, 1));
        for (uint32_t ci = 0; ci < logical_ic; ci++)
            w[co * logical_ic + ci] = hf(weights[((co / 32u) * pairs + ci / 2u) * 64u + lane * 2u + ci % 2u]) * scale;
    }
}
void qnn16_pw_tanh_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oc, const float *w, Qnn16Epilogue e, uint32_t part, uint32_t parts) {
    uint32_t pixels = xs.b * xs.h * xs.w;
    for (uint32_t p = part; p < pixels; p += parts)
        for (uint32_t co = 0; co < oc; co++) {
            float v = e.bias[co];
            for (uint32_t ci = 0; ci < xs.c; ci++) v += x[p * xs.c + ci] * w[co * xs.c + ci];
            y[p * oc + co] = tanhf(v);
        }
}
void qnn16_prepare_conv3x3_s2(float *w, const uint16_t *weights, const uint8_t *fi, uint32_t tables,
                              uint32_t ic, uint32_t oc) {
    static const uint8_t tile[9] = {1,4,0,7,8,6,3,5,2};
    uint32_t kg = ic / 32u;
    for (uint32_t co = 0; co < oc; co++)
        for (uint32_t k = 0; k < 9; k++)
            for (uint32_t ci = 0; ci < ic; ci++) {
                uint32_t wi = ((((co / 32u) * kg + ci / 32u) * 9u + tile[k]) * 16u +
                               (ci % 32u) / 2u) * 64u + (co % 32u) * 2u + ci % 2u;
                w[(co * 9u + k) * ic + ci] = hf(weights[wi]) * scale_of(fi, tables, co);
            }
}
void qnn16_prepare_pw(float *w, const uint16_t *weights, const uint8_t *fi, uint32_t tables, uint32_t ic,
                      uint32_t logical_ic, uint32_t oc) {
    uint32_t pairs = (ic + 1u) / 2u;
    for (uint32_t co = 0; co < oc; co++) {
        float s = scale_of(fi, tables, co);
        for (uint32_t ci = 0; ci < logical_ic; ci++)
            w[co * logical_ic + ci] = hf(weights[((co / 32u) * pairs + ci / 2u) * 64u + (co % 32u) * 2u + ci % 2u]) * s;
    }
}

static inline float finish(float acc, Qnn16Epilogue e, uint32_t c) {
    acc += e.bias[c];
    if (acc < e.lo[c])
        acc = e.lo[c];
    if (acc > e.hi[c])
        acc = e.hi[c];
    return acc;
}

void qnn16_dw3x3_f32(float *y, const float *x, Qnn16Shape xs, const float *w, Qnn16Epilogue e,
                     uint32_t part, uint32_t parts) {
    uint32_t oh = xs.h - 2u, ow = xs.w - 2u, c = xs.c;
    for (uint32_t r = part; r < xs.b * oh; r += parts) {
        uint32_t b = r / oh, oy = r % oh;
        for (uint32_t ox = 0; ox < ow; ox++) {
            float *out = y + (r * ow + ox) * c;
            const float *in = x + ((b * xs.h + oy) * xs.w + ox) * c;
            for (uint32_t ch = 0; ch < c; ch++) {
                const float *k = w + ch * 9u;
                float acc = 0;
                for (uint32_t ky = 0; ky < 3; ky++) {
                    const float *row = in + ky * xs.w * c + ch;
                    acc += row[0] * k[ky * 3u] + row[c] * k[ky * 3u + 1u] + row[2u * c] * k[ky * 3u + 2u];
                }
                out[ch] = finish(acc, e, ch);
            }
        }
    }
}
void qnn16_dw3x3_s2_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oh, uint32_t ow, uint32_t channels, const float *w,
                        Qnn16Epilogue e, uint32_t part, uint32_t parts) {
    /* Space-to-depth input (4C channels): tap row ky reads phase ky%2 of row
     * y + (ky==2); tap column kx reads phase (kx==1) of column x + (kx>0). */
    static const uint8_t off[3] = {0, 1, 1}, phase[3] = {0, 1, 0};
    uint32_t c = channels;
    for (uint32_t r = part; r < xs.b * oh; r += parts) {
        uint32_t b = r / oh, oy = r % oh;
        for (uint32_t ox = 0; ox < ow; ox++) {
            float *out = y + (r * ow + ox) * c;
            for (uint32_t ch = 0; ch < c; ch++) {
                uint32_t g = ch / 32u, lane = ch % 32u;
                float acc = 0;
                for (uint32_t ky = 0; ky < 3; ky++)
                    for (uint32_t kx = 0; kx < 3; kx++) {
                        uint32_t q = phase[ky] * 2u + (kx == 1);
                        uint32_t iy = oy + (ky == 2), ix = ox + off[kx];
                        acc += x[((b * xs.h + iy) * xs.w + ix) * xs.c + (g * 4u + q) * 32u + lane] *
                               w[ch * 9u + ky * 3u + kx];
                    }
                out[ch] = finish(acc, e, ch);
            }
        }
    }
}
void qnn16_conv3x3_s2_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oh, uint32_t ow,
                          uint32_t oc, const float *w, Qnn16Epilogue e, uint32_t part, uint32_t parts) {
    uint32_t ic = xs.c / 4u;
    for (uint32_t r = part; r < xs.b * oh; r += parts) {
        uint32_t b = r / oh, oy = r % oh;
        for (uint32_t ox = 0; ox < ow; ox++)
            for (uint32_t co = 0; co < oc; co++) {
                float acc = 0;
                for (uint32_t ky = 0; ky < 3; ky++)
                    for (uint32_t kx = 0; kx < 3; kx++) {
                        uint32_t phase = (ky % 2u) * 2u + (kx == 1u);
                        const float *in = x + ((b * xs.h + oy + (ky == 2u)) * xs.w + ox + (kx > 0u)) * xs.c;
                        const float *kw = w + (co * 9u + ky * 3u + kx) * ic;
                        for (uint32_t ci = 0; ci < ic; ci++)
                            acc += in[((ci / 32u) * 4u + phase) * 32u + ci % 32u] * kw[ci];
                    }
                y[(r * ow + ox) * oc + co] = finish(acc, e, co);
            }
    }
}
void qnn16_dw1x1_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oc, const float *w, Qnn16Epilogue e,
                     uint32_t part, uint32_t parts) {
    for (uint32_t r = part; r < xs.b * xs.h; r += parts)
        for (uint32_t px = 0; px < xs.w; px++) {
            const float *in = x + (r * xs.w + px) * xs.c;
            float *out = y + (r * xs.w + px) * oc;
            for (uint32_t co = 0; co < oc; co++)
                out[co] = finish(in[(co / 32u) * 64u + co % 32u] * w[co * 2u] + in[(co / 32u) * 64u + 32u + co % 32u] * w[co * 2u + 1u], e, co);
        }
}
void qnn16_conv3x3_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oc, const float *w, Qnn16Epilogue e,
                       uint32_t part, uint32_t parts) {
    uint32_t oh = xs.h - 2u, ow = xs.w - 2u, ic = xs.c;
    for (uint32_t r = part; r < xs.b * oh; r += parts) {
        uint32_t b = r / oh, oy = r % oh;
        for (uint32_t ox = 0; ox < ow; ox++) {
            float *out = y + (r * ow + ox) * oc;
            for (uint32_t co = 0; co < oc; co++) {
                float acc = 0;
                for (uint32_t ky = 0; ky < 3; ky++)
                    for (uint32_t kx = 0; kx < 3; kx++) {
                        const float *in = x + ((b * xs.h + oy + ky) * xs.w + ox + kx) * ic;
                        const float *k = w + (co * 9u + ky * 3u + kx) * ic;
                        for (uint32_t ci = 0; ci < ic; ci++)
                            acc += in[ci] * k[ci];
                    }
                out[co] = finish(acc, e, co);
            }
        }
    }
}
void qnn16_pw_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oc, const float *w, Qnn16Epilogue e,
                  uint32_t part, uint32_t parts) {
    for (uint32_t r = part; r < xs.b * xs.h; r += parts)
        for (uint32_t px = 0; px < xs.w; px++) {
            const float *in = x + (r * xs.w + px) * xs.c;
            float *out = y + (r * xs.w + px) * oc;
            for (uint32_t co = 0; co < oc; co++) {
                const float *k = w + co * xs.c;
                float acc = 0;
                for (uint32_t ci = 0; ci < xs.c; ci++)
                    acc += in[ci] * k[ci];
                out[co] = finish(acc, e, co);
            }
        }
}
void qnn16_s2d_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oh, uint32_t ow, uint32_t part,
                   uint32_t parts) {
    uint32_t oc = ((xs.c + 31u) / 32u) * 128u;
    for (uint32_t r = part; r < xs.b * oh; r += parts) {
        uint32_t b = r / oh, oy = r % oh;
        for (uint32_t ox = 0; ox < ow; ox++) {
            for (uint32_t j = 0; j < oc; j++) y[(r * ow + ox) * oc + j] = 0.f;
            for (uint32_t q = 0; q < 4; q++) {
                uint32_t iy = 2u * oy + (q >> 1);
                int64_t ix = (int64_t)2 * ox - (q & 1u);
                int in = iy < xs.h && ix >= 0 && ix < xs.w;
                const float *src = in ? x + ((b * xs.h + iy) * xs.w + (uint32_t)ix) * xs.c : 0;
                for (uint32_t ch = 0; ch < xs.c; ch++)
                    y[(r * ow + ox) * oc + ((ch / 32u) * 4u + q) * 32u + ch % 32u] = in ? src[ch] : 0.0f;
            }
        }
    }
}

/* Crouton half index of (y, x, c) inside a 2 KiB FP16 block. */
static inline uint32_t tile_half(uint32_t y, uint32_t x, uint32_t c) {
    return (((y & 7u) * 2u + ((x & 3u) >> 1)) * 32u + (c & 31u)) * 2u + (x & 1u);
}
static const uint16_t *block_of(const Qnn16Tensor *t, uint32_t b, uint32_t fy, uint32_t fx, uint32_t fc) {
    const Qnn16Shape f = t->full;
    uint32_t nh = (f.h + 7u) / 8u, nw = (f.w + 3u) / 4u, nc = (f.c + 31u) / 32u;
    uint32_t off = t->block_offsets[((b * nh + fy / 8u) * nw + fx / 4u) * nc + fc / 32u];
    return off == 0xffffffffu ? 0 : (const uint16_t *)(t->base + off);
}
int qnn16_gather_f32(float *out, const Qnn16Tensor *t, uint32_t part, uint32_t parts) {
    const Qnn16Shape d = t->dims, p = t->padding, f = t->full;
    for (uint32_t r = part; r < d.b * d.h; r += parts) {
        uint32_t b = r / d.h, yy = r % d.h, fb = b + p.b, fy = yy + p.h;
        float *row = out + r * d.w * d.c;
        for (uint32_t xx = 0; xx < d.w; xx++) {
            uint32_t fx = xx + p.w;
            float *o = row + xx * d.c;
            if (t->direct) {
                const uint16_t *src = (const uint16_t *)(t->base + t->direct_offset) +
                                      ((fb * f.h + fy) * f.w + fx) * f.c + p.c;
                for (uint32_t c = 0; c < d.c; c++)
                    o[c] = hf(src[c]);
                continue;
            }
            const uint16_t *blk = 0;
            for (uint32_t c = 0; c < d.c; c++) {
                uint32_t fc = c + p.c;
                if (!c || !(fc & 31u))
                    blk = block_of(t, fb, fy, fx, fc);
                o[c] = blk ? hf(blk[tile_half(fy, fx, fc)]) : 0.0f;
            }
        }
    }
    return 0;
}
int qnn16_scatter_f32(Qnn16Tensor *t, const float *in, uint32_t part, uint32_t parts) {
    const Qnn16Shape d = t->dims, p = t->padding, f = t->full;
    for (uint32_t r = part; r < d.b * d.h; r += parts) {
        uint32_t b = r / d.h, yy = r % d.h, fb = b + p.b, fy = yy + p.h;
        const float *row = in + r * d.w * d.c;
        for (uint32_t xx = 0; xx < d.w; xx++) {
            uint32_t fx = xx + p.w;
            const float *v = row + xx * d.c;
            if (t->direct) {
                uint16_t *dst = (uint16_t *)(t->base + t->direct_offset) + ((fb * f.h + fy) * f.w + fx) * f.c + p.c;
                for (uint32_t c = 0; c < d.c; c++)
                    dst[c] = fh(v[c]);
                continue;
            }
            uint16_t *blk = 0;
            for (uint32_t c = 0; c < d.c; c++) {
                uint32_t fc = c + p.c;
                if (!c || !(fc & 31u))
                    blk = (uint16_t *)block_of(t, fb, fy, fx, fc);
                if (blk)
                    blk[tile_half(fy, fx, fc)] = fh(v[c]);
            }
        }
    }
    return 0;
}
