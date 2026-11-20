// SPDX-License-Identifier: LGPL-3.0-or-later
// Host check of the im2col operand layout used by the DSP's blocked-FP16 dense convolutions.
// The matrix product is emulated from the documented operand layouts; the real matrix unit is
// exercised only by the simulator and device tests.
#include "tensor/fp16_pack.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
void check(bool ok, const char *what) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s\n", what);
        std::exit(1);
    }
}
uint16_t to_half(float f) { // round to nearest even, normal range only (inputs are small)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t e = int32_t((u >> 23) & 0xffu) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (e <= 0)
        return uint16_t(sign);
    uint32_t r = (m + 0xfffu + ((m >> 13) & 1u)) >> 13;
    if (r == 0x400u) {
        r = 0;
        e++;
    }
    return uint16_t(sign | (uint32_t(e) << 10) | r);
}
float from_half(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
    if (!e)
        return 0.0f;
    uint32_t u = sign | ((e + 112u) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float rounded(float f) { return from_half(to_half(f)); }

struct Case {
    uint32_t ih, iw, ic, co, kh, kw, sh, sw, pt, pl, oh, ow;
    uint32_t dh = 1, dw = 1; // dilation: tap (ky, kx) reads ky * dh, kx * dw
};
void run(const Case &c) {
    std::mt19937 rng(c.ih * 131 + c.iw * 17 + c.ic);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const uint32_t apw = (c.iw + 1) & ~1u, opw = (c.ow + 1) & ~1u;
    const uint32_t atiles = (c.ih * apw + 31) / 32, a_plane = atiles * 1024; // halfwords
    const uint32_t blocks_in = (c.ic + 31) / 32;
    std::vector<float> in(c.ih * c.iw * c.ic);
    for (auto &v : in)
        v = rounded(dist(rng));
    std::vector<uint16_t> src(blocks_in * a_plane, 0xdead); // poison: pad lanes must not leak
    for (uint32_t y = 0; y < c.ih; y++)
        for (uint32_t x = 0; x < c.iw; x++)
            for (uint32_t ch = 0; ch < c.ic; ch++) {
                uint32_t pix = y * apw + x;
                src[(ch >> 5) * a_plane + (pix >> 1) * 64 + (ch & 31) * 2 + (pix & 1)] =
                    to_half(in[(y * c.iw + x) * c.ic + ch]);
            }
    const uint32_t K = c.kh * c.kw * c.ic, kb = (K + 31) / 32, nb = (c.co + 31) / 32;
    std::vector<float> w(K * c.co);
    for (auto &v : w)
        v = rounded(dist(rng));
    // Packed weights exactly as hxfg_writer.pack_hmx_weights: [nblock][ktile][16][32][2].
    std::vector<float> packed(nb * kb * 16 * 32 * 2, 0.0f);
    for (uint32_t k = 0; k < K; k++)
        for (uint32_t n = 0; n < c.co; n++) {
            uint32_t kt = k / 32, r = (k % 32) / 2, s = k % 2, b = n / 32, nn = n % 32;
            packed[(((b * kb + kt) * 16 + r) * 32 + nn) * 2 + s] = w[k * c.co + n];
        }
    const uint32_t otiles = (c.oh * opw + 31) / 32;
    uint32_t p[11] = {c.kh, c.kw, c.sh, c.sw, c.pt, c.pl, 0, 0, 0, 0, kb};
    std::vector<uint16_t> tile(kb * 1024, 0xbeef);
    int checked = 0;
    for (uint32_t t = 0; t < otiles; t++) {
        hx_im2col_tile(tile.data(), src.data(), c.ih, c.iw, c.ic, a_plane, c.oh, c.ow, p, kb, t,
                       c.dh, c.dw);
        for (uint32_t m = 0; m < 32; m++) {
            uint32_t pixel = t * 32 + m, oy = pixel / opw, ox = pixel % opw;
            if (oy >= c.oh || ox >= c.ow)
                continue;
            for (uint32_t n = 0; n < c.co; n++) {
                // Emulated product from the packed operands.
                double got = 0;
                for (uint32_t k = 0; k < kb * 32; k++) {
                    float a = from_half(tile[(k / 32) * 1024 + (m >> 1) * 64 + (k & 31) * 2 + (m & 1)]);
                    uint32_t kt = k / 32, r = (k % 32) / 2, s = k % 2, b = n / 32, nn = n % 32;
                    got += a * packed[(((b * kb + kt) * 16 + r) * 32 + nn) * 2 + s];
                }
                // Independent direct convolution.
                double want = 0;
                for (uint32_t ky = 0; ky < c.kh; ky++)
                    for (uint32_t kx = 0; kx < c.kw; kx++) {
                        int32_t iy = int32_t(oy * c.sh + ky * c.dh) - int32_t(c.pt),
                                ix = int32_t(ox * c.sw + kx * c.dw) - int32_t(c.pl);
                        if (iy < 0 || iy >= int32_t(c.ih) || ix < 0 || ix >= int32_t(c.iw))
                            continue;
                        for (uint32_t ch = 0; ch < c.ic; ch++)
                            want += in[(iy * c.iw + ix) * c.ic + ch] *
                                    w[((ky * c.kw + kx) * c.ic + ch) * c.co + n];
                    }
                check(std::fabs(got - want) < 1e-4, "im2col product differs from direct convolution");
                checked++;
            }
        }
    }
    check(checked == int(c.oh * c.ow * c.co), "not every output was checked");
}
} // namespace
int main() {
    const Case cases[] = {
        {7, 9, 5, 6, 3, 3, 1, 1, 1, 1, 7, 9},      // odd width, small channel counts
        {8, 8, 40, 3, 3, 3, 2, 2, 1, 1, 4, 4},     // two channel blocks, stride 2
        {9, 7, 33, 4, 1, 1, 2, 2, 0, 0, 5, 4},     // strided pointwise
        {6, 6, 3, 2, 5, 5, 1, 1, 2, 2, 6, 6},      // 5x5
        {10, 11, 32, 40, 7, 7, 2, 2, 3, 3, 5, 6},  // 7x7 stride 2, two output blocks, K = 1568
        {8, 8, 8, 8, 3, 3, 2, 2, 0, 0, 3, 3},      // SAME_UPPER-style: no leading padding
        {5, 12, 4, 4, 3, 1, 1, 1, 1, 0, 5, 12},    // non-square kernel
        {1, 1, 64, 32, 3, 3, 1, 1, 1, 1, 1, 1},    // 1x1 map, padding only
        {9, 9, 4, 5, 3, 3, 1, 1, 2, 2, 9, 9, 2, 2},    // dilation 2, SAME-style padding
        {11, 8, 33, 3, 3, 3, 2, 2, 1, 0, 4, 3, 3, 1},  // dilation 3 rows only, stride 2, K = 297
        {7, 7, 3, 2, 3, 3, 1, 1, 3, 3, 7, 7, 3, 3},    // dilation 3, pad 3 (< extent 7)
        {20, 20, 3, 2, 3, 3, 1, 1, 0, 0, 4, 4, 8, 8},  // dilation 8, extent 17, no padding
        {12, 16, 40, 40, 5, 3, 1, 2, 2, 1, 8, 5, 2, 4},  // dilation 2x4, two blocks, stride (1,2)
    };
    for (const auto &c : cases)
        run(c);
    std::puts("im2col operand layout matches direct convolution for every case");
}
