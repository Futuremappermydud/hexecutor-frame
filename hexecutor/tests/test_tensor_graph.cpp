// SPDX-License-Identifier: LGPL-3.0-or-later
#include "tensor/graph.h"
#include "hexecutor/executor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
#define check(b) do { if (!(b)) { std::cerr << "check failed at line " << __LINE__ << std::endl; throw std::runtime_error("tensor test failed"); } } while(0)
void near(float a, float b) { check(std::abs(a - b) < 1e-5f); }
struct Fixture {
    std::vector<uint8_t> image, constants = std::vector<uint8_t>(4096), arena;
    HxGraph graph{};
    Fixture(std::vector<HxTensor> tensors, HxOp op, std::vector<uint32_t> inputs) {
        uint32_t total = 0;
        for (auto &t : tensors) {
            t.offset = total;
            t.stride = (t.c + 31) & ~31u;
            total += t.h * t.w * t.stride * 4;
        }
        arena.resize(total);
        HxHeader h{};
        h.magic = HXFG_MAGIC;
        h.version = 1;
        h.tensors = tensors.size();
        h.ops = 1;
        h.inputs = inputs.size();
        h.outputs = 1;
        h.arena_bytes = total;
        h.constant_bytes = constants.size();
        h.tensor_offset = 256;
        h.op_offset = 256 + tensors.size() * 32;
        h.io_offset = h.op_offset + 64;
        h.bytes = h.io_offset + (h.inputs + 1) * 4;
        image.resize(h.bytes);
        std::memcpy(image.data(), &h, 256);
        std::memcpy(image.data() + 256, tensors.data(), tensors.size() * 32);
        std::memcpy(image.data() + h.op_offset, &op, 64);
        inputs.push_back(op.out);
        std::memcpy(image.data() + h.io_offset, inputs.data(), inputs.size() * 4);
    }
    uint32_t bad = 0;
    uint32_t open() {
        return hx_graph_open(&graph, image.data(), image.size(), constants.data(), constants.size(),
                             arena.data(), arena.size(), &bad);
    }
    HxTensor &tensor(size_t i) { return reinterpret_cast<HxTensor *>(image.data() + 256)[i]; }
    HxOp &op() {
        return *reinterpret_cast<HxOp *>(image.data() +
                                         reinterpret_cast<HxHeader *>(image.data())->op_offset);
    }
    void constant(size_t off, std::initializer_list<float> v) {
        std::memcpy(constants.data() + off, v.begin(), v.size() * 4);
    }
    void constant_i32(size_t off, std::initializer_list<int32_t> v) {
        std::memcpy(constants.data() + off, v.begin(), v.size() * 4);
    }
    void constant(size_t off, const std::vector<float> &v) {
        std::memcpy(constants.data() + off, v.data(), v.size() * 4);
    }
    void input(size_t i, const std::vector<float> &v) {
        check(open() == 0);
        check(v.size() == hx_tensor_elements(&tensor(i)));
        hx_pack(reinterpret_cast<float *>(arena.data() + tensor(i).offset), v.data(), &tensor(i));
    }
    void input_i32(size_t i, const std::vector<int32_t> &v) {
        check(open() == 0);
        check(v.size() == hx_tensor_elements(&tensor(i)));
        hx_pack(reinterpret_cast<float *>(arena.data() + tensor(i).offset), reinterpret_cast<const float *>(v.data()), &tensor(i));
    }
    std::vector<float> run() {
        check(open() == 0);
        for (uint32_t part = 0; part < 3; part++)
            hx_execute_op(&graph, 0, part, 3);
        auto &t = tensor(op().out);
        std::vector<float> out(hx_tensor_elements(&t));
        hx_unpack(out.data(), reinterpret_cast<float *>(arena.data() + t.offset), &t);
        return out;
    }
    std::vector<int32_t> run_i32() {
        check(open() == 0);
        for (uint32_t part = 0; part < 3; part++)
            hx_execute_op(&graph, 0, part, 3);
        auto &t = tensor(op().out);
        std::vector<int32_t> out(hx_tensor_elements(&t));
        hx_unpack(reinterpret_cast<float *>(out.data()), reinterpret_cast<const float *>(arena.data() + t.offset), &t);
        return out;
    }
};
HxTensor shape(uint32_t h, uint32_t w, uint32_t c) { return {0, h, w, c, 0, {0, 0, 0}}; }
inline HxTensor shape_i32(uint32_t h, uint32_t w, uint32_t c) { return {0, h, w, c, 0, {0, 0, HX_DTYPE_INT32}}; }
inline HxTensor shape_bool(uint32_t h, uint32_t w, uint32_t c) { return {0, h, w, c, 0, {0, 0, HX_DTYPE_BOOL}}; }
HxOp op(uint32_t kind, uint32_t out = 1, uint32_t y = HXFG_NONE, uint32_t z = HXFG_NONE) {
    HxOp o{};
    o.kind = kind;
    o.x = 0;
    o.y = y;
    o.z = z;
    o.out = out;
    return o;
}
} // namespace
int main() {
    auto conv = op(HX_CONV);
    uint32_t p[] = {1, 1, 1, 1, 0, 0, 0, 128, 3};
    std::memcpy(conv.p, p, sizeof p);
    Fixture f({shape(1, 2, 1), shape(1, 2, 1)}, conv, {0});
    f.constant(0, {2});
    f.constant(128, {-1});
    f.input(0, {-1, 4});
    auto r = f.run();
    near(r[0], 0);
    near(r[1], 6);
    f.op().p[6] = 4000;
    check(f.open() == HX_SHAPE);
    f.op().p[6] = 0;
    f.tensor(1).offset = 0;
    check(f.open() == HX_ALIAS);
    f.tensor(1).offset = 256;
    f.tensor(0).stride = 1;
    check(f.open() == HX_SHAPE);
    f.tensor(0).stride = 32;
    f.op().kind = 999;
    check(f.open() == HX_SHAPE);
    auto dw = op(HX_DEPTHWISE);
    uint32_t pd[] = {2, 1, 1, 1, 0, 0, 0, 256, 0};
    std::memcpy(dw.p, pd, sizeof pd);
    Fixture d({shape(2, 1, 2), shape(1, 1, 2)}, dw, {0});
    d.constant(0, {1, 2});
    d.constant(128, {3, 4});
    d.constant(256, {5, 6});
    d.input(0, {1, 2, 3, 4});
    r = d.run();
    near(r[0], 15);
    near(r[1], 26);
    auto pool = op(HX_MAXPOOL);
    pool.p[0] = 2;
    pool.p[1] = 1;
    pool.p[2] = pool.p[3] = 1;
    Fixture m({shape(2, 1, 1), shape(1, 1, 1)}, pool, {0});
    m.input(0, {-4, -2});
    r = m.run();
    near(r[0], -2);
    auto plus = op(HX_ADD, 2, 1);
    plus.p[0] = 3;
    Fixture a({shape(1, 2, 1), shape(1, 2, 1), shape(1, 2, 1)}, plus, {0, 1});
    a.input(0, {-4, 7});
    a.input(1, {1, 1});
    r = a.run();
    near(r[0], 0);
    near(r[1], 6);
    auto pre = op(HX_PRELU);
    Fixture pr({shape(1, 1, 2), shape(1, 1, 2)}, pre, {0});
    pr.constant(0, {0.25, 0.5});
    pr.input(0, {-4, 3});
    r = pr.run();
    near(r[0], -1);
    near(r[1], 3);
    auto resize = op(HX_RESIZE);
    resize.p[1] = 1;
    Fixture re({shape(1, 2, 1), shape(1, 4, 1)}, resize, {0});
    re.input(0, {0, 4});
    r = re.run();
    near(r[0], 0);
    near(r[1], 1);
    near(r[2], 3);
    near(r[3], 4);
    {  // Nearest, asymmetric floor: 2x up, a 5 -> 3 ratio (sources 0, 1, 3), and both axes.
        auto nn = op(HX_RESIZE);
        nn.p[2] = 1;
        Fixture fn({shape(1, 2, 1), shape(1, 4, 1)}, nn, {0});
        fn.input(0, {0, 4});
        check(fn.run() == std::vector<float>({0, 0, 4, 4}));
        Fixture fq({shape(1, 5, 1), shape(1, 3, 1)}, nn, {0});
        fq.input(0, {0, 10, 20, 30, 40});
        check(fq.run() == std::vector<float>({0, 10, 30}));
        Fixture f2({shape(2, 3, 2), shape(3, 2, 2)}, nn, {0});
        std::vector<float> in(12), want;
        for (uint32_t i = 0; i < 12; i++)
            in[i] = float(i + 1);
        f2.input(0, in);
        for (uint32_t oy = 0; oy < 3; oy++)
            for (uint32_t ox = 0; ox < 2; ox++)
                for (uint32_t c = 0; c < 2; c++) {
                    uint32_t sy = oy * 2 / 3, sx = ox * 3 / 2;
                    want.push_back(float((sy * 3 + sx) * 2 + c + 1));
                }
        check(f2.run() == want);
        // Nearest takes no bilinear flags, and the flag is required to be 0 or 1.
        auto bad = op(HX_RESIZE);
        bad.p[0] = 1;
        bad.p[2] = 1;
        Fixture fb({shape(1, 2, 1), shape(1, 4, 1)}, bad, {0});
        check(fb.open() == HX_SHAPE);
        fb.op().p[0] = 0;
        fb.op().p[1] = 1;
        check(fb.open() == HX_SHAPE);
        fb.op().p[1] = 0;
        fb.op().p[2] = 2;
        check(fb.open() == HX_SHAPE);
        fb.op().p[2] = 1;
        check(fb.open() == 0);
    }
    auto mean = op(HX_MEAN);
    Fixture me({shape(2, 2, 1), shape(1, 1, 1)}, mean, {0});
    me.input(0, {1, 2, 3, 4});
    r = me.run();
    near(r[0], 2.5);
    auto pad = op(HX_PAD);
    pad.p[2] = 1;
    Fixture pa({shape(1, 1, 1), shape(1, 1, 3)}, pad, {0});
    pa.input(0, {4});
    r = pa.run();
    near(r[0], 0);
    near(r[1], 4);
    near(r[2], 0);
    auto reshape = op(HX_RESHAPE);
    Fixture rs({shape(1, 2, 3), shape(1, 3, 2)}, reshape, {0});
    rs.input(0, {1, 2, 3, 4, 5, 6});
    r = rs.run();
    for (int i = 0; i < 6; i++)
        near(r[i], i + 1);
    {  // Transpose: all six permutations of a 2x3x5 tensor; out[o] = in[s] with s[p[k]] = o[k].
        const uint32_t perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                      {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        const uint32_t ad[3] = {2, 3, 5};
        std::vector<float> src(2 * 3 * 5);
        for (uint32_t i = 0; i < src.size(); i++)
            src[i] = float(i);
        for (const auto &perm : perms) {
            const uint32_t bd[3] = {ad[perm[0]], ad[perm[1]], ad[perm[2]]};
            auto tr = op(HX_TRANSPOSE);
            for (uint32_t k = 0; k < 3; k++)
                tr.p[k] = perm[k];
            Fixture t({shape(2, 3, 5), shape(bd[0], bd[1], bd[2])}, tr, {0});
            t.input(0, src);
            auto r = t.run();
            check(r.size() == size_t(bd[0]) * bd[1] * bd[2]);
            for (uint32_t o0 = 0; o0 < bd[0]; o0++)
                for (uint32_t o1 = 0; o1 < bd[1]; o1++)
                    for (uint32_t o2 = 0; o2 < bd[2]; o2++) {
                        const uint32_t o[3] = {o0, o1, o2};
                        uint32_t s[3];
                        for (uint32_t k = 0; k < 3; k++)
                            s[perm[k]] = o[k];
                        near(r[(o0 * bd[1] + o1) * bd[2] + o2],
                             src[(s[0] * 3 + s[1]) * 5 + s[2]]);
                    }
        }
        // Rejections: not a permutation, a wrong output shape, a stray field, a stray operand.
        auto bad = op(HX_TRANSPOSE);
        bad.p[0] = 1;
        bad.p[1] = 0;
        bad.p[2] = 2;
        Fixture tb({shape(2, 3, 5), shape(3, 2, 5)}, bad, {0});
        check(tb.open() == HX_OK);
        tb.op().p[2] = 0;
        check(tb.open() == HX_SHAPE);
        tb.op().p[2] = 2;
        tb.op().p[3] = 1;
        check(tb.open() == HX_SHAPE);
        tb.op().p[3] = 0;
        tb.op().p[1] = 1;
        check(tb.open() == HX_SHAPE);
        tb.op().p[1] = 0;
        tb.op().p[0] = 3;
        check(tb.open() == HX_SHAPE);
        tb.op().p[0] = 1;
        tb.op().y = 0;
        check(tb.open() == HX_OPERATOR);
        Fixture tw({shape(2, 3, 5), shape(2, 5, 3)}, bad, {0});
        tw.op().p[0] = 0;
        tw.op().p[1] = 2;
        tw.op().p[2] = 1;
        check(tw.open() == HX_OK);
        tw.tensor(1).h = 3; /* keeps the output's offset and stride; only the dims change */
        tw.tensor(1).w = 2;
        tw.tensor(1).c = 5;
        check(tw.open() == HX_SHAPE);
    }
    {  // NCHW flatten (p[0] = 1): element n = c*H*W + h*W + w of the HxWxC source, into 1x1xN.
        auto fl = op(HX_RESHAPE);
        fl.p[0] = 1;
        Fixture ff({shape(2, 3, 5), shape(1, 1, 30)}, fl, {0});
        std::vector<float> src(30);
        for (uint32_t i = 0; i < 30; i++)
            src[i] = float(i);
        ff.input(0, src);
        auto r = ff.run();
        for (uint32_t n = 0; n < 30; n++) {
            const uint32_t c = n / 6, h = (n % 6) / 3, w = n % 3;
            near(r[n], src[(h * 3 + w) * 5 + c]);
        }
        // Unflatten (p[0] = 2): dst(h, w, c) = src[c*H*W + h*W + w] from a 1x1xN source.
        auto un = op(HX_RESHAPE);
        un.p[0] = 2;
        Fixture uf({shape(1, 1, 30), shape(2, 3, 5)}, un, {0});
        uf.input(0, src);
        r = uf.run();
        for (uint32_t h = 0; h < 2; h++)
            for (uint32_t w = 0; w < 3; w++)
                for (uint32_t c = 0; c < 5; c++)
                    near(r[(h * 3 + w) * 5 + c], src[c * 6 + h * 3 + w]);
        // Flatten then unflatten through the NCHW order returns the source (a round trip).
        Fixture rt({shape(1, 1, 30), shape(2, 3, 5)}, un, {0});
        rt.input(0, ff.run());
        check(rt.run() == src);
        // Rejections: flatten needs a 1x1 destination, unflatten a 1x1 source, flags are 0..2,
        // and the flag leaves p[1] and p[2] unused.
        Fixture f2({shape(2, 3, 5), shape(2, 15, 1)}, fl, {0});
        check(f2.open() == HX_SHAPE);
        Fixture f3({shape(2, 3, 5), shape(1, 1, 29)}, fl, {0});
        check(f3.open() == HX_SHAPE);
        Fixture f4({shape(2, 3, 5), shape(2, 3, 5)}, un, {0});
        check(f4.open() == HX_SHAPE);
        Fixture f5({shape(2, 3, 5), shape(1, 1, 30)}, fl, {0});
        f5.op().p[0] = 3;
        check(f5.open() == HX_SHAPE);
        f5.op().p[0] = 1;
        f5.op().p[1] = 1;
        check(f5.open() == HX_SHAPE);
        f5.op().p[1] = 0;
        check(f5.open() == HX_OK);
    }
    auto concat = op(HX_CONCAT, 2, 1);
    concat.p[0] = 2;
    Fixture ca({shape(1, 2, 1), shape(1, 2, 2), shape(1, 2, 3)}, concat, {0, 1});
    ca.input(0, {1, 2});
    ca.input(1, {3, 4, 5, 6});
    r = ca.run();
    check(r == std::vector<float>({1, 3, 4, 2, 5, 6}));
    // Slice: offsets on height, width and channels; the output is the sub-box.
    auto slice = op(HX_SLICE);
    slice.p[0] = 1;
    slice.p[1] = 2;
    slice.p[2] = 5;
    Fixture sl({shape(3, 4, 40), shape(2, 2, 20)}, slice, {0});
    std::vector<float> src(3 * 4 * 40);
    for (size_t i = 0; i < src.size(); i++)
        src[i] = float(i);
    sl.input(0, src);
    r = sl.run();
    check(r.size() == 2 * 2 * 20);
    for (uint32_t oy = 0; oy < 2; oy++)
        for (uint32_t ox = 0; ox < 2; ox++)
            for (uint32_t c = 0; c < 20; c++)
                near(r[(oy * 2 + ox) * 20 + c], float(((oy + 1) * 4 + ox + 2) * 40 + c + 5));
    Fixture sb({shape(3, 4, 40), shape(2, 2, 20)}, op(HX_SLICE), {0});
    sb.op().p[0] = 1;
    sb.op().p[1] = 2;
    sb.op().p[2] = 5;
    check(sb.open() == HX_OK);
    sb.op().p[2] = 21; /* 21 + 20 channels exceeds 40 */
    check(sb.open() == HX_SHAPE);
    sb.op().p[2] = 5;
    sb.op().p[0] = 2; /* 2 + 2 rows exceeds 3 */
    check(sb.open() == HX_SHAPE);
    sb.op().p[0] = 1;
    sb.op().p[1] = 3; /* 3 + 2 columns exceeds 4 */
    check(sb.open() == HX_SHAPE);
    sb.op().p[1] = 2;
    sb.op().p[3] = 1; /* unused field */
    check(sb.open() == HX_SHAPE);
    sb.op().p[3] = 0;
    sb.op().y = 0;
    check(sb.open() == HX_OPERATOR);
    {  // Zero insert (ConvTranspose upsampling): stride 2 with and without padding, two channels.
        auto zi = op(HX_ZEROINSERT);
        zi.p[0] = zi.p[1] = 2;
        const std::vector<float> in = {10, -10, 20, -20, 30, -30, 40, -40};
        Fixture z0({shape(2, 2, 2), shape(3, 3, 2)}, zi, {0});
        z0.input(0, in);
        check(z0.run() == std::vector<float>({10, -10, 0, 0, 20, -20,  //
                                              0, 0, 0, 0, 0, 0,        //
                                              30, -30, 0, 0, 40, -40}));
        // Pad 1 on top and left: source pixels land on odd rows and columns of the 4x4 output.
        zi.p[2] = zi.p[3] = 1;
        Fixture z1({shape(2, 2, 2), shape(4, 4, 2)}, zi, {0});
        z1.input(0, in);
        std::vector<float> want(4 * 4 * 2, 0.0f);
        for (uint32_t sy = 0; sy < 2; sy++)
            for (uint32_t sx = 0; sx < 2; sx++)
                for (uint32_t c = 0; c < 2; c++)
                    want[((1 + 2 * sy) * 4 + 1 + 2 * sx) * 2 + c] = in[(sy * 2 + sx) * 2 + c];
        check(z1.run() == want);
        // Pad 1 with output 3x3 leaves no room for the last source column (1 + 2 + 1 > 3).
        check(z0.open() == 0);
        z0.op().p[2] = z0.op().p[3] = 1;
        check(z0.open() == HX_SHAPE);
        z0.op().p[2] = z0.op().p[3] = 0;
        z0.op().p[0] = 0;
        check(z0.open() == HX_SHAPE);
        z0.op().p[0] = 9;
        check(z0.open() == HX_SHAPE);
        z0.op().p[0] = 2;
        z0.op().p[4] = 1; /* unused field */
        check(z0.open() == HX_SHAPE);
        z0.op().p[4] = 0;
        z0.tensor(1).c = 1; /* channels must be unchanged */
        check(z0.open() == HX_SHAPE);
        z0.tensor(1).c = 2;
        check(z0.open() == 0);
    }
    auto log = op(HX_LOGISTIC);
    Fixture l({shape(1, 1, 1), shape(1, 1, 1)}, log, {0});
    l.input(0, {0});
    near(l.run()[0], 0.5);
    // Standalone activations.
    auto act = [&](uint32_t fn, std::vector<float> in, float alpha = 0, float beta = 0) {
        auto o = op(HX_ACT);
        o.p[0] = fn;
        std::memcpy(&o.p[1], &alpha, 4);
        std::memcpy(&o.p[2], &beta, 4);
        Fixture fx({shape(1, in.size(), 1), shape(1, in.size(), 1)}, o, {0});
        fx.input(0, in);
        return fx.run();
    };
    r = act(HX_ACT_HARDSWISH, {-4, -3, 0, 3, 6});
    near(r[0], 0); near(r[1], 0); near(r[2], 0); near(r[3], 3); near(r[4], 6);
    r = act(HX_ACT_HARDSWISH, {1});
    near(r[0], 1.0f * 4.0f / 6.0f);
    r = act(HX_ACT_SILU, {0, 1, -2});
    near(r[0], 0); near(r[1], 1.0f / (1.0f + std::exp(-1.0f))); near(r[2], -2.0f / (1.0f + std::exp(2.0f)));
    r = act(HX_ACT_TANH, {0, 0.5f, -3, 50, -50});
    for (int i = 0; i < 5; i++) {
        float xs[] = {0, 0.5f, -3, 50, -50};
        near(r[i], std::tanh(xs[i]));
    }
    r = act(HX_ACT_LEAKY, {-2, 3}, 0.1f);
    near(r[0], -0.2f); near(r[1], 3);
    r = act(HX_ACT_HARDSIGMOID, {-5, 0, 1, 5}, 1.0f / 6.0f, 0.5f);
    near(r[0], 0); near(r[1], 0.5f); near(r[2], 0.5f + 1.0f / 6.0f); near(r[3], 1);
    r = act(HX_ACT_RELU6, {-1, 3, 9});
    near(r[0], 0); near(r[1], 3); near(r[2], 6);
    r = act(HX_ACT_GELU, {0, 1, -1, 2.5f});
    for (int i = 0; i < 4; i++) {
        float xs[] = {0, 1, -1, 2.5f};
        float want = 0.5f * xs[i] * (1.0f + std::erf(xs[i] / std::sqrt(2.0f)));
        check(std::abs(r[i] - want) < 1e-5f);
    }
    r = act(HX_ACT_EXP, {0, 1, -2, 0.5f});
    near(r[0], 1); near(r[1], std::exp(1.0f)); near(r[2], std::exp(-2.0f)); near(r[3], std::exp(0.5f));
    r = act(HX_ACT_SQRT, {0, 4, 2, 0.25f});  // positive inputs only: sqrt of negative is NaN
    near(r[0], 0); near(r[1], 2); near(r[2], std::sqrt(2.0f)); near(r[3], 0.5f);
    r = act(HX_ACT_ABS, {-3, 0, 2.5f, -0.5f});
    near(r[0], 3); near(r[1], 0); near(r[2], 2.5f); near(r[3], 0.5f);
    r = act(HX_ACT_NEG, {-2, 0, 5});
    near(r[0], 2); near(r[1], 0); near(r[2], -5);
    r = act(HX_ACT_ERF, {0, 1, -1, 2.5f});
    for (int i = 0; i < 4; i++) {
        float xs[] = {0, 1, -1, 2.5f};
        check(std::abs(r[i] - std::erf(xs[i])) < 1e-5f);
    }
    {  // Unknown function, and parameters on a function that takes none, are rejected.
        auto o = op(HX_ACT);
        o.p[0] = HX_ACT_COUNT;
        Fixture fx({shape(1, 1, 1), shape(1, 1, 1)}, o, {0});
        check(fx.open() == HX_SHAPE);
        fx.op().p[0] = HX_ACT_SILU;
        fx.op().p[1] = 1;
        check(fx.open() == HX_SHAPE);
    }
    // Binary: tensor/tensor, tensor/broadcast, constant, reversed, and rejected shapes.
    auto bin = [&](uint32_t opcode, uint32_t rev) {
        auto o = op(HX_BINARY, 2, 1);
        o.p[0] = opcode;
        o.p[3] = rev;
        return o;
    };
    {
        Fixture fx({shape(1, 2, 2), shape(1, 2, 2), shape(1, 2, 2)}, bin(HX_BIN_MUL, 0), {0, 1});
        fx.input(0, {1, 2, 3, 4});
        fx.input(1, {5, 6, 7, 8});
        check(fx.run() == std::vector<float>({5, 12, 21, 32}));
        fx.op().p[0] = HX_BIN_SUB;
        check(fx.run() == std::vector<float>({-4, -4, -4, -4}));
        fx.op().p[3] = 1;
        check(fx.run() == std::vector<float>({4, 4, 4, 4}));
    }
    {  // Squeeze-excite style: x (2x2x2) times a 1x1x2 gate, and times a 1x1x1 scalar.
        Fixture fx({shape(2, 2, 2), shape(1, 1, 2), shape(2, 2, 2)}, bin(HX_BIN_MUL, 0), {0, 1});
        fx.input(0, {1, 2, 3, 4, 5, 6, 7, 8});
        fx.input(1, {10, 100});
        check(fx.run() == std::vector<float>({10, 200, 30, 400, 50, 600, 70, 800}));
        Fixture fs({shape(1, 2, 2), shape(1, 1, 1), shape(1, 2, 2)}, bin(HX_BIN_ADD, 0), {0, 1});
        fs.input(0, {1, 2, 3, 4});
        fs.input(1, {10});
        check(fs.run() == std::vector<float>({11, 12, 13, 14}));
        Fixture fr({shape(2, 1, 1), shape(2, 1, 1), shape(2, 1, 1)}, bin(HX_BIN_ADD, 0), {0, 1});
        fr.input(0, {1, 2});
        fr.input(1, {10, 20});
        check(fr.run() == std::vector<float>({11, 22}));  // row broadcast not requested: same shape
        Fixture fh({shape(2, 2, 1), shape(2, 1, 1), shape(2, 2, 1)}, bin(HX_BIN_SUB, 0), {0, 1});
        fh.input(0, {1, 2, 3, 4});
        fh.input(1, {10, 20});
        check(fh.run() == std::vector<float>({-9, -8, -17, -16}));  // width broadcast
        fh.tensor(1).c = 2;  // a channel count that is neither one nor the output's
        fh.tensor(1).stride = 32;
        check(fh.open() == HX_SHAPE);
    }
    {  // Constant per-channel operand, plain and reversed; y must then be NONE.
        auto o = op(HX_BINARY, 1);
        o.p[0] = HX_BIN_SUB;
        o.p[1] = 1;
        o.p[2] = 0;
        Fixture fx({shape(1, 2, 2), shape(1, 2, 2)}, o, {0});
        fx.constant(0, {10, 20});
        fx.input(0, {1, 2, 3, 4});
        check(fx.run() == std::vector<float>({-9, -18, -7, -16}));
        fx.op().p[3] = 1;
        check(fx.run() == std::vector<float>({9, 18, 7, 16}));
        fx.op().p[2] = 4000;
        check(fx.open() == HX_SHAPE);
        fx.op().p[2] = 0;
        fx.op().y = 0;
        check(fx.open() == HX_OPERATOR);
    }
    // Average pooling: padded windows exclude or include the padding.
    {
        auto o = op(HX_AVGPOOL);
        o.p[0] = o.p[1] = 3;
        o.p[2] = o.p[3] = 1;
        o.p[4] = o.p[5] = 1;
        Fixture fx({shape(2, 2, 1), shape(2, 2, 1)}, o, {0});
        fx.input(0, {1, 2, 3, 4});
        r = fx.run();
        for (float v : r) near(v, 2.5f);  // all four taps are always valid: sum 10 / 4
        fx.op().p[6] = 1;
        r = fx.run();
        for (float v : r) near(v, 10.0f / 9.0f);
        fx.op().p[6] = 2;
        check(fx.open() == HX_SHAPE);
    }
    {  // Plain 2x2/2 average over channels that differ per pixel.
        auto o = op(HX_AVGPOOL);
        o.p[0] = o.p[1] = o.p[2] = o.p[3] = 2;
        Fixture fx({shape(2, 2, 2), shape(1, 1, 2)}, o, {0});
        fx.input(0, {1, 10, 2, 20, 3, 30, 4, 40});
        r = fx.run();
        near(r[0], 2.5f);
        near(r[1], 25.0f);
    }
    {  // Softmax over channels per pixel: stable at large magnitudes, padding lanes zero.
        auto ref = [](const std::vector<float> &v) {
            double peak = v[0], sum = 0;
            for (float x : v)
                peak = x > peak ? x : peak;
            for (float x : v)
                sum += std::exp(double(x) - peak);
            std::vector<float> out;
            for (float x : v)
                out.push_back(float(std::exp(double(x) - peak) / sum));
            return out;
        };
        Fixture fx({shape(1, 2, 3), shape(1, 2, 3)}, op(HX_SOFTMAX), {0});
        fx.input(0, {1000, 1001, 1002, -5, 0, 5});
        auto sm = fx.run();
        auto e0 = ref({1000, 1001, 1002}), e1 = ref({-5, 0, 5});
        for (int i = 0; i < 3; i++) {
            near(sm[i], e0[i]);
            near(sm[3 + i], e1[i]);
        }
        near(sm[0] + sm[1] + sm[2], 1.0f);
        near(sm[3] + sm[4] + sm[5], 1.0f);
        {  // 40 channels span two channel blocks; values spread far beyond exp's float range.
            std::vector<float> v;
            for (int i = 0; i < 40; i++)
                v.push_back(i % 2 ? 900.0f - float(i) : 0.5f * float(i));
            Fixture wide({shape(1, 1, 40), shape(1, 1, 40)}, op(HX_SOFTMAX), {0});
            wide.input(0, v);
            auto w = wide.run();
            auto e = ref(v);
            float total = 0;
            for (int i = 0; i < 40; i++) {
                near(w[i], e[i]);
                total += w[i];
            }
            near(total, 1.0f);
            const float *raw = reinterpret_cast<const float *>(wide.arena.data() + wide.tensor(1).offset);
            for (uint32_t k = 40; k < 64; k++)
                check(raw[k] == 0.0f);
        }
        Fixture bad({shape(1, 1, 3), shape(1, 1, 3)}, op(HX_SOFTMAX), {0});
        check(bad.open() == 0);
        bad.op().p[0] = 1;
        check(bad.open() == HX_SHAPE);
        bad.op().p[0] = 0;
        bad.op().y = 0;
        check(bad.open() == HX_OPERATOR);
        bad.op().y = HXFG_NONE;
        bad.tensor(1).c = 2;
        check(bad.open() == HX_SHAPE);
    }
    {  // HX_REDUCE: every operation over every non-empty HWC subset of 3x4x40, keepdims shapes,
       // padding lanes zero. The reference filters source coordinates independently.
        std::vector<float> src(3 * 4 * 40);
        for (uint32_t y = 0; y < 3; y++)
            for (uint32_t x = 0; x < 4; x++)
                for (uint32_t c = 0; c < 40; c++)
                    src[(y * 4 + x) * 40 + c] = float(int((y * 7 + x * 5 + c * 3) % 23) - 11);
        for (uint32_t kind = 0; kind < HX_RED_COUNT; kind++)
            for (uint32_t mask = 1; mask < 8; mask++) {
                const uint32_t oh = mask & 1u ? 1 : 3, ow = mask & 2u ? 1 : 4,
                               oc = mask & 4u ? 1 : 40;
                auto o = op(HX_REDUCE);
                o.p[0] = kind;
                o.p[1] = mask;
                Fixture fx({shape(3, 4, 40), shape(oh, ow, oc)}, o, {0});
                fx.input(0, src);
                auto r = fx.run();
                check(r.size() == oh * ow * oc);
                const HxTensor &ot = fx.tensor(1);
                const float *raw = reinterpret_cast<const float *>(fx.arena.data() + ot.offset);
                for (uint32_t oy = 0; oy < oh; oy++)
                    for (uint32_t ox = 0; ox < ow; ox++)
                        for (uint32_t c = 0; c < ot.stride; c++) {
                            if (c >= oc) {
                                check(raw[(oy * ow + ox) * ot.stride + c] == 0.0f);
                                continue;
                            }
                            double acc = kind == HX_RED_MAX   ? -1e30
                                         : kind == HX_RED_MIN ? 1e30
                                                              : 0.0,
                                   n = 0;
                            for (uint32_t y = 0; y < 3; y++)
                                for (uint32_t x = 0; x < 4; x++)
                                    for (uint32_t k = 0; k < 40; k++) {
                                        if (!((mask & 1u) || y == oy) || !((mask & 2u) || x == ox) ||
                                            !((mask & 4u) || k == c))
                                            continue;
                                        const double v = src[(y * 4 + x) * 40 + k];
                                        acc = kind == HX_RED_MAX   ? std::max(acc, v)
                                              : kind == HX_RED_MIN ? std::min(acc, v)
                                                                   : acc + v;
                                        n++;
                                    }
                            const double want = kind == HX_RED_MEAN ? acc / n : acc;
                            check(std::abs(double(r[(oy * ow + ox) * oc + c]) - want) < 1e-4);
                        }
            }
        {  // Shape and operand rejections: operation, axes, keepdims layout, y operand.
            auto o = op(HX_REDUCE);
            o.p[1] = 1;
            Fixture fx({shape(3, 4, 40), shape(1, 4, 40)}, o, {0});
            check(fx.open() == HX_OK);
            fx.op().p[0] = HX_RED_COUNT;
            check(fx.open() == HX_SHAPE);
            fx.op().p[0] = HX_RED_SUM;
            fx.op().p[1] = 0;
            check(fx.open() == HX_SHAPE);
            fx.op().p[1] = 8;
            check(fx.open() == HX_SHAPE);
            fx.op().p[1] = 1;
            fx.op().p[2] = 1;
            check(fx.open() == HX_SHAPE);
            fx.op().p[2] = 0;
            fx.tensor(1).h = 3;  // h is reduced, so the output keeps size one there
            check(fx.open() == HX_SHAPE);
            fx.tensor(1).h = 1;
            fx.tensor(1).w = 1;  // w is kept
            check(fx.open() == HX_SHAPE);
            fx.tensor(1).w = 4;
            fx.op().y = 0;
            check(fx.open() == HX_OPERATOR);
            fx.op().y = HXFG_NONE;
            check(fx.open() == HX_OK);
            Fixture ch({shape(3, 4, 40), shape(3, 4, 40)}, op(HX_REDUCE), {0});
            ch.op().p[1] = 4;  // channels reduced: the output must have one channel
            check(ch.open() == HX_SHAPE);
            ch.tensor(1).c = 1;
            ch.tensor(1).stride = 32;
            check(ch.open() == HX_OK);
        }
    }
    {  // Normalisation (HX_NORM): channel mode, group mode with groups 1, 2 and C, rejections.
        auto nrm = op(HX_NORM);
        nrm.p[0] = HX_NORM_CHANNEL;
        nrm.p[3] = 0;
        nrm.p[4] = 128;
        // Pixel (1, 3): mean 2, variance 1. Pixel (4, 0): mean 2, variance 4. eps = 0.
        Fixture cn({shape(1, 2, 2), shape(1, 2, 2)}, nrm, {0});
        cn.constant(0, {2, 1});
        cn.constant(128, {0.5f, -1});
        cn.input(0, {1, 3, 4, 0});
        check(cn.run() == std::vector<float>({-1.5f, 0.0f, 2.5f, -2.0f}));
        cn.op().p[1] = 2;  // channel mode takes no group count
        check(cn.open() == HX_SHAPE);
        cn.op().p[1] = 0;
        cn.op().p[5] = 1;  // unused field
        check(cn.open() == HX_SHAPE);
        cn.op().p[5] = 0;
        cn.op().y = 0;
        check(cn.open() == HX_OPERATOR);
        cn.op().y = HXFG_NONE;
        cn.tensor(1).w = 1;  // output shape must equal the input shape
        check(cn.open() == HX_SHAPE);
        cn.tensor(1).w = 2;
        cn.op().p[0] = HX_NORM_COUNT;
        check(cn.open() == HX_SHAPE);
        cn.op().p[0] = HX_NORM_CHANNEL;
        cn.op().p[3] = 4000;  // gamma outside the constant pool
        check(cn.open() == HX_SHAPE);
        cn.op().p[3] = 0;
        check(cn.open() == HX_OK);
    }
    {  // Group mode, groups = 1 (LayerNorm over C, H, W): one mean and variance over all values.
        auto gn = op(HX_NORM);
        gn.p[0] = HX_NORM_GROUP;
        gn.p[1] = 1;
        gn.p[3] = 0;
        gn.p[4] = 128;
        Fixture g1({shape(1, 2, 2), shape(1, 2, 2)}, gn, {0});
        g1.constant(0, {1, 1});
        g1.input(0, {1, 3, 4, 0});
        // Mean 2, variance 2.5 over {1, 3, 4, 0}.
        const float s = std::sqrt(2.5f);
        auto r1 = g1.run();
        near(r1[0], -1.0f / s);
        near(r1[1], 1.0f / s);
        near(r1[2], 2.0f / s);
        near(r1[3], -2.0f / s);
    }
    {  // Group mode, groups = C (InstanceNorm): per channel over the pixels, with gamma and beta.
        auto gn = op(HX_NORM);
        gn.p[0] = HX_NORM_GROUP;
        gn.p[1] = 2;
        gn.p[3] = 0;
        gn.p[4] = 128;
        Fixture ic({shape(1, 2, 2), shape(1, 2, 2)}, gn, {0});
        ic.constant(0, {3, 1});
        ic.constant(128, {0.5f, -1});
        ic.input(0, {1, 3, 4, 0});
        // Channel 0 over {1, 4}: mean 2.5, sd 1.5. Channel 1 over {3, 0}: mean 1.5, sd 1.5.
        check(ic.run() == std::vector<float>({-2.5f, 0.0f, 3.5f, -2.0f}));
    }
    {  // Group mode, two groups of two channels over four pixels' worth of values, padded lanes 0.
        auto gn = op(HX_NORM);
        gn.p[0] = HX_NORM_GROUP;
        gn.p[1] = 2;
        gn.p[3] = 0;
        gn.p[4] = 128;
        Fixture g2({shape(1, 2, 4), shape(1, 2, 4)}, gn, {0});
        g2.constant(0, {1, 1, 1, 1});
        g2.input(0, {1, 2, 10, 20, 3, 4, 30, 40});
        // Group 0 {1, 2, 3, 4}: mean 2.5, variance 1.25. Group 1 {10, 20, 30, 40}: mean 25,
        // variance 125.
        const float s0 = std::sqrt(1.25f), s1 = std::sqrt(125.0f);
        auto r2 = g2.run();
        const float want[] = {-1.5f / s0, -0.5f / s0, -15.0f / s1, -5.0f / s1,
                              0.5f / s0,  1.5f / s0,  5.0f / s1,  15.0f / s1};
        for (int i = 0; i < 8; i++)
            near(r2[i], want[i]);
        std::memset(g2.arena.data() + g2.tensor(1).offset, 0xff, hx_tensor_bytes(&g2.tensor(1)));
        g2.run();
        const float *raw = reinterpret_cast<const float *>(g2.arena.data() + g2.tensor(1).offset);
        for (uint32_t pixel = 0; pixel < 2; pixel++)
            for (uint32_t k = 4; k < 32; k++)
                check(raw[pixel * 32 + k] == 0.0f);
        gn.p[1] = 3;  // 4 channels do not split into 3 groups
        Fixture bad({shape(1, 2, 4), shape(1, 2, 4)}, gn, {0});
        check(bad.open() == HX_SHAPE);
        bad.op().p[1] = 0;  // group mode needs a group count
        check(bad.open() == HX_SHAPE);
        bad.op().p[1] = 5;  // more groups than channels
        check(bad.open() == HX_SHAPE);
        bad.op().p[1] = 2;
        bad.op().p[0] = HX_NORM_COUNT;
        check(bad.open() == HX_SHAPE);
        bad.op().p[0] = HX_NORM_GROUP;
        bad.op().p[3] = 64;  // not 128-byte aligned
        check(bad.open() == HX_SHAPE);
        bad.op().p[3] = 0;
        bad.op().p[5] = 1;  // unused field
        check(bad.open() == HX_SHAPE);
        bad.op().p[5] = 0;
        bad.op().y = 0;
        check(bad.open() == HX_OPERATOR);
        bad.op().y = HXFG_NONE;
        check(bad.open() == HX_OK);
    }
    {  // Constant pools up to 64 MiB open; one byte past that, or a session placing them past
       // 96 MiB, does not.
        auto op0 = op(HX_LOGISTIC);
        Fixture big({shape(1, 1, 1), shape(1, 1, 1)}, op0, {0});
        auto setup = [&](uint32_t bytes) {
            big.constants.assign(bytes, 0);
            reinterpret_cast<HxHeader *>(big.image.data())->constant_bytes = bytes;
        };
        setup(0x2000080u);
        check(big.open() == HX_OK);
        setup(HXFG_MAX_CONSTANT_BYTES);
        check(big.open() == HX_OK);
        setup(HXFG_MAX_CONSTANT_BYTES + 128u);
        check(big.open() == HX_HEADER);
        HxSessionConfig s{};
        s.magic = HXFG_CONFIG_MAGIC;
        s.version = 1;
        s.code_bytes = 4096;
        s.graphs = 1;
        s.graph[0] = {0x100000, 256, 0x2000000, HXFG_MAX_CONSTANT_BYTES, 0x1000000, 4096};
        check(hx_session_config_valid(&s)); // ends exactly at 96 MiB
        s.graph[0].constant_bytes += 128;
        check(!hx_session_config_valid(&s));
        s.graph[0].constants = 0x1ffff80; // below the region start
        s.graph[0].constant_bytes = 4096;
        check(!hx_session_config_valid(&s));
    }
    // Bootstrap must reject every memory ownership overlap and boundary.
    HxSessionConfig c{};
    c.magic = HXFG_CONFIG_MAGIC;
    c.version = 1;
    c.code_bytes = 4096;
    c.graphs = 2;
    c.graph[0] = {0x100000, 256, 0x2000000, 4096, 0x1000000, 4096};
    c.graph[1] = {0x101000, 256, 0x2001000, 4096, 0x1001000, 4096};
    check(hx_session_config_valid(&c));
    c.graph[1].arena = c.graph[0].arena;
    check(!hx_session_config_valid(&c));
    c.graphs = 1;
    c.graph[0].constant_bytes = 0xffffffff;
    check(!hx_session_config_valid(&c));
    // Dilated convolution: 1x2 kernel, dilation 2 along width. out[x] = in[x] + 10 * in[x + 2].
    {
        auto dconv = op(HX_CONV);
        uint32_t dp[] = {1, 2, 1, 1, 0, 0, 0, 256, 0};
        std::memcpy(dconv.p, dp, sizeof dp);
        dconv.z = (1u << 8) | 2u;
        Fixture dc({shape(1, 5, 1), shape(1, 3, 1)}, dconv, {0});
        dc.constant(0, {1});
        dc.constant(128, {10});
        dc.input(0, {1, 2, 3, 4, 5});
        r = dc.run();
        check(r.size() == 3);
        near(r[0], 31);
        near(r[1], 42);
        near(r[2], 53);
    }
    // Dilated convolution in both axes with top padding: 2x2 kernel, dilation (2, 1), pad_top 1.
    // Input 3x3 holds 1..9; taps are 1, 10, 100, 1000 in row-major order.
    {
        auto dconv = op(HX_CONV);
        uint32_t dp[] = {2, 2, 1, 1, 1, 0, 0, 512, 0};
        std::memcpy(dconv.p, dp, sizeof dp);
        dconv.z = (2u << 8) | 1u;
        Fixture dc({shape(3, 3, 1), shape(2, 2, 1)}, dconv, {0});
        dc.constant(0, {1});
        dc.constant(128, {10});
        dc.constant(256, {100});
        dc.constant(384, {1000});
        dc.input(0, {1, 2, 3, 4, 5, 6, 7, 8, 9});
        r = dc.run();
        check(r.size() == 4);
        near(r[0], 5400);
        near(r[1], 6500);
        near(r[2], 8721);
        near(r[3], 9832);
        // The window reads rows relative to the dilated extent: pad_top 3 >= extent 3 is invalid.
        dc.op().p[4] = 3;
        check(dc.open() == HX_SHAPE);
    }
    // Dilated depthwise: 2x1 kernel, dilation (2, 1) on a 4x1x2 input.
    {
        auto ddw = op(HX_DEPTHWISE);
        uint32_t dp[] = {2, 1, 1, 1, 0, 0, 0, 256, 0};
        std::memcpy(ddw.p, dp, sizeof dp);
        ddw.z = (2u << 8) | 1u;
        Fixture dd({shape(4, 1, 2), shape(2, 1, 2)}, ddw, {0});
        dd.constant(0, {1, 2});
        dd.constant(128, {10, 20});
        dd.constant(256, {0.5f, -1});
        dd.input(0, {1, 10, 2, 20, 3, 30, 4, 40});
        r = dd.run();
        check(r.size() == 4);
        near(r[0], 31.5f);
        near(r[1], 619);
        near(r[2], 42.5f);
        near(r[3], 839);
    }
    // New common general-model ops: Gather, Where, ArgMax, TopK, Tile, Cast, Binary full-size const, Resize.
    {  // Gather with constant table: 4 vocab, 2 embedding dim (each row padded to 32 floats = 128 bytes)
        auto gop = op(HX_GATHER);
        gop.p[0] = 1; // mode 1: constant table
        gop.p[1] = 0; // constant offset 0
        gop.p[2] = 4; // vocab_size 4
        gop.p[3] = 1; // index_dtype float32
        Fixture fg({shape(1, 4, 1), shape(1, 4, 2)}, gop, {0});
        fg.constant(0, {10, 11});
        fg.constant(128, {20, 21});
        fg.constant(256, {30, 31});
        fg.constant(384, {40, 41});
        fg.input(0, {2, 0, 3, 1});
        r = fg.run();
        check(r == std::vector<float>({30, 31, 10, 11, 40, 41, 20, 21}));
    }
    {  // Where: condition selects between X and Y tensors
        auto wop = op(HX_WHERE, 3, 1);
        wop.z = 2;
        wop.p[0] = 0; // tensor X
        wop.p[2] = 0; // tensor Y
        Fixture fw({shape(1, 4, 1), shape(1, 4, 1), shape(1, 4, 1), shape(1, 4, 1)}, wop, {0, 1, 2});
        fw.input(0, {1, 0, 1, 0});
        fw.input(1, {10, 20, 30, 40});
        fw.input(2, {-1, -2, -3, -4});
        r = fw.run();
        check(r == std::vector<float>({10, -2, 30, -4}));
    }
    {  // ArgMax: channel argmax
        auto aop = op(HX_ARGMAX);
        aop.p[0] = 0;
        aop.p[1] = 0; // output float32
        Fixture fa({shape(1, 2, 4), shape(1, 2, 1)}, aop, {0});
        fa.input(0, {1, 5, 2, 0, 7, 3, 4, 2});
        r = fa.run();
        check(r == std::vector<float>({1, 0}));
    }
    {  // TopK: top 2 elements along channel axis
        auto topk = op(HX_TOPK);
        topk.p[0] = 2; // K = 2
        topk.p[1] = 1; // largest
        topk.p[2] = 1; // sorted
        topk.p[3] = 0; // index_dtype float
        Fixture ft({shape(1, 1, 5), shape(1, 1, 2)}, topk, {0});
        ft.input(0, {3, 10, 1, 8, 5});
        r = ft.run();
        check(r == std::vector<float>({10, 8}));
    }
    {  // Tile: replicate input across H and W
        auto top = op(HX_TILE);
        Fixture fti({shape(1, 2, 2), shape(2, 4, 2)}, top, {0});
        fti.input(0, {1, 2, 3, 4});
        r = fti.run();
        check(r.size() == 16);
        check(r == std::vector<float>({1, 2, 3, 4, 1, 2, 3, 4,
                                       1, 2, 3, 4, 1, 2, 3, 4}));
    }
    {  // Cast: float <-> int roundtrip
        auto cop = op(HX_CAST);
        cop.p[0] = 0; // float to int32 bit pattern
        Fixture fc({shape(1, 2, 1), shape(1, 2, 1)}, cop, {0});
        fc.input(0, {3.0f, -5.0f});
        r = fc.run();
        int32_t v0, v1;
        std::memcpy(&v0, &r[0], 4);
        std::memcpy(&v1, &r[1], 4);
        check(v0 == 3 && v1 == -5);
    }
    {  // Binary mode 2: full-size constant tensor add and runtime div
        auto bop = op(HX_BINARY);
        bop.p[0] = HX_BIN_ADD;
        bop.p[1] = 2; // full-size constant
        bop.p[2] = 0; // constant offset 0
        Fixture fb2({shape(1, 2, 2), shape(1, 2, 2)}, bop, {0});
        // Row 0 at 0: {10, 20}, padded to stride 32 = 128 bytes
        fb2.constant(0, {10, 20});
        // Row 1 at 128: {30, 40}
        fb2.constant(128, {30, 40});
        fb2.input(0, {1, 2, 3, 4});
        r = fb2.run();
        check(r == std::vector<float>({11, 22, 33, 44}));

        // Runtime div
        auto dop = op(HX_BINARY, 2, 1);
        dop.p[0] = HX_BIN_DIV;
        Fixture fd({shape(1, 2, 2), shape(1, 2, 2), shape(1, 2, 2)}, dop, {0, 1});
        fd.input(0, {10, 20, 30, 40});
        fd.input(1, {2, 4, 5, 8});
        r = fd.run();
        check(r == std::vector<float>({5, 5, 6, 5}));
    }
    {  // Resize: nearest round_prefer_floor and bicubic
        auto rn = op(HX_RESIZE);
        rn.p[2] = 1; // nearest
        rn.p[3] = 0; // asymmetric
        rn.p[4] = 1; // round_prefer_floor
        Fixture frn({shape(1, 2, 1), shape(1, 4, 1)}, rn, {0});
        frn.input(0, {10, 20});
        r = frn.run();
        check(r == std::vector<float>({10, 10, 20, 20}));

        auto rc = op(HX_RESIZE);
        rc.p[5] = 1; // bicubic
        rc.p[3] = 0; // asymmetric
        Fixture frc({shape(1, 2, 1), shape(1, 4, 1)}, rc, {0});
        frc.input(0, {10, 20});
        r = frc.run();
        check(r.size() == 4);
    }
    {  // ArgMin: channel argmin (float input, int32 output)
        auto amin = op(HX_ARGMIN);
        amin.p[0] = 0;
        amin.p[1] = 1; // output int32
        Fixture fa({shape(1, 2, 4), shape_i32(1, 2, 1)}, amin, {0});
        fa.input(0, {10.f, 5.f, 20.f, 30.f, 7.f, 3.f, 4.f, 2.f});
        auto r_i = fa.run_i32();
        check(r_i == std::vector<int32_t>({1, 3}));
    }
    {  // ArgMax: channel argmax (int32 input, int32 output)
        auto amax = op(HX_ARGMAX);
        amax.p[0] = 0;
        amax.p[1] = 1; // output int32
        Fixture fa({shape_i32(1, 2, 4), shape_i32(1, 2, 1)}, amax, {0});
        fa.input_i32(0, {100, 500, 200, 300, -70, -30, -40, -20});
        auto r_i = fa.run_i32();
        check(r_i == std::vector<int32_t>({1, 3}));
    }
    {  // Integer binary arithmetic: Add, Sub, Mul, Min, Max
        for (auto bkind : {HX_BIN_ADD, HX_BIN_SUB, HX_BIN_MUL, HX_BIN_MIN, HX_BIN_MAX}) {
            auto bop = op(HX_BINARY, 2, 1);
            bop.p[0] = bkind;
            Fixture fb({shape_i32(1, 2, 2), shape_i32(1, 2, 2), shape_i32(1, 2, 2)}, bop, {0, 1});
            fb.input_i32(0, {10, -20, 30, -40});
            fb.input_i32(1, {5, 2, -15, 8});
            auto r_i = fb.run_i32();
            if (bkind == HX_BIN_ADD) check(r_i == std::vector<int32_t>({15, -18, 15, -32}));
            if (bkind == HX_BIN_SUB) check(r_i == std::vector<int32_t>({5, -22, 45, -48}));
            if (bkind == HX_BIN_MUL) check(r_i == std::vector<int32_t>({50, -40, -450, -320}));
            if (bkind == HX_BIN_MIN) check(r_i == std::vector<int32_t>({5, -20, -15, -40}));
            if (bkind == HX_BIN_MAX) check(r_i == std::vector<int32_t>({10, 2, 30, 8}));
        }
    }
    {  // Comparison ops on int32 and float32: Equal, Greater, Less, GE, LE
        for (auto bkind : {HX_BIN_EQUAL, HX_BIN_GREATER, HX_BIN_LESS, HX_BIN_GE, HX_BIN_LE}) {
            auto bop = op(HX_BINARY, 2, 1);
            bop.p[0] = bkind;
            Fixture fb({shape_i32(1, 2, 2), shape_i32(1, 2, 2), shape_bool(1, 2, 2)}, bop, {0, 1});
            fb.input_i32(0, {10, 20, 30, 40});
            fb.input_i32(1, {10, 15, 35, 40});
            auto r_i = fb.run_i32();
            if (bkind == HX_BIN_EQUAL) check(r_i == std::vector<int32_t>({1, 0, 0, 1}));
            if (bkind == HX_BIN_GREATER) check(r_i == std::vector<int32_t>({0, 1, 0, 0}));
            if (bkind == HX_BIN_LESS) check(r_i == std::vector<int32_t>({0, 0, 1, 0}));
            if (bkind == HX_BIN_GE) check(r_i == std::vector<int32_t>({1, 1, 0, 1}));
            if (bkind == HX_BIN_LE) check(r_i == std::vector<int32_t>({1, 0, 1, 1}));
        }
    }
    {  // Logical ops: Not, And, Or
        auto nop = op(HX_ACT);
        nop.p[0] = HX_ACT_NOT;
        Fixture fn({shape_bool(1, 1, 4), shape_bool(1, 1, 4)}, nop, {0});
        fn.input_i32(0, {1, 0, 1, 0});
        auto r_i = fn.run_i32();
        check(r_i == std::vector<int32_t>({0, 1, 0, 1}));

        for (auto bkind : {HX_BIN_AND, HX_BIN_OR}) {
            auto bop = op(HX_BINARY, 2, 1);
            bop.p[0] = bkind;
            Fixture fb({shape_bool(1, 1, 4), shape_bool(1, 1, 4), shape_bool(1, 1, 4)}, bop, {0, 1});
            fb.input_i32(0, {1, 1, 0, 0});
            fb.input_i32(1, {1, 0, 1, 0});
            auto r_b = fb.run_i32();
            if (bkind == HX_BIN_AND) check(r_b == std::vector<int32_t>({1, 0, 0, 0}));
            if (bkind == HX_BIN_OR) check(r_b == std::vector<int32_t>({1, 1, 1, 0}));
        }
    }
    {  // Where with integer tensors and constants
        auto wop = op(HX_WHERE, 3, 1, 2);
        wop.p[0] = 0; // y tensor
        wop.p[2] = 0; // z tensor
        Fixture fw({shape_bool(1, 1, 4), shape_i32(1, 1, 4), shape_i32(1, 1, 4), shape_i32(1, 1, 4)}, wop, {0, 1, 2});
        fw.input_i32(0, {1, 0, 1, 0});
        fw.input_i32(1, {100, 200, 300, 400});
        fw.input_i32(2, {-10, -20, -30, -40});
        auto r_i = fw.run_i32();
        check(r_i == std::vector<int32_t>({100, -20, 300, -40}));

        // Where with scalar constant
        auto wop2 = op(HX_WHERE, 2, HXFG_NONE, 1);
        wop2.p[0] = 1; // scalar constant
        wop2.p[1] = 999;
        wop2.p[2] = 0; // tensor
        Fixture fw2({shape_bool(1, 1, 4), shape_i32(1, 1, 4), shape_i32(1, 1, 4)}, wop2, {0, 1});
        fw2.input_i32(0, {1, 0, 0, 1});
        fw2.input_i32(1, {5, 6, 7, 8});
        r_i = fw2.run_i32();
        check(r_i == std::vector<int32_t>({999, 6, 7, 999}));
    }
    {  // Cast: all modes (0..5)
        // Mode 0: float -> int32
        auto c0 = op(HX_CAST); c0.p[0] = 0;
        Fixture fc0({shape(1, 1, 2), shape_i32(1, 1, 2)}, c0, {0});
        fc0.input(0, {4.2f, -3.8f});
        auto r0 = fc0.run_i32();
        check(r0 == std::vector<int32_t>({4, -3}));

        // Mode 1: int32 -> float
        auto c1 = op(HX_CAST); c1.p[0] = 1;
        Fixture fc1({shape_i32(1, 1, 2), shape(1, 1, 2)}, c1, {0});
        fc1.input_i32(0, {42, -17});
        auto r1 = fc1.run();
        check(r1 == std::vector<float>({42.0f, -17.0f}));

        // Mode 2: bool -> float
        auto c2 = op(HX_CAST); c2.p[0] = 2;
        Fixture fc2({shape_bool(1, 1, 2), shape(1, 1, 2)}, c2, {0});
        fc2.input_i32(0, {1, 0});
        auto r2 = fc2.run();
        check(r2 == std::vector<float>({1.0f, 0.0f}));

        // Mode 3: float -> bool
        auto c3 = op(HX_CAST); c3.p[0] = 3;
        Fixture fc3({shape(1, 1, 3), shape_bool(1, 1, 3)}, c3, {0});
        fc3.input(0, {0.0f, 3.5f, -0.0f});
        auto r3 = fc3.run_i32();
        check(r3 == std::vector<int32_t>({0, 1, 0}));

        // Mode 4: int32 -> bool
        auto c4 = op(HX_CAST); c4.p[0] = 4;
        Fixture fc4({shape_i32(1, 1, 3), shape_bool(1, 1, 3)}, c4, {0});
        fc4.input_i32(0, {0, 15, -7});
        auto r4 = fc4.run_i32();
        check(r4 == std::vector<int32_t>({0, 1, 1}));

        // Mode 5: bool -> int32
        auto c5 = op(HX_CAST); c5.p[0] = 5;
        Fixture fc5({shape_bool(1, 1, 2), shape_i32(1, 1, 2)}, c5, {0});
        fc5.input_i32(0, {1, 0});
        auto r5 = fc5.run_i32();
        check(r5 == std::vector<int32_t>({1, 0}));
    }
    // z is only a dilation on conv/depthwise/pool, and must encode 1..8 per axis with one above 1.
    {
        auto non_conv = op(HX_PAD);
        non_conv.z = (2u << 8) | 1u;
        Fixture pz({shape(2, 1, 1), shape(3, 1, 1)}, non_conv, {0});
        check(pz.open() == HX_OPERATOR);
        auto pool = op(HX_MAXPOOL);
        pool.p[0] = 2;
        pool.p[1] = 1;
        pool.p[2] = pool.p[3] = 1;
        pool.z = (2u << 8) | 1u;
        Fixture mp({shape(3, 1, 1), shape(1, 1, 1)}, pool, {0});
        check(mp.open() == HX_OK);
        mp.input(0, {10.0f, 50.0f, 30.0f});
        auto mpr = mp.run();
        check(mpr.size() == 1 && mpr[0] == 30.0f); // 50.0f skipped by dilation 2

        auto apool = op(HX_AVGPOOL);
        apool.p[0] = 2;
        apool.p[1] = 1;
        apool.p[2] = apool.p[3] = 1;
        apool.z = (2u << 8) | 1u;
        Fixture ap({shape(3, 1, 1), shape(1, 1, 1)}, apool, {0});
        check(ap.open() == HX_OK);
        ap.input(0, {10.0f, 50.0f, 30.0f});
        auto apr = ap.run();
        check(apr.size() == 1 && apr[0] == 20.0f); // (10 + 30) / 2
        auto conv = op(HX_CONV);
        uint32_t cp[] = {1, 1, 1, 1, 0, 0, 0, 128, 0};
        std::memcpy(conv.p, cp, sizeof cp);
        for (uint32_t z : {(9u << 8) | 1u, (1u << 8) | 9u, (1u << 8) | 1u, (0u << 8) | 2u,
                           (1u << 16) | (2u << 8) | 2u}) {
            conv.z = z;
            Fixture cz({shape(1, 2, 1), shape(1, 2, 1)}, conv, {0});
            check(cz.open() == HX_OPERATOR);
        }
    }
    // Matrix product (HX_MATMUL): unbatched and batched multi-head forms
    {
        constexpr uint32_t L = 7, K = 33, N = 19;
        std::vector<float> a(L * K), bk(K * N), bn(N * K);
        for (uint32_t i = 0; i < L; i++)
            for (uint32_t k = 0; k < K; k++)
                a[i * K + k] = ((int)((i * 5 + k * 3) % 13) - 6) * 0.125f;
        for (uint32_t k = 0; k < K; k++)
            for (uint32_t n = 0; n < N; n++)
                bk[k * N + n] = ((int)((k * 7 + n * 2) % 11) - 5) * 0.25f;
        for (uint32_t n = 0; n < N; n++)
            for (uint32_t k = 0; k < K; k++)
                bn[n * K + k] = bk[k * N + n];
        auto reference = [&](bool trans, float scale) {
            std::vector<float> out(L * N);
            for (uint32_t i = 0; i < L; i++)
                for (uint32_t n = 0; n < N; n++) {
                    double acc = 0;
                    for (uint32_t k = 0; k < K; k++)
                        acc += double(a[i * K + k]) * (trans ? bn[n * K + k] : bk[k * N + n]);
                    out[i * N + n] = float(acc * scale);
                }
            return out;
        };
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            return u;
        };
        auto mm = [&](uint32_t trans, uint32_t scale_bits) {
            auto o = op(HX_MATMUL, 2, 1);
            o.p[0] = trans;
            o.p[1] = scale_bits;
            return o;
        };
        struct Case {
            bool trans;
            uint32_t scale_bits;
            float scale;
        };
        // p[1] == 0 selects 1.0f; the explicit 1.0f pattern must agree with it.
        for (Case c : {Case{false, 0, 1.0f}, Case{true, 0, 1.0f}, Case{true, bits(1.0f), 1.0f},
                       Case{false, bits(0.5f), 0.5f}, Case{true, bits(-2.0f), -2.0f}}) {
            Fixture fx({shape(L, 1, K), c.trans ? shape(N, 1, K) : shape(K, 1, N), shape(L, 1, N)},
                       mm(c.trans, c.scale_bits), {0, 1});
            fx.input(0, a);
            fx.input(1, c.trans ? bn : bk);
            auto got = fx.run();
            auto want = reference(c.trans, c.scale);
            check(got.size() == want.size());
            for (size_t i = 0; i < want.size(); i++)
                check(std::abs(got[i] - want[i]) < 1e-4f);
        }
        // Rejections: transB and scalar fields, unused fields, width, operand liveness, aliasing.
        Fixture bad({shape(L, 1, K), shape(K, 1, N), shape(L, 1, N)}, mm(0, 0), {0, 1});
        bad.input(0, a);
        bad.input(1, bk);
        bad.op().p[0] = 2;
        check(bad.open() == HX_SHAPE);
        bad.op().p[0] = 1;  // [K,N] is not [N,K] unless N == K
        check(bad.open() == HX_SHAPE);
        bad.op().p[0] = 0;
        bad.op().p[2] = 1;  // batch count 1 is a single batch and valid
        check(bad.open() == HX_OK);
        bad.op().p[2] = 0;
        check(bad.run().size() == L * N);
        bad.op().y = HXFG_NONE;
        check(bad.open() == HX_OPERATOR);
        bad.op().y = 1;
        bad.tensor(2).offset = bad.tensor(1).offset;  // output over the live operand B
        check(bad.open() == HX_ALIAS);
        Fixture wide({shape(L, 2, K), shape(K, 1, N), shape(L, 1, N)}, mm(0, 0), {0, 1});
        check(wide.open() == HX_SHAPE);  // mixed w: A is batched (w > 1) but B is not
        Fixture unset({shape(L, 1, K), shape(K, 1, N), shape(L, 1, N)}, mm(0, 0), {0});
        check(unset.open() == HX_UNINITIALIZED);
        Fixture big({shape(1, 1, 4097), shape(4097, 1, 1), shape(1, 1, 1)}, mm(0, 0), {0, 1});
        check(big.open() == HX_SHAPE);
        // Batched multi-head MatMul with w > 1: [H, Lq, K] x [H, Lk, K]^T -> [H, Lq, Lk]
        {
            constexpr uint32_t H = 3, Lq = 4, Lk = 5, D = 8;
            std::vector<float> q(H * Lq * D), k(H * Lk * D);
            for (size_t i = 0; i < q.size(); i++) q[i] = (float)((i % 7) - 3) * 0.25f;
            for (size_t i = 0; i < k.size(); i++) k[i] = (float)((i % 5) - 2) * 0.5f;

            auto bmm_op = op(HX_MATMUL, 2, 1);
            bmm_op.p[0] = 1; // transB = 1
            bmm_op.p[1] = bits(0.5f); // scale

            Fixture fxb({shape(H, Lq, D), shape(H, Lk, D), shape(H, Lq, Lk)}, bmm_op, {0, 1});
            fxb.input(0, q);
            fxb.input(1, k);
            auto got_b = fxb.run();
            check(got_b.size() == H * Lq * Lk);

            for (uint32_t h = 0; h < H; h++)
                for (uint32_t qi = 0; qi < Lq; qi++)
                    for (uint32_t ki = 0; ki < Lk; ki++) {
                        double acc = 0;
                        for (uint32_t d = 0; d < D; d++)
                            acc += double(q[(h * Lq + qi) * D + d]) * double(k[(h * Lk + ki) * D + d]);
                        float want = float(acc * 0.5f);
                        float got = got_b[(h * Lq + qi) * Lk + ki];
                        check(std::abs(got - want) < 1e-4f);
                    }
        }

        // Batched multi-head MatMul with w = 1 and p[2] = H: [H*Lq, 1, K] x [H*Lk, 1, K]^T -> [H*Lq, 1, Lk]
        {
            constexpr uint32_t H = 2, Lq = 3, Lk = 4, D = 6;
            std::vector<float> q(H * Lq * D), k(H * Lk * D);
            for (size_t i = 0; i < q.size(); i++) q[i] = (float)(i + 1) * 0.1f;
            for (size_t i = 0; i < k.size(); i++) k[i] = (float)(i % 4) * 0.2f;

            auto bmm_op = op(HX_MATMUL, 2, 1);
            bmm_op.p[0] = 1; // transB = 1
            bmm_op.p[1] = 0; // scale 1.0f
            bmm_op.p[2] = H; // batch count H

            Fixture fxb({shape(H * Lq, 1, D), shape(H * Lk, 1, D), shape(H * Lq, 1, Lk)}, bmm_op, {0, 1});
            fxb.input(0, q);
            fxb.input(1, k);
            auto got_b = fxb.run();
            check(got_b.size() == H * Lq * Lk);
            for (uint32_t h = 0; h < H; h++)
                for (uint32_t qi = 0; qi < Lq; qi++)
                    for (uint32_t ki = 0; ki < Lk; ki++) {
                        double acc = 0;
                        for (uint32_t d = 0; d < D; d++)
                            acc += double(q[(h * Lq + qi) * D + d]) * double(k[(h * Lk + ki) * D + d]);
                        float want = float(acc);
                        float got = got_b[(h * Lq + qi) * Lk + ki];
                        check(std::abs(got - want) < 1e-4f);
                    }
        }
    }

    // Softmax with causal mask and additive mask (HX_SOFTMAX and HX_ATTN_SOFTMAX)
    {
        constexpr uint32_t H = 2, Lq = 3, Lk = 3;
        std::vector<float> logits(H * Lq * Lk);
        for (size_t i = 0; i < logits.size(); i++) logits[i] = (float)(i % 5) - 2.0f;

        // Causal softmax: p[1] = 1
        auto sm = op(HX_ATTN_SOFTMAX);
        sm.p[0] = 2; // channel/key axis
        sm.p[1] = 1; // causal mask
        Fixture fc({shape(H, Lq, Lk), shape(H, Lq, Lk)}, sm, {0});
        fc.input(0, logits);
        auto got_c = fc.run();
        check(got_c.size() == H * Lq * Lk);
        for (uint32_t h = 0; h < H; h++)
            for (uint32_t q = 0; q < Lq; q++) {
                float row_sum = 0.0f;
                for (uint32_t k = 0; k < Lk; k++) {
                    float v = got_c[(h * Lq + q) * Lk + k];
                    if (k > q) {
                        check(v == 0.0f); // future tokens masked out
                    } else {
                        check(v > 0.0f);
                        row_sum += v;
                    }
                }
                check(std::abs(row_sum - 1.0f) < 1e-4f);
            }

        // Additive mask with HX_ATTN_SOFTMAX
        auto asm_op = op(HX_ATTN_SOFTMAX, 2, 1);
        asm_op.p[0] = 2; // channel axis
        asm_op.p[1] = 0; // non-causal
        std::vector<float> mask(H * Lq * Lk, 0.0f);
        // mask out last key position with large negative value
        for (uint32_t h = 0; h < H; h++)
            for (uint32_t q = 0; q < Lq; q++)
                mask[(h * Lq + q) * Lk + 2] = -10000.0f;

        Fixture fa({shape(H, Lq, Lk), shape(H, Lq, Lk), shape(H, Lq, Lk)}, asm_op, {0, 1});
        fa.input(0, logits);
        fa.input(1, mask);
        auto got_a = fa.run();
        for (uint32_t h = 0; h < H; h++)
            for (uint32_t q = 0; q < Lq; q++) {
                check(got_a[(h * Lq + q) * Lk + 2] < 1e-6f);
                float row_sum = got_a[(h * Lq + q) * Lk + 0] + got_a[(h * Lq + q) * Lk + 1];
                check(std::abs(row_sum - 1.0f) < 1e-4f);
            }
    }

    // Fused Scaled Dot-Product Attention (HX_SDPA, 27)
    {
        constexpr uint32_t H = 2, Lq = 4, Lk = 4, D = 8;
        std::vector<float> q(H * Lq * D), k(H * Lk * D), v(H * Lk * D);
        for (size_t i = 0; i < q.size(); i++) q[i] = (float)((i * 3) % 7 - 3) * 0.25f;
        for (size_t i = 0; i < k.size(); i++) k[i] = (float)((i * 5) % 9 - 4) * 0.25f;
        for (size_t i = 0; i < v.size(); i++) v[i] = (float)((i * 7) % 11 - 5) * 0.25f;

        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            return u;
        };

        for (uint32_t causal : {0u, 1u}) {
            auto sdpa = op(HX_SDPA, 3, 1, 2); // out=3, y=1 (K), z=2 (V)
            sdpa.p[0] = HXFG_NONE; // no mask
            float scale = 1.0f / std::sqrt((float)D);
            sdpa.p[1] = bits(scale);
            sdpa.p[2] = causal;

            Fixture f({shape(H, Lq, D), shape(H, Lk, D), shape(H, Lk, D), shape(H, Lq, D)}, sdpa, {0, 1, 2});
            f.input(0, q);
            f.input(1, k);
            f.input(2, v);
            auto got = f.run();
            check(got.size() == H * Lq * D);

            // Compute reference output using composed formula:
            for (uint32_t h = 0; h < H; h++) {
                for (uint32_t qi = 0; qi < Lq; qi++) {
                    std::vector<float> scores(Lk);
                    float peak = -1e30f;
                    for (uint32_t ki = 0; ki < Lk; ki++) {
                        if (causal && ki > qi) {
                            scores[ki] = -1e30f;
                            continue;
                        }
                        double dot = 0;
                        for (uint32_t d = 0; d < D; d++)
                            dot += double(q[(h * Lq + qi) * D + d]) * double(k[(h * Lk + ki) * D + d]);
                        scores[ki] = float(dot * scale);
                        if (scores[ki] > peak) peak = scores[ki];
                    }
                    float sum = 0;
                    for (uint32_t ki = 0; ki < Lk; ki++) {
                        if (causal && ki > qi) {
                            scores[ki] = 0;
                        } else {
                            scores[ki] = std::exp(scores[ki] - peak);
                            sum += scores[ki];
                        }
                    }
                    for (uint32_t ki = 0; ki < Lk; ki++)
                        scores[ki] /= sum;

                    for (uint32_t d = 0; d < D; d++) {
                        double acc = 0;
                        for (uint32_t ki = 0; ki < Lk; ki++)
                            acc += double(scores[ki]) * double(v[(h * Lk + ki) * D + d]);
                        float want = float(acc);
                        float val = got[(h * Lq + qi) * D + d];
                        check(std::abs(val - want) < 1e-4f);
                    }
                }
            }
        }
    }
    // RMSNorm (HX_RMSNORM, 22)
    {
        constexpr uint32_t D = 32;
        auto rms = op(HX_RMSNORM);
        float eps = 1e-5f;
        uint32_t eps_u;
        std::memcpy(&eps_u, &eps, 4);
        rms.p[0] = eps_u;
        rms.p[1] = 0;       // gamma offset
        rms.p[2] = HXFG_NONE; // no beta
        Fixture f({shape(1, 1, D), shape(1, 1, D)}, rms, {0});
        std::vector<float> gamma(D, 1.5f), in(D);
        for (uint32_t i = 0; i < D; i++) in[i] = (float)(i + 1) * 0.1f;
        f.constant(0, {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f,
                       1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f,
                       1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f,
                       1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f});
        f.input(0, in);
        auto out = f.run();
        check(out.size() == D);
        double ssq = 0;
        for (float v : in) ssq += double(v) * double(v);
        float inv = (float)(1.0 / std::sqrt(ssq / double(D) + double(eps)));
        for (uint32_t i = 0; i < D; i++)
            check(std::abs(out[i] - (in[i] * inv * 1.5f)) < 1e-5f);
    }
    // RMSNorm without weight (p[1] = HXFG_NONE) and rank-3 token layout (H=L, W=1, C=D)
    {
        constexpr uint32_t L = 4, D = 32;
        auto rms = op(HX_RMSNORM);
        rms.p[0] = 0;         // default eps 1e-5
        rms.p[1] = HXFG_NONE; // no weight
        rms.p[2] = HXFG_NONE; // no beta
        Fixture f({shape(L, 1, D), shape(L, 1, D)}, rms, {0});
        std::vector<float> in(L * D);
        for (uint32_t i = 0; i < L * D; i++) in[i] = (float)(i + 1) * 0.05f;
        f.input(0, in);
        auto out = f.run();
        check(out.size() == L * D);
        float eps = 1e-5f;
        for (uint32_t t = 0; t < L; t++) {
            double ssq = 0;
            for (uint32_t i = 0; i < D; i++) {
                float v = in[t * D + i];
                ssq += double(v) * double(v);
            }
            float inv = (float)(1.0 / std::sqrt(ssq / double(D) + double(eps)));
            for (uint32_t i = 0; i < D; i++) {
                float expected = in[t * D + i] * inv;
                check(std::abs(out[t * D + i] - expected) < 1e-5f);
            }
        }
    }
    // LayerNorm over last axis for rank-3 tokens (H=L, W=1, C=D)
    {
        constexpr uint32_t L = 4, D = 32;
        auto ln = op(HX_NORM);
        ln.p[0] = HX_NORM_CHANNEL;
        ln.p[1] = 0;
        float eps = 1e-5f;
        uint32_t eps_u;
        std::memcpy(&eps_u, &eps, 4);
        ln.p[2] = eps_u;
        ln.p[3] = 0;       // gamma offset
        ln.p[4] = D * 4;   // beta offset
        Fixture f({shape(L, 1, D), shape(L, 1, D)}, ln, {0});
        std::vector<float> gamma(D, 1.2f), beta(D, 0.3f);
        std::vector<float> consts(2 * D);
        for (uint32_t i = 0; i < D; i++) {
            consts[i] = gamma[i];
            consts[D + i] = beta[i];
        }
        f.constant(0, consts);
        std::vector<float> in(L * D);
        for (uint32_t i = 0; i < L * D; i++) in[i] = (float)(i + 1) * 0.05f;
        f.input(0, in);
        auto out = f.run();
        check(out.size() == L * D);
        for (uint32_t t = 0; t < L; t++) {
            double sum = 0, sum_sq = 0;
            for (uint32_t i = 0; i < D; i++) {
                sum += in[t * D + i];
            }
            double mean = sum / double(D);
            for (uint32_t i = 0; i < D; i++) {
                double diff = in[t * D + i] - mean;
                sum_sq += diff * diff;
            }
            double var = sum_sq / double(D);
            float inv = (float)(1.0 / std::sqrt(var + double(eps)));
            for (uint32_t i = 0; i < D; i++) {
                float expected = (float)((in[t * D + i] - mean) * inv * gamma[i] + beta[i]);
                check(std::abs(out[t * D + i] - expected) < 1e-5f);
            }
        }
    }
    // RoPE (HX_ROPE, 23)
    {
        constexpr uint32_t head_dim = 16, num_heads = 2, C = head_dim * num_heads;
        auto rope = op(HX_ROPE);
        rope.p[0] = head_dim;
        rope.p[1] = num_heads;
        rope.p[2] = 5; // pos = 5
        rope.p[3] = 0; // NeoX
        Fixture f({shape(1, 1, C), shape(1, 1, C)}, rope, {0});
        std::vector<float> in(C);
        for (uint32_t i = 0; i < C; i++) in[i] = (float)(i + 1) * 0.2f;
        f.input(0, in);
        auto out = f.run();
        check(out.size() == C);
        for (uint32_t h = 0; h < num_heads; h++) {
            for (uint32_t j = 0; j < head_dim / 2; j++) {
                double theta = std::pow(10000.0, -2.0 * double(j) / double(head_dim));
                double ang = 5.0 * theta;
                float cos_v = float(std::cos(ang)), sin_v = float(std::sin(ang));
                float v0 = in[h * head_dim + j], v1 = in[h * head_dim + head_dim / 2 + j];
                float exp0 = v0 * cos_v - v1 * sin_v;
                float exp1 = v0 * sin_v + v1 * cos_v;
                check(std::abs(out[h * head_dim + j] - exp0) < 1e-4f);
                check(std::abs(out[h * head_dim + head_dim / 2 + j] - exp1) < 1e-4f);
            }
        }
    }
    // ShortConv (HX_SHORTCONV, 24) with persistent state ring
    {
        constexpr uint32_t D = 32, K = 3, S = K - 1; // 2 state tokens
        auto sc = op(HX_SHORTCONV, 2, 1); // out=2, state=1
        sc.p[0] = K;
        sc.p[1] = 0; // weights offset
        sc.p[2] = HXFG_NONE; // no bias
        sc.p[3] = 0; // no activation
        // Tensors: 0: in [1, 1, D], 1: state [1, S, D], 2: out [1, 1, D]
        std::vector<HxTensor> tensors = {shape(1, 1, D), shape(1, S, D), shape(1, 1, D)};
        tensors[1].reserved[0] = 1; // persistent state tensor!
        Fixture f(tensors, sc, {0});
        // Weights: K * D = 3 * 32
        // Tap 0: 0.25, Tap 1: 0.5, Tap 2: 1.0 (current token)
        std::vector<float> w(K * D);
        for (uint32_t d = 0; d < D; d++) {
            w[0 * D + d] = 0.25f;
            w[1 * D + d] = 0.5f;
            w[2 * D + d] = 1.0f;
        }
        std::memcpy(f.constants.data(), w.data(), w.size() * 4);

        // Step 1: feed token 1 (all 2.0f)
        std::vector<float> tok1(D, 2.0f);
        f.input(0, tok1);
        auto out1 = f.run();
        // Since state was 0, out = 0*0.25 + 0*0.5 + 2.0*1.0 = 2.0
        check(out1.size() == D);
        near(out1[0], 2.0f);

        // Step 2: feed token 2 (all 4.0f)
        std::vector<float> tok2(D, 4.0f);
        f.input(0, tok2);
        auto out2 = f.run();
        // State now has [0, 2.0]. out = 0*0.25 + 2.0*0.5 + 4.0*1.0 = 1.0 + 4.0 = 5.0!
        near(out2[0], 5.0f);

        // Step 3: feed token 3 (all 8.0f)
        std::vector<float> tok3(D, 8.0f);
        f.input(0, tok3);
        auto out3 = f.run();
        // State now has [2.0, 4.0]. out = 2.0*0.25 + 4.0*0.5 + 8.0*1.0 = 0.5 + 2.0 + 8.0 = 10.5!
        near(out3[0], 10.5f);
    }
    // QuantLinear Q4_0 (HX_LINEAR_Q, 25)
    {
        constexpr uint32_t K = 32, N = 32;
        auto ql = op(HX_LINEAR_Q);
        ql.p[0] = 0; // Q4_0
        ql.p[1] = 0; // weights offset
        ql.p[2] = HXFG_NONE; // no bias
        ql.p[3] = K;
        ql.p[4] = N;
        ql.p[5] = 0; // no activation
        Fixture f({shape(1, 1, K), shape(1, 1, N)}, ql, {0});
        // 1 block of 32 inputs per row, N rows.
        // Block has 16 bytes nibbles + 2 bytes fp16 scale = 18 bytes. Total = 32 * 18 = 576 bytes.
        std::vector<uint8_t> qdata(N * 18);
        for (uint32_t r = 0; r < N; r++) {
            uint8_t *b = qdata.data() + r * 18;
            // nibbles: 0x99 -> 9 and 9, which minus 8 is 1 and 1
            std::memset(b, 0x99, 16);
            // fp16 scale: 1.0f is 0x3c00
            b[16] = 0x00;
            b[17] = 0x3c;
        }
        std::memcpy(f.constants.data(), qdata.data(), qdata.size());
        // input: all 1.0f
        std::vector<float> in(K, 1.0f);
        f.input(0, in);
        auto out = f.run();
        check(out.size() == N);
        // Each weight is (9 - 8) * 1.0 = 1.0. Sum over 32 inputs is 32.0!
        near(out[0], 32.0f);
        near(out[N - 1], 32.0f);
    }
    // KV-cache write/append (HX_KV_WRITE, 26) with persistent state buffer
    {
        constexpr uint32_t D = 32, MAX_TOKENS = 4;
        // Tensors:
        // 0: in token [1, 1, D]
        // 1: pos scalar [1, 1, 1]
        // 2: persistent KV cache [1, MAX_TOKENS, D]
        auto kvw = op(HX_KV_WRITE, 2, 1); // out=2, y=1
        kvw.p[0] = 0; // static pos fallback
        kvw.p[1] = 0; // axis
        kvw.p[2] = MAX_TOKENS;
        std::vector<HxTensor> tensors = {shape(1, 1, D), shape(1, 1, 1), shape(1, MAX_TOKENS, D)};
        tensors[2].reserved[0] = HX_TENSOR_PERSISTENT;
        tensors[2].reserved[1] = 0x1234; // state_id / name_hash

        std::vector<uint8_t> state_buf(MAX_TOKENS * D * 4, 0);

        Fixture f(tensors, kvw, {0, 1});
        f.tensor(2).offset = 0; // offset within persistent state region
        hx_graph_set_state(&f.graph, state_buf.data(), state_buf.size());

        // Step 0: write token 0 at pos 0 (with value 3.0f)
        std::vector<float> tok0(D, 3.0f);
        f.input(0, tok0);
        f.input(1, {0.0f});
        f.run();
        const float *state_f = reinterpret_cast<const float *>(state_buf.data());
        near(state_f[0 * D], 3.0f);
        near(state_f[1 * D], 0.0f);

        // Step 1: write token 1 at pos 1 (with value 7.0f)
        std::vector<float> tok1(D, 7.0f);
        f.input(0, tok1);
        f.input(1, {1.0f});
        f.run();
        near(state_f[0 * D], 3.0f);
        near(state_f[1 * D], 7.0f);
        near(state_f[2 * D], 0.0f);

        // Step 2: write token 2 at pos 2 (with value 11.0f)
        std::vector<float> tok2(D, 11.0f);
        f.input(0, tok2);
        f.input(1, {2.0f});
        f.run();
        near(state_f[0 * D], 3.0f);
        near(state_f[1 * D], 7.0f);
        near(state_f[2 * D], 11.0f);
        near(state_f[3 * D], 0.0f);

        // Verify rejection when target tensor is not marked persistent
        Fixture bad_target(tensors, kvw, {0, 1});
        bad_target.tensor(2).reserved[0] = 0; // NOT persistent
        check(bad_target.open() == HX_SHAPE);
    }
    // Two graphs sharing persistent state by name/id (prefill -> decode)
    {
        constexpr uint32_t D = 32, MAX_TOKENS = 4;
        // Shared state memory
        std::vector<uint8_t> shared_state(MAX_TOKENS * D * 4, 0);

        // Graph A (prefill): writes 2 tokens into persistent cache
        // Tensor 0: input [1, 2, D], Tensor 1: pos [1, 1, 1], Tensor 2: persistent cache [1, MAX_TOKENS, D]
        auto opA = op(HX_KV_WRITE, 2, 1);
        opA.p[2] = MAX_TOKENS;
        std::vector<HxTensor> tensorsA = {shape(1, 2, D), shape(1, 1, 1), shape(1, MAX_TOKENS, D)};
        tensorsA[2].reserved[0] = HX_TENSOR_PERSISTENT;
        tensorsA[2].reserved[1] = 0x9999; // shared name hash

        Fixture gA(tensorsA, opA, {0, 1});
        gA.tensor(2).offset = 0;
        hx_graph_set_state(&gA.graph, shared_state.data(), shared_state.size());
        std::vector<float> tokens_in(2 * D);
        for (uint32_t d = 0; d < D; d++) {
            tokens_in[0 * D + d] = 1.5f;
            tokens_in[1 * D + d] = 2.5f;
        }
        gA.input(0, tokens_in);
        gA.input(1, {0.0f});
        gA.run();

        // Graph B (decode): reads persistent cache via MatMul and appends 1 token at pos 2
        // We'll test that Graph B sees the exact tokens written by Graph A!
        auto opB = op(HX_KV_WRITE, 2, 1);
        opB.p[2] = MAX_TOKENS;
        std::vector<HxTensor> tensorsB = {shape(1, 1, D), shape(1, 1, 1), shape(1, MAX_TOKENS, D)};
        tensorsB[2].reserved[0] = HX_TENSOR_PERSISTENT;
        tensorsB[2].reserved[1] = 0x9999; // shared name hash

        Fixture gB(tensorsB, opB, {0, 1});
        gB.tensor(2).offset = 0;
        hx_graph_set_state(&gB.graph, shared_state.data(), shared_state.size());

        // Check that Graph B sees tokens written by Graph A
        const float *shared_f = reinterpret_cast<const float *>(shared_state.data());
        near(shared_f[0 * D], 1.5f);
        near(shared_f[1 * D], 2.5f);
        near(shared_f[2 * D], 0.0f);

        // Now Graph B appends token at pos 2 (value 4.0f)
        std::vector<float> tok_b(D, 4.0f);
        gB.input(0, tok_b);
        gB.input(1, {2.0f});
        gB.run();

        near(shared_f[0 * D], 1.5f);
        near(shared_f[1 * D], 2.5f);
        near(shared_f[2 * D], 4.0f);

        // Reset state
        std::memset(shared_state.data(), 0, shared_state.size());
        near(shared_f[0 * D], 0.0f);
        near(shared_f[1 * D], 0.0f);
        near(shared_f[2 * D], 0.0f);
    }
    {
        // HX_GATHER: constant pool embedding lookup
        auto gop = op(HX_GATHER, 1);
        gop.p[0] = 1; // mode 1: constant pool
        gop.p[1] = 128; // offset (128-byte aligned)
        gop.p[2] = 4; // vocab
        gop.p[3] = 1; // float indices
        gop.p[4] = 0; // axis 0
        gop.p[5] = 2; // inner_dim
        gop.p[7] = 32; // table stride
        Fixture f({shape(1, 2, 1), shape(1, 2, 2)}, gop, {0});
        // vocab 0: [10, 11], vocab 1: [20, 21], vocab 2: [30, 31], vocab 3: [40, 41]
        f.constant(128 + 0 * 32 * 4, {10.0f, 11.0f});
        f.constant(128 + 1 * 32 * 4, {20.0f, 21.0f});
        f.constant(128 + 2 * 32 * 4, {30.0f, 31.0f});
        f.constant(128 + 3 * 32 * 4, {40.0f, 41.0f});
        f.input(0, {2.0f, 0.0f});
        auto r = f.run();
        near(r[0], 30.0f);
        near(r[1], 31.0f);
        near(r[2], 10.0f);
        near(r[3], 11.0f);
    }
    {
        // HX_GATHER_ELEMENTS: axis = 1
        auto geop = op(HX_GATHER_ELEMENTS, 2, 1);
        geop.p[0] = 1; // axis 1 (W)
        geop.p[1] = 1; // float indices
        Fixture f({shape(2, 2, 1), shape(2, 2, 1), shape(2, 2, 1)}, geop, {0, 1});
        f.input(0, {1.0f, 2.0f,
                    3.0f, 4.0f});
        f.input(1, {1.0f, 0.0f,
                    0.0f, 1.0f});
        auto r = f.run();
        near(r[0], 2.0f);
        near(r[1], 1.0f);
        near(r[2], 3.0f);
        near(r[3], 4.0f);
    }
    {
        // HX_GATHER_ND: k = 2
        auto gnd = op(HX_GATHER_ND, 2, 1);
        gnd.p[1] = 1; // float indices
        gnd.p[4] = 2; // k = 2
        Fixture f({shape(2, 2, 2), shape(2, 1, 2), shape(2, 1, 2)}, gnd, {0, 1});
        f.input(0, {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f});
        // Index 0: coords (0, 1) -> data[0, 1] which is [3.0, 4.0]
        // Index 1: coords (1, 0) -> data[1, 0] which is [5.0, 6.0]
        f.input(1, {0.0f, 1.0f,
                    1.0f, 0.0f});
        auto r = f.run();
        near(r[0], 3.0f);
        near(r[1], 4.0f);
        near(r[2], 5.0f);
        near(r[3], 6.0f);
    }
    {
        // HX_SCATTER_ND: k = 2, assignment
        auto snd = op(HX_SCATTER_ND, 3, 1, 2);
        snd.p[0] = 0; // distinct out
        snd.p[1] = 1; // float indices
        snd.p[2] = 2; // k = 2
        snd.p[4] = 0; // assignment
        Fixture f({shape(2, 2, 2), shape(1, 1, 2), shape(1, 1, 2), shape(2, 2, 2)}, snd, {0, 1, 2});
        f.input(0, {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f});
        // Index coords (0, 1) -> update with [99.0, 100.0]
        f.input(1, {0.0f, 1.0f});
        f.input(2, {99.0f, 100.0f});
        auto r = f.run();
        near(r[0], 1.0f);
        near(r[1], 2.0f);
        near(r[2], 99.0f);
        near(r[3], 100.0f);
        near(r[4], 5.0f);
        near(r[5], 6.0f);
        near(r[6], 7.0f);
        near(r[7], 8.0f);
    }
    {
        // HX_SCATTER_ND: in-place row update on persistent tensor (KV-cache style)
        std::vector<uint8_t> state(4 * 1 * 32 * sizeof(float), 0);
        float *state_f = reinterpret_cast<float *>(state.data());
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                state_f[r * 32 + c] = (float)(r + 1);

        auto snd_inplace = op(HX_SCATTER_ND, 0, 1, 2);
        snd_inplace.p[0] = 1; // mode 1: in-place
        snd_inplace.p[1] = 1; // float indices
        snd_inplace.p[2] = 1; // k = 1 (row update)
        snd_inplace.p[4] = 0; // assignment

        std::vector<HxTensor> tensors = {shape(4, 1, 4), shape(1, 1, 1), shape(1, 1, 4)};
        tensors[0].reserved[0] = HX_TENSOR_PERSISTENT;

        Fixture f(tensors, snd_inplace, {1, 2});
        f.tensor(0).offset = 0;
        hx_graph_set_state(&f.graph, state.data(), state.size());

        // Row 2 -> update with [7, 8, 9, 10]
        f.input(1, {2.0f});
        f.input(2, {7.0f, 8.0f, 9.0f, 10.0f});
        f.run();

        near(state_f[0 * 32 + 0], 1.0f);
        near(state_f[1 * 32 + 0], 2.0f);
        near(state_f[2 * 32 + 0], 7.0f);
        near(state_f[2 * 32 + 1], 8.0f);
        near(state_f[2 * 32 + 2], 9.0f);
        near(state_f[2 * 32 + 3], 10.0f);
        near(state_f[3 * 32 + 0], 4.0f);
    }
    std::cout
        << "Tensor kernels, padding, layout, aliases, ranges, and bootstrap ownership passed\n";
}
