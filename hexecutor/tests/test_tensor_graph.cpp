#include "tensor/graph.h"
#include "hexecutor/executor.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void check(bool b) {
    if (!b)
        throw std::runtime_error("tensor test failed");
}
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
    uint32_t open() {
        uint32_t bad;
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
    void input(size_t i, const std::vector<float> &v) {
        check(open() == 0);
        check(v.size() == hx_tensor_elements(&tensor(i)));
        hx_pack(reinterpret_cast<float *>(arena.data() + tensor(i).offset), v.data(), &tensor(i));
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
};
HxTensor shape(uint32_t h, uint32_t w, uint32_t c) { return {0, h, w, c, 0, {0, 0, 0}}; }
HxOp op(uint32_t kind, uint32_t out = 1, uint32_t y = HXFG_NONE) {
    HxOp o{};
    o.kind = kind;
    o.x = 0;
    o.y = y;
    o.z = HXFG_NONE;
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
    auto concat = op(HX_CONCAT, 2, 1);
    concat.p[0] = 2;
    Fixture ca({shape(1, 2, 1), shape(1, 2, 2), shape(1, 2, 3)}, concat, {0, 1});
    ca.input(0, {1, 2});
    ca.input(1, {3, 4, 5, 6});
    r = ca.run();
    check(r == std::vector<float>({1, 3, 4, 2, 5, 6}));
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
    std::cout
        << "Tensor kernels, padding, layout, aliases, ranges, and bootstrap ownership passed\n";
}
