#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The decode graph of gen_hxfg_decode_golden.py as an ONNX model, lowered by lower_onnx_float.py.

Same weights, same independent float64 NumPy golden, same simulator test: the only difference is that
the HXFG file comes from the ONNX lowering (MatMulNBits, RotaryEmbedding, hexecutor::ShortConv,
hexecutor::KVCacheWrite, RMSNormalization, ScaledDotProductAttention) instead of straight from the
writer. Needs onnx; run it with a Python that has it.

Usage: gen_onnx_decode_golden.py OUT_DIR
"""

import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))
import gen_hxfg_decode_golden as base  # noqa: E402
from hexecutor.offline import lower_onnx_float as lowering  # noqa: E402

D, DH, LMAX, ROT, THETA, K_TAPS = base.D, base.DH, base.LMAX, base.ROT, base.THETA, base.K_TAPS
OPSET = 23


def rope_tables(rows=64):
    j = np.arange(ROT // 2)
    ang = np.arange(rows)[:, None] * np.float64(np.float32(THETA)) ** (-2.0 * j / ROT)[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def build_onnx(model, theta_scale=1.0):
    """ONNX ModelProto for the recipe recorded by gen_hxfg_decode_golden.build()."""
    nodes, inits = [], []
    name = lambda i: f"t{i}"
    counter = [0]

    def const(array, label):
        counter[0] += 1
        n = f"{label}_{counter[0]}"
        inits.append(numpy_helper.from_array(np.asarray(array), n))
        return n

    cos, sin = rope_tables()
    cos, sin = const(cos, "cos"), const(sin * 1.0, "sin")
    ids = model["ids"]
    for item in model["recipe"]:
        kind, out = item[0], name(item[1])
        if kind == "rms":
            nodes.append(helper.make_node("RMSNormalization", [name(item[2]), const(item[3], "gamma")], [out],
                                          epsilon=base.EPS))
        elif kind == "linear":
            _, _, x, blob, fmt, k, n, bias, act = item
            nb = k // 32
            raw = np.frombuffer(blob, np.uint8).reshape(n, nb, 18 if fmt == 0 else 34)
            data = raw[..., :16] if fmt == 0 else (raw[..., :32].view(np.int8).astype(np.int16) + 128).astype(np.uint8)
            scales = np.ascontiguousarray(raw[..., -2:]).view(np.float16).reshape(n * nb)
            pre = out if act == 0 else out + "_pre"
            nodes.append(helper.make_node(
                "MatMulNBits", [name(x), const(np.ascontiguousarray(data), "B"), const(scales, "S"), "", "",
                                const(bias, "bias")], [pre],
                domain="com.microsoft", K=k, N=n, bits=4 if fmt == 0 else 8, block_size=32))
            if act == 3:  # SiLU, as an exporter writes it
                nodes.append(helper.make_node("Sigmoid", [pre], [pre + "_s"]))
                nodes.append(helper.make_node("Mul", [pre, pre + "_s"], [out]))
        elif kind == "bin":
            nodes.append(helper.make_node("Add" if item[4] == base.hx.BIN_ADD else "Mul",
                                          [name(item[2]), name(item[3])], [out]))
        elif kind == "shortconv":
            _, _, u, gate, w, b, _ring = item
            nodes.append(helper.make_node(
                "ShortConv", [name(u), const(w.T.reshape(D, 1, K_TAPS), "convw"), const(b, "convb"), name(gate)],
                [out], domain="hexecutor", state="conv0", activation="none"))
        elif kind == "rope":
            nodes.append(helper.make_node(
                "RotaryEmbedding", [name(item[2]), cos, sin, name(item[3])], [out],
                num_heads=1, rotary_embedding_dim=ROT, interleaved=0))
        elif kind == "kvw":
            nodes.append(helper.make_node("KVCacheWrite", [name(item[2]), name(item[3])], [out],
                                          domain="hexecutor", state=item[4], max_len=LMAX))
        elif kind == "sdpa":
            nodes.append(helper.make_node("ScaledDotProductAttention",
                                          [name(item[2]), name(item[3]), name(item[4]), name(item[5])], [out]))
        else:
            raise ValueError(kind)
    graph = helper.make_graph(
        nodes, "lfm_decode",
        [helper.make_tensor_value_info(name(ids["x"]), T.FLOAT, [1, 1, D]),
         helper.make_tensor_value_info(name(ids["pos"]), T.INT64, [1, 1]),
         helper.make_tensor_value_info(name(ids["mask"]), T.FLOAT, [1, 1, LMAX])],
        [helper.make_tensor_value_info(name(ids["out"]), T.FLOAT, [1, 1, D])], inits)
    m = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", OPSET), helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("hexecutor", 1)])
    m.ir_version = 10
    return m


def main(out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = base.build()
    rng = np.random.default_rng(99)
    xs = rng.normal(0, 1, (base.STEPS, D)).astype(np.float32)
    expected, (ring, kc, vc) = base.golden(model, xs)
    source = out_dir / "lfm_decode.onnx"
    onnx.save(build_onnx(model), source)
    result = lowering.lower(source, out_dir / "lfm_decode")
    base.finish(out_dir, "lfm_decode", result, xs, expected, ring, kc, vc)
    print("state names:", result["state_names"], "fp16 plan:", result["fp16_plan"]["eligible"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hexecutor/native/build/lfm/onnx_decode")
