#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare layers [lo, hi) run eagerly against the same layers lowered to HXFG and run by the CPU reference.

Usage: check_stage.py MODEL_DIR LO HI RUNNER [STEPS] [LMAX]     (RUNNER = path of hexecutor_decode_run)
Inputs are random activations at the scale of real hidden states; weights are Q8 so quantization stays small.
"""
import pathlib, subprocess, sys, tempfile
import numpy as np
import torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "lfm"))
import lfm3_stages as L
from gen_executorch_decode_golden import export_and_capture
from hexecutor.offline.executorch import ops

model, lo, hi, runner = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
lmax = int(sys.argv[6]) if len(sys.argv) > 6 else 64
ck = L.Checkpoint(model)
kinds = "q k v o in_b in_c in_x out w1 w2 w3".split()
st = L.Stage(ck, lo, hi, lmax, final=False, q8=kinds, log=lambda *_: None).eval()
rng = np.random.default_rng(3)
xs = rng.normal(0, 1.0, (steps, L.D)).astype(np.float32)
import os
if os.environ.get("TOKENS"):  # real embeddings of these token ids as the input
    emb = ck.f.get_tensor("model.embed_tokens.weight").float().numpy()
    xs = np.stack([emb[int(t)] for t in os.environ["TOKENS"].split(",")][:steps]).astype(np.float32)
ex = (torch.zeros(1, 1, L.D), torch.zeros(1, 1), torch.zeros(1, 1, 1, lmax), torch.zeros(1, 1, 1, lmax))
(summary, image, constants), _ = export_and_capture(st, ex)
d = pathlib.Path(tempfile.mkdtemp())
(d / "g.hxfg").write_bytes(image); (d / "g.bin").write_bytes(constants)
if os.environ.get("DUMP_OPS"):
    import struct
    nt, no = struct.unpack_from("<2I", image, 12)
    T = [struct.unpack_from("<8I", image, 256 + 32 * i) for i in range(nt)]
    for i in range(no):
        o = struct.unpack_from("<16I", image, 256 + 32 * nt + 64 * i)
        print("op", i, "kind", o[0], "x", o[1], T[o[1]][1:5] if o[1] < nt else "-", "y", o[2] if o[2] != 0xffffffff else "-", "out", o[4], T[o[4]][1:5], "p", o[5:11])
print("ops", summary["operations"], "state", list(summary["state_names"])[:3], flush=True)
ops.reset_state()
eager, mf, ms = [], [], []
with torch.no_grad():
    for p in range(steps):
        a, b = L.masks(p, lmax)
        mf.append(a.ravel()); ms.append(b.ravel())
        eager.append(st(torch.from_numpy(xs[p]).reshape(1, 1, L.D), torch.full((1, 1), float(p)),
                        torch.from_numpy(a), torch.from_numpy(b)).numpy().ravel())
(d / "x.f32").write_bytes(xs.tobytes())
(d / "pos.f32").write_bytes(np.arange(steps, dtype="<f4").tobytes())
(d / "mf.f32").write_bytes(np.concatenate(mf).astype("<f4").tobytes())
(d / "ms.f32").write_bytes(np.concatenate(ms).astype("<f4").tobytes())
xf = [str(d / "x.f32")]
cands = [xf, xf + [str(d / "pos.f32")], xf + [str(d / "pos.f32"), str(d / "mf.f32")],
         xf + [str(d / "pos.f32"), str(d / "ms.f32")], xf + [str(d / "pos.f32"), str(d / "mf.f32"), str(d / "ms.f32")]]
for c in cands:  # the graph keeps only the inputs it uses
    r = subprocess.run([runner, str(d / "g.hxfg"), str(d / "g.bin"), str(steps), str(d / "o_")] + c, capture_output=True, text=True)
    if r.returncode == 0:
        break
if r.returncode:
    print(r.stderr[-500:]); sys.exit(1)
out = np.fromfile(d / "o_0.f32", "<f4").reshape(steps, L.D)
for p in range(steps):
    e = np.array(eager[p])
    print(p, "cos %.6f  maxabs %.5f (scale %.3f)" % (out[p] @ e / np.linalg.norm(out[p]) / np.linalg.norm(e),
                                                      np.abs(out[p] - e).max(), np.abs(e).max()))
