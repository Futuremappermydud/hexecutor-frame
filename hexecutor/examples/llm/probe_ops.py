#!/usr/bin/env python3
"""Validate the NPU's non-matmul Qwen3 operators against NumPy."""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "depth-model/python"),
                             str(_source_root / "depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

HERE = (PATHS.repo / 'hexecutor/examples/llm')
FRAME = (PATHS.native)
sys.path.insert(0, str(FRAME))
from hexecutor.offline.frame_transport import FrameTransport


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--gguf-python", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.gguf_python))
    import gguf
    reader = gguf.GGUFReader(args.model, mode="r")
    rng = np.random.default_rng(42)
    results = []

    with FrameTransport(FRAME / "build/qd_loader.elf", FRAME / "build/hmx_q4_matvec.elf", 20) as transport:
        def run(name, op, input_data, weight_data, count, fields=(), reference=None):
            raw_input = np.asarray(input_data, dtype=np.float32).tobytes()
            raw_weights = weight_data if isinstance(weight_data, bytes) else np.asarray(weight_data, dtype=np.float32).tobytes()
            if len(raw_input) > 0x4000:
                raise ValueError("input overlaps weights")
            def prepare(seq):
                transport.arena[:256] = bytes(256)
                transport.arena[2048:2176] = bytes(128)
                struct.pack_into("<8I", transport.arena, 0, 0x34514d48, 1, seq, op, *fields, *([0] * (4-len(fields))))
                transport.arena[0x20000:0x20000 + len(raw_input)] = raw_input
                transport.arena[0xc0000:0xc0000 + len(raw_weights)] = raw_weights
                transport.fence()
            def collect():
                status = list(struct.unpack_from("<16I", transport.arena, 2048))
                values = np.frombuffer(bytes(transport.arena[0x20000:0x20000 + count*4]), np.float32)
                return status, values
            status, actual = transport.invoke(prepare, lambda: transport.workers_done() and transport.workers_exited(), collect)
            if status[1]:
                raise RuntimeError(f"{name}: DSP error {status[1]}")
            row = {"name": name, "status": status[:2]}
            if reference is not None:
                expected = np.asarray(reference, dtype=np.float32)
                row["max_abs_error"] = float(np.max(np.abs(actual-expected)))
                row["max_rel_error"] = float(np.max(np.abs(actual-expected)/np.maximum(0.1, np.abs(expected))))
                row["sample_actual"] = actual[:6].tolist()
                row["sample_expected"] = expected[:6].tolist()
            results.append(row)
            print(json.dumps(row), flush=True)
            return actual

        x = rng.normal(size=1024).astype(np.float32)
        w = rng.normal(1, 0.1, size=1024).astype(np.float32)
        ref = x / np.sqrt(np.mean(x*x)+1e-6) * w
        run("rmsnorm_1024", 1, x, w, 1024, (1024, 1), ref)
        x = rng.normal(size=(16,128)).astype(np.float32)
        w = rng.normal(1, 0.1, size=128).astype(np.float32)
        ref = x / np.sqrt(np.mean(x*x, axis=1, keepdims=True)+1e-6) * w
        run("qnorm_16x128", 1, x, w, 2048, (128, 16), ref.reshape(-1))
        x = rng.normal(size=1024).astype(np.float32)
        y = rng.normal(size=1024).astype(np.float32)
        run("residual_add", 2, x, y, 1024, (1024,), x+y)
        gate = rng.normal(size=3072).astype(np.float32)
        up = rng.normal(size=3072).astype(np.float32)
        ref = (gate/(1+np.exp(-gate)))*up
        run("swiglu", 3, gate, up, 3072, (3072,), ref)
        qk = rng.normal(size=(24,128)).astype(np.float32)
        theta = np.arange(64, dtype=np.float32)/64
        angle = 7/np.power(1000000.0, theta)
        pairs = np.stack([np.cos(angle), np.sin(angle)], axis=1).astype(np.float32)
        ref = qk.reshape(24,2,64).copy()
        a,b=ref[:,0,:].copy(),ref[:,1,:].copy()
        ref[:,0,:]=a*pairs[:,0]-b*pairs[:,1]
        ref[:,1,:]=a*pairs[:,1]+b*pairs[:,0]
        run("rope", 4, qk, pairs, 3072, (), ref.reshape(-1))
        q = rng.normal(size=(16,128)).astype(np.float32)
        k = rng.normal(size=(8,128)).astype(np.float32)
        v = rng.normal(size=(8,128)).astype(np.float32)
        prior = 3
        history = rng.normal(size=(prior,2,8,128)).astype(np.float32)
        ref=np.empty((16,128),np.float32)
        for h in range(16):
            kh=np.concatenate([history[:,0,h//2,:],k[h//2:h//2+1]],axis=0)
            vh=np.concatenate([history[:,1,h//2,:],v[h//2:h//2+1]],axis=0)
            scores=kh@q[h] / np.sqrt(128.0)
            probabilities=np.exp(scores-np.max(scores))
            probabilities/=np.sum(probabilities)
            ref[h]=probabilities@vh
        run("attention_t3", 5, np.concatenate([q.reshape(-1),k.reshape(-1),v.reshape(-1)]),
            history, 2048, (prior,), ref.reshape(-1))
        embedding = next(t for t in reader.tensors if t.name == "token_embd.weight")
        token_id = 1000
        raw = embedding.data[token_id].tobytes()
        ref = gguf.dequantize(embedding.data[token_id:token_id+1], embedding.tensor_type).reshape(-1)
        run("embedding_token1000", 6, np.zeros(1024, np.float32), raw, 1024, (1024,), ref)

    args.report.write_text(json.dumps(results, indent=2) + "\n")
    if any(r.get("max_rel_error", 0) > 0.002 for r in results):
        raise RuntimeError("NPU operator mismatch")


if __name__ == "__main__":
    main()
