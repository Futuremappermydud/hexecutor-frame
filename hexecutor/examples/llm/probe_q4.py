#!/usr/bin/env python3
"""Compare a real Qwen GGUF projection with the Frame's HMX Q4_0 operator."""

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
import time

import numpy as np

HERE = (PATHS.repo / 'hexecutor/examples/llm')
FRAME = (PATHS.native)
sys.path.insert(0, str(FRAME))
from hexecutor.offline.frame_transport import FrameTransport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gguf-python", type=Path, required=True)
    parser.add_argument("--tensor", default="blk.0.attn_k.weight")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--start", type=int, default=0, help="Diagnostic first K index")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.gguf_python))
    import gguf

    reader = gguf.GGUFReader(args.model, mode="r")
    tensor = next(t for t in reader.tensors if t.name == args.tensor)
    k, total_n = map(int, tensor.shape)
    n = min(args.rows, total_n)
    if n % 32 or k % 32 or n > 3072 or k > 3072 or not 1<=args.batch<=32:
        parser.error("operator supports K<=3072, N<=3072, batch<=32")
    if tensor.tensor_type != gguf.GGMLQuantizationType.Q4_0:
        parser.error("tensor must be Q4_0")
    x = np.random.default_rng(2026).normal(0, 0.1, (args.batch,k)).astype(np.float32)
    raw = tensor.data[:n].tobytes()
    expected = x.astype(np.float16).astype(np.float32)[:,args.start:] @ gguf.dequantize(tensor.data[:n], tensor.tensor_type)[:, args.start:].T
    row = {"model": str(args.model), "tensor": args.tensor, "k": k, "n": n, "batch":args.batch,
           "input_bytes": len(raw) + x.nbytes}

    with FrameTransport(FRAME / "build/qd_loader.elf", FRAME / "build/hmx_q4_matvec.elf", 20) as transport:
        def prepare(seq):
            transport.arena[:256] = bytes(256)
            transport.arena[2048:2176] = bytes(128)
            struct.pack_into("<8I", transport.arena, 0, 0x34514d48, 1, seq, 0, k, n, args.start,args.batch)
            transport.arena[0x20000:0x20000 + x.nbytes] = x.tobytes()
            transport.arena[0xc0000:0xc0000 + len(raw)] = raw
            transport.fence()

        def collect():
            row["status"] = list(struct.unpack_from("<32I", transport.arena, 2048))
            row["actual"] = np.frombuffer(bytes(transport.arena[0x20000:0x20000+args.batch*n*4]), np.float32).reshape(args.batch,n)

        started = time.monotonic()
        transport.invoke(prepare, lambda: transport.workers_done() and transport.workers_exited(), collect)
        row["wall_seconds"] = time.monotonic() - started
    actual = row.pop("actual")
    row["max_abs_error"] = float(np.max(np.abs(actual - expected)))
    row["max_rel_error"] = float(np.max(np.abs(actual - expected) / np.maximum(0.1, np.abs(expected))))
    row["rms_error"] = float(np.sqrt(np.mean(np.square(actual - expected))))
    row["sample_actual"] = actual[0,:16].tolist()
    row["sample_expected"] = expected[0,:16].tolist()
    args.report.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row))
    status = row["status"]
    if status[1] or status[2] or status[3] != 32 or status[7] or status[10] or status[11] or status[12] != 68:
        raise RuntimeError("HMX operator status failure")
    if row["max_rel_error"] > 0.03:
        raise RuntimeError("HMX Q4 result differs from FP32 reference")


if __name__ == "__main__":
    main()
