#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Prepare HMX Q4_0 tiles offline; no model arithmetic runs on the host at inference time."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from npu_qwen import MODEL_DEFAULT, GGUF_PY_DEFAULT


def pack_rows(raw, k):
    """Match llama.cpp's 32x32 Q4 tile: 512 nibble bytes, 64 scale bytes, 64 pad."""
    rows = len(raw)
    blocks = raw.reshape(rows // 32, 32, k // 32, 18).transpose(0, 2, 1, 3)
    codes = blocks[..., 2:18]
    values = np.concatenate((codes & 15, codes >> 4), axis=-1)
    pairs = values[..., 0::2] | (values[..., 1::2] << 4)
    tiles = np.zeros((rows // 32, k // 32, 640), dtype=np.uint8)
    tiles[..., :512] = pairs.transpose(0, 1, 3, 2).reshape(rows // 32, k // 32, 512)
    tiles[..., 512:576] = blocks[..., :2].reshape(rows // 32, k // 32, 64)
    return tiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--gguf-python", type=Path, default=GGUF_PY_DEFAULT)
    parser.add_argument("--output", type=Path, default=MODEL_DEFAULT.parent / "Qwen3-0.6B-HMX-Q4")
    parser.add_argument("--only", help="prepare one matrix")
    args = parser.parse_args()
    sys.path.insert(0, str(args.gguf_python))
    import gguf

    reader = gguf.GGUFReader(args.model, mode="r")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    metadata = {}
    for tensor in reader.tensors:
        if len(tensor.shape) != 2 or tensor.name == "token_embd.weight":
            continue
        if args.only and tensor.name != args.only:
            continue
        if tensor.tensor_type != gguf.GGMLQuantizationType.Q4_0:
            raise ValueError(f"{tensor.name}: expected Q4_0")
        k, n = map(int, tensor.shape)
        if k % 32 or n % 32:
            raise ValueError(f"{tensor.name}: unsupported shape")
        size = (k // 32) * (n // 32) * 640
        target = args.output / (tensor.name + ".hmx-q4")
        if not target.exists() or target.stat().st_size != size:
            part = target.with_suffix(target.suffix + ".part")
            with part.open("wb") as stream:
                for row in range(0, n, 1024):
                    count = min(1024, n - row)
                    pack_rows(tensor.data[row:row + count].copy(), k).tofile(stream)
            if part.stat().st_size != size:
                raise RuntimeError(f"{tensor.name}: packed size mismatch")
            part.replace(target)
        metadata[tensor.name] = {"k": k, "n": n, "bytes": size}
        print(f"{tensor.name}: {size / 1048576:.1f} MiB", flush=True)
    manifest = {"source": str(args.model), "source_bytes": args.model.stat().st_size,
                "tensors": metadata, "seconds": time.monotonic() - started}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"complete: {len(metadata)} matrices, {sum(v['bytes'] for v in metadata.values()) / 1048576:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
