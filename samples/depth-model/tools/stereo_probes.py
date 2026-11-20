#!/usr/bin/env python3
"""Generate synthetic stereo probes with known shifts, or summarize their outputs.

    stereo_probes.py make DIR        write probe inputs and DIR/input_list.txt
    stereo_probes.py summarize DIR OUTPUT_DIR [--float-name NAME]

Each probe crops two windows from one procedural texture. In `w+d`, y is x
shifted so a feature at column c in x appears at column c-d in y. Hidden is
zero and t holds two identity matrices unless the case name says otherwise.
These are synthetic stimuli for probing behavior, not camera data.
"""
import argparse
from pathlib import Path
import struct

import numpy as np

H, W, PAD = 320, 256, 48
IDENTITY = struct.pack("<32f", *([1.0 if i % 5 == 0 else 0.0 for i in range(16)] * 2))
ZERO_T = struct.pack("<32f", *[0.0] * 32)
U8_SCALE, U8_ZERO = 0.05533868819475174, 136


def texture(seed=0):
    rng = np.random.default_rng(seed)
    hh, ww = H + 2 * PAD, W + 2 * PAD
    img = np.zeros((hh, ww))
    for s in [2, 4, 8, 16, 32]:
        up = np.kron(rng.standard_normal((hh // s + 2, ww // s + 2)), np.ones((s, s)))[:hh, :ww]
        k = max(1, s // 2)
        for axis in (0, 1):
            up = np.cumsum(up, axis=axis)
            up = (np.roll(up, -k, axis=axis) - up) / k
        img += up * s ** 0.5
    img = (img - img.mean()) / img.std()
    return np.clip(128 + 40 * img, 0, 255).astype(np.uint8)


def make(out):
    big = texture()
    crop = lambda dy, dx: big[PAD + dy:PAD + dy + H, PAD + dx:PAD + dx + W].copy()
    base = crop(0, 0)
    flat = np.full((H, W), 128, np.uint8)
    zero_h = np.zeros((1, 2, H, W), np.float32)
    cases = [("same", base, base, zero_h, IDENTITY)]
    for d in [2, 4, 8, 16, 32]:
        cases += [(f"w+{d}", base, crop(0, d), zero_h, IDENTITY), (f"w-{d}", base, crop(0, -d), zero_h, IDENTITY),
                  (f"h+{d}", base, crop(d, 0), zero_h, IDENTITY), (f"h-{d}", base, crop(-d, 0), zero_h, IDENTITY)]
    for hv in [0.0, 16.0]:
        for tn, t in [("id", IDENTITY), ("zero", ZERO_T)]:
            cases.append((f"flat_h{hv:g}_{tn}", flat, flat, zero_h + np.float32(hv), t))
    lines = []
    for name, x, y, hidden, t in cases:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        x.tofile(d / "x.raw")
        y.tofile(d / "y.raw")
        hidden.astype(np.float32).tofile(d / "hidden.raw")
        (d / "t.raw").write_bytes(t)
        lines.append(" ".join(f"{k}:={d / (k + '.raw')}" for k in ["x", "y", "hidden", "t"]))
    (out / "input_list.txt").write_text("\n".join(lines) + "\n")
    (out / "cases.txt").write_text("\n".join(c[0] for c in cases) + "\n")
    print(out / "input_list.txt")


def summarize(probe_dir, results, float_name):
    cases = (probe_dir / "cases.txt").read_text().split()
    inner = (slice(40, 280), slice(40, 216))
    print(f"{'case':14s} {'float0':>8s} {'float1':>8s} {'u8deq0':>8s} {'u8deq1':>8s}   (medians over interior)")
    for i, name in enumerate(cases):
        r = results / f"Result_{i}"
        q = np.fromfile(r / "boltnn_permute_from_boltnn_tensor_81_native.raw", np.uint8).reshape(2, H, W)
        f = np.fromfile(r / float_name, np.float32).reshape(2, H, W)
        dq = (q.astype(np.float32) - U8_ZERO) * U8_SCALE
        print(f"{name:14s} {np.median(f[0][inner]):8.3f} {np.median(f[1][inner]):8.3f} "
              f"{np.median(dq[0][inner]):8.3f} {np.median(dq[1][inner]):8.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("dir", type=Path)
    s = sub.add_parser("summarize")
    s.add_argument("dir", type=Path)
    s.add_argument("results", type=Path)
    s.add_argument("--float-name", default="quantized_decomposed_dequantize_per_tensor_default_333.raw",
                   help="alternate model: ..._default_335.raw")
    args = parser.parse_args()
    if args.cmd == "make":
        make(args.dir.resolve())
    else:
        summarize(args.dir.resolve(), args.results.resolve(), args.float_name)


if __name__ == "__main__":
    main()
