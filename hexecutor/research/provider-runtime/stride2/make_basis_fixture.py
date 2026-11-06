#!/usr/bin/env python3
"""Build one raw-S2 one-hot fixture for the corrected V69 generic leaf harness."""
from __future__ import annotations

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
import struct
import sys
from pathlib import Path
import numpy as np

HERE = (PATHS.repo / 'hexecutor/research/provider-runtime/stride2')
FRAME = (PATHS.native)
sys.path.insert(0, str(FRAME / "native_stride2"))
import check_s2_oracle as oracle  # noqa: E402

PAGE = 2048
HEADER = 48 * 4

def write_blocked(dst: bytearray, max_w: int, max_c: int,
                  y: int, x: int, c: int, value: int) -> None:
    nbw, nbc = (max_w + 7) // 8, max_c // 32
    block = ((y // 8) * nbw + x // 8) * nbc + c // 32
    off = block * PAGE + (y % 8) * 256 + (x % 8) * 32 + c % 32
    dst[off] = value

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ky", type=int)
    ap.add_argument("kx", type=int)
    ap.add_argument("--target-y", type=int, default=4)
    ap.add_argument("--target-x", type=int, default=4)
    ap.add_argument("--dense-input", action="store_true")
    ap.add_argument("--spatial-grid", action="store_true")
    ap.add_argument("--lane-grid", action="store_true")
    ap.add_argument("--all-channel-basis", action="store_true")
    ap.add_argument("--random-seed", type=int)
    ap.add_argument("--random-one-tap", action="store_true",
                    help="randomize X and W only at the selected spatial tap")
    ap.add_argument("--ones-one-tap", action="store_true",
                    help="set every CI/CO coefficient at only the selected tap to one")
    ap.add_argument("--diagonal-random", action="store_true",
                    help="random diagonal raw weights at one tap with random X")
    ap.add_argument("--permuted-basis", action="store_true",
                    help="one-hot cyclic CI->CO channel matrix at one tap")
    ap.add_argument("--channel-group", type=int, choices=range(4))
    ap.add_argument("--output-group", type=int, choices=range(2), default=0)
    a = ap.parse_args()
    if not (0 <= a.ky < 3 and 0 <= a.kx < 3):
        raise SystemExit("tap out of range")
    src = (FRAME / "native_stride2/s2_fixture_224.bin").read_bytes()
    h = struct.unpack_from("<48I", src)
    if h[1] != 224:
        raise SystemExit("wrong fixture")
    xbytes, wb, fb, ybytes = h[15] * PAGE, h[35], h[36], h[29] * PAGE
    ci, co = h[32], h[33]
    if (ci, co, wb, fb, ybytes) != (32, 64, 18432, 512, 36 * PAGE):
        raise SystemExit("basis harness supports only sequence 224 geometry")

    raww = np.zeros((3, 3, ci, co), dtype=np.int8)
    if a.permuted_basis:
        for ch in range(ci):
            raww[a.ky, a.kx, ch, a.output_group * 32 + (ch + 1) % 32] = 1
    elif a.diagonal_random:
        rng = np.random.default_rng(a.random_seed if a.random_seed is not None else 123)
        for ch in range(min(ci, co)):
            raww[a.ky, a.kx, ch, a.output_group * 32 + ch] = rng.integers(-2, 3, dtype=np.int8)
    elif a.ones_one_tap:
        raww[a.ky, a.kx] = 1
    elif a.random_seed is not None:
        rng = np.random.default_rng(a.random_seed)
        if a.random_one_tap:
            raww[a.ky, a.kx] = rng.integers(-2, 3, size=(ci, co), dtype=np.int8)
        else:
            raww[:] = rng.integers(-2, 3, size=raww.shape, dtype=np.int8)
    elif a.all_channel_basis:
        for ch in range(ci):
            raww[a.ky, a.kx, ch, ch] = 1
    else:
        raww[a.ky, a.kx, 0, 0] = 1
    packed_w = oracle.pack_xwb(raww)
    tap_probe = np.zeros((3, 3, ci, co), dtype=np.int8)
    tap_probe[a.ky, a.kx, 0, 0] = 1
    lowered = oracle.lower_s2(oracle.pack_xwb(tap_probe), ci, co)
    nonzero = np.argwhere(lowered[:, :, :, 0] != 0)
    if len(nonzero) != 1:
        raise SystemExit(f"expected one lowered coefficient; got {len(nonzero)}")
    block_y, block_x, input_channel = map(int, nonzero[0])
    if a.channel_group is not None:
        input_channel = a.channel_group * 32
    x = bytearray(xbytes)
    input_y, input_x = a.target_y + block_y, a.target_x + block_x
    if a.random_seed is not None:
        rng = np.random.default_rng(a.random_seed + 1)
        logical_x = rng.integers(0, 4, size=(9, 130, 128), dtype=np.uint8)
        for yy in range(9):
            for xx in range(130):
                for cc in range(128):
                    write_blocked(x, 136, 128, yy, xx, cc, int(logical_x[yy, xx, cc]))
    elif a.dense_input:
        for yy in range(9):
            for xx in range(130):
                if a.channel_group is None:
                    channels = range(128)
                else:
                    channels = range(a.channel_group * 32, (a.channel_group + 1) * 32)
                for cc in channels:
                    write_blocked(x, 136, 128, yy, xx, cc, 1)
    elif a.spatial_grid:
        lo = 0 if a.channel_group is None else a.channel_group * 32
        hi = 128 if a.channel_group is None else (a.channel_group + 1) * 32
        for oy in range(-2, 3):
            for ox in range(-2, 3):
                val = 1 + (oy + 2) * 5 + (ox + 2)
                for cc in range(lo, hi):
                    write_blocked(x, 136, 128, a.target_y + oy, a.target_x + ox, cc, val)
    elif a.lane_grid or a.all_channel_basis or a.permuted_basis:
        lo = 0 if a.channel_group is None else a.channel_group * 32
        for lane in range(32):
            write_blocked(x, 136, 128, input_y, input_x, lo + lane, lane + 1)
    else:
        write_blocked(x, 136, 128, input_y, input_x, input_channel, 1)
    fi = bytearray(fb)
    for cb in range(co // 32):
        for ch in range(32):
            struct.pack_into("<i", fi, (cb * 64 + ch) * 4, 0x6000)
    y = bytearray([0] * ybytes)
    if a.random_seed is not None:
        dense_x = oracle.read_blocked(bytes(x), 16, 136, 128, 136, 128)
        lowered = oracle.lower_s2(packed_w, ci, co)
        accum = np.zeros((8, 128, co), dtype=np.int64)
        for ay in range(2):
            for ax in range(2):
                accum += dense_x[ay:ay + 8, ax:ax + 128, :4 * ci].astype(np.int64) @ lowered[ay, ax].astype(np.int64)
        expected = np.clip(accum, 0, 255).astype(np.uint8)
        for yy in range(8):
            for xx in range(128):
                for cc in range(co):
                    write_blocked(y, 128, 64, yy, xx, cc, int(expected[yy, xx, cc]))
    elif a.dense_input:
        for yy in range(8):
            for xx in range(128):
                write_blocked(y, 128, 64, yy, xx, 0, 1)
    for i in range(32 * PAGE, ybytes):
        y[i] = 0xA5  # output descriptor exposes only first 32 pages
    expected_off = ((a.target_y // 8) * 16 + a.target_x // 8) * 2 * PAGE
    expected_off += (a.target_y % 8) * 256 + (a.target_x % 8) * 32
    if a.random_seed is not None or a.dense_input:
        pass
    elif a.spatial_grid:
        y[expected_off] = 1 + (block_y + 2) * 5 + (block_x + 2)
    elif a.all_channel_basis or a.permuted_basis:
        for ch in range(32):
            off = expected_off + ch
            y[off] = ch + 1
    else:
        y[expected_off] = 1
    payload = x + packed_w + fi + y
    ((PATHS.repo / 'hexecutor/research/provider-runtime/stride2/s2_basis_fixture.bin')).write_bytes(src[:HEADER] + payload)
    print(f"basis tap=({a.ky},{a.kx}) lowered_block=({block_y},{block_x}) "
          f"input_channel={input_channel} input=({input_y},{input_x},{input_channel}) "
          f"random_seed={a.random_seed} dense_input={a.dense_input} spatial_grid={a.spatial_grid} "
          f"expected_output=unit")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
