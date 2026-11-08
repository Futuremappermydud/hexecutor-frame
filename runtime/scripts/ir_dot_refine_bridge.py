#!/usr/bin/env python3
"""CPU IR+dot refine worker for the live TSDF. RGB disparity is the prior; no DSP."""

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
import os
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str((PATHS.repo / 'runtime/scripts')))
from ir_dot_lib import (  # noqa: E402
    FOOT_N, H, H_IR, K, MAX_D, RECT_F, RECT_W, W, W_IR, ir_points_to_rgb_depth, load_rgb_chain,
    Rectifier, rgb_disparity_to_ir, search, unwrapped_offset, zncc_volume,
)

HELLO_MAGIC, REQUEST_MAGIC, REPLY_MAGIC = 0x30444F54, 0x31444F54, 0x34444F54
REPLY_FORMAT = "<5I16f"  # must match ReplyHeader in src/ir_dot_bridge.cpp
REQUEST_SIZE = 20
DEBUG = os.environ.get("IR_DOT_DEBUG") == "1"


def receive(sock, size, allow_eof=False):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        n = sock.recv_into(view[offset:])
        if not n:
            if allow_eof and offset == 0:
                return None
            raise EOFError("incomplete IR-dot request")
        offset += n
    return data


def refine(rect, chain, disp, unc, left, right, rng=None):
    """Returns (kept pixel count, depth, plane_info)."""
    dd = np.arange(MAX_D + 1)[:, None, None]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    L, R, okl, okr = rect.rectify(left, right)
    prior = rgb_disparity_to_ir(disp, unc, rect, chain)
    vol = zncc_volume(L, R, okl, okr)
    anywhere = okl & np.isfinite(prior)
    ref, kept = search(vol, dd, prior, anywhere)
    if DEBUG:
        print(f"[IrDotDiag] prior_finite={np.isfinite(prior).sum()} pass1 kept={kept.sum()} vol_max={vol.max():.3f}", flush=True)
    prior = prior + unwrapped_offset(ref - prior, kept)
    ref, kept = search(vol, dd, prior, anywhere)
    depth, plane_info = ir_points_to_rgb_depth(rect, u[kept], v[kept], ref[kept], chain, rng=rng)
    return int(kept.sum()), depth, plane_info


def serve(sock, rect, chain):
    sock.sendall(struct.pack("<5I2f", HELLO_MAGIC, W, H, RECT_W, FOOT_N, RECT_F, np.degrees(K)))
    print(f"[IrDot] bridge ready yaw={os.environ.get('IR_YAW_DEG', '0')} "
          f"pitch={os.environ.get('IR_PITCH', '0.25')} roll={os.environ.get('IR_ROLL_DEG', '0.4')}", flush=True)
    while True:
        header = receive(sock, REQUEST_SIZE, allow_eof=True)
        if header is None:
            break
        magic, width, height, rect_w, sequence = struct.unpack("<5I", header)
        if magic != REQUEST_MAGIC or (width, height, rect_w) != (W, H, RECT_W):
            raise ValueError("invalid IR-dot request")
        disp = np.frombuffer(receive(sock, W * H * 4), "<f4").reshape(H, W).copy()
        unc = np.frombuffer(receive(sock, W * H), np.uint8).reshape(H, W).copy()
        left = np.frombuffer(receive(sock, W_IR * H_IR), np.uint8).reshape(H_IR, W_IR).copy()
        right = np.frombuffer(receive(sock, W_IR * H_IR), np.uint8).reshape(H_IR, W_IR).copy()
        t0 = time.monotonic()
        kept, depth, plane_info = refine(rect, chain, disp, unc, left, right)
        ms = (time.monotonic() - t0) * 1000
        if plane_info is not None:
            fit, grid = plane_info
            header = struct.pack(REPLY_FORMAT, REPLY_MAGIC, sequence, kept, 1, fit.count, ms, *fit.n, fit.c,
                                 *fit.cen, *fit.u, *fit.v, fit.radius, fit.rms)
            print(f"[IrDot] kept {kept} IR pixels, plane n=[{fit.n[0]:.3f},{fit.n[1]:.3f},{fit.n[2]:.3f}] "
                  f"c={fit.c:.3f}m rad={fit.radius:.2f}m ({fit.count} inliers, rms={fit.rms * 1000:.1f}mm), "
                  f"{ms:.0f} ms", flush=True)
        else:
            grid = np.zeros((FOOT_N, FOOT_N), np.uint8)
            header = struct.pack(REPLY_FORMAT, REPLY_MAGIC, sequence, kept, 0, 0, ms, *([0.0] * 15))
            print(f"[IrDot] kept {kept} IR pixels, no plane, {ms:.0f} ms", flush=True)
        sock.sendall(header + grid.astype(np.uint8).tobytes()
                     + np.ascontiguousarray(depth, dtype="<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", type=int, required=True)
    args = ap.parse_args()
    rect = Rectifier()
    chain = load_rgb_chain()
    with socket.socket(fileno=args.fd) as sock:
        serve(sock, rect, chain)


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, ConnectionResetError):
        pass
    except KeyboardInterrupt:
        sys.exit(130)
