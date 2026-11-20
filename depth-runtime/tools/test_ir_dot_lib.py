#!/usr/bin/env python3
"""Plane fit, footprint and bridge wire format. Run: python3 tools/test_ir_dot_lib.py"""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import json
import math
import os
import socket
import struct
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str((PATHS.repo / 'depth-runtime/scripts')))
os.environ.setdefault("IR_YAW_DEG", "0")
import ir_dot_lib as lib  # noqa: E402
import ir_dot_refine_bridge as bridge  # noqa: E402

RNG = np.random.default_rng(7)


def wall(n, c, count=4000, half=(1.5, 1.2), noise=0.008):
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = np.cross(n, [0.3, 1, 0.2]); a /= np.linalg.norm(a)
    b = np.cross(n, a)
    s, t = RNG.uniform(-half[0], half[0], count), RNG.uniform(-half[1], half[1], count)
    return n * c + np.outer(s, a) + np.outer(t, b) + RNG.normal(0, noise, (count, 3)) * n[None]


def test_plane_faces_any_way():
    # Floor, 1.5 m below a camera pitched up 10 degrees: its n has a negative z.
    p = math.radians(10)
    floor = wall([0, math.cos(p), -math.sin(p)], 1.5)
    fit = lib.fit_dominant_plane(floor, rng=np.random.default_rng(0))
    assert fit is not None and abs(fit.c - 1.5) < 0.01 and fit.n[2] < 0 and fit.n[1] > 0.9
    for tilted in ([0.75, -0.32, 0.57], [-0.54, -0.04, 0.84], [0, 0, 1]):
        fit = lib.fit_dominant_plane(wall(tilted, 2.0), rng=np.random.default_rng(0))
        assert fit is not None and abs(fit.c - 2.0) < 0.01
        assert abs(abs(fit.n @ (np.array(tilted) / np.linalg.norm(tilted))) - 1) < 1e-3
        assert abs(np.linalg.det(np.stack([fit.u, fit.v, fit.n]))) > 0.999


def test_rejects():
    assert lib.fit_dominant_plane(wall([0, 0, 1], 2.0, count=30)) is None
    strip = wall([0, 0, 1], 2.0, half=(1.5, 0.02))
    assert lib.fit_dominant_plane(strip, rng=np.random.default_rng(0)) is None
    assert lib.fit_dominant_plane(wall([0, 0, 1], 0.2, half=(0.5, 0.5)), rng=np.random.default_rng(0)) is None


def test_refit_tightens():
    pts = np.vstack([wall([0, 0, 1], 2.0, noise=0.004), RNG.uniform(-1, 3, (1500, 3))])
    fit = lib.fit_dominant_plane(pts, rng=np.random.default_rng(0))
    assert abs(fit.c - 2.0) < 0.004 and abs(fit.n[2]) > 0.9999


def test_footprint_follows_the_surface():
    # L-shaped observed area: the radius test would cover the empty corner.
    a = wall([0, 0, 1], 2.0, count=3000, half=(1.5, 0.3))
    b = wall([0, 0, 1], 2.0, count=3000, half=(0.3, 1.5))
    fit = lib.fit_dominant_plane(np.vstack([a, b]), rng=np.random.default_rng(0))
    grid = lib.plane_footprint(np.vstack([a, b]), fit)
    def cell(x, y):
        iv, iu = lib.plane_cells(np.array([[x, y, 2.0]]), fit)
        return grid[iv[0], iu[0]]
    assert cell(1.2, 0.0) == 1 and cell(0.0, 1.2) == 1 and cell(0.0, 0.0) == 1
    assert cell(1.2, 1.2) == 0 and cell(-1.2, 1.2) == 0


def test_foreground_dots_block_their_cells():
    pts_wall = wall([0, 0, 1], 2.0)
    # A chair 0.5 m in front of the wall hides what is behind it.
    chair = np.column_stack([RNG.uniform(-0.2, 0.2, 800), RNG.uniform(-0.2, 0.2, 800), RNG.normal(1.5, 0.01, 800)])
    visible = pts_wall[~((np.abs(pts_wall[:, 0]) < 0.27) & (np.abs(pts_wall[:, 1]) < 0.27))]
    pts = np.vstack([visible, chair])
    fit = lib.fit_dominant_plane(pts, rng=np.random.default_rng(0))
    grid = lib.plane_footprint(pts, fit)
    iv, iu = lib.plane_cells(np.array([[0.0, 0.0, 2.0]]), fit)
    assert grid[iv[0], iu[0]] == 2
    iv, iu = lib.plane_cells(np.array([[1.0, 0.8, 2.0]]), fit)
    assert grid[iv[0], iu[0]] == 1
    # The dense depth has a hole on the chair, then the chair's own dots.
    depth = lib.plane_depth(fit, grid)
    assert depth[308, 308] == 0.0 and abs(depth[308 + 150, 308 + 180] - 2.0) < 0.02
    ui = np.round(lib.RECT_F * chair[:, 0] / chair[:, 2] + lib.RECT_CX).astype(int)
    vi = np.round(lib.RECT_F * chair[:, 1] / chair[:, 2] + lib.RECT_CY).astype(int)
    lib.stamp_points(depth, ui, vi, chair[:, 2].astype(np.float32))
    assert abs(depth[vi[0], ui[0]] - 1.5) < 0.05


def test_near_dot_wins_a_pixel():
    depth = np.full((lib.RECT_H, lib.RECT_W), 2.0, np.float32)
    lib.stamp_points(depth, np.array([50, 50]), np.array([60, 60]), np.array([1.2, 1.0], np.float32))
    assert depth[60, 50] == 1.0 and depth[60, 51] == 2.0
    lib.stamp_points(depth, np.array([90]), np.array([90]), np.array([3.0], np.float32))
    assert depth[90, 90] == 2.0  # a dot behind the surface does not replace it


def quat(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = math.sin(math.radians(deg) / 2)
    return [*(axis * s), math.cos(math.radians(deg) / 2)]


def history(yaws):
    return {"devices": [{"history": [{"passthroughCam0FromPassthroughCam1Extrinsics": [-0.064, 0, 0, *quat([0.1, 1, 0.05], y)],
                                      "frontCam0FromPassthroughCam0Extrinsics": [0.01, 0.02, 0.0, 0, 0, 0, 1]} for y in yaws]}]}


def test_rgb_basis_orthonormal():
    entry = np.array([-0.064, 0.001, 0.002, *quat([0.2, 1, 0.3], 3.0)])
    Rrgb, _ = lib.rgb_basis(entry)
    assert np.abs(Rrgb.T @ Rrgb - np.eye(3)).max() < 1e-12


def test_calibration_policy_matches_the_rectifier():
    es = lib.entries(history([0.1, 0.5, 0.11, 0.12, 2.0]), "passthroughCam0FromPassthroughCam1Extrinsics")
    assert np.array_equal(lib.stereo_entry(es, "medoid"), es[3]) or np.array_equal(lib.stereo_entry(es, "medoid"), es[2])
    assert np.array_equal(lib.stereo_entry(es, "latest"), es[-1])
    assert np.array_equal(lib.stereo_entry(es[:2], "medoid"), es[1])  # C++ takes the medoid only above two


def test_wire_format_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        cams = [{"sourceCamera": name, "extrinsics": {"plus_x": [1, 0, 0], "plus_z": [0, 0, 1], "position": [x, 0, 0]},
                 "intrinsics": [{"fx": 280.0, "fy": 280.0, "cx": 528.0, "cy": 512.0, "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0}]}
                for name, x in (("slam_left", -50.0), ("slam_right", 50.0))]
        rig = Path(d) / "rig.json"
        rig.write_text(json.dumps({"cameras": cams}))
        rgb = Path(d) / "history.json"
        rgb.write_text(json.dumps(history([0.1, 0.12, 0.11])))
        rect = lib.Rectifier(lib.load_rig(rig))
        chain = lib.load_rgb_chain(rgb)
        a, b = socket.socketpair()
        worker = threading.Thread(target=lambda: bridge.serve(a, rect, chain), daemon=True)
        worker.start()
        hello = struct.unpack("<5I2f", bridge.receive(b, 28))
        assert hello[0] == bridge.HELLO_MAGIC and hello[1:4] == (lib.W, lib.H, lib.RECT_W) and hello[4] == lib.FOOT_N
        assert abs(hello[5] - lib.RECT_F) < 1e-3 and abs(hello[6] - 0.29) < 1e-4
        b.sendall(struct.pack("<5I", bridge.REQUEST_MAGIC, lib.W, lib.H, lib.RECT_W, 41))
        b.sendall(np.full((lib.H, lib.W), 20.0, "<f4").tobytes())
        b.sendall(np.zeros((lib.H, lib.W), np.uint8).tobytes())
        flat = np.full((lib.H_IR, lib.W_IR), 40, np.uint8).tobytes()
        b.sendall(flat + flat)
        size = struct.calcsize(bridge.REPLY_FORMAT)
        assert size == 84, "must match ReplyHeader in src/ir_dot_bridge.cpp"
        head = struct.unpack(bridge.REPLY_FORMAT, bridge.receive(b, size))
        assert head[0] == bridge.REPLY_MAGIC and head[1] == 41 and head[3] == 0  # echoes the sequence; no plane
        body = bridge.receive(b, lib.FOOT_N ** 2 + lib.RECT_W ** 2 * 4)
        assert len(body) == lib.FOOT_N ** 2 + lib.RECT_W ** 2 * 4
        b.close()
        worker.join(5)
        assert not worker.is_alive()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: PASS")
    print("All IR-dot library tests PASS")
