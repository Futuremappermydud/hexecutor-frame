"""Check the native C++ pipeline (samples/hand-tracking/nirhands) against nirhands_pipeline.py on the same pairs.

Runs the Python pipeline with the headset's factory calibration expressed in the OpenVR head frame (the
native pipeline's world frame), reads the native replay's JSON lines, and reports per-frame agreement:
seed centres, track IDs/sides and joint positions.

    NIR=/Users/owenelliott/orca/workspaces/synthetic-hand-render/cowfish/stereo_nir_hands
    "$NIR/.venv/bin/python" samples/hand-tracking/tools/compare_native_nirhands.py --project "$NIR" \\
        --calibration samples/hand-tracking/.work/device-calibration --input pairs.raw --native native.jsonl

``--to-raw IN OUT`` converts a JPEG (FRNIR002) recording to the raw FRNIR001 stream the native replay reads.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def head_frame_cameras(calibration_dir: Path):
    from nirhands.camera.calibration import CameraCalibration
    from nirhands.frametop.session import Session
    session = Session(calibration_dir)
    head_from_world = session.head_from_world()
    world_from_head = np.linalg.inv(head_from_world)
    out = []
    for name in ("slam_left", "slam_right"):
        c = session.cameras[name]
        R = c.R_cw @ world_from_head[:3, :3]
        t = c.R_cw @ world_from_head[:3, 3] + c.t_cw
        out.append(CameraCalibration(K=c.K, D=c.D, R_cw=R, t_cw=t, distortion_model=c.distortion_model))
    return tuple(out)


def to_raw(src: Path, dst: Path) -> int:
    from nirhands_live import open_input, pairs
    header = struct.Struct("<8sqqII")
    n = 0
    with open_input(src) as f, dst.open("wb") as out:
        for images, times, seqs in pairs(f, truncated_ok=True):
            out.write(header.pack(b"FRNIR001", times[0], times[1], seqs[0], seqs[1]))
            out.write(np.ascontiguousarray(images[0]).tobytes())
            out.write(np.ascontiguousarray(images[1]).tobytes())
            n += 1
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--to-raw", nargs=2, type=Path, metavar=("IN", "OUT"))
    p.add_argument("--calibration", type=Path, help="directory with calibration.json, device.json, manifest.json")
    p.add_argument("--input", type=Path, help="raw FRNIR001 stream given to the native replay")
    p.add_argument("--native", type=Path, help="native replay JSON lines")
    p.add_argument("--detector", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    from nirhands_pipeline import HandPipeline, TrackerConfig, add_project
    add_project(args.project)
    if args.to_raw:
        print(f"wrote {to_raw(*args.to_raw)} raw pairs to {args.to_raw[1]}")
        return
    from nirhands_live import pairs

    cameras = head_frame_cameras(args.calibration)
    runs = args.project.resolve().parent / "data" / "runs"
    pipe = HandPipeline(args.project, args.detector or runs / "detector_s224" / "last.pt",
                        args.checkpoint or runs / "cotrain_v5_labels_fah" / "best.pt", cameras, args.device,
                        TrackerConfig.legacy())  # the native port predates crop refinement and split smoothing
    native = [json.loads(line) for line in args.native.read_text().splitlines() if line.startswith("{")]
    seed_err, joint_err, raw_err = [], [], []
    id_mismatch = side_mismatch = count_mismatch = detect_mismatch = 0
    first_divergence = None
    with args.input.open("rb") as f:
        for i, (images, times, _) in enumerate(pairs(f, truncated_ok=True)):
            if i >= len(native):
                break
            r = pipe.step(images, times[0] / 1e9)
            n = native[i]
            detect_mismatch += int(bool(n["detected"]) != (r.seeds is not None))
            if r.seeds is not None and len(r.seeds) == len(n["seeds"]):
                for s, ns in zip(r.seeds, n["seeds"]):
                    seed_err.append(np.linalg.norm(s.center_world - np.asarray(ns["center"])) * 1000)
            py = {h.id: h for h in r.hands}
            nat = {h["id"]: h for h in n["hands"]}
            if set(py) != set(nat):
                count_mismatch += int(len(py) != len(nat))
                id_mismatch += 1
                if first_divergence is None:
                    first_divergence = (i, sorted(py), sorted(nat))
            for hid in set(py) & set(nat):
                side_mismatch += int(py[hid].is_right != bool(nat[hid]["right"]))
                raw_err.append(np.linalg.norm(py[hid].raw - np.asarray(nat[hid]["raw"]), axis=1).max() * 1000)
                joint_err.append(np.linalg.norm(py[hid].joints - np.asarray(nat[hid]["joints"]), axis=1).max() * 1000)
    q = lambda a: (np.percentile(a, [50, 90, 99, 100]).round(3).tolist() if a else None)
    report = {
        "frames": min(i + 1, len(native)),
        "detector_run_mismatch_frames": detect_mismatch,
        "seed_centre_mm_p50_p90_p99_max": q(seed_err),
        "frames_with_different_track_ids": id_mismatch,
        "frames_with_different_hand_count": count_mismatch,
        "first_divergence(frame, python_ids, native_ids)": first_divergence,
        "side_mismatches": side_mismatch,
        "raw_joint_max_err_mm_p50_p90_p99_max": q(raw_err),
        "smoothed_joint_max_err_mm_p50_p90_p99_max": q(joint_err),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
