"""Measure the wearer's hand profile (palm shape and bone lengths) from a recorded live stream.

A skeleton checkpoint poses one mean hand at a predicted scale unless it is given the wearer's profile. This
tool measures that profile once per person: it replays a ``nirhands_live.py --record`` stream through the
pipeline to find the hands, runs the FAH-L labeller on both views of sampled frames, triangulates its 2D
keypoints and takes the median profile over the hands whose two views agree. Left and right hands are pooled
(right hands mirrored), as in training.

Record 20-30 s of open, spread hands turned slowly in front of the headset; curled fingers triangulate worse.
FAH-L's 2D joints are not exactly the MANO joints the model trained on (its knuckle row is a few percent wider).

Run with the FAH venv of the ``stereo_nir_hands`` project (it has FAH and the pipeline's dependencies):

    NIR=/Users/owenelliott/orca/workspaces/synthetic-hand-render/cowfish/stereo_nir_hands
    "$NIR/.deps/fah_venv/bin/python" samples/hand-tracking/tools/nirhands_profile.py --project "$NIR" \\
        --rig "$NIR/config/steam_frame_tracking.yaml" --checkpoint <skeleton checkpoint> \\
        --input samples/hand-tracking/.work/live-nir-<stamp>.frnir --output samples/hand-tracking/.work/hand-profile.json

Then pass ``--profile samples/hand-tracking/.work/hand-profile.json`` to ``nirhands_live.py`` or ``nirhands_take.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nirhands_pipeline import add_project  # noqa: E402


def main(argv=None):
    from nirhands_take import build_pipeline, pipeline_args
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pipeline_args(p)
    calib = p.add_mutually_exclusive_group(required=True)
    calib.add_argument("--rig", type=Path)
    calib.add_argument("--session-calibration", type=Path)
    p.add_argument("--input", type=Path, required=True, help="recorded stream (.frnir, .raw or .raw.zst)")
    p.add_argument("--output", type=Path, required=True, help="profile JSON to write")
    p.add_argument("--every", type=int, default=4, help="label every Nth pair")
    p.add_argument("--max-reproj-px", type=float, default=1.5,
                   help="keep hands whose median two-view reprojection error is below this")
    p.add_argument("--min-hands", type=int, default=30, help="fail below this many accepted hands")
    args = p.parse_args(argv)
    if args.profile:
        p.error("--profile is this tool's output, not an input")
    add_project(args.project)
    import torch
    from nirhands.frametop.fah_teacher import FahTeacher
    from nirhands.frametop.labeller import triangulate
    from nirhands.frametop.wilor_teacher import make_hand_crop
    from nirhands.training import skeleton as sk
    from nirhands.training.real_eval import _headset_up
    from nirhands_live import load_cameras, open_input, pairs

    cameras = load_cameras(args)
    pipe = build_pipeline(args, cameras)
    if pipe.keypoints.kind != "skeleton":
        raise SystemExit("a hand profile only applies to skeleton checkpoints")
    up = _headset_up(*cameras)
    teacher = FahTeacher()
    profiles, sides, seen = [], [], 0
    for index, (images, times, _) in enumerate(pairs(open_input(args.input), truncated_ok=True)):
        hands = pipe.step(images, times[0] / 1e9).hands
        if index % args.every:
            continue
        for hand in hands:
            if hand.branch != "stereo":
                continue
            seen += 1
            centre = (hand.raw.min(0) + hand.raw.max(0)) / 2
            crops = [make_hand_crop(images[v], cameras[v], centre, up, not hand.is_right, 256, 0.10) for v in (0, 1)]
            if min(c.valid_fraction for c in crops) < 0.9:
                continue
            pixels = np.stack([q.keypoints_raw for q in teacher.predict(crops, list(cameras),
                                                                        [not hand.is_right] * 2)])
            if not np.isfinite(pixels).all():
                continue
            joints, error = triangulate(list(cameras), pixels)
            if not np.isfinite(joints).all() or np.median(error) > args.max_reproj_px:
                continue
            if hand.is_right:
                joints = joints * [-1.0, 1.0, 1.0]  # left-hand convention; the profile is frame-independent
            profiles.append(sk.skeleton_profile(torch.from_numpy(joints).float()).numpy())
            sides.append(hand.is_right)
        if index % (args.every * 25) == 0:
            print(f"pair {index}: {len(profiles)} of {seen} hands accepted", flush=True)
    if len(profiles) < args.min_hands:
        raise SystemExit(f"only {len(profiles)} of {seen} hands accepted (need {args.min_hands}); record open "
                         "hands seen by both cameras")
    profiles, sides = np.array(profiles), np.array(sides)
    profile = np.median(profiles, axis=0)
    mean = pipe.keypoints.model.mean_profile.detach().cpu().numpy()
    fingers = lambda q: q[18:].reshape(5, 3).sum(1) * 1000  # noqa: E731
    report = {
        "hands": len(profiles), "left": int((~sides).sum()), "right": int(sides.sum()),
        "finger_length_mm": np.round(fingers(profile).astype(float), 1).tolist(),
        "model_mean_finger_length_mm": np.round(fingers(mean).astype(float), 1).tolist(),
        "wrist_to_middle_mcp_mm": round(float(np.linalg.norm(profile[9:12]) * 1000), 1),
        "model_mean_wrist_to_middle_mcp_mm": round(float(np.linalg.norm(mean[9:12]) * 1000), 1),
        "bone_length_spread_mm_median": round(float(np.median(profiles[:, 18:].std(0)) * 1000), 2),
    }
    for name, mask in (("left", ~sides), ("right", sides)):
        if mask.sum() >= 10:
            report[f"finger_length_mm_{name}"] = np.round(fingers(np.median(profiles[mask], axis=0)).astype(float), 1).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"profile": [round(float(v), 5) for v in profile], "source": str(args.input),
                                       "labeller": "fah-l two-view triangulation", **report}, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
