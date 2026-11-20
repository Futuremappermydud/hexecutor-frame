"""Run the end-to-end NIR hand pipeline over a whole Frametop take (consecutive frame sets).

Reads ``sets.bin.zst`` and the session's factory calibration from the Frametop dataset, so it exercises the
same detector -> keypoints -> tracking -> smoothing path as the live viewer, at the recording's frame rate,
without the headset. Prints dropout/jitter/latency statistics; optionally shows the viewer or writes a video.

Frametop Hands is CC BY-NC 4.0 with privacy terms: keep rendered videos local.

Example (from the repository root):

    NIR=/Users/owenelliott/orca/workspaces/synthetic-hand-render/cowfish/stereo_nir_hands
    "$NIR/.venv/bin/python" samples/hand-tracking/tools/nirhands_take.py --project "$NIR" \\
        --contributor 884acd76 --take 03-gestures --display
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nirhands_pipeline import HandPipeline, TrackerConfig, add_project, load_profile  # noqa: E402

DATASET = Path.home() / (".cache/huggingface/hub/datasets--DeeJanuz--frametop-hands/snapshots/"
                         "10e42078c8d58108209e20653b0c11d49f499711/contributions")


def pipeline_args(p: argparse.ArgumentParser) -> None:
    """Model and tracker options shared by the take replay and the live viewer."""
    p.add_argument("--project", type=Path, required=True, help="read-only stereo_nir_hands project")
    p.add_argument("--detector", type=Path, default=None,
                   help="detector checkpoint (default <project>/../data/runs/detector_s224/last.pt)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="keypoint checkpoint (default <project>/../data/runs/cotrain_v5_labels_fah/best.pt)")
    p.add_argument("--device", default=None, help="mps or cpu (default mps when available)")
    p.add_argument("--profile", type=Path, default=None,
                   help="wearer's hand profile JSON from nirhands_profile.py (skeleton checkpoints only); "
                        "default: the model's mean hand at a predicted scale")
    p.add_argument("--detect-every", type=int, default=3, help="detector cadence once all hands are confirmed")
    p.add_argument("--detect-threshold", type=float, default=0.3)
    p.add_argument("--single-view-threshold", type=float, default=0.3)
    p.add_argument("--min-cutoff", type=float, default=1.5, help="One Euro minimum cutoff (Hz)")
    p.add_argument("--beta", type=float, default=10.0, help="One Euro speed coefficient (Hz per m/s)")
    p.add_argument("--smoothing", choices=("split", "world"), default="split",
                   help="split: palm pose and hand-local fingers filtered separately; world: one filter on joints")
    p.add_argument("--finger-min-cutoff", type=float, default=2.5, help="hand-local One Euro minimum cutoff (Hz)")
    p.add_argument("--finger-beta", type=float, default=20.0, help="hand-local One Euro speed coefficient")
    p.add_argument("--refine-m", type=float, default=0.015,
                   help="re-crop when the predicted root lands this far (m) from the crop centre; 0 disables")
    p.add_argument("--legacy", action="store_true",
                   help="tracker behaviour of the native port: no crop refinement, world smoothing")


def build_pipeline(args, cameras) -> HandPipeline:
    data = args.project.resolve().parent / "data" / "runs"
    detector = args.detector or data / "detector_s224" / "last.pt"
    checkpoint = args.checkpoint or data / "cotrain_v5_labels_fah" / "best.pt"
    kw = dict(detect_every=args.detect_every, min_cutoff=args.min_cutoff, beta=args.beta)
    if args.legacy:
        cfg = TrackerConfig.legacy(**kw)
    else:
        cfg = TrackerConfig(**kw, smoothing=args.smoothing, finger_min_cutoff=args.finger_min_cutoff,
                            finger_beta=args.finger_beta, refine_m=args.refine_m)
    profile = load_profile(args.profile) if args.profile else None
    pipe = HandPipeline(args.project, detector, checkpoint, cameras, args.device, cfg, profile=profile,
                        threshold=args.detect_threshold, single_view_threshold=args.single_view_threshold)
    print(f"detector {detector}\nkeypoints {checkpoint} ({pipe.keypoints.kind} model)\n"
          f"hand profile {args.profile or 'model mean'}\ndevice {pipe.device}", flush=True)
    return pipe


def find_session(dataset: Path, contributor: str, session: str | None) -> Path:
    roots = sorted(p for c in dataset.iterdir() if c.name.startswith(contributor) for p in c.iterdir() if p.is_dir())
    if session:
        roots = [r for r in roots if r.name.startswith(session)]
    if not roots:
        raise SystemExit(f"no session for contributor {contributor!r} session {session!r} under {dataset}")
    return roots[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pipeline_args(p)
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--contributor", required=True, help="contributor id prefix")
    p.add_argument("--session", default=None, help="session name prefix (default: first)")
    p.add_argument("--take", required=True, help="take name prefix, e.g. 03-gestures")
    p.add_argument("--start", type=int, default=0, help="first frame set")
    p.add_argument("--limit", type=int, default=0, help="frame sets to process (0 = all)")
    p.add_argument("--display", action="store_true")
    p.add_argument("--realtime", action="store_true", help="pace display at the recording's frame rate")
    p.add_argument("--video", type=Path, default=None, help="write the rendered view to this .mp4 (local only)")
    p.add_argument("--json", type=Path, default=None, help="write summary statistics and per-frame hands")
    p.add_argument("--labels", type=Path, default=None,
                   help="label root (e.g. <project>/../data/frametop_labels/fah_v1): score tracked hands on the "
                        "labelled frame sets (MPJPE, PA-MPJPE, per-finger flexion)")
    p.add_argument("--max-quality", type=int, default=1, help="label quality to score (1 = confident two-view)")
    args = p.parse_args(argv)
    add_project(args.project)
    from nirhands.frametop.session import Session, read_sets

    session = Session(find_session(args.dataset, args.contributor, args.session))
    takes = [t for t in session.takes() if t.name.startswith(args.take)]
    if not takes:
        raise SystemExit(f"no take {args.take!r}; have {[t.name for t in session.takes()]}")
    take = takes[0]
    cameras = (session.cameras["slam_left"], session.cameras["slam_right"])
    pipe = build_pipeline(args, cameras)
    print(f"take {session.name}/{take.name}", flush=True)

    viewer = writer = None
    if args.display or args.video:
        from nirhands_view import Viewer
        viewer = Viewer(cameras)
    per_frame = []
    shown = 0
    scorer = None
    if args.labels:
        from nirhands_metrics import TakeScorer
        npz = args.labels / session.root.parent.name / session.root.name / f"{take.name}.npz"
        if not npz.exists():
            raise SystemExit(f"no labels {npz}")
        scorer = TakeScorer(npz, args.max_quality)
    wall0 = t_first = None
    for fs in read_sets(take / "sets.bin.zst", cameras=("slam_left", "slam_right")):
        if fs.index < args.start:
            continue
        if args.limit and shown >= args.limit:
            break
        if set(fs.images) != {"slam_left", "slam_right"}:
            continue
        images = np.stack([fs.images["slam_left"], fs.images["slam_right"]])
        t = fs.capture_ns["slam_left"] / 1e9
        result = pipe.step(images, t)
        shown += 1
        if scorer is not None:
            scorer.add(fs.index, result.hands)
        per_frame.append({"set": fs.index, "t": t, "timings_ms": result.timings_ms,
                          "hands": [{"id": h.id, "right": h.is_right, "branch": h.branch,
                                     "visibility_mean": float(h.visibility.mean()),
                                     "joints": np.round(h.joints, 5).tolist(),
                                     "raw": np.round(h.raw, 5).tolist()} for h in result.hands]})
        if viewer is None:
            if shown % 50 == 0:
                print(f"set {fs.index}: hands {len(result.hands)} total {result.timings_ms['total_ms']:.1f} ms",
                      flush=True)
            continue
        frame = viewer.render(images, result, pipe.stats, f"set {fs.index}")
        if args.video:
            import cv2
            if writer is None:
                args.video.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                         (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise SystemExit(f"cannot write video {args.video}")
            writer.write(frame)
        if args.display:
            if args.realtime:
                if wall0 is None:
                    wall0, t_first = time.monotonic(), t
                wait = (t - t_first) - (time.monotonic() - wall0)
                if wait > 0:
                    time.sleep(wait)
            if not viewer.show(frame):
                break
    if writer is not None:
        writer.release()
    summary = {"take": f"{session.name}/{take.name}", **pipe.stats.summary()}
    if scorer is not None:
        summary["labels"] = scorer.summary()
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "frames": per_frame}) + "\n")


if __name__ == "__main__":
    main()
