"""Measure Mercury/MediaPipe seeds and NIR keypoints on cached Frametop hands."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nirhands_live import HandSeeder
from nirhands_replay import NirHandsBackend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--seed", choices=("mercury", "mediapipe"), default="mercury")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--match-px", type=float, default=100)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    backend = NirHandsBackend(a.checkpoint, a.project, "cpu")
    dataset, frames, labels, manifest, loader, _, collect = backend.reference
    items = collect(manifest, labels, max_quality=1)
    load = loader(dataset, frames)
    from nirhands.frametop.labeller import triangulate
    from nirhands.frametop.session import unproject_kb
    seen = {}
    stats = {"hands": 0, "per_view_visible": [0, 0], "per_view_detected": [0, 0],
             "handed_correct": 0, "handed_total": 0, "stereo": 0, "mono": 0,
             "detector_ms": [], "center_error_mm": [], "joint_error_mm": [],
             "oracle_joint_error_mm": []}
    for item in items:
        if stats["hands"] >= a.limit:
            break
        key = (item["contributor"], item["session"], item["take"], item["set_index"])
        if key not in seen:
            loaded = load(item)
            if loaded is None:
                continue
            images, left, right = loaded
            seeder = HandSeeder(a.seed, (left, right), a.model_dir)
            start = time.perf_counter()
            detections = seeder.detect(images)
            stats["detector_ms"].append((time.perf_counter() - start) * 1000)
            seen[key] = (images, (left, right), detections)
        images, cameras, detections = seen[key]
        label = np.asarray(item["joints_world"], np.float64)
        center_gt = label[9]
        selected = []
        for view, cam in enumerate(cameras):
            pixel_gt = cam.project(center_gt)
            visible = (0 <= pixel_gt[0] < images.shape[2] and 0 <= pixel_gt[1] < images.shape[1])
            stats["per_view_visible"][view] += int(visible)
            closest = min(detections[view], key=lambda d: np.linalg.norm(d[0] - pixel_gt), default=None)
            hit = visible and closest is not None and np.linalg.norm(closest[0] - pixel_gt) <= a.match_px
            selected.append(closest if hit else None)
            if hit:
                stats["per_view_detected"][view] += 1
                stats["handed_total"] += 1
                stats["handed_correct"] += int(closest[1] == bool(item["hand"]))
        if selected[0] and selected[1]:
            seeded, _ = triangulate(list(cameras), np.array([[selected[0][0]], [selected[1][0]]]))
            center = seeded[0]
            visible = (True, True)
            stats["stereo"] += 1
        elif selected[0] or selected[1]:
            view = 0 if selected[0] else 1
            cam = cameras[view]
            # Fixed depth is the actual live mono policy; labels do not set it.
            ray = unproject_kb(cam, selected[view][0]) * .45
            center = cam.R_cw.T @ (ray - cam.t_cw)
            visible = (view == 0, view == 1)
            stats["mono"] += 1
        else:
            center = None
        if center is not None and np.isfinite(center).all():
            stats["center_error_mm"].append(float(np.linalg.norm(center - center_gt) * 1000))
            right = selected[0][1] if selected[0] else selected[1][1]
            pred = backend.predict(images, cameras, center, right, visible)
            if pred is not None:
                stats["joint_error_mm"].append(float(np.linalg.norm(pred[0] - label, axis=1).mean() * 1000))
        oracle = backend.predict(images, cameras, center_gt, bool(item["hand"]))
        if oracle is not None:
            stats["oracle_joint_error_mm"].append(float(np.linalg.norm(oracle[0] - label, axis=1).mean() * 1000))
        stats["hands"] += 1
    def summary(values):
        return {"count": len(values), "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None}
    result = {k: v for k, v in stats.items() if not isinstance(v, list)}
    result["per_view_visible"] = stats["per_view_visible"]
    result["per_view_detected"] = stats["per_view_detected"]
    result["handedness_accuracy"] = stats["handed_correct"] / max(stats["handed_total"], 1)
    for name in ("detector_ms", "center_error_mm", "joint_error_mm", "oracle_joint_error_mm"):
        result[name] = summary(stats[name])
    result["match_px"] = a.match_px
    print(json.dumps(result, indent=2), flush=True)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
