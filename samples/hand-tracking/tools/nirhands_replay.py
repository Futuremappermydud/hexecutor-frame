"""Preview the stereo NIR keypoint checkpoint on cached Steam Frame tracking frames.

This is a host-side replay diagnostic.  It deliberately imports the training
package's crop and decode code; the repo's OpenXR viewer currently captures
the separate color cameras and cannot supply these tracking-camera images.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from nirhands_pipeline import load_keypoint_model, run_keypoint_model


def _imports(project: Path):
    sys.path.insert(0, str(project / "src"))
    from nirhands.training.samples import ROOT_INDEX, build_sample, decode_prediction
    from nirhands.training.real_eval import (
        DEFAULT_DATASET, DEFAULT_FRAMES, DEFAULT_LABELS, DEFAULT_MANIFEST,
        _default_loader, _network_input, collect_hands,
    )
    return ROOT_INDEX, build_sample, decode_prediction, (
        DEFAULT_DATASET, DEFAULT_FRAMES, DEFAULT_LABELS, DEFAULT_MANIFEST,
        _default_loader, _network_input, collect_hands,
    )


class NirHandsBackend:
    """Per-hand inference using the reference sample and inverse geometry."""

    def __init__(self, checkpoint: Path, project: Path, device: str = "cpu"):
        self.root_index, self.build_sample, self.decode, imports = _imports(project)
        self.reference = imports
        self.normalize = imports[5]
        self.device = device
        self.model, self.size, self.kind = load_keypoint_model(checkpoint, device)

    def predict(self, images: np.ndarray, cameras: tuple, center_world: np.ndarray,
                is_right: bool, visible_views: tuple[bool, bool] = (True, True)):
        """Return world joints, visibility, and selected branch for one seeded hand."""
        if not any(visible_views):
            return None
        center = np.asarray(center_world, np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            return None
        # build_sample needs joints only to form training targets; its crop and
        # decode depend on center_world.  Synthetic joints at that center avoid
        # feeding any label to inference.
        dummy = np.broadcast_to(center, (21, 3)).copy()
        swapped = visible_views == (False, True)
        if swapped:
            images = images[[1, 0]]
            cameras = (cameras[1], cameras[0])
        sample = self.build_sample(images, *cameras, dummy, is_right=is_right,
                                   center_world=center, size=self.size,
                                   force_mono=not all(visible_views))
        x, second_valid = self.normalize(sample)
        pred = run_keypoint_model(self.model, x[None], [second_valid], [sample["geometry"]], self.device)
        joints = self.decode(sample["geometry"], pred["keypoints"][0].float().cpu().numpy(),
                             pred["disparity"][0].float().cpu().numpy())
        visibility = torch.sigmoid(pred["visibility_logit"][0]).float().cpu().numpy()
        if not np.isfinite(joints).all():
            return None
        return joints, visibility, sample["geometry"].branch


def _draw(images, cameras, hands):
    # MediaPipe's and nirhands' 21-joint order agree.
    bones = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
             (0,17),(17,18),(18,19),(19,20))
    panes = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in images]
    for joints, _, side in hands:
        color = (255, 100, 20) if side else (10, 120, 255)
        for pane, camera in zip(panes, cameras):
            pixels = camera.project(joints)
            for a, b in bones:
                pa, pb = tuple(np.rint(pixels[a]).astype(int)), tuple(np.rint(pixels[b]).astype(int))
                cv2.line(pane, pa, pb, color, 2, cv2.LINE_AA)
            for pixel in pixels:
                cv2.circle(pane, tuple(np.rint(pixel).astype(int)), 3, color, -1, cv2.LINE_AA)
    return np.hstack(panes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="read-only stereo_nir_hands project")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", choices=("mercury", "mediapipe", "oracle"), default="mercury")
    parser.add_argument("--model-dir", type=Path, help="existing detector ONNX assets")
    parser.add_argument("--display", action="store_true", help="show the two tracking views and reprojected joints")
    parser.add_argument("--verify", action="store_true", help="compare with eval_frametop predict_model at identical centers")
    args = parser.parse_args(argv)
    backend = NirHandsBackend(args.checkpoint, args.project, args.device)
    dataset, frames, labels, manifest, loader, _, collect = backend.reference
    items = collect(manifest, labels, max_quality=1)
    load = loader(dataset, frames)
    tracker = None
    if args.seed != "oracle" and args.model_dir is None:
        parser.error("--model-dir is required for detector seeding")
    count = 0
    timings = []
    predictions = []
    selected = []
    last_key = None
    for item in items:
        if count >= args.limit:
            break
        loaded = load(item)
        if loaded is None:
            continue
        images, left, right = loaded
        key = (item["contributor"], item["session"], item["take"], item["set_index"])
        if args.seed != "oracle":
            if key == last_key:
                continue
            if tracker is None or key[:2] != last_key[:2]:
                from nirhands_live import SeedTracker
                tracker = SeedTracker(backend, (left, right), args.model_dir, seed=args.seed)
            last_key = key
            start = time.perf_counter()
            hands = tracker.step(images)
            timings.append((time.perf_counter() - start) * 1000)
            print(f"{count}: {item['session']}/{item['take']}/{item['set_index']} "
                  f"hands={len(hands)} latency_ms={timings[-1]:.1f}", flush=True)
            if args.display:
                cv2.imshow("Steam Frame NIR: left | right", _draw(images, (left, right), hands))
                if cv2.waitKey(0) in (27, ord('q')):
                    break
            count += 1
            continue
        center = item["joints_world"][backend.root_index]
        start = time.perf_counter()
        outcome = backend.predict(images, (left, right), center, item["hand"] == 1)
        timings.append((time.perf_counter() - start) * 1000)
        if outcome is None:
            raise RuntimeError("model failed on a cached real Frame hand")
        joints, visibility, branch = outcome
        selected.append(item)
        predictions.append(joints)
        print(f"{count}: {item['session']}/{item['take']}/{item['set_index']} "
              f"hand={item['hand']} branch={branch} visible={int((visibility > .5).sum())} "
              f"latency_ms={timings[-1]:.1f}", flush=True)
        if args.display:
            cv2.imshow("Steam Frame NIR: left | right", _draw(images, (left, right), [(joints, visibility, item["hand"])]))
            if cv2.waitKey(0) in (27, ord('q')):
                break
        count += 1
    if args.seed == "oracle" and not predictions:
        raise RuntimeError("no cached labeled Frame hands found")
    if args.verify and args.seed != "oracle":
        parser.error("--verify requires --seed oracle")
    if args.verify:
        from nirhands.training.real_eval import predict_model
        oracle = predict_model(selected, backend.model, backend.size, dataset, frames, args.device,
                               lateral_sigma_m=0, depth_sigma_m=0, batch_size=1, load=load)
        error_mm = np.linalg.norm(np.asarray(predictions) - oracle, axis=2) * 1000
        print(f"reference max joint delta: {error_mm.max():.4f} mm", flush=True)
        if not np.isfinite(error_mm).all() or error_mm.max() > 1:
            raise RuntimeError("backend disagrees with eval_frametop by over 1 mm")
    print(f"median processing latency: {np.median(timings):.1f} ms ({args.device})")


if __name__ == "__main__":
    main()
