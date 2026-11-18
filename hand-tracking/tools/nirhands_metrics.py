"""Accuracy of tracked hands against Frametop labels, including per-finger articulation.

``TakeScorer`` collects, on each labelled frame set of a take, the tracked hand nearest each labelled hand
(middle MCP within ``match_m``) and reports:

- **recall** (labelled hands with a tracked hand nearby) and **side accuracy**;
- **MPJPE** and **PA-MPJPE** (similarity Procrustes), for raw and smoothed joints;
- **per-finger flexion**: the sum of the three bend angles along each finger (thumb: CMC, MCP, IP; others: MCP,
  PIP, DIP), in degrees. For each finger it gives the mean absolute error, the Pearson correlation with the label
  across labelled hands, and the ratio of predicted to labelled standard deviation. A ratio well under 1 with a
  low correlation means the model drifts towards an average pose instead of following that finger.

The labels come from a teacher model (FAH or WiLoR plus a MANO fit), not motion capture, so these numbers measure
agreement with that teacher. A teacher that itself under-articulates fingers would hide the same failure.
"""
from __future__ import annotations

import numpy as np

FINGERS = {"thumb": (0, 1, 2, 3, 4), "index": (0, 5, 6, 7, 8), "middle": (0, 9, 10, 11, 12),
           "ring": (0, 13, 14, 15, 16), "pinky": (0, 17, 18, 19, 20)}
ROOT = 9


def flexion_deg(joints: np.ndarray) -> np.ndarray:
    """Per-finger total bend [5] in degrees: the sum of angles between consecutive bones along the chain."""
    out = []
    for chain in FINGERS.values():
        bones = np.diff(joints[list(chain)], axis=0)
        bones = bones / np.maximum(np.linalg.norm(bones, axis=-1, keepdims=True), 1e-9)
        cos = np.clip(np.sum(bones[:-1] * bones[1:], -1), -1.0, 1.0)
        out.append(np.degrees(np.arccos(cos)).sum())
    return np.asarray(out)


def procrustes_error_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean joint error after the best similarity transform of ``pred`` onto ``gt``."""
    p, g = pred - pred.mean(0), gt - gt.mean(0)
    u, s, vt = np.linalg.svd(p.T @ g)
    d = np.sign(np.linalg.det(u @ vt))
    s[-1] *= d
    u[:, -1] *= d
    R = u @ vt
    scale = s.sum() / max((p ** 2).sum(), 1e-12)
    return float(np.linalg.norm(scale * p @ R - g, axis=-1).mean() * 1000)


class TakeScorer:
    def __init__(self, labels_npz, max_quality: int = 1, match_m: float = 0.10):
        z = np.load(labels_npz, allow_pickle=True)
        self.rows = {int(s): i for i, s in enumerate(z["set_index"])}
        self.joints = z["joints_world"].astype(np.float64)   # [N, 2 (left, right), 21, 3]
        self.quality = z["quality"]
        self.max_quality = max_quality
        self.match_m = match_m
        self.records: list[dict] = []

    def add(self, set_index: int, hands) -> None:
        row = self.rows.get(int(set_index))
        if row is None:
            return
        for side in (0, 1):
            q = int(self.quality[row, side])
            gt = self.joints[row, side]
            if not 1 <= q <= self.max_quality or not np.isfinite(gt).all():
                continue
            rec = {"set": int(set_index), "right": bool(side), "matched": False}
            dists = [np.linalg.norm(h.raw[ROOT] - gt[ROOT]) for h in hands]
            if dists and min(dists) <= self.match_m:
                h = hands[int(np.argmin(dists))]
                rec.update(matched=True, side_ok=bool(h.is_right) == bool(side), gt_flex=flexion_deg(gt))
                for key, j in (("raw", h.raw), ("smoothed", h.joints)):
                    rec[f"mpjpe_{key}"] = float(np.linalg.norm(j - gt, axis=-1).mean() * 1000)
                    rec[f"pa_{key}"] = procrustes_error_mm(j, gt)
                    rec[f"flex_{key}"] = flexion_deg(j)
            self.records.append(rec)

    def summary(self) -> dict:
        m = [r for r in self.records if r["matched"]]
        out = {"labelled_hands": len(self.records), "matched": len(m),
               "recall": len(m) / len(self.records) if self.records else None}
        if not m:
            return out
        out["side_accuracy"] = float(np.mean([r["side_ok"] for r in m]))
        gt = np.stack([r["gt_flex"] for r in m])
        for key in ("raw", "smoothed"):
            out[f"mpjpe_mm_median_{key}"] = float(np.median([r[f"mpjpe_{key}"] for r in m]))
            out[f"pa_mpjpe_mm_median_{key}"] = float(np.median([r[f"pa_{key}"] for r in m]))
            pred = np.stack([r[f"flex_{key}"] for r in m])
            fingers = {}
            for i, name in enumerate(FINGERS):
                g, p = gt[:, i], pred[:, i]
                corr = float(np.corrcoef(g, p)[0, 1]) if len(m) > 2 and g.std() > 0 and p.std() > 0 else None
                fingers[name] = {"mae_deg": float(np.abs(p - g).mean()), "corr": corr,
                                 "std_ratio": float(p.std() / g.std()) if g.std() > 0 else None}
            out[f"flexion_{key}"] = fingers
        return out
