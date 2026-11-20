"""End-to-end stereo NIR hand pipeline: detector -> 3D seeds -> keypoint model -> tracks -> smoothing.

Host-side development pipeline for the Steam Frame slam (tracking) camera pair. It imports the read-only
``stereo_nir_hands`` project (``--project``) for the detector, crop geometry and keypoint network, so the
crop/decode code is exactly what training and evaluation use.

Per frame pair:

1. **Detect** (every frame while acquiring, every ``detect_every`` frames otherwise): the per-view CenterNet
   detector runs on both raw views and ``seed_hands`` pairs detections by ray intersection into 3D seeds
   (single-view seeds use the predicted distance).
2. **Associate** seeds with tracks by 3D distance. Unmatched seeds start tentative tracks.
3. **Keypoints**: every track is cropped around its predicted centre (last middle MCP + constant velocity) and
   all tracks run as one batch through the keypoint network. A track whose predicted middle MCP lands more than
   ``refine_m`` from its crop centre is cropped again at that point and re-run (``refine``), so the network sees
   the hand about as well centred as in training (1 cm lateral jitter) even when frames arrive slowly.
4. **Track management**: a tentative track is confirmed after ``confirm_hits`` detector hits; a track is
   dropped after ``max_missed`` consecutive detector runs without a matching seed, after low joint
   visibility, or when it duplicates a better track.
5. **Smoothing** on capture timestamps. ``smoothing="split"`` (default) filters the palm pose (wrist position and
   palm axes) and the hand-local joints separately, so whole-hand motion does not open the finger filter and
   finger motion is not tied to hand speed. ``"world"`` is the earlier single One Euro on world joints.

Nothing here is device code; it defines the behaviour a device port has to reproduce.
"""
from __future__ import annotations

import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20))
ROOT = 9  # middle MCP: crop centre and track position


def add_project(project: Path) -> None:
    src = str(Path(project) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ------------------------------------------------------------------------------------------- smoothing


class OneEuro:
    """One Euro filter (Casiez et al. 2012) over an array of any shape; time in seconds."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 10.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.x is None or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        # Speed per joint (norm over xyz) so all three axes of a joint share one cutoff.
        speed = np.linalg.norm(self.dx, axis=-1, keepdims=True)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


def palm_frame(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(origin [3], R [3,3]) of a hand-local frame: origin at the wrist, x towards the middle MCP, z the palm
    normal from index and pinky MCPs. Columns of R are the local axes in world coordinates."""
    o = joints[0]
    x = joints[ROOT] - o
    x = x / max(np.linalg.norm(x), 1e-9)
    z = np.cross(joints[5] - o, joints[17] - o)
    z = z - (z @ x) * x
    z = z / max(np.linalg.norm(z), 1e-9)
    return o, np.stack((x, np.cross(z, x), z), axis=1)


def to_local(joints: np.ndarray) -> np.ndarray:
    o, R = palm_frame(joints)
    return (joints - o) @ R


class HandSmoother:
    """Palm pose and hand-local joints filtered separately (each a One Euro filter), then recombined.

    Palm position uses the translation parameters; palm axes are unit vectors whose speed is in rad/s, so their
    speed coefficient is ``beta * lever_m`` (a point ``lever_m`` from the wrist moving at the same speed).
    Hand-local joints use their own cutoff and speed coefficient: the palm is static there, and fingers open the
    filter only when they actually move.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float, finger_min_cutoff: float,
                 finger_beta: float, lever_m: float = 0.15):
        self.origin = OneEuro(min_cutoff, beta, d_cutoff)
        self.axes = OneEuro(min_cutoff, beta * lever_m, d_cutoff)
        self.local = OneEuro(finger_min_cutoff, finger_beta, d_cutoff)

    def __call__(self, joints: np.ndarray, t: float) -> np.ndarray:
        o, R = palm_frame(joints)
        local = self.local((joints - o) @ R, t)
        o = self.origin(o, t)
        x, z = self.axes(np.stack((R[:, 0], R[:, 2])), t)
        x = x / max(np.linalg.norm(x), 1e-9)
        z = z - (z @ x) * x
        z = z / max(np.linalg.norm(z), 1e-9)
        R = np.stack((x, np.cross(z, x), z), axis=1)
        return o + local @ R.T


# ------------------------------------------------------------------------------------------- models


class Detector:
    """Per-view hand detector plus stereo seeding (``nirhands.training.detector``)."""

    def __init__(self, checkpoint: Path, device: str, threshold: float = 0.3, candidate_threshold: float = 0.15,
                 top_k: int = 3, single_view_threshold: float = 0.3):
        from nirhands.training.detector import SeedConfig, decode_view, preprocess, seed_hands
        from nirhands.training.detector_train import load_detector
        self.model, cfg = load_detector(checkpoint, device)
        self.size = int(cfg["size"])
        self.device = device
        self.threshold = threshold
        self.candidate_threshold = candidate_threshold
        self.top_k = top_k
        self.seed_cfg = SeedConfig(single_view_threshold=single_view_threshold)
        self._preprocess, self._decode, self._seed = preprocess, decode_view, seed_hands

    def __call__(self, images: np.ndarray, cameras) -> tuple[list, list]:
        """(3D seeds, per-view display detections) for one uint8 pair [2,H,W]."""
        x = np.stack([self._preprocess(images[v], self.size) for v in (0, 1)])[:, None]
        with torch.inference_mode():
            out = self.model(torch.from_numpy(x).to(self.device)).float().cpu().numpy()
        wh = [(images[v].shape[1], images[v].shape[0]) for v in (0, 1)]
        cands = [self._decode(out[v], wh[v], self.candidate_threshold, top_k=self.top_k) for v in (0, 1)]
        shown = [[d for d in c if d.score >= self.threshold] for c in cands]
        return self._seed(cands, cameras, self.seed_cfg), shown


def resolve_checkpoint(checkpoint: Path | str) -> Path:
    """Resolve a checkpoint argument from a local path, directory, run name, HF repo, or HF URL."""
    s = str(checkpoint)
    if "huggingface.co/" in s:
        s = s.split("huggingface.co/", 1)[1].strip("/")
    if s.startswith("FutureMapper/") or s.startswith("hf:") or ("/" in s and not Path(s).exists()):
        repo = s[3:] if s.startswith("hf:") else s
        repo_name = repo.split("/", 1)[-1]
        runs = Path("/Users/owenelliott/orca/workspaces/synthetic-hand-render/cowfish/data/runs")
        for cand in (runs / repo_name / "best.pt", runs / repo_name.replace("-", "_") / "best.pt"):
            if cand.exists():
                return cand
        try:
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(repo_id=repo, filename="best.pt"))
        except Exception:
            pass
    p = Path(checkpoint)
    if p.is_dir():
        for name in ("best.pt", "last.pt"):
            if (p / name).exists():
                return p / name
    if not p.exists():
        runs = Path("/Users/owenelliott/orca/workspaces/synthetic-hand-render/cowfish/data/runs")
        for cand in (runs / s / "best.pt", runs / s.replace("-", "_") / "best.pt", runs / s, runs / s.replace("-", "_")):
            if cand.is_file():
                return cand
            if cand.is_dir():
                for name in ("best.pt", "last.pt"):
                    if (cand / name).exists():
                        return cand / name
    return p


def load_keypoint_model(checkpoint: Path | str, device: str):
    """(model in eval mode, crop size, kind) for a keypoint checkpoint. ``kind`` is ``"skeleton"`` for
    ``SkeletonHandNet`` (``args["arch"] == "skeleton"``), else ``"keypoint"`` (``StereoKeypointNet``)."""
    from nirhands.training import model as model_module
    ckpt_path = resolve_checkpoint(checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if hasattr(model_module, "model_from_checkpoint"):
        net = model_module.model_from_checkpoint(ckpt)
    elif ckpt["args"].get("arch") == "skeleton":
        raise SystemExit(f"{checkpoint} is a skeleton checkpoint, but --project has no skeleton model code")
    else:
        net = model_module.StereoKeypointNet(width=int(ckpt["args"]["width"]))
        net.load_state_dict(ckpt["model"])
    kind = "skeleton" if getattr(net, "needs_geometry", False) else "keypoint"
    return net.to(device).eval(), int(ckpt["args"]["size"]), kind


def load_profile(path: Path) -> np.ndarray:
    """Hand profile [33] (palm points and bone lengths, metres, left-hand convention) written by
    ``nirhands_profile.py``."""
    import json
    profile = np.asarray(json.loads(Path(path).read_text())["profile"], np.float32)
    if profile.shape != (33,) or not np.isfinite(profile).all():
        raise SystemExit(f"{path}: not a 33-value hand profile")
    return profile


def run_keypoint_model(model, crops: np.ndarray, second_valid: list, geometries: list, device: str,
                       profile: np.ndarray | None = None) -> dict:
    """Forward pass for a batch of crops [B,2,S,S]. A skeleton model also gets each crop's geometry, so its
    ``keypoints``/``disparity`` outputs are its skeleton joints re-encoded into crop coordinates and decode with
    ``decode_prediction`` like the keypoint model's. Without geometry it would return its auxiliary heatmap path.
    ``profile`` [33] is the wearer's hand shape for a skeleton model; without it the model poses its mean hand
    at a predicted scale."""
    x = torch.from_numpy(crops).to(device)
    second = torch.tensor([float(v) for v in second_valid], dtype=torch.float32, device=device)
    with torch.inference_mode():
        if getattr(model, "needs_geometry", False):
            from nirhands.training.skeleton import geometry_from_samples
            geometry = geometry_from_samples(geometries).to(device=device, dtype=torch.float32)
            if profile is None:
                return model(x, second, geometry)
            shape = torch.from_numpy(np.asarray(profile, np.float32)).to(device).expand(len(crops), -1)
            return model(x, second, geometry, profile=shape)
        return model(x, second)


class Keypoints:
    """Batched stereo keypoint or skeleton network (``nirhands.training``) with the training crop/decode."""

    def __init__(self, checkpoint: Path, device: str, profile: np.ndarray | None = None):
        from nirhands.training.real_eval import _network_input
        from nirhands.training.samples import build_sample, decode_prediction
        self.model, self.size, self.kind = load_keypoint_model(checkpoint, device)
        self.device = device
        if profile is not None and self.kind != "skeleton":
            raise SystemExit("a hand profile needs a skeleton checkpoint; this one predicts free keypoints")
        self.profile = profile
        self._build, self._input, self._decode = build_sample, _network_input, decode_prediction
        # Checkpoints trained with --tight-crop expect a crop fitted to the hand (samples.tight_crop).
        self.tight = bool(getattr(self.model, "tight_crop", False))

    def crop_request(self, center: np.ndarray, is_right: bool, joints: np.ndarray | None, root: int = 9) -> tuple:
        """Request for a track predicted at root position ``center``. With a tight-crop model and previous
        ``joints``, the crop is fitted to those joints moved to the predicted root; else the loose default."""
        if self.tight and joints is not None:
            from nirhands.training.samples import tight_crop
            anchor, radius = tight_crop(np.asarray(joints, np.float64) + (center - joints[root]))
            return anchor, is_right, radius
        return center, is_right

    def __call__(self, images: np.ndarray, cameras, requests: list[tuple[np.ndarray, bool]]) -> list:
        """For each (centre_world, is_right[, hand_radius_m]): (joints [21,3], visibility [21], branch) or None."""
        if not requests:
            return []
        samples, xs, second = [], [], []
        for center, is_right, *radius in requests:
            dummy = np.broadcast_to(np.asarray(center, np.float64), (21, 3)).copy()  # targets unused
            extra = {"hand_radius_m": radius[0]} if radius else {}
            sample = self._build(images, cameras[0], cameras[1], dummy, is_right=is_right,
                                 center_world=center, size=self.size, **extra)
            x, valid = self._input(sample)
            samples.append(sample)
            xs.append(x)
            second.append(float(valid))
        pred = run_keypoint_model(self.model, np.stack(xs), second, [s["geometry"] for s in samples], self.device,
                                  self.profile)
        kp = pred["keypoints"].float().cpu().numpy().astype(np.float64)
        disp = pred["disparity"].float().cpu().numpy().astype(np.float64)
        vis = torch.sigmoid(pred["visibility_logit"]).float().cpu().numpy()
        out = []
        for i, sample in enumerate(samples):
            joints = self._decode(sample["geometry"], kp[i], disp[i])
            out.append((joints, vis[i], sample["geometry"].branch) if np.isfinite(joints).all() else None)
        return out


# ------------------------------------------------------------------------------------------- tracking


@dataclass
class TrackerConfig:
    max_hands: int = 2             # confirmed (shown) tracks
    max_tentative: int = 2         # candidate tracks on top, so a rejected false positive cannot block a real hand
    detect_every: int = 3          # detector cadence once every track is confirmed
    associate_m: float = 0.10      # stereo seed <-> track gate (seed centre error is ~40 mm median, ~65 mm p90)
    associate_radius: float = 0.8  # single-view seed <-> track gate, in detected hand radii, in that image
                                   # (a single-view seed's depth comes from the detector and is often 5-10 cm off)
    confirm_hits: int = 2          # detector hits before a track is shown
    max_missed: int = 3            # consecutive detector runs without a matching seed ...
    max_unseen_s: float = 0.3      # ... and at least this long since the last matching seed
    max_unseen_visible_s: float = 1.0  # ... or this long while the keypoint model still sees the hand clearly
    visible_mean: float = 0.6      # mean joint visibility counted as "clearly"
    start_stereo_score: float = 0.35  # new tracks need a stereo seed this strong ...
    start_mono_score: float = 0.5     # ... or a single-view seed this strong (Frametop no-hands FP scores: <=0.34 / <=0.41)
    min_visibility: float = 0.25   # mean joint visibility; below this counts as a weak frame
    max_weak: int = 3              # consecutive weak frames before drop
    duplicate_m: float = 0.06      # two tracks whose middle MCPs AND wrists are this close are one hand
                                   # (a mirrored left/right reading of one hand shares both; touching hands don't)
    confirm_visibility: float = 0.45  # mean joint visibility needed when a track is confirmed
    max_tentative_frames: int = 8  # an unconfirmed track is discarded after this many frames
    # Index-to-pinky MCP distance does not change with finger curl. Real Frametop hands: per-frame 5th percentile
    # 4.6-6.4 cm, median 5.9-6.8 cm. A collapsed skeleton on a bright window in a live capture: median 3.0 cm.
    confirm_span_m: float = 0.045  # median over at least confirm_span_frames tentative frames
    confirm_span_frames: int = 3
    keep_span_m: float = 0.035     # confirmed tracks: median over the last span_window frames
    span_window: int = 10
    reuse_id_s: float = 1.0        # a new same-side track near a recently dropped one takes its id
    reuse_id_m: float = 0.15
    max_speed_mps: float = 3.0     # clamp on the constant-velocity prediction
    max_jump_m: float = 0.12       # keypoint root further than this from the crop centre = weak frame
    refine_m: float = 0.015        # re-crop at the predicted root when it lands this far from the crop centre (0 = off)
    min_cutoff: float = 1.5        # One Euro, Hz (palm position and axes; all joints with smoothing="world")
    beta: float = 10.0             # One Euro, Hz per m/s
    d_cutoff: float = 1.0
    smoothing: str = "split"       # "split": palm pose + hand-local joints; "world": one filter on world joints
    finger_min_cutoff: float = 2.5  # hand-local joints (smoothing="split"), Hz
    finger_beta: float = 20.0       # Hz per m/s of hand-local joint speed

    @classmethod
    def legacy(cls, **kw) -> "TrackerConfig":
        """Behaviour before crop refinement and split smoothing (what the native C++ port implements)."""
        return cls(**{"refine_m": 0.0, "smoothing": "world", **kw})


@dataclass
class Track:
    id: int
    center: np.ndarray
    side_vote: float
    t: float
    hits: int = 1
    missed: int = 0
    weak: int = 0
    age: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joints: np.ndarray | None = None
    smoothed: np.ndarray | None = None
    visibility: np.ndarray | None = None
    branch: str = ""
    score: float = 0.0
    views: tuple = ()
    filter: OneEuro | HandSmoother | None = None
    confirmed: bool = False
    last_hit_t: float = 0.0
    spans: deque = field(default_factory=lambda: deque(maxlen=TrackerConfig.span_window))  # recent knuckle spans

    @property
    def is_right(self) -> bool:
        return self.side_vote > 0


@dataclass
class HandResult:
    id: int
    is_right: bool
    joints: np.ndarray      # smoothed world joints [21,3] (metres, rig world frame)
    raw: np.ndarray         # unsmoothed network output
    visibility: np.ndarray
    branch: str
    score: float
    views: tuple
    age: int


@dataclass
class FrameResult:
    hands: list
    detections: list | None     # per-view Detection lists when the detector ran, else None
    seeds: list | None
    timings_ms: dict


class HandPipeline:
    """Stateful tracker. Call ``reset()`` between unrelated streams."""

    def __init__(self, project: Path, detector: Path, keypoints: Path, cameras, device: str | None = None,
                 config: TrackerConfig | None = None, profile: np.ndarray | None = None, **detector_kwargs):
        add_project(project)
        self.device = device or default_device()
        self.cameras = tuple(cameras)
        self.cfg = config or TrackerConfig()
        self.detector = Detector(detector, self.device, **detector_kwargs)
        self.keypoints = Keypoints(keypoints, self.device, profile)
        self.stats = PipelineStats()
        self.reset()

    def reset(self) -> None:
        self.tracks: list[Track] = []
        self.recent: list[tuple] = []   # (id, is_right, centre, drop time) of dropped confirmed tracks
        self.frame = 0
        self.next_id = 1
        self.t = None

    def _retire(self, reason: str, tr: Track) -> None:
        self.stats.drop(reason, tr)
        if tr.confirmed:
            self.recent.append((tr.id, tr.is_right, tr.center.copy(), self.t))

    def _need_detect(self) -> bool:
        confirmed = [t for t in self.tracks if t.confirmed]
        if len(confirmed) < self.cfg.max_hands or any(not t.confirmed or t.weak for t in self.tracks):
            return True
        return self.frame % self.cfg.detect_every == 0

    def step(self, images: np.ndarray, t: float) -> FrameResult:
        """One synchronized uint8 pair [2,H,W] at capture time ``t`` (seconds, increasing)."""
        cfg = self.cfg
        timings = {}
        t0 = time.perf_counter()
        detections = seeds = None
        self.t = t
        self.recent = [r for r in self.recent if t - r[3] <= cfg.reuse_id_s]
        # Predict each track forward to this frame.
        for tr in self.tracks:
            dt = min(max(t - tr.t, 0.0), 0.2)
            v = tr.velocity
            speed = float(np.linalg.norm(v))
            if speed > cfg.max_speed_mps:
                v = v * (cfg.max_speed_mps / speed)
            tr.center = tr.center + v * dt
        if self._need_detect():
            seeds, detections = self.detector(images, self.cameras)
            timings["detect_ms"] = (time.perf_counter() - t0) * 1000
            self._associate(seeds, t)
        t1 = time.perf_counter()
        outs = self.keypoints(images, self.cameras,
                              [self.keypoints.crop_request(tr.center, tr.is_right, tr.joints) for tr in self.tracks])
        timings["keypoints_ms"] = (time.perf_counter() - t1) * 1000
        if cfg.refine_m > 0:
            redo = [i for i, (tr, out) in enumerate(zip(self.tracks, outs)) if out is not None
                    and cfg.refine_m < np.linalg.norm(out[0][ROOT] - tr.center) <= cfg.max_jump_m]
            if redo:
                t2 = time.perf_counter()
                again = self.keypoints(images, self.cameras,
                                       [self.keypoints.crop_request(outs[i][0][ROOT], self.tracks[i].is_right,
                                                                    outs[i][0]) for i in redo])
                for i, out in zip(redo, again):
                    if out is not None:
                        self.tracks[i].center = outs[i][0][ROOT].copy()
                        outs[i] = out
                timings["refine_ms"] = (time.perf_counter() - t2) * 1000
                self.stats.refined += len(redo)
        survivors = []
        for tr, out in zip(self.tracks, outs):
            if out is None:
                continue
            joints, vis, branch = out
            root = joints[ROOT]
            if np.linalg.norm(root - tr.center) > cfg.max_jump_m:
                # Usually fast motion out of the crop. Hold the last pose and let the detector recentre.
                tr.weak += 1
                tr.velocity = np.zeros(3)
                if tr.joints is not None:
                    tr.center = tr.joints[ROOT].copy()
                if tr.weak >= cfg.max_weak or tr.joints is None:
                    self._retire("jump", tr)
                    continue
                survivors.append(tr)
                continue
            dt = t - tr.t
            if tr.joints is not None and dt > 1e-4:
                inst = (root - tr.joints[ROOT]) / dt
                tr.velocity = 0.5 * tr.velocity + 0.5 * inst
            tr.joints, tr.visibility, tr.branch, tr.t = joints, vis, branch, t
            tr.center = root.copy()
            tr.weak = tr.weak + 1 if float(vis.mean()) < cfg.min_visibility else 0
            if tr.weak >= cfg.max_weak:
                self._retire("visibility", tr)
                continue
            if tr.filter is None:
                tr.filter = (HandSmoother(cfg.min_cutoff, cfg.beta, cfg.d_cutoff, cfg.finger_min_cutoff,
                                          cfg.finger_beta)
                             if cfg.smoothing == "split" else OneEuro(cfg.min_cutoff, cfg.beta, cfg.d_cutoff))
            tr.smoothed = tr.filter(joints, t)
            tr.spans.append(float(np.linalg.norm(joints[5] - joints[17])))
            if (tr.confirmed and len(tr.spans) == tr.spans.maxlen
                    and float(np.median(tr.spans)) < cfg.keep_span_m):
                self._retire("implausible", tr)
                continue
            tr.age += 1
            survivors.append(tr)
        self.tracks = self._dedupe(survivors)
        self.frame += 1
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        hands = [HandResult(tr.id, tr.is_right, tr.smoothed, tr.joints, tr.visibility, tr.branch, tr.score,
                            tr.views, tr.age) for tr in self.tracks if tr.confirmed]
        self.stats.update(t, hands, timings, detected=seeds is not None)
        return FrameResult(hands, detections, seeds, timings)

    def _seed_cost(self, seed, center) -> float:
        """Seed-to-track distance normalized by its gate (<= 1 matches): 3D for stereo seeds, image for single-view."""
        if len(seed.views) == 2:
            return float(np.linalg.norm(seed.center_world - center)) / self.cfg.associate_m
        view, det = seed.views[0], seed.detections[0]
        camera = self.cameras[view]
        if camera.world_to_camera(center)[2] <= 0:
            return np.inf
        try:
            u, v = camera.project(center)
        except ValueError:
            return np.inf
        return math.hypot(u - det.u, v - det.v) / (self.cfg.associate_radius * max(det.radius_px, 20.0))

    def _associate(self, seeds, t) -> None:
        cfg = self.cfg
        pairs = sorted((self._seed_cost(s, tr.center), i, k)
                       for i, tr in enumerate(self.tracks) for k, s in enumerate(seeds))
        used_t, used_s = set(), set()
        for cost, i, k in pairs:
            if cost > 1.0 or i in used_t or k in used_s:
                continue
            used_t.add(i)
            used_s.add(k)
            tr, s = self.tracks[i], seeds[k]
            tr.hits += 1
            tr.missed = 0
            tr.score = s.score
            tr.views = s.views
            tr.last_hit_t = t
            if tr.weak:
                tr.center = np.asarray(s.center_world, np.float64).copy()
            tr.side_vote = 0.8 * tr.side_vote + (s.score if s.is_right else -s.score)
            if (not tr.confirmed and tr.hits >= cfg.confirm_hits and tr.visibility is not None
                    and sum(o.confirmed for o in self.tracks) < cfg.max_hands
                    and float(tr.visibility.mean()) >= cfg.confirm_visibility
                    and len(tr.spans) >= cfg.confirm_span_frames
                    and float(np.median(tr.spans)) >= cfg.confirm_span_m):
                tr.confirmed = True
        kept = []
        for i, tr in enumerate(self.tracks):
            if i not in used_t:
                tr.missed += 1
                if not tr.confirmed and tr.missed >= 2:
                    continue
                clear = tr.visibility is not None and float(tr.visibility.mean()) >= cfg.visible_mean
                unseen = cfg.max_unseen_visible_s if clear else cfg.max_unseen_s
                if tr.missed >= cfg.max_missed and t - tr.last_hit_t >= unseen:
                    self._retire("missed", tr)
                    continue
            if not tr.confirmed and tr.age >= cfg.max_tentative_frames:
                continue
            kept.append(tr)
        self.tracks = kept
        for k in sorted(range(len(seeds)), key=lambda k: -seeds[k].score):
            s = seeds[k]
            if k in used_s or len(self.tracks) >= cfg.max_hands + cfg.max_tentative:
                continue
            if s.score < (cfg.start_stereo_score if len(s.views) == 2 else cfg.start_mono_score):
                continue
            if any(self._seed_cost(s, tr.center) <= 1.0 for tr in self.tracks):
                continue
            center = np.asarray(s.center_world, np.float64).copy()
            tr = Track(self.next_id, center, s.score if s.is_right else -s.score, t, score=s.score, views=s.views,
                       last_hit_t=t)
            tr.confirmed = cfg.confirm_hits <= 1
            reuse = [r for r in self.recent if r[1] == s.is_right and np.linalg.norm(r[2] - center) < cfg.reuse_id_m
                     and all(r[0] != other.id for other in self.tracks)]
            if reuse and sum(o.confirmed for o in self.tracks) < cfg.max_hands:
                # Re-acquiring a hand that was just lost: keep its id and skip confirmation.
                tr.id, tr.confirmed = reuse[-1][0], True
                self.recent.remove(reuse[-1])
            else:
                self.next_id += 1
            self.tracks.append(tr)

    def _dedupe(self, tracks: list[Track]) -> list[Track]:
        order = sorted(tracks, key=lambda tr: (not tr.confirmed, -tr.hits, -float(tr.visibility.mean())))
        kept: list[Track] = []
        def duplicate(a, b):
            if np.linalg.norm(a.center - b.center) >= self.cfg.duplicate_m:
                return False
            if a.joints is None or b.joints is None:
                return True
            return float(np.linalg.norm(a.joints[0] - b.joints[0])) < self.cfg.duplicate_m

        for tr in order:
            if any(duplicate(tr, k) for k in kept):
                self._retire("duplicate", tr)
                continue
            kept.append(tr)
        confirmed = sorted((tr for tr in kept if tr.confirmed), key=lambda tr: tr.id)[: self.cfg.max_hands]
        tentative = [tr for tr in kept if not tr.confirmed][: self.cfg.max_tentative]
        return confirmed + tentative


# ------------------------------------------------------------------------------------------- statistics


class PipelineStats:
    """Running dropout, jitter and latency statistics for a stream.

    Jitter is the RMS second difference of joint positions between consecutive frames of one track (mm), which
    is zero for constant-velocity motion; it is reported for raw and smoothed joints. It mixes real hand
    acceleration with noise, so compare it between configurations on the same footage, not as an absolute.
    Local jitter is the same in the hand's own palm frame (``to_local``): whole-hand motion drops out, so it is
    mostly finger noise plus real finger motion.
    """

    def __init__(self, window: int = 120):
        self.frames = 0
        self.frames_with_hands = 0
        self.detector_runs = 0
        self.tracks_started: set[int] = set()
        self.drops: dict[str, int] = {}
        self.track_frames: dict[int, int] = {}
        self.history: dict[int, deque] = {}
        self.jitter = {"raw": [], "smoothed": [], "local_raw": [], "local_smoothed": []}
        self.refined = 0
        self.timings: dict[str, deque] = {}
        self.window = window
        self.times: deque = deque(maxlen=window)

    def drop(self, reason: str, track: Track) -> None:
        if track.confirmed:
            self.drops[reason] = self.drops.get(reason, 0) + 1

    def update(self, t, hands, timings, detected) -> None:
        self.frames += 1
        self.detector_runs += int(detected)
        self.frames_with_hands += int(bool(hands))
        self.times.append(time.perf_counter())
        for k, v in timings.items():
            self.timings.setdefault(k, deque(maxlen=self.window)).append(v)
        live = set()
        for h in hands:
            live.add(h.id)
            self.tracks_started.add(h.id)
            self.track_frames[h.id] = self.track_frames.get(h.id, 0) + 1
            hist = self.history.setdefault(h.id, deque(maxlen=3))
            hist.append((h.raw.copy(), h.joints.copy(), to_local(h.raw), to_local(h.joints)))
            if len(hist) == 3:
                for i, key in enumerate(("raw", "smoothed", "local_raw", "local_smoothed")):
                    a, b, c = (x[i] for x in hist)
                    self.jitter[key].append(float(np.sqrt(np.mean(np.sum((a - 2 * b + c) ** 2, -1)))) * 1000)
        for tid in list(self.history):
            if tid not in live:
                del self.history[tid]

    def fps(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-6)

    def median_ms(self, key: str) -> float:
        v = self.timings.get(key)
        return float(np.median(v)) if v else float("nan")

    def summary(self) -> dict:
        lengths = list(self.track_frames.values())
        med = lambda k: float(np.median(self.jitter[k])) if self.jitter[k] else None  # noqa: E731
        return {
            "frames": self.frames,
            "frames_with_hands": self.frames_with_hands,
            "detector_runs": self.detector_runs,
            "confirmed_tracks": len(self.tracks_started),
            "track_length_frames_median": float(np.median(lengths)) if lengths else 0.0,
            "drops": dict(self.drops),
            "jitter_mm_rms_raw_median": med("raw"),
            "jitter_mm_rms_smoothed_median": med("smoothed"),
            "local_jitter_mm_rms_raw_median": med("local_raw"),
            "local_jitter_mm_rms_smoothed_median": med("local_smoothed"),
            "refined_crops": self.refined,
            "latency_ms_median": {k: self.median_ms(k) for k in self.timings},
        }
