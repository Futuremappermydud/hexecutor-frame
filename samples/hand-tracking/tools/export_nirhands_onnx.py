"""Export the NIR hand detector and stereo keypoint network to ONNX for the native on-device pipeline.

Writes into ``--output`` (an ignored local directory):

* ``detector.onnx``: ``image`` float32 [2,1,S,S] (area-resized, per-image standardized views) -> ``maps``
  float32 [2,10,S/4,S/4] logits. BatchNorm folded.
* ``keypoints.onnx``: ``crops`` float32 [B,2,K,K] (robust-normalized crops, slot 1 zero when mono) and
  ``second_valid`` float32 [B]. A keypoint checkpoint returns ``keypoints``, ``disparity`` and
  ``visibility_logit``. A skeleton checkpoint returns auxiliary keypoints/disparity, visibility, absolute palm
  points, angles and scale; the native host applies forward kinematics and Procrustes, keeping SVD off the CDSP.
* ``manifest.json``: source checkpoints with SHA-256, architecture, input sizes, the skeleton's mean profile,
  and the ONNX Runtime vs PyTorch check.

The native pipeline reads sizes from the manifest. Example from the repository root:

    NIR=/Users/owenelliott/orca/workspaces/synthetic-hand-render/cowfish/stereo_nir_hands
    "$NIR/.venv/bin/python" samples/hand-tracking/tools/export_nirhands_onnx.py --project "$NIR" \\
        --output samples/hand-tracking/.work/nirhands-models
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class KeypointExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, crops, second_valid):
        out = self.model(crops, second_valid)
        return out["keypoints"], out["disparity"], out["visibility_logit"]


class SkeletonKeypointExport(nn.Module):
    """Skeleton network up to its CDSP-friendly outputs; the host owns FK and the 3x3 Procrustes solve."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, crops, second_valid):
        from nirhands.training import skeleton as sk
        from nirhands.training.model import soft_argmax

        m = self.model
        b, _, s, _ = crops.shape
        if hasattr(m, "temporal_cell"):
            raise SystemExit("temporal skeleton checkpoints are not exportable yet (recurrent state)")
        u0, d2 = m._trunk(crops, second_valid)[:2]
        heatmaps = m.heatmap_head(u0)
        kp_hm, probs = soft_argmax(heatmaps * m.temperature, s)
        disp_hm = (probs * m.disparity_head(u0)).sum(dim=(2, 3))
        visibility_logit = m.visibility_head(u0.mean(dim=(2, 3)))
        pool_target = getattr(m, "pool", 4)
        kernel = m.pose_conv(d2).shape[-1] // pool_target
        pooled = F.avg_pool2d(m.pose_conv(d2), kernel).flatten(1)
        aux = torch.cat([kp_hm.detach().flatten(1), disp_hm.detach()], dim=1)
        feat = torch.cat([pooled, aux], dim=1)
        scale = 1.0 + 0.25 * torch.tanh(m.scale_mlp(feat)[:, 0])
        profile = m.mean_profile[None].to(feat.dtype) * scale.detach()[:, None]
        hidden = m.pose_mlp(torch.cat([feat, m.profile_embed(profile / 0.1)], dim=1))
        palm_points = (m.init_palm + m.palm_out(hidden)).reshape(b, len(sk.PALM_JOINTS), 3) * 0.1
        angles = m.angle_out(hidden).reshape(b, sk.NUM_BONES, 2)
        return kp_hm, disp_hm, visibility_logit, palm_points.reshape(b, -1), angles.reshape(b, -1), scale


def decode_skeleton_export(raw, mean_profile, geometry, size):
    """Apply the same host-side FK/Procrustes/root/encode contract as the native pipeline."""
    from nirhands.training import skeleton as sk

    kp_hm, disp_hm, visibility, palm, angles, scale = raw
    palm = palm.reshape(-1, len(sk.PALM_JOINTS), 3)
    palm = palm - palm[:, sk.PALM_JOINTS.index(sk.ROOT_INDEX):sk.PALM_JOINTS.index(sk.ROOT_INDEX) + 1]
    profile = mean_profile[None].to(scale.dtype) * scale.detach()[:, None]
    joints_rel = sk.pose_joints(profile, angles.reshape(-1, sk.NUM_BONES, 2), palm)
    joints_rel = joints_rel - joints_rel[:, sk.ROOT_INDEX:sk.ROOT_INDEX + 1]
    root = sk.decode_model_frame(kp_hm[:, sk.ROOT_INDEX:sk.ROOT_INDEX + 1],
                                 disp_hm[:, sk.ROOT_INDEX:sk.ROOT_INDEX + 1], geometry, size)
    keypoints, disparity = sk.encode_model_frame(joints_rel + root, geometry, size)
    return keypoints, disparity, visibility


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--detector", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args(argv)
    sys.path.insert(0, str(args.project / "src"))
    import onnxruntime as ort
    from nirhands.training import model as model_module
    from nirhands.training.detector import fold_batchnorm
    from nirhands.training.detector_train import load_detector

    runs = args.project.resolve().parent / "data" / "runs"
    det_path = args.detector or runs / "detector_s224" / "last.pt"
    from nirhands_pipeline import resolve_checkpoint
    if args.checkpoint:
        kp_path = resolve_checkpoint(args.checkpoint)
    else:
        local_tight96 = runs / "skeleton_tight96_ft4_fg" / "best.pt"
        if not local_tight96.exists():
            local_tight96 = runs / "skeleton_tight96_full_v1" / "best.pt"
        if local_tight96.exists():
            kp_path = local_tight96
        else:
            from huggingface_hub import hf_hub_download
            kp_path = Path(hf_hub_download(repo_id="FutureMapper/skeleton_tight96_ft4-fg", filename="best.pt"))
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    det, det_cfg = load_detector(det_path, "cpu")
    det_size = int(det_cfg["size"])
    det_onnx = args.output / "detector.onnx"
    torch.onnx.export(fold_batchnorm(det), (torch.zeros(2, 1, det_size, det_size),), str(det_onnx),
                      opset_version=args.opset, dynamo=False, input_names=["image"], output_names=["maps"],
                      do_constant_folding=True)
    x = rng.standard_normal((2, 1, det_size, det_size)).astype(np.float32)
    with torch.no_grad():
        ref = det(torch.from_numpy(x)).numpy()
    got = ort.InferenceSession(str(det_onnx), providers=["CPUExecutionProvider"]).run(None, {"image": x})[0]
    det_diff = float(np.abs(ref - got).max())

    ckpt = torch.load(kp_path, map_location="cpu", weights_only=False)
    if hasattr(model_module, "model_from_checkpoint"):
        kp = model_module.model_from_checkpoint(ckpt).eval()
    else:
        kp = model_module.StereoKeypointNet(width=int(ckpt["args"]["width"])).eval()
        kp.load_state_dict(ckpt["model"])
    kp_size = int(ckpt["args"]["size"])
    skeleton = bool(getattr(kp, "needs_geometry", False))
    # eval(): torch.onnx.export restores the wrapper's mode afterwards, and a fresh wrapper is in train mode,
    # which would leave BatchNorm models on batch statistics for the PyTorch reference below.
    export = (SkeletonKeypointExport(kp) if skeleton else KeypointExport(kp)).eval()
    output_names = (["keypoints_hm", "disparity_hm", "visibility_logit", "palm_points", "angles", "scale"]
                    if skeleton else ["keypoints", "disparity", "visibility_logit"])
    kp_onnx = args.output / "keypoints.onnx"
    torch.onnx.export(export, (torch.zeros(2, 2, kp_size, kp_size), torch.ones(2)), str(kp_onnx),
                      opset_version=args.opset, dynamo=False, input_names=["crops", "second_valid"],
                      output_names=output_names,
                      dynamic_axes={n: {0: "batch"} for n in ("crops", "second_valid", *output_names)},
                      do_constant_folding=True)
    sess = ort.InferenceSession(str(kp_onnx), providers=["CPUExecutionProvider"])
    kp_diff = raw_diff = 0.0
    for batch, valid in ((1, [1.0]), (3, [1.0, 0.0, 1.0])):
        crops = rng.uniform(-0.5, 1.5, (batch, 2, kp_size, kp_size)).astype(np.float32)
        sv = np.asarray(valid, np.float32)
        crops[sv == 0, 1] = 0
        geometry = torch.tensor([[2.0 * kp_size, -0.5, 0.25, 0.064, 0.45, float(i % 2)]
                                 for i in range(batch)], dtype=torch.float32)
        with torch.no_grad():
            raw_ref = export(torch.from_numpy(crops), torch.from_numpy(sv))
        got = sess.run(None, {"crops": crops, "second_valid": sv})
        raw_diff = max(raw_diff, *(float(np.abs(r.numpy() - g).max()) for r, g in zip(raw_ref, got)))
        if skeleton:
            from nirhands.training import skeleton as sk
            with torch.no_grad():
                ref = kp(torch.from_numpy(crops), torch.from_numpy(sv), geometry)
                decoded = decode_skeleton_export(raw_ref, kp.mean_profile, geometry, kp_size)
            kp_diff = max(kp_diff, *(float(np.abs(r.numpy() - g.numpy()).max())
                                     for r, g in zip((ref["keypoints"], ref["disparity"]), decoded[:2])))
        else:
            kp_diff = raw_diff

    manifest = {
        "detector": {"onnx": det_onnx.name, "size": det_size, "checkpoint": str(det_path), "sha256": sha256(det_path),
                     "max_abs_diff_vs_pytorch": det_diff},
        "keypoints": {"onnx": kp_onnx.name, "size": kp_size, "arch": "skeleton" if skeleton else "keypoint",
                      "checkpoint": str(kp_path), "sha256": sha256(kp_path),
                      "max_abs_diff_vs_pytorch": kp_diff, "max_abs_diff_raw_vs_pytorch": raw_diff},
        "opset": args.opset,
    }
    if skeleton:
        manifest["keypoints"]["mean_profile"] = [float(v) for v in kp.mean_profile]
        manifest["keypoints"]["outputs"] = output_names
        manifest["keypoints"]["pool"] = int(getattr(kp, "pool", 4))
    tight_crop = bool(getattr(kp, "tight_crop", False) or ckpt.get("args", {}).get("tight_crop", False))
    manifest["keypoints"]["tight_crop"] = tight_crop
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if det_diff > 1e-3 or kp_diff > 1e-3:
        raise SystemExit("ONNX Runtime output does not match PyTorch")


if __name__ == "__main__":
    main()
