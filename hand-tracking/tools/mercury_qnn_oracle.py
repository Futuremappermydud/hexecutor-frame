#!/usr/bin/env python3
"""Compare Monado Mercury's ONNX nets under ONNX Runtime with their QNN HTP contexts.

Offline only. ONNX Runtime is the float32 reference; QAIRT's x86 HTP simulator
executes the V75 contexts produced by hexecutor/offline/compile_qnn.py and is the
oracle that native DSP handlers for those run lists must reproduce. Inputs are
built with Mercury's preprocessing (letterbox and grayscale normalization) from
a local image. Every model file, image and tensor stays under an ignored .work
directory.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np
from PIL import Image
import onnxruntime as ort

DETECTION = 160
KEYPOINT = 128
TARGET = "x86_64-linux-clang"


def normalize(gray_u8):
    """Mercury normalizeGrayscaleImage: std scaled to 0.25, mean moved to 0.5."""
    x = gray_u8.astype(np.float32) / 255.0
    std = float(x.std())
    if std == 0.0:
        raise ValueError("image has zero standard deviation")
    x *= np.float32(0.25 / std)
    x += np.float32(0.5 - float(x.mean()))
    return x


def letterbox(gray, size):
    h, w = gray.shape
    scale = min(size / w, size / h)
    iw, ih = max(1, round(w * scale)), max(1, round(h * scale))
    small = np.asarray(Image.fromarray(gray).resize((iw, ih), Image.BILINEAR))
    out = np.zeros((size, size), np.uint8)
    x0, y0 = (size - iw) // 2, (size - ih) // 2
    out[y0 : y0 + ih, x0 : x0 + iw] = small
    return out, scale, x0, y0


def crop(gray, cx, cy, side, size):
    box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    return np.asarray(
        Image.fromarray(gray).transform((size, size), Image.EXTENT, box, Image.BILINEAR)
    )


def run_sim(sdk, image, context, inputs, out_dir):
    """Run qnn-net-run on the simulator; inputs is an ordered {name: float32 array}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = []
    for name, value in inputs.items():
        path = out_dir / f"in-{name}.raw"
        np.ascontiguousarray(value, np.float32).tofile(path)
        fields.append(f"{name}:=/out/{path.name}")
    (out_dir / "input_list.txt").write_text(" ".join(fields) + "\n")
    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "-v", f"{sdk}:/sdk:ro", "-v", f"{context.parent}:/ctx:ro", "-v", f"{out_dir}:/out",
        "-e", f"LD_LIBRARY_PATH=/sdk/lib/{TARGET}:/usr/lib/llvm-14/lib", image,
        f"/sdk/bin/{TARGET}/qnn-net-run", "--backend", f"/sdk/lib/{TARGET}/libQnnHtp.so",
        "--retrieve_context", f"/ctx/{context.name}", "--input_list", "/out/input_list.txt",
        "--use_native_input_files", "--use_native_output_files", "--output_dir", "/out",
        "--log_level", "warn",
    ]
    (out_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
    with (out_dir / "sim.log").open("w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    result = out_dir / "Result_0"
    return {p.stem: np.fromfile(p, np.float32) for p in result.glob("*.raw")}


def compare(reference, simulated, layouts):
    report = {}
    for name, ref in reference.items():
        got = simulated[name]
        if name in layouts:
            got = got.reshape(layouts[name][0]).transpose(layouts[name][1])
        got = got.reshape(ref.shape)
        diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        report[name] = {
            "shape": list(ref.shape),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "ref_abs_max": float(np.abs(ref).max()),
            "nonfinite": int((~np.isfinite(got)).sum()),
        }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True, help="directory with the Mercury ONNX files")
    p.add_argument("--detection-context", type=Path, required=True)
    p.add_argument("--keypoint-context", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--docker-image", default="hexecutor-qairt-tools")
    a = p.parse_args()
    work = a.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    gray = np.asarray(Image.open(a.image).convert("L"))

    boxed, scale, x0, y0 = letterbox(gray, DETECTION)
    det_in = normalize(boxed)[None, None]
    det = ort.InferenceSession(str(a.models / "grayscale_detection_160x160.onnx"))
    det_names = [o.name for o in det.get_outputs()]
    det_ref = dict(zip(det_names, det.run(None, {"inputImg": det_in})))
    det_sim = run_sim(a.sdk.resolve(), a.docker_image, a.detection_context.resolve(),
                      {"inputImg": det_in}, work / "detection")
    report = {"detection": compare(det_ref, det_sim, {})}

    # Crop the most confident detection the way Mercury sizes it (2x the predicted size).
    hand = int(np.argmax(det_ref["hand_exists"][0]))
    cx = (det_ref["cx"][0, hand] + 1) / 2 * DETECTION
    cy = (det_ref["cy"][0, hand] + 1) / 2 * DETECTION
    side = det_ref["size"][0, hand] * DETECTION * 2.0
    cx, cy, side = (cx - x0) / scale, (cy - y0) / scale, side / scale
    patch = crop(gray, cx, cy, side, KEYPOINT)
    if hand == 1:
        patch = patch[:, ::-1]  # Mercury mirrors the second hand slot.
    kp_in = {
        "inputImg": normalize(np.ascontiguousarray(patch))[None, None],
        "lastKeypoints": np.zeros((1, 42), np.float32),
        "useLastKeypoints": np.zeros((1,), np.float32),
    }
    kp = ort.InferenceSession(str(a.models / "grayscale_keypoint_jan18.onnx"))
    kp_names = [o.name for o in kp.get_outputs()]
    kp_ref = dict(zip(kp_names, kp.run(None, kp_in)))
    kp_sim = run_sim(a.sdk.resolve(), a.docker_image, a.keypoint_context.resolve(), kp_in,
                     work / "keypoint")
    # The context returns heatmap_xy as NHWC [1,22,22,21]; ONNX is NCHW.
    report["keypoint"] = compare(kp_ref, kp_sim, {"heatmap_xy": ((1, 22, 22, 21), (0, 3, 1, 2))})
    argmax_ref = kp_ref["heatmap_xy"].reshape(21, -1).argmax(1)
    argmax_sim = (kp_sim["heatmap_xy"].reshape(22, 22, 21).transpose(2, 0, 1)
                  .reshape(21, -1).argmax(1))
    report["keypoint"]["heatmap_argmax_agree"] = int((argmax_ref == argmax_sim).sum())
    report["detection_used"] = {"slot": hand, "cx": float(cx), "cy": float(cy), "side": float(side),
                                "confidence": float(det_ref["hand_exists"][0, hand])}
    for name, arrays in (("detection", det_ref), ("keypoint", kp_ref)):
        for key, value in arrays.items():
            value.astype(np.float32).tofile(work / f"{name}-ref-{key}.f32")
    for key, value in kp_in.items():
        value.astype(np.float32).tofile(work / f"keypoint-in-{key}.f32")
    det_in.astype(np.float32).tofile(work / "detection-in-inputImg.f32")
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
