#!/usr/bin/env python3
"""Offline ONNX Runtime responder for mercury_replay --exchange.

Development only: it lets the C++ Mercury port run against the reference
networks on recorded images before the DSP backend exists. It is never part of
the live viewer. Protocol on stdin/stdout, little-endian float32:

  "detect N\\n"   N x 160*160 image        -> N x (hand_exists[2], cx[2], cy[2], size[2])
  "keypoint N\\n" N x (128*128, 42, 1)     -> N x (heatmap_xy 21*22*22 NCHW, depth 21*22,
                                              scalar_extras 8, curls 10)
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, required=True,
                   help="directory with grayscale_detection_160x160.onnx and grayscale_keypoint_jan18.onnx")
    a = p.parse_args()
    det = ort.InferenceSession(str(a.models / "grayscale_detection_160x160.onnx"))
    kp = ort.InferenceSession(str(a.models / "grayscale_keypoint_jan18.onnx"))
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    while True:
        line = inp.readline()
        if not line:
            return
        kind, count = line.decode().split()
        count = int(count)
        if kind == "detect":
            data = np.frombuffer(inp.read(count * 160 * 160 * 4), np.float32).reshape(count, 1, 1, 160, 160)
            for i in range(count):
                hand_exists, cx, cy, size = det.run(["hand_exists", "cx", "cy", "size"], {"inputImg": data[i]})
                out.write(np.concatenate([hand_exists[0], cx[0], cy[0], size[0]]).astype(np.float32).tobytes())
        elif kind == "keypoint":
            stride = 128 * 128 + 42 + 1
            data = np.frombuffer(inp.read(count * stride * 4), np.float32).reshape(count, stride)
            for row in data:
                feeds = {
                    "inputImg": row[: 128 * 128].reshape(1, 1, 128, 128),
                    "lastKeypoints": row[128 * 128 : 128 * 128 + 42].reshape(1, 42),
                    "useLastKeypoints": row[-1:].reshape(1),
                }
                xy, depth, extras, curls = kp.run(["heatmap_xy", "heatmap_depth", "scalar_extras", "curls"], feeds)
                out.write(np.concatenate([xy.ravel(), depth.ravel(), extras.ravel(), curls.ravel()])
                          .astype(np.float32).tobytes())
        else:
            raise ValueError(f"unknown request {kind!r}")
        out.flush()


if __name__ == "__main__":
    main()
