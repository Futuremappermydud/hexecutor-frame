#!/usr/bin/env python3
"""Fetch Google's versioned Hand Landmarker bundle and lower both graphs offline.

Model source: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
MediaPipe attribution and license: ../NOTICE and ../LICENSE.mediapipe
"""

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hexecutor.offline.compact_hxfg import compact
from hexecutor.offline.lower_tflite import lower

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--assets",
        type=Path,
        required=True,
        help="explicit ignored model asset directory",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="explicit ignored deployment directory",
    )
    args = p.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    bundle = args.assets / "hand_landmarker.task"
    data = (
        bundle.read_bytes()
        if bundle.exists()
        else urllib.request.urlopen(URL, timeout=60).read()
    )
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("bundle hash mismatch; refusing to lower a different model")
    bundle.write_bytes(data)
    with zipfile.ZipFile(bundle) as z:
        for name, prefix in [
            ("hand_detector.tflite", "palm"),
            ("hand_landmarks_detector.tflite", "landmark"),
        ]:
            path = args.assets / name
            path.write_bytes(z.read(name))
            lower(path, args.output / prefix)
            # Pruned channels cost matrix tiles, depthwise blocks and VTCM.
            m = compact(args.output / prefix, args.output / prefix)
            print(
                f'{prefix}: {len(m["ops"])} ops, {m["arena_bytes"]} arena bytes, {m["constant_bytes"]} constant bytes'
            )
    (args.output / "SOURCE.txt").write_text(
        f"{URL}\nsha256 {SHA256}\nMediaPipe, Apache-2.0; see hand-tracking/NOTICE\n"
    )


if __name__ == "__main__":
    main()
