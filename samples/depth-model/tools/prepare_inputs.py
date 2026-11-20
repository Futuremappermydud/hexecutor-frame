#!/usr/bin/env python3
"""Validate four native QNN input files, or generate an explicit synthetic smoke case."""
import argparse
import json
from pathlib import Path
import struct

SIZES = {"x": 81920, "y": 81920, "hidden": 655360, "t": 128}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true",
                        help="Synthetic tensors for runtime testing; do not represent valid depth inputs")
    for name in SIZES:
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.smoke and any(getattr(args, k) for k in SIZES):
        parser.error("choose --smoke or all four input paths")
    if not args.smoke and not all(getattr(args, k) for k in SIZES):
        parser.error("supply --x --y --hidden --t, or --smoke")
    output = args.output.resolve()
    if any(c.isspace() for c in str(output)):
        parser.error("QNN input-list paths must not contain whitespace")
    blobs = {}
    if args.smoke:
        blobs["x"] = blobs["y"] = bytes(range(256)) * 320
        blobs["hidden"] = bytes(SIZES["hidden"])
        # Identity matrices are a placeholder, not a recovered transform convention.
        blobs["t"] = struct.pack("<32f", *([1.0 if i % 5 == 0 else 0.0 for i in range(16)] * 2))
    else:
        for name, size in SIZES.items():
            blobs[name] = getattr(args, name).read_bytes()
            if len(blobs[name]) != size:
                parser.error(f"{name}: expected {size} bytes, got {len(blobs[name])}")
    output.mkdir(parents=True, exist_ok=True)
    for name, blob in blobs.items():
        (output / f"{name}.raw").write_bytes(blob)
    (output / "input_list.txt").write_text(
        " ".join(f"{name}:={output / (name + '.raw')}" for name in SIZES) + "\n")
    (output / "inputs.json").write_text(json.dumps({
        "synthetic_smoke_only": args.smoke,
        "byte_sizes": SIZES,
        "semantics": "Caller must supply rectification, temporal state and transform conventions; these remain unverified."
    }, indent=2) + "\n")
    print(output / "input_list.txt")


if __name__ == "__main__":
    main()
