#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Export spatially bounded HXFG tiles of the SHARP multiresolution decoders."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from export_fusion import write_hwc


class DisparityTile(torch.nn.Module):
    def __init__(self, projection, fusion, crop, combine):
        super().__init__()
        self.projection = projection
        self.fusion = fusion
        self.crop = crop
        self.combine = combine

    def forward(self, value, skip=None):
        if self.combine:
            decoded = self.fusion(value, self.projection(skip))
        else:
            decoded = self.fusion(self.projection(value))
        y, x, size = self.crop
        return decoded[:, :, y:y + size, x:x + size]


def read_capture(directory, level):
    sides = (768, 384, 192, 96, 48)
    channels = (256, 256, 512, 1024, 1024)
    shape = (1, channels[level], sides[level], sides[level])
    data = np.fromfile(directory / f"rest_a-{level}.f32", dtype="<f4")
    return torch.from_numpy(data.reshape(shape))


def target(backbone, captures, level, device):
    decoder = backbone.decoder.to(device)
    coarse = None
    for index in range(4, max(level, 0) - 1, -1):
        skip = read_capture(captures, index).to(device)
        projected = decoder.convs[index](skip)
        if coarse is None:
            result = decoder.fusions[index](projected)
        else:
            result = decoder.fusions[index](coarse, projected)
        if index == level:
            inputs = (skip,) if coarse is None else (coarse, skip)
            return decoder.convs[index], decoder.fusions[index], inputs, result
        coarse = result
    head = backbone.head.to(device)
    return torch.nn.Identity(), head, (coarse,), head(coarse)


def export_tile(projection, fusion, inputs, expected, level, y, x, tile, output):
    side = inputs[0].shape[-1]
    # Receptive-field radii for the head, deepest stage and skip stages respectively.
    halo = 2 if level == -1 else 3 if level == 4 else 5
    scale = 1 if level == 0 else 2
    y0, x0 = max(0, y - halo), max(0, x - halo)
    y1, x1 = min(side, y + tile + halo), min(side, x + tile + halo)
    cropped = tuple(v[:, :, y0:y1, x0:x1].cpu().contiguous() for v in inputs)
    module = DisparityTile(projection, fusion, ((y - y0) * scale, (x - x0) * scale, tile * scale),
                           len(cropped) == 2).eval()
    oracle = expected[:, :, y * scale:(y + tile) * scale, x * scale:(x + tile) * scale].cpu()
    torch.testing.assert_close(module(*cropped), oracle, rtol=5e-4, atol=2e-4)
    output.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(module, cropped, output / "disparity.onnx",
                      input_names=[f"input_{i}" for i in range(len(cropped))],
                      output_names=["disparity_features"], opset_version=17, dynamo=False)
    for i, value in enumerate(cropped):
        write_hwc(output / f"input_{i}.f32", value)
    write_hwc(output / "expected.f32", oracle)
    manifest = {"level": level, "input_crop": [y0, x0, y1, x1], "origin": [y, x],
                "tile": tile, "scale": scale, "inputs": [list(t.shape) for t in cropped],
                "output": list(oracle.shape)}
    (output / "tile.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apple-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=range(-1, 5), default=4,
                        help="decoder level 4..0, or -1 for the disparity head")
    parser.add_argument("--origin", type=int, nargs=2, metavar=("Y", "X"), default=(0, 0))
    parser.add_argument("--tile-size", choices=(8, 16, 32), type=int, default=8)
    parser.add_argument("--all-boundaries", action="store_true",
                        help="export each top/interior/bottom by left/interior/right placement")
    parser.add_argument("--reference-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--gaussian", action="store_true",
                        help="use the trained Gaussian feature decoder")
    args = parser.parse_args()
    if args.gaussian and args.level == -1:
        parser.error("the Gaussian feature decoder supports levels 4..0")
    sys.path.insert(0, str(args.apple_source.resolve() / "src"))
    from sharp.models import create_predictor
    from sharp.models.params import PredictorParams

    torch.set_num_threads(4)
    predictor = create_predictor(PredictorParams()).eval()
    predictor.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True))
    backbone = predictor.feature_model if args.gaussian else predictor.monodepth_model.monodepth_predictor
    with torch.inference_mode():
        projection, fusion, inputs, expected = target(backbone, args.captures, args.level, args.reference_device)
        projection, fusion = projection.cpu(), fusion.cpu()
        side = inputs[0].shape[-1]
        y, x = args.origin
        tile = args.tile_size
        if min(y, x) < 0 or max(y, x) + tile > side:
            parser.error("tile origin lies outside the layer")
        if args.all_boundaries:
            if side % tile or side < 3 * tile:
                parser.error("boundary export requires three or more whole tiles per axis")
            positions = (0, tile, side - tile)
            for row, y in enumerate(positions):
                for col, x in enumerate(positions):
                    export_tile(projection, fusion, inputs, expected, args.level, y, x, tile,
                                args.output / f"r{row}c{col}")
        else:
            export_tile(projection, fusion, inputs, expected, args.level, y, x, tile, args.output)


if __name__ == "__main__":
    main()
