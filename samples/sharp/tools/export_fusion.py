#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Export SHARP rest_a as independently tiled feature-upsample graphs."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

class LowResolutionFusion(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.patch = encoder.upsample2
        self.image = encoder.upsample_lowres
        self.fuse = encoder.fuse_lowres

    def forward(self, patch, image):
        return self.fuse(torch.cat((self.patch(patch), self.image(image)), dim=1))


def load_tile(path, side, tile):
    feature = np.memmap(path, mode="r", dtype="<f4", shape=(1, 1024, side, side))
    return torch.from_numpy(np.ascontiguousarray(feature[:, :, :tile, :tile]))


def write_hwc(path, feature):
    feature.detach().cpu().numpy().transpose(0, 2, 3, 1).astype("<f4").tofile(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apple-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True, help="offline captured SHARP stage boundaries")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.apple_source.resolve() / "src"))
    from sharp.models import create_predictor
    from sharp.models.params import PredictorParams

    torch.set_num_threads(4)
    predictor = create_predictor(PredictorParams()).eval()
    predictor.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True))
    encoder = predictor.monodepth_model.monodepth_predictor.encoder
    args.output.mkdir(parents=True, exist_ok=True)
    parts = (encoder.upsample_latent0, encoder.upsample_latent1, encoder.upsample0,
             encoder.upsample1, LowResolutionFusion(encoder).eval())
    sides = (96, 96, 96, 48, 24)
    tile = 8
    with torch.inference_mode():
        for i, (stage, side) in enumerate(zip(parts, sides)):
            inputs = (load_tile(args.captures / f"enc-{i}.f32", side, tile),)
            if i == 4:
                inputs += (load_tile(args.captures / "enc-5.f32", side, tile),)
            output = stage(*inputs)
            _, channels, h, w = output.shape
            oracle_side = h * side // tile
            oracle = np.memmap(args.captures / f"rest_a-{i}.f32", mode="r", dtype="<f4",
                               shape=(1, channels, oracle_side, oracle_side))
            reference = np.ascontiguousarray(oracle[:, :, :h, :w])
            np.testing.assert_allclose(output.detach().numpy(), reference, rtol=5e-4, atol=1e-4)
            stem = f"fusion_{i}"
            names = [f"input_{j}" for j in range(len(inputs))]
            torch.onnx.export(stage, inputs, args.output / f"{stem}.onnx",
                              input_names=names, output_names=["features"], opset_version=17, dynamo=False)
            for j, tensor in enumerate(inputs):
                write_hwc(args.output / f"{stem}_input_{j}.f32", tensor)
            write_hwc(args.output / f"{stem}_expected.f32", torch.from_numpy(reference))
            print(stem, tuple(output.shape), flush=True)


if __name__ == "__main__":
    main()
