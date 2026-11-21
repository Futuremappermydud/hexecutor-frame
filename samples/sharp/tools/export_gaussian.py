#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Export one trained SHARP Gaussian head phase and its matched spatial tiles."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from export_fusion import write_hwc


class GaussianFusion(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.image_encoder = features.image_encoder
        self.fusion = features.fusion

    def forward(self, coarse, image_depth):
        return self.fusion(coarse, self.image_encoder(image_depth).texture_features)


class DeltaHead(torch.nn.Module):
    def __init__(self, features, prediction):
        super().__init__()
        self.geometry = torch.nn.Sequential(features.geometry_head[2:], prediction.geometry_prediction_head)
        self.texture = torch.nn.Sequential(features.texture_head[2:], prediction.texture_prediction_head)

    def forward(self, geometry, texture):
        return torch.cat((self.geometry(geometry), self.texture(texture)), dim=1)


class GaussianTile(torch.nn.Module):
    def __init__(self, phase, offset, size, residual):
        super().__init__()
        self.phase = phase
        self.offset = offset
        self.size = size
        self.residual = residual

    def forward(self, value, other=None):
        y, x = self.offset
        if other is None or self.residual:
            result = self.phase(value)
        else:
            result = self.phase(value, other)
        if self.residual:
            result = result + other
        return result[:, :, y:y + self.size, x:x + self.size].contiguous()


def read_hwc(path, side, channels, device):
    matrix = np.fromfile(path, dtype="<f4").reshape(1, side, side, channels)
    return torch.from_numpy(matrix).permute(0, 3, 1, 2).contiguous().to(device)


def select_phase(model, phase, fused):
    if phase == "delta":
        inputs = []
        for head in (model.feature_model.geometry_head, model.feature_model.texture_head):
            inputs.append(head[1](head[0](fused)))
        module = DeltaHead(model.feature_model, model.prediction_head)
        return module, inputs, module(*inputs), 0, False, None

    branch, block, convolution = phase.split("-")
    head = getattr(model.feature_model, branch + "_head")
    x = fused if block == "0" else head[0](fused)
    body = head[int(block)].residual
    if convolution == "0":
        normalization, module, inputs = body[0], body[1:3], [body[0](x)]
        expected = module(inputs[0])
    else:
        normalization, module, inputs = body[3], body[4:6], [body[:4](x), x]
        expected = module(inputs[0]) + x
    return module, inputs, expected, 1, convolution == "1", normalization


def export_tile(module, inputs, expected, y, x, tile, halo, residual, fusion, output):
    side = expected.shape[-1]
    y0, x0 = max(0, y - halo), max(0, x - halo)
    y1, x1 = min(side, y + tile + halo), min(side, x + tile + halo)
    cropped = []
    for i, value in enumerate(inputs):
        scale = 2 if fusion and i == 1 else 1
        cropped.append(value[:, :, scale*y0:scale*y1, scale*x0:scale*x1].cpu().contiguous())
    phase = GaussianTile(module, (y - y0, x - x0), tile, residual).eval()
    oracle = expected[:, :, y:y + tile, x:x + tile].cpu()
    torch.testing.assert_close(phase(*cropped), oracle, rtol=5e-4, atol=2e-4)
    output.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(phase, tuple(cropped), output / "phase.onnx",
                      input_names=[f"input_{i}" for i in range(len(cropped))],
                      output_names=["features"], opset_version=17, dynamo=False)
    for i, value in enumerate(cropped):
        write_hwc(output / f"input_{i}.f32", value)
    write_hwc(output / "expected.f32", oracle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apple-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True, help="directory of five HWC rest_a inputs")
    parser.add_argument("--image", type=Path, required=True, help="1536x1536x3 HWC RGB [0,1]")
    parser.add_argument("--disparity", type=Path, required=True, help="1536x1536x2 HWC")
    parser.add_argument("--focal-over-width", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    branches = [f"{branch}-{i}-{j}" for branch in ("texture", "geometry") for i in range(2) for j in range(2)]
    parser.add_argument("--phase", choices=["fusion", "delta", *branches], required=True)
    parser.add_argument("--tile-size", choices=(32, 64), type=int, default=32)
    parser.add_argument("--reference-device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if not 0 < args.focal_over_width < float("inf"):
        parser.error("focal-over-width must be finite and positive")
    sys.path.insert(0, str(args.apple_source.resolve() / "src"))
    from sharp.models import create_predictor
    from sharp.models.params import PredictorParams

    torch.set_num_threads(4)
    model = create_predictor(PredictorParams()).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True))
    device = args.reference_device
    with torch.inference_mode():
        features = model.feature_model.to(device)
        model.prediction_head.to(device)
        image = read_hwc(args.image, 1536, 3, device)
        disparity = read_hwc(args.disparity, 1536, 2, device).clamp(1e-4, 1e4)
        depth = args.focal_over_width / disparity
        image_depth = model.init_model(image, depth).feature_input
        encodings = [read_hwc(args.fusion / f"rest_a-{i}.f32", 768 // (2**i), channels, device)
                     for i, channels in enumerate((256, 256, 512, 1024, 1024))]
        coarse = features.upsample(features.decoder(encodings))
        fusion = GaussianFusion(features)
        fused = fusion(coarse, image_depth)
        is_fusion = args.phase == "fusion"
        if is_fusion:
            phase, inputs, expected, halo, residual, normalization = fusion, [coarse, image_depth], fused, 4, False, None
        else:
            phase, inputs, expected, halo, residual, normalization = select_phase(model, args.phase, fused)
        phase.cpu()
        args.output.mkdir(parents=True, exist_ok=True)
        if normalization is not None:
            torch.cat((normalization.weight, normalization.bias)).cpu().numpy().astype("<f4").tofile(
                args.output / "affine.f32")
        tile = args.tile_size
        positions = (0, tile, expected.shape[-1] - tile)
        for row, y in enumerate(positions):
            for col, x in enumerate(positions):
                export_tile(phase, inputs, expected, y, x, tile, halo, residual, is_fusion, args.output / f"r{row}c{col}")


if __name__ == "__main__":
    main()
