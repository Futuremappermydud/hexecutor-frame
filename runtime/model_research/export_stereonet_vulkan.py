#!/usr/bin/env python3
"""Export Qualcomm's trained StereoNet weights for the Frame Vulkan backend.

The Qualcomm ONNX release replaces each Conv3d with a very large Conv2d
"unfold" graph to satisfy QNN.  That representation consumes several GB when
prepared by a general Vulkan runtime.  MNN supports Conv3d, so this export keeps
the native layers and only substitutes the cost-volume construction with one
ONNX-friendly GatherElements operation.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from stereonet.model import StereoNet as StereoNetModel
from stereonet.model import soft_argmin
from torch import nn


def compute_volume(
    reference: torch.Tensor,
    target: torch.Tensor,
    disparities: int,
) -> torch.Tensor:
    """Return a left-reference cost volume in B,C,D,H,W order."""
    batch, channels, height, width = reference.shape
    reference_expanded = (
        reference.reshape(batch * channels, height, width)
        .unsqueeze(-1)
        .expand(-1, -1, -1, disparities)
    )
    disparity = torch.arange(disparities, device=reference.device)
    x = torch.arange(width, device=reference.device).unsqueeze(1)
    shifted_x = torch.clamp(x - disparity, min=0)
    valid = x >= disparity

    target_expanded = (
        target.reshape(batch * channels, height, width)
        .unsqueeze(-1)
        .expand(-1, -1, -1, disparities)
    )
    indices = shifted_x.view(1, 1, width, disparities).expand(
        batch * channels, height, width, disparities
    )
    shifted = torch.gather(target_expanded, dim=2, index=indices)
    cost = (reference_expanded - shifted) * valid.view(1, 1, width, disparities)
    return cost.reshape(batch, channels, height, width, disparities).permute(
        0, 1, 4, 2, 3
    )


class VulkanCostVolume(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.disparities = source._max_downsampled_disps
        self.net = source.net

    def forward(self, pair: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        reference, target = pair
        cost = compute_volume(reference, target, self.disparities)
        return self.net(cost).squeeze(1)


class StereoNetVulkan(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        source = StereoNetModel(in_channels=1)
        self.feature_extractor = source.feature_extractor
        self.cost_volumizer: nn.Module = source.cost_volumizer
        self.refiners = source.refiners
        self.refinement_layers = source.k_refinement_layers
        self.disparities = source.candidate_disparities
        self.register_buffer(
            "mean", torch.tensor([111.5684, 113.6528]).view(1, 2, 1, 1) / 255.0
        )
        self.register_buffer(
            "std", torch.tensor([61.9625, 62.0313]).view(1, 2, 1, 1) / 255.0
        )

    def enable_vulkan_cost_volume(self) -> None:
        self.cost_volumizer = VulkanCostVolume(self.cost_volumizer)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        reference = image[:, :1]
        target = image[:, 1:2]
        reference_features = self.feature_extractor(reference)
        target_features = self.feature_extractor(target)
        cost = self.cost_volumizer((reference_features, target_features))
        disparity = soft_argmin(cost, self.disparities)

        for index, refiner in enumerate(self.refiners, start=1):
            scale = (2**self.refinement_layers) / (2**index)
            size = [int(reference.shape[2] // scale), int(reference.shape[3] // scale)]
            disparity_resized = F.interpolate(
                disparity, size, mode="bilinear", align_corners=True
            )
            reference_resized = F.interpolate(
                reference, size, mode="bilinear", align_corners=True
            )
            disparity = F.relu(
                refiner(torch.cat((reference_resized, disparity_resized), dim=1))
                + disparity_resized
            )
        return disparity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    args = parser.parse_args()

    model = StereoNetVulkan()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"mean", "std"} or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.enable_vulkan_cost_volume()
    model.eval()

    example = torch.zeros(1, 2, args.height, args.width, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            args.output,
            input_names=["image"],
            output_names=["disparity"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"Exported {args.height}x{args.width} StereoNet to {args.output}")


if __name__ == "__main__":
    main()
