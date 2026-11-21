#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Export trained SHARP token embeddings and individual ViT blocks offline."""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
import timm
import torch
from torch.nn import functional as F


class TokenEmbedding(torch.nn.Module):
    """Project packed RGB patches, then add the learned position and class tokens."""

    def __init__(self, encoder):
        super().__init__()
        conv = encoder.patch_embed.proj
        positions = encoder._pos_embed(torch.zeros(1, 576, 1024))
        self.register_buffer("weight", conv.weight.detach().flatten(1))
        self.register_buffer(
            "offset", torch.cat((positions[:, :1], positions[:, 1:] + conv.bias.detach()), dim=1)
        )

    def forward(self, pixels):
        return F.linear(pixels, self.weight) + self.offset


def write_float(path, tensor):
    np.asarray(tensor.detach().cpu().numpy(), dtype="<f4").tofile(path)


def export_onnx(module, value, path, input_name, output_name):
    torch.onnx.export(
        module, (value,), path, input_names=[input_name], output_names=[output_name],
        opset_version=17, dynamo=False,
    )


def export_encoder(encoder, images, destination, blocks):
    destination.mkdir(parents=True, exist_ok=True)
    embed = TokenEmbedding(encoder).eval()
    pixels = F.unfold(images, kernel_size=16, stride=16).transpose(1, 2).contiguous()
    pixels = F.pad(pixels, (0, 0, 1, 0))
    tokens = encoder.norm_pre(encoder.patch_drop(encoder._pos_embed(encoder.patch_embed(images))))
    torch.testing.assert_close(embed(pixels), tokens, rtol=1e-5, atol=4e-6)
    write_float(destination / "embed_input.f32", pixels[:1])
    write_float(destination / "embed_expected.f32", tokens[:1])
    export_onnx(embed, pixels[:1], destination / "embed.onnx", "pixels", "tokens")
    export_onnx(encoder.norm, tokens[:1], destination / "norm.onnx", "tokens", "normalized_tokens")

    for index in range(blocks):
        name = f"block_{index:02d}"
        block = encoder.blocks[index]
        expected = block(tokens)
        export_onnx(block, tokens[:1], destination / f"{name}.onnx", "tokens", "next_tokens")
        write_float(destination / f"{name}_expected.f32", expected[:1])
        calibration = []
        for sample in range(tokens.shape[0]):
            path = destination / f"{name}_input_{sample:02d}.f32"
            write_float(path, tokens[sample:sample + 1])
            calibration.append(str(path))
        (destination / f"{name}_calibration.txt").write_text("\n".join(calibration) + "\n")
        tokens = expected
        print(f"{destination.name} {name}", flush=True)
    if blocks == len(encoder.blocks):
        write_float(destination / "norm_input.f32", tokens[:1])
        write_float(destination / "norm_expected.f32", encoder.norm(tokens[:1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apple-source", type=Path, required=True, help="local apple/ml-sharp checkout")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="calibration RGB image")
    parser.add_argument("--output", type=Path, required=True, help="untracked artifact directory")
    parser.add_argument("--blocks", type=int, choices=range(25), default=24,
                        metavar="0..24", help="blocks to export per encoder")
    args = parser.parse_args()
    sys.path.insert(0, str(args.apple_source.resolve() / "src"))
    from sharp.models import create_predictor
    from sharp.models.encoders.spn_encoder import split
    from sharp.models.params import PredictorParams

    torch.set_num_threads(4)
    timm.layers.set_fused_attn(False)
    predictor = create_predictor(PredictorParams()).eval()
    predictor.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True))
    backbone = predictor.monodepth_model.monodepth_predictor
    encoder = backbone.encoder
    with Image.open(args.image) as source:
        rgb = np.array(ImageOps.exif_transpose(source).convert("RGB"))
    image = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255
    with torch.inference_mode():
        image = F.interpolate(image, size=(1536, 1536), mode="bilinear", align_corners=True)
        high, middle, low = encoder._create_pyramid(backbone.normalizer(image))
        patches = torch.cat((split(high, overlap_ratio=0.25), split(middle, overlap_ratio=0.5), low))
        # Sample both corners of the two overlapping grids and the entire low-resolution view.
        export_encoder(encoder.patch_encoder, patches[[0, 24, 25, 33, 34]], args.output / "pe", args.blocks)
        export_encoder(encoder.image_encoder, low, args.output / "ie", args.blocks)


if __name__ == "__main__":
    main()
