"""Export LFM3-VL vision encoder and multimodal projector to ONNX float.

Static shape, batch 1, fixed tile size 32x32 (smallest supported tile:
patch_size=16, spatial_merge_size=2 -> 2x2 = 4 patches/tokens).
Supports exporting individual subgraphs (patchifier, mlp, attention, conv, projector)
and the full vision tower + projector, applying module surgery for HXFG-supported ops.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Symbolic RMSNorm mapping to HXFG HX_RMSNORM (kind 22)
class RMSNormSymbolic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * weight

    @staticmethod
    def symbolic(g, x, weight, eps):
        return g.op("RMSNormalization", x, weight, epsilon_f=eps)


class WrappedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x):
        return RMSNormSymbolic.apply(x, self.weight, self.eps)


class PatchifierSubgraph(nn.Module):
    """Linear patch projection from raw patch pixels to hidden dimension followed by RMSNorm."""
    def __init__(self, in_channels=3, temporal_patch_size=2, patch_size=16, hidden_size=1024, eps=1e-5):
        super().__init__()
        patch_dim = in_channels * temporal_patch_size * patch_size * patch_size  # 1536
        self.patchifier = nn.Linear(patch_dim, hidden_size, bias=False)
        self.norm = WrappedRMSNorm(hidden_size, eps=eps)

    def forward(self, pixel_values):
        # pixel_values: [1, L, 1536] (L=4 for 32x32 image)
        return self.norm(self.patchifier(pixel_values))


class MLPSubgraph(nn.Module):
    """LFM3-VL Feed-Forward Network: SiLU-gated linear projections with RMSNorm and residual add."""
    def __init__(self, hidden_size=1024, intermediate_size=2560, eps=1e-5):
        super().__init__()
        self.norm = WrappedRMSNorm(hidden_size, eps=eps)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # x: [1, L, hidden_size]
        h = self.norm(x)
        # SwiGLU / SiLU gate
        gate = F.silu(self.w1(h)) * self.w3(h)
        return x + self.w2(gate)


class AttentionSubgraph(nn.Module):
    """Multi-head attention with GQA surgery (unrolled 8 KV heads -> 16 heads) matching HXFG heads chain."""
    def __init__(self, hidden_size=1024, num_heads=16, head_dim=64, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / np.sqrt(head_dim)
        self.norm = WrappedRMSNorm(hidden_size, eps=eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        # x: [1, L, hidden_size] (static L=4, hidden_size=1024)
        h = self.norm(x)
        q = self.q_proj(h).reshape(1, 4, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).reshape(1, 4, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).reshape(1, 4, self.num_heads, self.head_dim).transpose(1, 2)

        # Per-head attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)
        merged = ctx.transpose(1, 2).reshape(1, 4, self.hidden_size)
        return x + self.out_proj(merged)



class ConvSubgraph(nn.Module):
    """Native LFM3-VL gated depthwise conv layer over patches."""
    def __init__(self, hidden_size=1024, kernel_size=3, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = WrappedRMSNorm(hidden_size, eps=eps)

        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.spatial_conv = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2,
            groups=hidden_size, bias=False
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        # x: [1, 4, 1024]
        h = self.norm(x)
        proj = self.in_proj(h)
        B, C, v = torch.split(proj, [self.hidden_size, self.hidden_size, self.hidden_size], dim=-1)
        gated = B * v
        # 2x2 patch grid to raster NCHW [1, 1024, 2, 2]
        frames = gated.view(1, 1, 1, 2, 2, self.hidden_size).permute(0, 5, 1, 3, 2, 4).reshape(1, self.hidden_size, 2, 2)
        conv = self.spatial_conv(frames)
        conv_hw = conv.view(1, self.hidden_size, 1, 2, 1, 2).permute(0, 2, 4, 3, 5, 1).reshape(1, 4, self.hidden_size)
        return x + self.out_proj(C * conv_hw)



class ProjectorSubgraph(nn.Module):
    """Multimodal projector: LayerNorm, pixel-unshuffle reshape (4 tokens -> 1 token), MLP."""
    def __init__(self, hidden_size=1024, downsample_factor=2, proj_hidden=4096, text_dim=2048, eps=1e-6):
        super().__init__()
        self.in_features = hidden_size * (downsample_factor ** 2)  # 4096
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.linear_1 = nn.Linear(self.in_features, proj_hidden, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(proj_hidden, text_dim, bias=True)

    def forward(self, x):
        # x: [1, 4, 1024]
        norm = self.layer_norm(x)
        # Pixel unshuffle downsample: 4 tokens of 1024 -> 1 token of 4096
        reshaped = norm.reshape(1, 1, self.in_features)  # [1, 1, 4096]
        h = self.linear_1(reshaped)
        h = self.act(h)
        return self.linear_2(h)  # [1, 1, 2048]



class VisionTowerAndProjector(nn.Module):
    """Full LFM3-VL Vision Tower (14 layers) + Multimodal Projector."""
    def __init__(self, hidden_size=1024, num_layers=14):
        super().__init__()
        self.patchifier = PatchifierSubgraph(hidden_size=hidden_size)
        # Layer schedule from config.json: 8 conv, 6 attention
        # layers 0, 1, 3, 5, 7, 9, 11, 13 are conv
        # layers 2, 4, 6, 8, 10, 12 are full_attention
        layers = []
        for i in range(num_layers):
            if i in (2, 4, 6, 8, 10, 12):
                layers.append(AttentionSubgraph(hidden_size=hidden_size))
            else:
                layers.append(ConvSubgraph(hidden_size=hidden_size))
        self.layers = nn.ModuleList(layers)
        self.projector = ProjectorSubgraph(hidden_size=hidden_size)

    def forward(self, pixel_values):
        h = self.patchifier(pixel_values)
        for layer in self.layers:
            h = layer(h)
        return self.projector(h)


COMPONENTS = {
    "patchifier": (lambda: PatchifierSubgraph(), (1, 4, 1536)),
    "mlp": (lambda: MLPSubgraph(), (1, 4, 1024)),
    "attention": (lambda: AttentionSubgraph(), (1, 4, 1024)),
    "conv": (lambda: ConvSubgraph(), (1, 4, 1024)),
    "projector": (lambda: ProjectorSubgraph(), (1, 4, 1024)),
    "tower": (lambda: VisionTowerAndProjector(), (1, 4, 1536)),
}


def export_component(name, out_path, model=None, opset=17):
    if name not in COMPONENTS:
        raise ValueError(f"Unknown component: {name}. Available: {list(COMPONENTS.keys())}")
    builder, input_shape = COMPONENTS[name]
    if model is None:
        model = builder()
    model.eval()
    dummy_input = torch.randn(*input_shape)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported {name} to {out_path} (input shape: {input_shape})")
    return out_path, model



def main():
    parser = argparse.ArgumentParser(description="Export LFM3-VL vision components to ONNX float.")
    parser.add_argument("--component", choices=list(COMPONENTS.keys()) + ["all"], default="all",
                        help="Component to export (default: all)")
    parser.add_argument("--out-dir", default="hexecutor/examples/llm/lfm/vision/models",
                        help="Output directory for exported ONNX models")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = list(COMPONENTS.keys()) if args.component == "all" else [args.component]
    for comp in targets:
        out_path = out_dir / f"{comp}.onnx"
        export_component(comp, str(out_path), opset=args.opset)


if __name__ == "__main__":
    main()
