# SPDX-License-Identifier: LGPL-3.0-or-later
"""Export a diffusers UNet2DModel (DDPM-style denoiser) to a fully delegated hexecutor ExecuTorch .pte.

Run from the repository root with the ExecuTorch venv (plus `diffusers==0.35.2`):

    hexecutor/.work/executorch-venv/bin/python hexecutor/examples/diffusion/unet2d_export.py \
        --out hexecutor/.work/executorch/diffusion_unet32.pte [--hxfg-dir DIR] [--check]

The model has random weights (seed 0): the point is the graph, not the samples. One forward is
one denoising step, eps = unet(x_t, t), with inputs x [1, 3, 32, 32] and t as a float [1, 1].

Three export-time rewrites keep every operation on the DSP; each is checked against stock diffusers
(`--check` prints the differences, which are 0):
- the sinusoidal timestep embedding is linear(t -> t * freqs) + sin/cos + cat (no arange or rank-1 math);
- attention applies its group norm to the 4-D tensor before the [1, C, H, W] -> [1, HW, C] view
  (the same arithmetic), and uses one head, so SDPA lowers to two HVX matmuls and a softmax;
- the UNet forward is replayed block by block so the graph takes t directly.

`--hxfg-dir` also writes the delegate's HXFG image, constants and example inputs for
`hexecutor_tensor_run` (inputs are NHWC x then t). Outputs belong under hexecutor/.work/.
"""

import argparse
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.embeddings import get_timestep_embedding

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from hexecutor.offline.executorch import blob  # noqa: E402
from hexecutor.offline.executorch.export import export_program  # noqa: E402


class TokenAttnProcessor:
    """AttnProcessor2_0 for 4-D input and one head, with the group norm applied before the token view."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 *args, **kwargs):
        assert encoder_hidden_states is None and attention_mask is None and attn.heads == 1
        residual = hidden_states
        b, c, h, w = hidden_states.shape
        x = attn.group_norm(hidden_states) if attn.group_norm is not None else hidden_states
        x = x.view(b, c, h * w).transpose(1, 2)
        q, k, v = (f(x).view(b, 1, h * w, c) for f in (attn.to_q, attn.to_k, attn.to_v))
        o = attn.to_out[0](F.scaled_dot_product_attention(q, k, v).view(b, h * w, c))
        o = o.transpose(1, 2).reshape(b, c, h, w)
        if attn.residual_connection:
            o = o + residual
        return o / attn.rescale_output_factor


class TimeProj(torch.nn.Module):
    """diffusers get_timestep_embedding as a bias-free linear layer (weights = the frequencies) and sin/cos."""

    def __init__(self, tp):
        super().__init__()
        half = tp.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32) / (half - tp.downscale_freq_shift)
        self.lin = torch.nn.Linear(1, half, bias=False)
        self.lin.weight.data = torch.exp(exponent)[:, None]
        self.flip = tp.flip_sin_to_cos

    def forward(self, t):  # [1, 1]
        a = self.lin(t)
        s, c = torch.sin(a), torch.cos(a)
        return torch.cat([c, s], dim=1) if self.flip else torch.cat([s, c], dim=1)


def set_processor(unet, processor):
    for m in unet.modules():
        if isinstance(m, Attention):
            m.set_processor(processor)


class Denoiser(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.u = unet
        self.time_proj = TimeProj(unet.time_proj)
        set_processor(unet, TokenAttnProcessor())

    def forward(self, x, t):
        u = self.u
        emb = u.time_embedding(self.time_proj(t))
        s = u.conv_in(x)
        res = (s,)
        for block in u.down_blocks:
            s, r = block(hidden_states=s, temb=emb)
            res += r
        s = u.mid_block(s, emb)
        for block in u.up_blocks:
            r, res = res[-len(block.resnets):], res[:-len(block.resnets)]
            s = block(s, r, emb)
        return u.conv_out(u.conv_act(u.conv_norm_out(s)))


def build_unet(size=32):
    torch.manual_seed(0)
    return UNet2DModel(
        sample_size=size, in_channels=3, out_channels=3, layers_per_block=1,
        block_out_channels=(32, 64, 64), norm_num_groups=8, attention_head_dim=None,  # one head
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).eval()


def check_against_diffusers(unet, model, x, t):
    with torch.no_grad():
        tp = unet.time_proj
        stock_emb = get_timestep_embedding(t.reshape(1), tp.num_channels, tp.flip_sin_to_cos, tp.downscale_freq_shift)
        print("timestep embedding max diff", (model.time_proj(t) - stock_emb).abs().max().item())
        set_processor(unet, AttnProcessor2_0())
        stock = unet(x, t.reshape(1)).sample
        set_processor(unet, TokenAttnProcessor())
        print("denoiser vs stock diffusers max diff", (model(x, t) - stock).abs().max().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hxfg-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="compare with stock diffusers and the CPU reference")
    args = parser.parse_args()

    unet = build_unet()
    model = Denoiser(unet).eval()
    x, t = torch.randn(1, 3, 32, 32), torch.tensor([[500.0]])
    if args.check:
        check_against_diffusers(unet, model, x, t)
    program = export_program(model, (x, t))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(program.buffer)
    plan = program.executorch_program.execution_plan[0]
    payloads = [bytes(program.executorch_program.backend_delegate_data[d.processed.index].data) for d in plan.delegates]
    ops = [op.name for op in plan.operators]
    print(f"wrote {args.out} ({len(program.buffer)} bytes); delegates {len(payloads)}; CPU operators {ops}")
    parsed = blob.unpack(payloads[0])
    n_tensors, n_ops = struct.unpack_from("<II", parsed.image, 12)
    print(f"HXFG ops {n_ops}, tensors {n_tensors}, constants {len(parsed.constants) / 2**20:.2f} MiB")

    def files(d):
        d.mkdir(parents=True, exist_ok=True)
        (d / "g.hxfg").write_bytes(parsed.image)
        (d / "g.bin").write_bytes(parsed.constants)
        (d / "x.f32").write_bytes(np.ascontiguousarray(x[0].numpy().transpose(1, 2, 0)).tobytes())
        (d / "t.f32").write_bytes(t.numpy().tobytes())

    if args.hxfg_dir:
        files(args.hxfg_dir)
    runner = ROOT / "build/bin/hexecutor_tensor_run"
    if args.check and runner.exists():
        with tempfile.TemporaryDirectory() as root:
            d = Path(root)
            files(d)
            subprocess.run([runner, d / "g.hxfg", d / "g.bin", d / "x.f32", d / "t.f32", d / "o_"],
                           check=True, capture_output=True)
            got = np.fromfile(d / "o_0.f32", np.float32).reshape(32, 32, 3).transpose(2, 0, 1)[None]
        with torch.no_grad():
            want = model(x, t).numpy()
        print("CPU reference vs PyTorch max diff", float(np.abs(got - want).max()))


if __name__ == "__main__":
    main()
