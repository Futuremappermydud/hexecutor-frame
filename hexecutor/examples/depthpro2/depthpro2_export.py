# SPDX-License-Identifier: LGPL-3.0-or-later
"""Export Apple Depth Pro 2 (single frame) to a fully delegated hexecutor ExecuTorch .pte.

Run from the repository root with the ExecuTorch venv (plus timm >= 1.0.27):

    hexecutor/.work/executorch-venv/bin/python hexecutor/examples/depthpro2/depthpro2_export.py \
        --src ML_DEPTH_PRO_2_CHECKOUT --weights DepthPro2-S/model.safetensors --size 128 \
        --out hexecutor/.work/depthpro2/dp2s_128.pte [--hxfg-dir DIR] [--check]

The weights (apple/DepthPro2-S, Apple ML Research licence, research use only) are read from a
local path and never written into the repository. Only `network/encoder.py` and
`network/decoder.py` of the upstream checkout are imported.

With one frame the cross-frame blocks are ordinary ViT blocks (the pooled prefix is the frame's
own), so the graph is ViT-S/16 (DINOv3, RoPE, 4 register tokens) plus the DPT decoder and head.
Export-time rewrites, each checked against the upstream modules by `--check`:
- the class and register tokens are a second input [1, 5, 384] (constants of the model), so the
  token concat is between two runtime tensors;
- RoPE is folded into the projections: rotate_half(W x) = (W^T M)^T x, so each head computes
  q = (W_q y) * cos + (W_q M y) * sin with full-size constant cos/sin (1/0 on prefix rows);
- each block runs its 6 heads as single-head SDPAs, concatenates them, and applies one
  output projection;
- cross-frame blocks (5, 7, 9, 11) keep their incoming class and register tokens, as upstream's
  cross_frame_attention does;
- the MLP is split in two along its hidden width, and convolutions over the per-op HMX budget
  (the 3x3 stride-2 384 -> 384 reassemble conv) into output-channel chunks plus a concat.

Inputs: patch tokens [1, N, 768] of raw RGB in [0, 1] ((c, ky, kx) order per 16 x 16 patch, row-major
patches; the patch embedding is a linear with the ImageNet normalisation folded in, since HXFG
convolutions stop at 15 x 15 kernels and stride 8), and prefix [1, 5, 384]. Output: inverse depth
[1, 1, S, S], focal-normalised. Metric depth = f_px / (S * inv), using the focal length
of the model input in pixels. With --arcturus-calibration and --camera, --image is sampled
onto a centered square pinhole and --hxfg-dir also receives its intrinsics in pinhole.json.
"""

import argparse
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from hexecutor.offline.executorch import blob  # noqa: E402
from hexecutor.offline.executorch.export import export_program  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
S_CONFIG = dict(backbone_preset="vit_small_patch16_dinov3", reassembled_blocks=(2, 5, 8, 11),
                cross_frame_blocks=(5, 7, 9, 11), dims_reassemble=(48, 96, 192, 384), dim_decoder=64)


def arcturus_pinhole(image, calibration, side, size):
    """Sample an unrotated Arcturus V3 color capture onto an upright square pinhole."""
    cameras = json.loads(calibration.read_text())["cameras"]
    source = f"passthrough_{side}"
    matches = [c for c in cameras if c["sourceCamera"] == source]
    if len(matches) != 1:
        raise ValueError(f"expected one calibration for {source}")
    camera = matches[0]
    if (camera["hardwareIdentifier"], camera["cameraModel"], camera["preRotateClockwiseByDeg"]) != (
        "Arcturus Camera V3", "kb", 0
    ):
        raise ValueError(f"{source} must use unrotated Arcturus V3 Kannala–Brandt calibration")
    width, height = camera["width"], camera["height"]
    if image.size != (width, height):
        raise ValueError(f"{source} capture must be {width} × {height}")

    intrinsics = camera["intrinsics"]
    focal_px = intrinsics["fx"] * size / width
    center = (size - 1) / 2
    axis = (np.arange(size, dtype=np.float64) - center) / focal_px
    # The right sensor is mounted upside down. The calibration still uses its raw pixel axes.
    if side == "right":
        axis = -axis
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x, y)
    theta = np.arctan(radius)
    t2 = theta * theta
    k1, k2, k3, k4 = (intrinsics[f"k{i}"] for i in range(1, 5))
    distorted = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    scale = np.divide(distorted, radius, out=np.ones_like(radius), where=radius != 0)
    u = intrinsics["cx"] + intrinsics["fx"] * x * scale
    v = intrinsics["cy"] + intrinsics["fy"] * y * scale
    if u.min() < 0 or u.max() >= width - 1 or v.min() < 0 or v.max() >= height - 1:
        raise ValueError(f"pinhole view extends beyond the {source} capture")

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    col, row = u.astype(np.intp), v.astype(np.intp)
    dx, dy = (u - col)[..., None], (v - row)[..., None]
    top = (1 - dx) * rgb[row, col] + dx * rgb[row, col + 1]
    bottom = (1 - dx) * rgb[row + 1, col] + dx * rgb[row + 1, col + 1]
    sampled = (1 - dy) * top + dy * bottom
    pinhole = {"source_camera": source, "focal_px": focal_px, "cx": center, "cy": center,
               "width": size, "height": size}
    return sampled.astype(np.float32) / 255, pinhole


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Reference(nn.Module):
    """Upstream encoder, decoder and head, as DepthPro2.forward runs them for one frame in float32."""

    def __init__(self, src, weights):
        super().__init__()
        net = Path(src) / "src/depth_pro_2/network"
        enc = load_module(net / "encoder.py", "dp2_encoder")
        dec = load_module(net / "decoder.py", "dp2_decoder")
        c = S_CONFIG
        self.encoder = enc.MultiViewEncoder(c["backbone_preset"], c["reassembled_blocks"],
                                            c["cross_frame_blocks"], c["dims_reassemble"])
        self.decoder = dec.DPTDecoder(c["dims_reassemble"], c["dim_decoder"])
        self.head = dec.create_inverse_depth_head(c["dim_decoder"])
        from safetensors.torch import load_file
        sd = load_file(str(weights))
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not missing and all(k.startswith("fov.") for k in unexpected), (missing, unexpected)
        self.eval()

    def forward(self, x):
        features, grid = self.encoder.encode(x)
        return self.head(self.decoder(self.encoder.reassemble_tokens(features, grid)))


class Attention1(nn.Module):
    """One EVA block's attention: fused projections with RoPE folded in, then single-head SDPAs.

    q|k = W_qk y and its rotation W_qk' y are each one linear over all heads (rotate_half(W x) = (W^T M)^T x
    per head), roped once with full-size cos/sin; v is one linear. Each head is a channel slice, the heads
    are concatenated and one output projection (LayerScale folded in) follows.
    """

    def __init__(self, block, cos, sin, gamma=None):
        super().__init__()
        attn = block.attn
        assert getattr(attn, "gate", None) is None and isinstance(attn.q_norm, nn.Identity)
        assert isinstance(attn.k_norm, nn.Identity) and isinstance(attn.norm, nn.Identity)
        heads, dim = attn.num_heads, attn.qkv.in_features
        d = dim // heads
        w = attn.qkv.weight.detach()  # [3 * dim, dim]
        if attn.q_bias is not None:
            bias = torch.cat((attn.q_bias, attn.k_bias, attn.v_bias)).detach()
        elif attn.qkv.bias is not None:
            bias = attn.qkv.bias.detach()
        else:
            bias = torch.zeros(3 * dim)
        eye = torch.eye(d)
        if getattr(attn, "rotate_half", False):
            m = torch.cat([-eye[:, d // 2:], eye[:, : d // 2]], dim=1)  # x @ m = cat(-x2, x1)
        else:
            m = torch.stack([-eye[:, 1::2], eye[:, ::2]], -1).reshape(d, d)
        mb = torch.block_diag(*([m] * (2 * heads)))  # per-head rotation of q|k
        self.heads, self.dim, self.d = heads, dim, d

        def lin(wt, bt):
            layer = nn.Linear(wt.shape[1], wt.shape[0])
            layer.weight.data = wt.clone()
            layer.bias.data = bt.clone()
            return layer

        wqk, bqk = w[: 2 * dim], bias[: 2 * dim]
        self.qk = lin(wqk, bqk)
        self.qkr = lin(mb.T @ wqk, bqk @ mb)
        self.v = lin(w[2 * dim:], bias[2 * dim:])
        pw, pb = attn.proj.weight.detach(), attn.proj.bias.detach()
        if gamma is not None:
            pw, pb = pw * gamma[:, None], pb * gamma
        self.proj = lin(pw, pb)
        self.register_buffer("cos", cos.repeat(1, 1, 2 * heads))
        self.register_buffer("sin", sin.repeat(1, 1, 2 * heads))

    def forward(self, y):
        qk = self.qk(y) * self.cos + self.qkr(y) * self.sin
        v = self.v(y)
        dim, d = self.dim, self.d
        out = []
        for h in range(self.heads):
            q = qk[:, :, h * d:(h + 1) * d]
            k = qk[:, :, dim + h * d:dim + (h + 1) * d]
            vh = v[:, :, h * d:(h + 1) * d]
            out.append(F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), vh.unsqueeze(1)).squeeze(1))
        return self.proj(torch.cat(out, dim=2))


class Mlp1(nn.Module):
    """timm Mlp (fc1, exact GELU, fc2) split along the hidden width so each linear fits the per-op HMX budget."""

    def __init__(self, mlp, parts=2, gamma=None):
        super().__init__()
        assert isinstance(mlp.act, nn.GELU) and mlp.act.approximate == "none"
        assert isinstance(mlp.norm, nn.Identity)
        hidden = mlp.fc1.out_features
        step = hidden // parts
        assert step * parts == hidden
        self.fc1, self.fc2 = nn.ModuleList(), nn.ModuleList()
        for i in range(parts):
            r = slice(i * step, (i + 1) * step)
            a = nn.Linear(mlp.fc1.in_features, step)
            a.weight.data = mlp.fc1.weight.detach()[r].clone()
            a.bias.data = mlp.fc1.bias.detach()[r].clone()
            b = nn.Linear(step, mlp.fc2.out_features, bias=i == 0)
            g = torch.ones(mlp.fc2.out_features) if gamma is None else gamma
            b.weight.data = mlp.fc2.weight.detach()[:, r] * g[:, None]
            if i == 0:
                b.bias.data = mlp.fc2.bias.detach() * g
            self.fc1.append(a)
            self.fc2.append(b)

    def forward(self, x):
        out = None
        for a, b in zip(self.fc1, self.fc2):
            y = b(F.gelu(a(x)))
            out = y if out is None else out + y
        return out


class Block1(nn.Module):
    """EvaBlock with LayerScale folded into the attention and MLP output projections.

    `keep_prefix` (cross-frame blocks) returns the incoming class and register tokens unchanged, as upstream's
    cross_frame_attention does with one frame."""

    def __init__(self, block, cos, sin, npt, keep_prefix):
        super().__init__()
        g1 = getattr(block, "gamma_1", None)
        g2 = getattr(block, "gamma_2", None)
        self.norm1, self.norm2 = block.norm1, block.norm2
        self.attn = Attention1(block, cos, sin, None if g1 is None else g1.detach())
        self.mlp = Mlp1(block.mlp, gamma=None if g2 is None else g2.detach())
        self.npt, self.keep_prefix = npt, keep_prefix

    def forward(self, x):
        y = x + self.attn(self.norm1(x))
        y = y + self.mlp(self.norm2(y))
        if self.keep_prefix:
            y = torch.cat([x[:, :self.npt], y[:, self.npt:]], dim=1)
        return y


def hmx_conv_fits(cin_k, cout):
    from hexecutor.offline import hxfg_writer as hx
    return hx.hmx_fits((cin_k + 31) // 32, (cout + 31) & ~31)


class SplitConv(nn.Module):
    """A dense Conv2d computed as output-channel chunks (each within the HMX budget) and concatenated."""

    def __init__(self, conv, parts):
        super().__init__()
        step = conv.out_channels // parts
        assert step * parts == conv.out_channels and step % 32 == 0
        self.parts = nn.ModuleList()
        for i in range(parts):
            c = nn.Conv2d(conv.in_channels, step, conv.kernel_size, conv.stride, conv.padding,
                          conv.dilation, bias=conv.bias is not None)
            c.weight.data = conv.weight.detach()[i * step:(i + 1) * step].clone()
            if conv.bias is not None:
                c.bias.data = conv.bias.detach()[i * step:(i + 1) * step].clone()
            self.parts.append(c)

    def forward(self, x):
        return torch.cat([c(x) for c in self.parts], dim=1)


def split_big_convs(module):
    for name, child in module.named_children():
        if type(child) is nn.Conv2d and child.groups == 1:
            k = child.in_channels * child.kernel_size[0] * child.kernel_size[1]
            parts = 1
            while not hmx_conv_fits(k, child.out_channels // parts):
                parts *= 2
            if parts > 1:
                setattr(module, name, SplitConv(child, parts))
        else:
            split_big_convs(child)


def fold_fusion_upsample(decoder):
    """FeatureFusionBlock2d's bias-free 2x2 ConvTranspose and the 1x1 out_conv after it are both linear:
    fold them into one ConvTranspose (bias = out_conv's), so the 2x-resolution pair is never live at once."""
    for f in decoder.fusions:
        d, o = f.deconv, f.out_conv
        assert d.bias is None and o.kernel_size == (1, 1)
        w = torch.einsum("imhw,om->iohw", d.weight.detach(), o.weight.detach()[:, :, 0, 0])
        c = nn.ConvTranspose2d(d.in_channels, o.out_channels, d.kernel_size, d.stride, bias=True)
        c.weight.data = w
        c.bias.data = o.bias.detach().clone()
        f.deconv, f.out_conv = c, nn.Identity()


class TailStrip(nn.Module):
    """Rows [r0, r1) (half resolution) of the inverse-depth head after its first conv: ConvTranspose 2x,
    3x3 conv + ReLU, 1x1 conv + ReLU, on the rows plus a one-row halo, cropped to output rows [2 r0, 2 r1)."""

    def __init__(self, head, r0, r1, rows):
        super().__init__()
        self.up, self.conv, self.out = head[1], head[2], head[4]
        self.a, self.b = max(r0 - 1, 0), min(r1 + 1, rows)
        self.crop0, self.crop1 = 2 * (r0 - self.a), 2 * (r0 - self.a) + 2 * (r1 - r0)

    def forward(self, h0):
        x = h0[:, :, self.a:self.b]
        x = F.relu(self.conv(self.up(x)))[:, :, self.crop0:self.crop1]
        return F.relu(self.out(x))


class DepthPro2Frame(nn.Module):
    """One-frame Depth Pro 2 in the form the hexecutor lowering takes."""

    def __init__(self, ref, size, strips=0):
        super().__init__()
        enc = ref.encoder
        p = enc.patch_size
        self.grid = (size // p, size // p)
        self.npt = enc.num_prefix_tokens
        rope = enc.rope.get_embed(shape=self.grid).detach().float()  # [N, 2 d]: sin, cos
        sin, cos = rope.chunk(2, -1)
        d = sin.shape[-1]
        cos = torch.cat([torch.ones(self.npt, d), cos])[None]
        sin = torch.cat([torch.zeros(self.npt, d), sin])[None]
        pe = enc.patch_embed.proj  # Conv2d(3, C, 16, stride 16) on the normalised image
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        w = (pe.weight.detach() / std)  # [C, 3, p, p], acting on raw RGB in [0, 1]
        self.patch_embed = nn.Linear(3 * p * p, pe.out_channels)
        self.patch_embed.weight.data = w.reshape(pe.out_channels, -1).clone()
        self.patch_embed.bias.data = pe.bias.detach() - (w * mean).sum(dim=(1, 2, 3))
        self.patch = p
        self.blocks = nn.ModuleList(Block1(b, cos, sin, self.npt, i in enc.cross_frame_blocks)
                                    for i, b in enumerate(enc.blocks))
        self.taps = enc.reassembled_blocks
        self.cross = enc.cross_frame_blocks
        self.reassemble = enc.reassemble
        self.decoder, self.head = ref.decoder, ref.head
        split_big_convs(self.reassemble)
        split_big_convs(self.decoder)
        split_big_convs(self.head)
        fold_fusion_upsample(self.decoder)
        half = size // 2
        assert strips == 0 or half % strips == 0
        self.tail = nn.ModuleList(TailStrip(self.head, i * half // strips, (i + 1) * half // strips, half)
                                  for i in range(strips))
        self.register_buffer("prefix_tokens", torch.cat([enc.cls_token, enc.reg_token], dim=1).detach())

    def patches(self, rgb):
        """Host-side layout: raw RGB [1, 3, S, S] in [0, 1] -> patch tokens [1, N, 3 * p * p], (c, ky, kx) order."""
        p = self.patch
        h, w = self.grid
        return rgb.reshape(1, 3, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(1, h * w, 3 * p * p)

    def forward(self, patches, prefix):
        h, w = self.grid
        t = self.patch_embed(patches)  # [1, N, C]
        c = t.shape[2]
        t = torch.cat([prefix, t], dim=1)
        feats = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in self.taps:
                f = t[:, self.npt:]
                feats.append(f.transpose(1, 2).reshape(1, c, h, w))
        pyramid = [layer(f) for layer, f in zip(self.reassemble, feats)]
        decoded = self.decoder(pyramid)
        if not len(self.tail):
            return self.head(decoded)
        h0 = self.head[0](decoded)
        return tuple(strip(h0) for strip in self.tail)


def segment_partitioner(model):
    """HexecutorPartitioner whose selection is split into one delegate (one HXFG graph) for the body and one
    per head-tail strip, so each graph's arena fits the VTCM FP16 plan. A node goes to the strip whose
    module it belongs to (nn_module_stack 'tail.<k>'), else to the latest segment among its inputs."""
    from executorch.exir.backend.partitioner import PartitionResult
    from executorch.exir.backend.utils import tag_constant_data
    from hexecutor.offline.executorch.lowering import select_nodes
    from hexecutor.offline.executorch.partitioner import TAG, HexecutorPartitioner

    class SegmentPartitioner(HexecutorPartitioner):
        def partition(self, ep):
            nodes = select_nodes(ep)
            chosen = set(nodes)
            seg = {}
            for n in nodes:
                own = 0
                for path, _ in (n.meta.get("nn_module_stack") or {}).values():
                    parts = path.split(".")
                    if len(parts) >= 2 and parts[0] in ("tail", "L['self'].tail") and parts[1].isdigit():
                        own = int(parts[1]) + 1
                    elif "tail." in path:
                        tail = path.split("tail.")[1].split(".")[0]
                        if tail.isdigit():
                            own = int(tail) + 1
                ins = [seg[a] for a in n.all_input_nodes if a in seg]
                seg[n] = max([own] + ins)
            tags = {}
            for n in nodes:
                tag = f"{TAG}_{seg[n]}"
                n.meta["delegation_tag"] = tag
                tags[tag] = self.delegation_spec
            assert all(n in chosen for n in seg)
            if nodes:
                tag_constant_data(ep)
            return PartitionResult(tagged_exported_program=ep, partition_tags=tags)

    return SegmentPartitioner()


def write_strips(args, model, payloads, x, prefix, want):
    """Per-delegate HXFG graphs for a strip export: body/ (inputs x, p; output h0) and strip<k>/ (input h0).
    Reference files hold the PyTorch tensors in HWC order; want.f32 is the assembled inverse depth."""
    assert len(payloads) == args.strips + 1, f"{len(payloads)} delegates for {args.strips} strips"
    with torch.no_grad():
        h, w = model.grid
        # recompute the body output h0 exactly as forward does
        feats = []
        tok = torch.cat([prefix, model.patch_embed(x)], dim=1)
        for i, blk in enumerate(model.blocks):
            tok = blk(tok)
            if i in model.taps:
                feats.append(tok[:, model.npt:].transpose(1, 2).reshape(1, tok.shape[2], h, w))
        decoded = model.decoder([layer(f) for layer, f in zip(model.reassemble, feats)])
        h0 = model.head[0](decoded)
        strips = [s(h0) for s in model.tail]

    def hwc(t):
        return np.ascontiguousarray(t[0].permute(1, 2, 0).numpy()).tobytes()

    out = args.hxfg_dir
    out.mkdir(parents=True, exist_ok=True)
    by_inputs = {}
    for payload in payloads:
        parsed = blob.unpack(payload)
        by_inputs.setdefault(len(parsed.inputs), []).append(parsed)
    body = by_inputs[2][0]
    parts = [("body", body, [h0])] + [(f"strip{k}", g, [strips[k]]) for k, g in enumerate(by_inputs[1])]
    for name, parsed, refs in parts:
        d = out / name
        d.mkdir(exist_ok=True)
        n_tensors, n_ops = struct.unpack_from("<II", parsed.image, 12)
        print(f"{name}: ops {n_ops}, tensors {n_tensors}, constants {len(parsed.constants) / 2**20:.2f} MiB")
        (d / "g.hxfg").write_bytes(parsed.image)
        (d / "g.bin").write_bytes(parsed.constants)
        for i, r in enumerate(refs):
            (d / f"ref_{i}.f32").write_bytes(hwc(r))
    (out / "x.f32").write_bytes(np.ascontiguousarray(x[0].numpy()).tobytes())
    (out / "p.f32").write_bytes(np.ascontiguousarray(prefix[0].numpy()).tobytes())
    (out / "want.f32").write_bytes(np.ascontiguousarray(want[0, 0].numpy()).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="ml-depth-pro-2 checkout")
    ap.add_argument("--weights", type=Path, required=True, help="DepthPro2-S model.safetensors")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hxfg-dir", type=Path)
    ap.add_argument("--image", type=Path, help="optional RGB image for the example input")
    ap.add_argument("--arcturus-calibration", type=Path, help="sensor-rig-calibration.json for a raw passthrough capture")
    ap.add_argument("--camera", choices=("left", "right"), help="camera for the unrotated RGB capture")
    ap.add_argument("--strips", type=int, default=0,
                    help="split the head tail into this many row strips, each its own delegate/graph "
                         "(needed above ~224 px, where one graph exceeds the 16 MiB HXFG cap); "
                         "writes body/ and strip<k>/ under --hxfg-dir")
    ap.add_argument("--score-scale", type=float, default=1 / 1024,
                    help="power-of-two factor stored on attention scores so FP16 does not overflow "
                         "(block 0 reaches 2e6); the softmax multiplies it back in float. 1 disables")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    from hexecutor.offline.executorch import lowering
    lowering.set_attention_score_scale(args.score_scale)
    assert args.size % 32 == 0
    if (args.arcturus_calibration and not (args.camera and args.image)) or (
        args.camera and not args.arcturus_calibration
    ):
        ap.error("--arcturus-calibration, --camera and --image are required together for a passthrough capture")

    torch.manual_seed(0)
    ref = Reference(args.src, args.weights)
    model = DepthPro2Frame(ref, args.size, args.strips).eval()
    pinhole = None
    if args.image:
        from PIL import Image

        with Image.open(args.image) as img:
            if args.arcturus_calibration:
                pixels, pinhole = arcturus_pinhole(img, args.arcturus_calibration, args.camera, args.size)
                print(f"{pinhole['source_camera']} pinhole focal length: {pinhole['focal_px']:.6f} px")
            else:
                img = img.convert("RGB").resize((args.size, args.size), Image.BILINEAR)
                pixels = np.asarray(img, dtype=np.float32) / 255
        rgb = torch.from_numpy(pixels).permute(2, 0, 1)[None]
    else:
        rgb = torch.rand(1, 3, args.size, args.size)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = model.patches(rgb)
    prefix = model.prefix_tokens.clone()
    with torch.no_grad():
        want = model(x, prefix)
        if args.strips:
            want = torch.cat(want, dim=2)  # assembled rows, for comparison with the stock model
        if args.check:
            stock = ref((rgb - mean) / std)
            rel = ((want - stock).abs().max() / stock.abs().max()).item()
            print(f"rewrite vs upstream: max abs {(want - stock).abs().max().item():.3e} rel {rel:.3e}")

    program = export_program(model, (x, prefix),
                             segment_partitioner(model) if args.strips else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(program.buffer)
    plan = program.executorch_program.execution_plan[0]
    payloads = [bytes(program.executorch_program.backend_delegate_data[d.processed.index].data)
                for d in plan.delegates]
    ops = [op.name for op in plan.operators]
    print(f"wrote {args.out} ({len(program.buffer)} bytes); delegates {len(payloads)}; CPU operators {ops}")
    if args.strips:
        write_strips(args, model, payloads, x, prefix, want)
        return
    parsed = blob.unpack(payloads[0])
    n_tensors, n_ops = struct.unpack_from("<II", parsed.image, 12)
    print(f"HXFG ops {n_ops}, tensors {n_tensors}, constants {len(parsed.constants) / 2**20:.2f} MiB")

    def files(d):
        d.mkdir(parents=True, exist_ok=True)
        (d / "g.hxfg").write_bytes(parsed.image)
        (d / "g.bin").write_bytes(parsed.constants)
        (d / "x.f32").write_bytes(np.ascontiguousarray(x[0].numpy()).tobytes())
        (d / "p.f32").write_bytes(np.ascontiguousarray(prefix[0].numpy()).tobytes())
        (d / "want.f32").write_bytes(np.ascontiguousarray(want[0, 0].numpy()).tobytes())

    if args.hxfg_dir:
        files(args.hxfg_dir)
        if pinhole is not None:
            (args.hxfg_dir / "pinhole.json").write_text(json.dumps(pinhole, indent=2) + "\n")
    runner = ROOT / "build/bin/hexecutor_tensor_run"
    if args.check and runner.exists() and len(payloads) == 1:
        with tempfile.TemporaryDirectory() as root:
            d = Path(root)
            files(d)
            subprocess.run([runner, d / "g.hxfg", d / "g.bin", d / "x.f32", d / "p.f32", d / "o_"],
                           check=True, capture_output=True)
            got = np.fromfile(d / "o_0.f32", np.float32).reshape(args.size, args.size)
        w = want[0, 0].numpy()
        print(f"CPU reference vs PyTorch: max abs {np.abs(got - w).max():.3e} (output max {np.abs(w).max():.3e})")


if __name__ == "__main__":
    main()
