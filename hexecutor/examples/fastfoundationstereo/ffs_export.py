# SPDX-License-Identifier: LGPL-3.0-or-later
"""Export NVIDIA C-Fast-FoundationStereo to a fully delegated hexecutor ExecuTorch .pte.

Run from the repository root with the ExecuTorch venv (the upstream code also imports timm, einops,
omegaconf, imageio, scipy and OpenCV; install them with `uv pip install --python ... --target DIR` and
put DIR on PYTHONPATH rather than into the pinned venv):

    hexecutor/.work/executorch-venv/bin/python hexecutor/examples/fastfoundationstereo/ffs_export.py \
        --src FAST_FOUNDATIONSTEREO_CHECKOUT --weights c-fast-foundationstereo/model_best_bp2_serialize.pth \
        --height 128 --width 192 --iters 4 --max-disp 192 \
        --out hexecutor/.work/ffs/ffs_128x192.pte [--hxfg-dir DIR] [--check]

Sources: code https://github.com/NVlabs/Fast-FoundationStereo (Apache-2.0); weights
https://huggingface.co/nvidia/c-fast-foundationstereo (NVIDIA Open Model Agreement; the research-only
checkpoints in the upstream Google Drive folder are not used). Keep the weights and every derived .pte or
HXFG image under the ignored `hexecutor/.work/`; they are never written into the repository.

The model is the "single ONNX" configuration of upstream `scripts/make_single_onnx.py` (ImageNet
normalisation stripped, so the two inputs are pre-normalised `[1, 3, H, W]` images). Everything the
delegate cannot take is rewritten here, and `--check` compares each rewrite with the unmodified upstream
model:

* the feature extractor runs on the left and the right image separately (HXFG tensors are batch one);
* the EdgeNeXt cross-covariance attention works per head on `[1, N, d]` token slices (the upstream
  `[3, B, heads, d, N]` permute is rank five);
* the cost-volume disparity attention keeps upstream's quirk of letting `scaled_dot_product_attention` read
  `[B, L, heads, d]` as `[batch B, heads L, sequence = heads, d]`, written as a bmm over `[B*L, heads, d]`;
* the convex-upsampling `unsqueeze(1)` and the per-pixel `grid_sample` lookups of the iterative update
  (`Combined_Geo_Encoding_Volume`, a batch of H*W one-row images) become gathers along the depth rows of
  `[1, C, D, H*W]` and along the channels of `[1, H*W, W2]` token rows (`GeoEnc4`), and `convc1` is split to
  read the sampled tensors directly (`SplitConv`).

Rank-5 cost volumes and their 3D convolutions, channels-last blocks, oversized convolutions, vector norms,
`im2col` and global max pools need no model change: `hexecutor/offline/executorch/volume.py` and
`decompositions.py` rewrite them.
"""

import argparse
import contextlib
import inspect
import os
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F


class _NoAutocast(contextlib.ContextDecorator):
    """The upstream forward wraps itself in CUDA autocast; export traces plain float32."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def load_upstream(src):
    """Import the upstream modules from a checkout; returns (foundation_stereo, make_single_onnx, edgenext)."""
    sys.path[:0] = [str(Path(src) / "scripts"), str(Path(src))]
    torch.autocast = torch.cuda.amp.autocast = torch.amp.autocast = _NoAutocast
    import core.foundation_stereo as fs
    import make_single_onnx as onnx_export
    import timm.models.edgenext as edgenext
    return fs, onnx_export, edgenext


# ---------------------------------------------------------------------------- model rewrites


def xca_forward(self, x):
    """CrossCovarianceAttn per head with batch-one rank-3/4 tensors: the channels of head h are a slice of the
    qkv token matrix, q and k are normalised over the tokens, and the d x d attention is a bmm."""
    B, N, C = x.shape
    H = self.num_heads
    d = C // H
    qkv = self.qkv(x)                                     # [1, N, 3C]
    heads = []
    for h in range(H):
        def tok(i):                                       # [1, N, d] token slice of the qkv matrix
            return qkv[..., i * C + h * d:i * C + (h + 1) * d]

        def unit(t):                                      # [1, N, d] -> L2-normalised over tokens, [1, N, d]
            m = t.transpose(1, 2).reshape(1, d, 1, N)     # token pair -> map (zero copy)
            m = m / torch.sqrt((m * m).sum(dim=3, keepdim=True)).clamp_min(1e-12)
            return m.reshape(1, d, N).transpose(1, 2)     # map -> token pair (zero copy)
        qn, kn = unit(tok(0)), unit(tok(1))
        attn = torch.bmm(qn.transpose(1, 2), kn) * self._temps[h]      # [1, d, d]
        attn = attn.softmax(dim=-1)
        heads.append(torch.bmm(attn, tok(2).transpose(1, 2)))          # [1, d, N]
    return self.proj(torch.cat(heads, dim=1).transpose(1, 2))          # [1, N, C]


def fmha_forward(self, query, key, value, attn_mask=None, window_size=(-1, -1)):
    """FlashMultiheadAttention: B*L independent attentions of `num_heads` rows (see the module docstring)."""
    B, L, C = query.shape
    nh, hd = self.num_heads, self.head_dim
    Q = self.q_proj(query).reshape(B * L, nh, hd)
    K = self.k_proj(key).reshape(B * L, nh, hd)
    V = self.v_proj(value).reshape(B * L, nh, hd)
    probs = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * (hd ** -0.5), dim=-1)
    return self.out_proj(torch.bmm(probs, V).reshape(B, L, C))


def context_upsample(disp_low, up_weights):
    b, c, h, w = disp_low.shape
    disp_unfold = F.unfold(disp_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
    disp_unfold = F.interpolate(disp_unfold, (h * 4, w * 4), mode="nearest").reshape(b, 9, h * 4, w * 4)
    return (disp_unfold * up_weights).sum(1, keepdim=True)


def hat_weights(pos, size):
    """Linear interpolation of a zero-padded line of `size` samples at fractional position `pos`:
    (floor index, ceil index, weights), the weights already zero for out-of-range corners."""
    f = torch.floor(pos)
    frac = pos - f
    va = (f + 1).clamp(0, 1) * (size - f).clamp(0, 1)
    vb = (f + 2).clamp(0, 1) * (size - 1 - f).clamp(0, 1)
    return f.clamp(0, size - 1), (f + 1).clamp(0, size - 1), (1 - frac) * va, frac * vb


class GeoEnc4:
    """Combined_Geo_Encoding_Volume with batch-one rank-4 / token tensors (see the module docstring)."""

    def __init__(self, fmap1, fmap2, geo_volume, num_levels=2):
        self.num_levels = num_levels
        b, cf, h, w1 = fmap1.shape
        c, d = geo_volume.shape[1], geo_volume.shape[2]
        self.h, self.w = h, w1
        n1, n2 = F.normalize(fmap1, dim=1), F.normalize(fmap2, dim=1)
        self.corr_pyr, self.geo_pyr = [], []
        for lvl in range(num_levels):
            rows = []
            for r in range(h):  # one [W1, W2] correlation per image row
                a = n1[:, :, r:r + 1, :].reshape(1, cf, w1).transpose(1, 2)               # [1, W1, Cf]
                bt = n2[:, :, r:r + 1, :].reshape(1, cf, n2.shape[3]).transpose(1, 2)     # [1, W2, Cf]
                rows.append(torch.bmm(a, bt.transpose(1, 2)))
            self.corr_pyr.append(torch.cat(rows, dim=1))                                  # [1, H*W1, W2/2^lvl]
            if lvl + 1 < num_levels:
                n2 = F.avg_pool2d(n2, [1, 2], stride=[1, 2])
        g = geo_volume.reshape(b, c, d, h * w1)
        for lvl in range(num_levels):
            self.geo_pyr.append(g)
            if lvl + 1 < num_levels:
                g = F.avg_pool2d(g, [2, 1], stride=[2, 1])

    def __call__(self, disp, coords, dx, low_memory=True):
        p = self.h * self.w
        k = dx.shape[2]
        dxf = dx.to(torch.float32)
        disp_tok = disp.reshape(1, 1, p).transpose(1, 2)                                  # [1, P, 1]
        coord_tok = coords.reshape(1, p, 1)
        outs = []
        for lvl in range(self.num_levels):
            scale = 2.0 ** lvl
            g = self.geo_pyr[lvl]                                                         # [1, C, Dl, P]
            c, dl = g.shape[1], g.shape[2]
            pos = disp.reshape(1, 1, 1, p).expand(1, 1, k, p) / scale + dxf.reshape(1, 1, k, 1)
            ia, ib, wa, wb = hat_weights(pos, dl)
            ga = torch.gather(g, 2, ia.expand(1, c, k, p).to(torch.int64))
            gb = torch.gather(g, 2, ib.expand(1, c, k, p).to(torch.int64))
            outs.append(ga * wa + gb * wb)                                                # [1, C, K, P]
            tab = self.corr_pyr[lvl]                                                      # [1, P, W2l]
            cpos = ((coord_tok - disp_tok) / scale).expand(1, p, k) + dxf.reshape(1, 1, k)
            ia, ib, wa, wb = hat_weights(cpos, tab.shape[2])
            ca = torch.gather(tab, 2, ia.to(torch.int64))
            cb = torch.gather(tab, 2, ib.to(torch.int64))
            outs.append((ca * wa + cb * wb).transpose(1, 2).reshape(1, k, 1, p))         # [1, K, 1, P]
        return outs


class SplitConv(nn.Module):
    """The 1x1 `convc1` over cat([geo0 (C*K), corr0 (K), geo1 (C*K), corr1 (K)]) as four convolutions that read
    the sampled tensors directly: [1, C, K, P] with a (K, 1) kernel, [1, K, 1, P] with 1x1."""

    def __init__(self, conv, c, k, h, w):
        super().__init__()
        wt, bias = conv.weight.detach(), conv.bias.detach()
        o = wt.shape[0]
        self.h, self.w = h, w
        parts, off = [], 0
        for n, kind in ((c * k, "geo"), (k, "corr"), (c * k, "geo"), (k, "corr")):
            sl = wt[:, off:off + n, 0, 0]
            off += n
            part = nn.Conv2d(c if kind == "geo" else k, o, (k, 1) if kind == "geo" else 1, bias=False)
            part.weight.data = (sl.reshape(o, c, k, 1) if kind == "geo" else sl.reshape(o, k, 1, 1)).clone()
            parts.append(part)
        self.parts = nn.ModuleList(parts)
        self.bias = nn.Parameter(bias.clone())

    def forward(self, outs):
        acc = None
        for conv, t in zip(self.parts, outs):
            y = conv(t)                                                                   # [1, O, 1, P]
            acc = y if acc is None else acc + y
        return acc.reshape(1, acc.shape[1], self.h, self.w) + self.bias.reshape(1, -1, 1, 1)


def encoder_forward(self, disp, corr):
    cor = F.relu(self.convc1(corr))
    cor = F.relu(self.convc2(cor))
    disp_ = F.relu(self.convd1(disp))
    disp_ = F.relu(self.convd2(disp_))
    out = F.relu(self.conv(torch.cat([cor, disp_], dim=1)))
    return torch.cat([out, disp], dim=1)


def _recompile(fs, method, replacements):
    src = textwrap.dedent(inspect.getsource(getattr(fs.FastFoundationStereo, method)))
    for old, new in replacements:
        if callable(old):
            src = old(src)
            continue
        assert old in src, f"upstream {method} changed: {old!r}"
        src = src.replace(old, new)
    local = {}
    exec(src, fs.__dict__, local)
    setattr(fs.FastFoundationStereo, method, local[method])


def load_model(fs, onnx_export, weights, iters, max_disp):
    """The checkpoint as the single-ONNX wrapper (pre-normalised inputs, ONNX-friendly cost volumes)."""
    from omegaconf import open_dict
    model = torch.load(weights, map_location="cpu", weights_only=False)
    with open_dict(model.args):
        model.args.normalize = True       # the C-Fast config omits it; the shipped ONNX normalises
    model.args.max_disp, model.args.valid_iters, model.args.mixed_precision = max_disp, iters, False
    fs.normalize_image = lambda image: image
    fs.build_gwc_volume_optimized_pytorch1 = onnx_export._build_gwc_volume_onnx
    fs.build_concat_volume_optimized_pytorch1 = onnx_export._build_concat_volume_onnx
    return onnx_export.FastFoundationStereoSingleOnnx(model.eval()).eval()


class VolumeStem(nn.Module):
    """`build_gwc_volume`, `build_concat_volume` and their channel concatenation, one disparity at a time, so the
    `[1, 8, Cg, D, H, W]` products and the `[1, C, D, H, W]` shifted-target volume (110 MiB each at 224 x 288,
    `max_disp` 128) never exist.

    Per disparity `d` the group-wise cosine similarity is
    `sum_c(ref * shift_d(tgt)) / (|ref_g| * |shift_d(tgt)_g|)`, the product of the two normalised volumes upstream
    builds (a zero column has norm zero and a zero numerator, so the zero-filled shift is exact). It is computed
    on the `W - d` columns where the shifted target exists, from a 1x1 convolution of ones that sums each group's
    channels, and zero-padded back. The slice `[gwc_d, left, shift_d(right)]` has 8 + 12 + 12 channels; the D
    slices (`[1, 32, 1, H * W]`) are joined by a balanced tree of two-input concatenations along the depth rows,
    which copies the volume log2(D) times instead of D times.
    """

    def __init__(self, model, max_disp):
        super().__init__()
        self.depth = max_disp // 4
        self.groups = model.cv_group
        self.proj = model.proj_cmb

    @staticmethod
    def _gsum(c, groups, like):
        m = torch.zeros(groups, c, 1, 1)
        cg = c // groups
        for g in range(groups):
            m[g, g * cg:(g + 1) * cg] = 1.0
        return m.to(like.dtype)

    def forward(self, ref, tgt):
        b, c, h, w = ref.shape
        gs = self._gsum(c, self.groups, ref)
        nr = torch.sqrt(F.conv2d(ref * ref, gs)).clamp_min(1e-12)                # [1, G, H, W]
        nt = torch.sqrt(F.conv2d(tgt * tgt, gs)).clamp_min(1e-12)
        left, right = self.proj(ref), self.proj(tgt)
        rows = []
        for d in range(self.depth):
            dot = F.conv2d(ref[:, :, :, d:] * tgt[:, :, :, :w - d], gs)          # [1, G, H, W - d]
            sim = dot / (nr[:, :, :, d:] * nt[:, :, :, :w - d])
            rs = right[:, :, :, :w - d]
            if d:
                sim, rs = F.pad(sim, (d, 0, 0, 0)), F.pad(rs, (d, 0, 0, 0))
            rows.append(torch.cat([torch.cat([sim, left], dim=1), rs], dim=1).reshape(1, -1, 1, h * w))
        while len(rows) > 1:
            rows = [torch.cat(rows[i:i + 2], dim=2) if i + 1 < len(rows) else rows[i] for i in range(0, len(rows), 2)]
        return rows[0].reshape(1, -1, self.depth, h, w)


def _between(start, end, new):
    """Source transform: replace from the line containing `start` up to and including `end` with `new`."""
    def apply(src):
        i = src.index(start)
        i = src.rfind("\n", 0, i) + 1
        j = src.index(end, i) + len(end)
        line = src[i:src.index("\n", i)]
        return src[:i] + line[:len(line) - len(line.lstrip())] + new + src[j:]
    return apply


def rewrite_model(fs, edgenext, wrapper, height, width):
    """Apply the model rewrites (class-level, so call it after any reference run of the upstream model)."""
    import core.submodule as submodule
    model = wrapper.model
    edgenext.CrossCovarianceAttn.forward = xca_forward
    submodule.FlashMultiheadAttention.forward = fmha_forward
    for mod in model.modules():
        if isinstance(mod, edgenext.CrossCovarianceAttn):
            mod._temps = [float(t) for t in mod.temperature.detach().flatten()]
    _recompile(fs, "forward", [
        ("out = self.feature(torch.cat([image1, image2], dim=0))", "out = None"),
        ("geo_feat.to(self.dtype)", "geo_feat"),
        ("features_left = [o[:B] for o in out]", "features_left = self.feature(image1)"),
        ("features_right = [o[B:] for o in out]", "features_right = self.feature(image2)"),
        (_between("if optimize_build_volume=='pytorch1':", "del concat_volume, gwc_volume",
                  "comb_volume = self.volume_stem(features_left[0], features_right[0])"), None),
    ])
    model.volume_stem = VolumeStem(model, model.args.max_disp)
    fs.context_upsample = context_upsample
    _recompile(fs, "upsample_disp", [("context_upsample(disp*4., spx_pred).unsqueeze(1)",
                                      "context_upsample(disp*4., spx_pred)")])
    fs.Combined_Geo_Encoding_Volume = GeoEnc4
    enc = model.update_block.encoder
    k = 2 * model.args.corr_radius + 1
    enc.convc1 = SplitConv(enc.convc1, model.volume_dim, k, height // 4, width // 4)
    type(enc).forward = encoder_forward
    return wrapper


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="Fast-FoundationStereo checkout")
    ap.add_argument("--weights", required=True, help="c-fast-foundationstereo model_best_bp2_serialize.pth")
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--max-disp", type=int, default=192)
    ap.add_argument("--out", required=True, help="output .pte (under hexecutor/.work/)")
    ap.add_argument("--hxfg-dir", help="also write the HXFG image, constants, inputs and PyTorch output here")
    ap.add_argument("--check", action="store_true", help="compare the rewritten model with the upstream one")
    args = ap.parse_args()
    if args.height % 32 or args.width % 32:
        ap.error("height and width must be multiples of 32")

    fs, onnx_export, edgenext = load_upstream(args.src)
    torch.manual_seed(1)
    left = (torch.rand(1, 3, args.height, args.width) * 255 - 123.675) / 58.0
    right = torch.roll(left, -3, 3) + 0.01 * torch.randn(1, 3, args.height, args.width)
    reference = None
    if args.check or args.hxfg_dir:
        with torch.no_grad():
            reference = load_model(fs, onnx_export, args.weights, args.iters, args.max_disp)(left, right)
    wrapper = rewrite_model(fs, edgenext, load_model(fs, onnx_export, args.weights, args.iters, args.max_disp),
                            args.height, args.width)
    if reference is not None:
        with torch.no_grad():
            got = wrapper(left, right)
        err = float((got - reference).abs().max())
        print(f"rewritten model vs upstream: max abs {err:.3e} on disparities of {float(reference.abs().max()):.2f}")
        if args.check and err > 1e-4:
            raise SystemExit("the rewrites disagree with the upstream model")

    from hexecutor.offline.executorch import blob
    from hexecutor.offline.executorch.export import export_program
    program = export_program(wrapper, (left, right))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(program.buffer)
    data = program.buffer
    start = data.find(blob.MAGIC)
    parsed = blob.unpack(data[start:])
    print(f"wrote {out} ({len(data) / 1e6:.1f} MB); HXFG image {len(parsed.image) / 1e3:.0f} kB, "
          f"constants {len(parsed.constants) / 1e6:.1f} MB")
    if args.hxfg_dir:
        import numpy as np
        d = Path(args.hxfg_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "ffs.hxfg").write_bytes(parsed.image)
        (d / "ffs.bin").write_bytes(parsed.constants)
        for name, t in (("left", left), ("right", right)):
            (d / f"{name}.f32").write_bytes(np.ascontiguousarray(t[0].numpy().transpose(1, 2, 0), np.float32).tobytes())
        (d / "want.f32").write_bytes(reference.numpy().astype(np.float32).tobytes())
        print(f"HXFG dump in {d}: run hexecutor_tensor_run ffs.hxfg ffs.bin left.f32 right.f32 out_")


if __name__ == "__main__":
    main()
