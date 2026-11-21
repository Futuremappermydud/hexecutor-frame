# SPDX-License-Identifier: LGPL-3.0-or-later
"""Offline rewrite of rank-5 (and a few rank-6) cost-volume graphs, channels-last blocks and oversized
convolutions into rank-4 ATen that HXFG lowers.

HXFG tensors are batch-one `[1, C, H, W]` maps, so a stereo cost volume `[1, C, D, H, W]` and its 3D
convolutions cannot be lowered as they are. This pass re-traces the decomposed program with every
rank-5 value held as a rank-4 *depth-row layout* `[1, C, D, H*W]`: channels stay dim 1, the depth axis
becomes the rows and each depth slab's pixels are one row of H*W columns (the same row-major pixel
storage as `[1, C, D, H, W]`). In that layout

* pointwise operations, batch norm, concatenation along C or D and broadcasts are unchanged;
* a convolution along D, `(kd, 1, 1)`, is one 2D convolution with a `(kd, 1)` kernel;
* the spatial taps `(ky, kx)` of any 3D convolution are one `index_select` over the H*W columns with a
  constant index (out-of-range and zero-inserted positions read an appended zero column), followed by the
  `(kd, 1)` convolution of that tap's weights; the taps are summed. Strides, padding and dilation in
  H and W live in the index, in D in the 2D convolution. Transposed 3D convolutions use the same gather
  for H and W and a transposed `(kd, 1)` convolution for D;
* indexing with separable constant indices (trilinear resize) is an `index_select` over D and one over
  the columns.

Beyond volumes: channels-last `[1, H, W, C]` blocks (ConvNeXt/EdgeNeXt) run as token matrices
`[1, H*W, C]`; a dense convolution over the HMX scratch budget is split into input-channel chunks (summed)
and output-channel chunks (concatenated) and a kernel above 15 rows into tap groups; constant subgraphs are
folded into buffers; per-pixel-batched `[B*L, ...]` linears run as `[B*L, C]` matrices.

The rewrite is exact up to float re-association. It runs eagerly once (creating the new constant tensors
and checking the result against the original program) and is then exported again, so the output is an
ordinary ExportedProgram that the delegate lowers as usual. Programs that need none of this are
returned untouched. See `rewrite_volumes`.
"""

import operator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import hxfg_writer as hx

aten = torch.ops.aten

MAX_COLUMNS = 4096  # HXFG limit on any tensor dimension (hxfg_writer.MAX_CHANNELS)
MAX_KERNEL = 15  # HXFG convolution kernel size limit (graph.c)
MAX_TEMP_BYTES = 6 << 20  # target size of one column chunk of a gathered 3D-convolution tap (0: never chunk)


class VolumeRewriteError(ValueError):
    """An operation on a rank-5/6 tensor that the rewrite has no rank-4 form for."""


class Pm:
    """A rank-5 value permuted to (1, H, W, D, C). `kind` "vol" holds the depth-row layout `[1, C, D, H*W]`
    it came from; "tok" holds the plain token tensor `[H*W, D, C]` (pixels as the batch) it will become."""

    __slots__ = ("t", "shape", "kind")

    def __init__(self, t, shape, kind):
        self.t, self.shape, self.kind = t, tuple(int(s) for s in shape), kind


class Tok:
    """A channels-last `[1, H, W, C]` tensor (a permute of NCHW) held as the token matrix `[1, H*W, C]`."""

    __slots__ = ("t", "shape")

    def __init__(self, t, shape):
        self.t, self.shape = t, tuple(int(s) for s in shape)
        if tuple(t.shape) != (1, self.shape[1] * self.shape[2], self.shape[3]):
            raise VolumeRewriteError(f"layout {tuple(t.shape)} does not hold channels-last {self.shape}")


class Vol:
    """A rank-5 `[1, C, D, H, W]` (or rank-6 `[1, G, Cg, D, H, W]`) value held as `[1, C, D, H*W]`."""

    __slots__ = ("t", "shape", "split")

    def __init__(self, t, shape, split=None):
        self.t, self.shape, self.split = t, tuple(int(s) for s in shape), split
        n, c, d, hw = t.shape
        if n != 1:
            raise VolumeRewriteError(f"batch must be one, got {tuple(t.shape)}")
        if len(self.shape) == 5:
            exp = (1, self.shape[1], self.shape[2], self.shape[3] * self.shape[4])
        else:
            exp = (1, self.shape[1] * self.shape[2], self.shape[3], self.shape[4] * self.shape[5])
        if tuple(t.shape) != exp:
            raise VolumeRewriteError(f"layout {tuple(t.shape)} does not hold logical shape {self.shape}")

    # logical dims (D, H, W) and channels of a rank-5 value
    @property
    def c(self):
        return self.t.shape[1]

    @property
    def d(self):
        return self.t.shape[2]

    @property
    def hw(self):
        h, w = self.shape[-2:]
        return h, w


def conv_chunks(ci, co, kh, kw):
    """(input-channel chunk, output-channel chunk) that make one dense convolution fit the HMX scratch
    budget (`hxfg_writer.hmx_fits`), or None when the whole convolution already fits."""
    def fits(cic, coc):
        return hx.hmx_fits((kh * kw * cic + 31) // 32, (coc + 31) & ~31)

    if fits(ci, co):
        return None
    best = None
    for parts in range(1, co + 1):  # output chunks: fewest first
        coc = -(-co // parts)
        cic = ci
        while cic > 1 and not fits(cic, coc):
            cic = -(-cic // 2) if cic > 64 else cic - 1
        if fits(cic, coc):
            cost = parts * -(-ci // cic)
            if best is None or cost < best[0]:
                best = (cost, cic, coc)
        if best is not None and parts > 4:
            break
    return None if best is None else (best[1], best[2])


def _pair(v, n):
    v = [int(x) for x in v]
    return v if len(v) == n else v * n


def _tap_index(h, w, ky, kx, stride, pad, dil, oh, ow):
    """Flat source column of tap (ky, kx) for every output pixel; `h * w` is the zero column."""
    iy = np.arange(oh) * stride[0] - pad[0] + ky * dil[0]
    ix = np.arange(ow) * stride[1] - pad[1] + kx * dil[1]
    ok = ((iy >= 0) & (iy < h))[:, None] & ((ix >= 0) & (ix < w))[None, :]
    flat = np.where(ok, iy[:, None] * w + ix[None, :], h * w)
    return flat.reshape(-1).astype(np.int64), bool(ok.any())


def _transposed_tap_index(h, w, ky, kx, stride, pad, dil, oh, ow):
    """Source column of tap (ky, kx) of a transposed convolution for every output pixel."""
    ny = np.arange(oh) + pad[0] - ky * dil[0]
    nx = np.arange(ow) + pad[1] - kx * dil[1]
    iy, ix = ny // stride[0], nx // stride[1]
    oky = (ny % stride[0] == 0) & (iy >= 0) & (iy < h)
    okx = (nx % stride[1] == 0) & (ix >= 0) & (ix < w)
    ok = oky[:, None] & okx[None, :]
    flat = np.where(ok, iy[:, None] * w + ix[None, :], h * w)
    return flat.reshape(-1).astype(np.int64), bool(ok.any())


POINTWISE = {
    aten.relu.default, aten.leaky_relu.default, aten.sigmoid.default, aten.tanh.default, aten.clamp.default,
    aten.gelu.default, aten.silu.default, aten.neg.default, aten.abs.default, aten.exp.default,
    aten.sqrt.default, aten.rsqrt.default, aten.reciprocal.default, aten.hardtanh.default, aten.erf.default,
    aten.sin.default, aten.cos.default, aten.hardswish.default, aten.hardsigmoid.default,
    aten.clone.default, aten.alias.default, aten.detach.default, aten.copy.default,
    aten.clamp_min.default, aten.clamp_max.default,
}
BINARY = {
    aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor, aten.div.Tensor, aten.minimum.default,
    aten.maximum.default, aten.pow.Tensor_Scalar, aten.pow.Tensor_Tensor,
}
IGNORED = {aten._assert_tensor_metadata.default}


class _Const:
    """Marker for a value known at rewrite time (parameter, constant or folded constant expression)."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


FOLD_LIMIT = 1 << 24  # elements; larger constant expressions stay in the graph


class _Rewriter(nn.Module):
    """Interprets the decomposed graph, holding rank-5 values as `Vol`.

    Every node whose inputs are all constants is evaluated once in `__init__` (constant folding: arange,
    clamp, sin/cos tables, casts) and read through `_Const`; a constant is registered as a buffer of the
    re-exported program only when a rank-4 operation consumes it."""

    def __init__(self, ep):
        super().__init__()
        self.sig = ep.graph_signature
        self.nodes = list(ep.graph.nodes)
        self.cv = {}  # constant name -> real tensor
        for ph, fqn in {**self.sig.inputs_to_parameters, **self.sig.inputs_to_buffers}.items():
            self.cv[ph] = ep.state_dict[fqn]
        for ph, fqn in self.sig.inputs_to_lifted_tensor_constants.items():
            self.cv[ph] = ep.constants[fqn]
        for node in self.nodes:
            if node.op != "call_function" or node.target in IGNORED:
                continue
            ins = [a for a in node.all_input_nodes]
            if ins and all(a.name in self.cv for a in ins) or (not ins and _is_factory(node)):
                if self._numel(node) > FOLD_LIMIT:
                    continue
                args = _map(node.args, {a: _Const(a.name) for a in ins}, None)
                args = _map_const(args, self.cv)
                kwargs = _map_const(_map(node.kwargs, {a: _Const(a.name) for a in ins}, None), self.cv)
                with torch.no_grad():
                    out = node.target(*args, **kwargs)
                if isinstance(out, torch.Tensor):
                    self.cv[node.name] = out
        self.cvn = {k: v.detach().numpy() for k, v in self.cv.items()}  # untraced views for index arithmetic
        self.user_inputs = list(self.sig.user_inputs)
        self.setup = True  # first (eager) run creates constants
        self._made = set()
        self.stats = {}

    @staticmethod
    def _numel(node):
        v = node.meta.get("val")
        return v.numel() if isinstance(v, torch.Tensor) else 0

    # ---- constants -------------------------------------------------------------------------------
    def const(self, key, make):
        """A new constant tensor: created on the eager run, a registered buffer afterwards."""
        name = "vc_" + "".join(ch if ch.isalnum() else "_" for ch in key)
        if name not in self._made:
            if not self.setup:
                raise RuntimeError(f"constant {key} missing on the traced run")
            tensor = make().detach().clone().contiguous()
            self.register_buffer(name, tensor, persistent=True)
            self._made.add(name)
        return getattr(self, name)

    def val(self, x):
        """Real tensor behind a `_Const` (for reading numbers); other values pass through."""
        return self.cv[x.name] if isinstance(x, _Const) else x

    def mat(self, x):
        """A `_Const` as a buffer tensor usable inside the traced program."""
        return self.const("c_" + x.name, lambda: self.cv[x.name]) if isinstance(x, _Const) else x

    # ---- helpers ---------------------------------------------------------------------------------
    def _count(self, key):
        self.stats[key] = self.stats.get(key, 0) + 1

    @staticmethod
    def _val(node):
        return node.meta.get("val")

    def _is_high(self, node):
        v = self._val(node)
        if isinstance(v, torch.Tensor):
            return v.dim() >= 5
        if isinstance(v, (tuple, list)):
            return any(isinstance(x, torch.Tensor) and x.dim() >= 5 for x in v)
        return False

    def _oversized(self, node, args):
        w = args[1]
        if not isinstance(w, _Const) or args[8] != 1:
            return False
        shape = tuple(self.cv[w.name].shape)
        if len(shape) != 4:
            return False
        ci, co = (shape[0], shape[1]) if args[6] else (shape[1], shape[0])
        return conv_chunks(ci, co, shape[2], shape[3]) is not None

    def _nhwc_start(self, node, args):
        if node.target in (aten.view.default, aten.reshape.default, aten._unsafe_view.default):
            x, out = args[0], tuple(self._val(node).shape)
            xs = tuple(self.cv[x.name].shape) if isinstance(x, _Const) else tuple(getattr(x, "shape", ()))
            return (isinstance(x, torch.Tensor) and len(xs) == 3 and xs[0] == 1 and len(out) == 4 and out[0] == 1
                    and out[3] == xs[2] and out[1] * out[2] == xs[1] and out[1] > 1 and out[2] > 1)
        return node.target is aten.permute.default and self._is_nchw_to_nhwc(node, args)

    def _lift(self, node, shape, x, tag):
        """A plain operand of a rank-5 binary op as a tensor broadcastable against `[1, C, D, H*W]`."""
        if isinstance(x, _Const):
            real = self.cv[x.name]
            if real.dim() == 0:
                return self.mat(x)
            if len(shape) == 6:
                raise VolumeRewriteError(f"{node.name}: constant against a rank-6 value")
            lead = (1,) * (len(shape) - real.dim())
            if len(lead) + real.dim() != 5:
                raise VolumeRewriteError(f"{node.name}: operand rank {real.dim()} against rank {len(shape)}")

            def make():
                full5 = real.reshape(lead + tuple(real.shape))
                _, c, d, h, w = full5.shape
                if h == 1 and w == 1:
                    return full5.reshape(1, c, d, 1)
                full = full5.expand(1, c, d, shape[3], shape[4])
                return full.reshape(1, c, d, shape[3] * shape[4])

            return self.const(f"{node.name}.lift{tag}", make)
        if isinstance(x, torch.Tensor):
            if x.dim() == 0:
                return x
            raise VolumeRewriteError(f"{node.name}: runtime plain operand of rank {x.dim()} against a volume")
        return x

    # ---- convolution -----------------------------------------------------------------------------
    def conv2d(self, key, x, w, bias, stride, pad, dil, transposed, outpad, groups, wshape=None):
        """aten.convolution on a rank-4 value; a dense convolution too large for the HMX scratch budget is
        split into input-channel chunks (summed) and output-channel chunks (concatenated). `w` is a tensor or
        a function that builds it (called only on the eager run, so no weight slicing enters the trace);
        `wshape` is then its shape."""
        wfn = w if callable(w) else (lambda: w)
        wshape = tuple(wshape if wshape is not None else w.shape)
        if transposed:
            ci, co, kh, kw = wshape
        else:
            co, ci, kh, kw = wshape
        if not transposed and kh > MAX_KERNEL:
            # HXFG kernels are at most 15 rows: split the taps into groups read from the padded input
            ph = pad[0]
            xp = F.pad(x, (0, 0, ph, ph)) if ph else x
            oh = (x.shape[2] + 2 * ph - dil[0] * (kh - 1) - 1) // stride[0] + 1
            acc = None
            for k0 in range(0, kh, MAX_KERNEL):
                k1 = min(kh, k0 + MAX_KERNEL)
                length = (oh - 1) * stride[0] + (k1 - k0 - 1) * dil[0] + 1
                part = xp[:, :, k0 * dil[0]:k0 * dil[0] + length]
                y = self.conv2d(f"{key}.k{k0}", part, (lambda k0=k0, k1=k1: wfn()[:, :, k0:k1]),
                                bias if acc is None else None, stride, [0, pad[1]], dil, False, outpad, groups,
                                wshape=(co, ci, k1 - k0, kw))
                acc = y if acc is None else acc + y
            self._count("conv_taps_split")
            return acc
        plan = conv_chunks(ci, co, kh, kw) if groups == 1 else None
        if plan is None:
            wt = self.const(f"{key}.w", wfn)
            bt = None if bias is None else self.const(f"{key}.b", lambda: bias)
            return aten.convolution.default(x, wt, bt, list(stride), list(pad), list(dil), bool(transposed),
                                            list(outpad), groups)
        cic, coc = plan
        pieces = []
        for o in range(0, co, coc):
            acc = None
            for i in range(0, ci, cic):
                xs = x[:, i:i + cic]
                ws = ((lambda o=o, i=i: wfn()[i:i + cic, o:o + coc]) if transposed
                      else (lambda o=o, i=i: wfn()[o:o + coc, i:i + cic]))
                wt = self.const(f"{key}.w{o}_{i}", ws)
                bt = self.const(f"{key}.b{o}", lambda o=o: bias[o:o + coc]) if (bias is not None and acc is None) else None
                y = aten.convolution.default(xs, wt, bt, list(stride), list(pad), list(dil), bool(transposed),
                                             list(outpad), 1)
                acc = y if acc is None else acc + y
            pieces.append(acc)
        self._count("conv_split")
        out = pieces[0]
        for extra in pieces[1:]:
            out = torch.cat([out, extra], dim=1)
        return out

    def _conv(self, node, v, w, bias, stride, pad, dil, transposed, outpad, groups):
        n, c, d, h, wd = v.shape
        kd, kh, kw = w.shape[2:]
        stride, pad, dil, outpad = (_pair(a, 3) for a in (stride, pad, dil, outpad))
        t = v.t
        if transposed:
            co = w.shape[1] * groups
            od = (d - 1) * stride[0] - 2 * pad[0] + dil[0] * (kd - 1) + outpad[0] + 1
            oh = (h - 1) * stride[1] - 2 * pad[1] + dil[1] * (kh - 1) + outpad[1] + 1
            ow = (wd - 1) * stride[2] - 2 * pad[2] + dil[2] * (kw - 1) + outpad[2] + 1
        else:
            co = w.shape[0]
            od = (d + 2 * pad[0] - dil[0] * (kd - 1) - 1) // stride[0] + 1
            oh = (h + 2 * pad[1] - dil[1] * (kh - 1) - 1) // stride[1] + 1
            ow = (wd + 2 * pad[2] - dil[2] * (kw - 1) - 1) // stride[2] + 1
        if oh * ow > MAX_COLUMNS or h * wd > MAX_COLUMNS:
            raise VolumeRewriteError(
                f"{node.name}: {h}x{wd} -> {oh}x{ow} pixels exceed the {MAX_COLUMNS}-column HXFG limit")
        spatial = (kh, kw) != (1, 1) or stride[1:] != [1, 1] or pad[1:] != [0, 0] or transposed
        src = t
        taps = []
        for ky in range(kh):
            for kx in range(kw):
                if transposed:
                    idx, used = _transposed_tap_index(h, wd, ky, kx, stride[1:], pad[1:], dil[1:], oh, ow)
                else:
                    idx, used = _tap_index(h, wd, ky, kx, stride[1:], pad[1:], dil[1:], oh, ow)
                if used:
                    taps.append((ky, kx, idx))
        if not taps:
            raise VolumeRewriteError(f"{node.name}: convolution reads only padding")
        # A gathered tap is `[1, C, D, columns]`; the output pixels are independent, so a spatial convolution runs
        # on column chunks sized to MAX_TEMP_BYTES and the chunks are joined at the end. This keeps the
        # (zero-inserted, for a transposed convolution) taps small instead of volume-sized.
        rows = max(d, od, (d - 1) * stride[0] + kd if transposed else d)
        width = max(int(c), int(co))
        per_column = rows * (-(-width // 32) * 32) * 4
        chunks = 1
        if spatial and MAX_TEMP_BYTES:
            chunks = max(1, min(oh * ow, -(-per_column * oh * ow // MAX_TEMP_BYTES)))
        while spatial and chunks < oh * ow:
            # the source range of a chunk (its pixels' taps, plus the zero column) must fit one tensor
            edges = [oh * ow * i // chunks for i in range(chunks + 1)]
            widest = 0
            for c0, c1 in zip(edges, edges[1:]):
                reads = np.concatenate([idx[c0:c1][idx[c0:c1] != h * wd] for _, _, idx in taps])
                widest = max(widest, int(reads.max()) - int(reads.min()) + 2 if reads.size else 1)
            if widest <= MAX_COLUMNS:
                break
            chunks *= 2
        chunks = min(chunks, oh * ow)
        bounds = [oh * ow * i // chunks for i in range(chunks + 1)]
        pieces = []
        for ci in range(chunks):
            c0, c1 = bounds[ci], bounds[ci + 1]
            acc = None
            subs = [(ky, kx, idx[c0:c1] if chunks > 1 else idx) for ky, kx, idx in taps]
            subs = [item for item in subs if chunks == 1 or not bool(np.all(item[2] == h * wd))]
            tile, lo, zero = None, 0, h * wd
            if spatial:
                # The chunk's taps read a contiguous column range of the volume (a few image rows around its
                # pixels): copy just that range plus one zero column for out-of-image taps, instead of padding
                # the whole volume (which would also overflow MAX_COLUMNS for a volume of exactly that many).
                reads = np.concatenate([sub[sub != h * wd] for _, _, sub in subs]) if subs else np.zeros(0, np.int64)
                lo, hi = (int(reads.min()), int(reads.max()) + 1) if reads.size else (0, 1)
                piece = t if (lo == 0 and hi == h * wd) else t[:, :, :, lo:hi]
                needs_zero = any(bool(np.any(sub == h * wd)) for _, _, sub in subs)
                tile = F.pad(piece, (0, 1, 0, 0)) if needs_zero else piece
                zero = hi - lo
                if tile.shape[3] > MAX_COLUMNS:
                    raise VolumeRewriteError(f"{node.name}: a column chunk reads {tile.shape[3]} pixels, over the "
                                             f"{MAX_COLUMNS}-column HXFG limit (lower MAX_TEMP_BYTES)")
            for ky, kx, sub in subs:
                if not spatial or (sub.size == h * wd and np.array_equal(sub, np.arange(h * wd))):
                    g = t
                else:
                    rel = np.where(sub == h * wd, zero, sub - lo)
                    ib = self.const(f"{node.name}.idx{ky}_{kx}_{ci}", lambda rel=rel: torch.from_numpy(rel))
                    g = torch.index_select(tile, 3, ib)
                y = self.conv2d(f"{node.name}.t{ky}_{kx}_{ci}", g,
                                (lambda ky=ky, kx=kx: w[:, :, :, ky, kx].unsqueeze(-1)),
                                bias if acc is None else None, [stride[0], 1], [pad[0], 0], [dil[0], 1], transposed,
                                [outpad[0], 0] if transposed else [0, 0], groups,
                                wshape=(tuple(w.shape[:3]) + (1,)))
                acc = y if acc is None else acc + y
            if acc is None:
                raise VolumeRewriteError(f"{node.name}: column chunk {ci} reads only padding")
            pieces.append(acc)
        while len(pieces) > 1:
            pieces = [torch.cat(pieces[i:i + 2], dim=3) if i + 1 < len(pieces) else pieces[i]
                      for i in range(0, len(pieces), 2)]
        acc = pieces[0]
        self._count("conv3d_transposed" if transposed else "conv3d")
        return Vol(acc, (1, co, od, oh, ow))

    # ---- node handlers ---------------------------------------------------------------------------
    def _handle(self, node, args, kwargs):
        tgt = node.target
        if tgt is operator.getitem:
            return args[0][args[1]]
        vols = [a for a in _flat(args) if isinstance(a, (Vol, Pm))]
        toks = [a for a in _flat(args) if isinstance(a, Tok)]
        if toks or self._nhwc_start(node, args):
            return self._tok(node, args, kwargs)
        if tgt is aten.convolution.default and vols:
            x, w, bias, stride, pad, dil, transposed, outpad, groups = args
            return self._conv(node, x, self.val(w), None if bias is None else self.val(bias),
                              stride, pad, dil, transposed, outpad, groups)
        if tgt is aten._native_batch_norm_legit_no_training.default and vols:
            x = args[0]
            out = tgt(x.t, *(self.mat(a) for a in args[1:]))
            return (Vol(out[0], x.shape, x.split), None, None)  # the statistics outputs stay unused
        if tgt in POINTWISE and vols:
            x = args[0]
            return Vol(tgt(x.t, *args[1:], **kwargs), x.shape, x.split)
        if tgt in BINARY and vols:
            return self._binary(node, args, kwargs)
        if tgt is aten.cat.default and vols:
            return self._cat(node, args)
        if tgt is aten.unsqueeze.default:
            return self._unsqueeze(node, args)
        if tgt in (aten.squeeze.dims, aten.squeeze.dim) and vols:
            return self._squeeze(node, args)
        if tgt is aten.expand.default and (vols or self._is_high(node)):
            return self._expand(node, args)
        if tgt in (aten.view.default, aten.reshape.default, aten._unsafe_view.default):
            return self._view(node, args)
        if tgt is aten.permute.default and (vols or isinstance(args[0], Pm)):
            return self._permute(node, args)

        if tgt is aten.index.Tensor and vols:
            return self._index(node, args)
        if tgt is aten.index_select.default and vols:
            return self._index_select(node, args)
        if tgt is aten.linalg_vector_norm.default and vols:
            return self._group_norm(node, args)
        if tgt is aten.sum.dim_IntList and vols:
            return self._group_sum(node, args)
        shape = getattr(self._val(node), "shape", None)
        raise VolumeRewriteError(f"{node.name}: no rank-4 form for {tgt} on rank-5 values (output {shape})")

    def _binary(self, node, args, kwargs):
        a, b = args[0], args[1]
        shape = tuple(self._val(node).shape)
        la = a.t if isinstance(a, Vol) else self._lift(node, shape, a, "a")
        lb = b.t if isinstance(b, Vol) else self._lift(node, shape, b, "b")
        out = node.target(la, lb, *args[2:], **kwargs)
        split = next((x.split for x in (a, b) if isinstance(x, Vol) and x.split), None)
        return Vol(out, shape, split)

    def _cat(self, node, args):
        tensors, dim = args[0], (args[1] if len(args) > 1 else 0)
        rank = len(self._val(node).shape)
        dim = dim + rank if dim < 0 else dim
        if rank != 5 or dim not in (1, 2) or not all(isinstance(t, Vol) for t in tensors):
            raise VolumeRewriteError(f"{node.name}: cat along dim {dim} of rank {rank}")
        out = torch.cat([t.t for t in tensors], dim=dim)
        return Vol(out, tuple(self._val(node).shape))

    def _unsqueeze(self, node, args):
        x, dim = args[0], args[1]
        out_shape = tuple(self._val(node).shape)
        if isinstance(x, Vol):
            raise VolumeRewriteError(f"{node.name}: unsqueeze of a rank-{len(x.shape)} value")
        if len(out_shape) != 5:
            return aten.unsqueeze.default(self.mat(x), dim)
        dim = dim + 5 if dim < 0 else dim
        if dim < 2:
            raise VolumeRewriteError(f"{node.name}: unsqueeze at dim {dim} would move the channel axis")
        if isinstance(x, _Const):
            raise VolumeRewriteError(f"{node.name}: unsqueeze of a constant into a volume")
        return Vol(x.reshape(1, out_shape[1], out_shape[2], out_shape[3] * out_shape[4]), out_shape)

    def _squeeze(self, node, args):
        x = args[0]
        out = tuple(self._val(node).shape)
        if len(out) != 4 or out[0] != 1:
            raise VolumeRewriteError(f"{node.name}: squeeze to {out}")
        if out[1] == x.shape[1]:  # H or W of one: the channels stay dim 1
            return x.t.reshape(out)
        if x.shape[1] == 1 and out[1] == x.shape[2]:
            # [1, 1, D, HW] -> [1, D, H, W]: the depth rows become channels, a real transposition
            return x.t.permute(0, 2, 1, 3).reshape(out)
        raise VolumeRewriteError(f"{node.name}: squeeze {x.shape} -> {out}")

    @staticmethod
    def _is_nchw_to_nhwc(node, args):
        x = args[0]
        return (isinstance(x, torch.Tensor) and x.dim() == 4 and x.shape[0] == 1
                and [int(p) % 4 for p in args[1]] == [0, 2, 3, 1])

    def _tok(self, node, args, kwargs):
        """Channels-last blocks (permute, layer norm, linear, GELU, per-channel scale, permute back) as
        token matrices, which HXFG lowers; a NCHW <-> token transpose pair is one HWC copy."""
        tgt = node.target
        x = args[0]
        if tgt in (aten.view.default, aten.reshape.default, aten._unsafe_view.default):
            out = tuple(self._val(node).shape)
            if isinstance(x, Tok):  # back to tokens [1, N, C]
                if len(out) == 3 and out[0] == 1 and out[2] == x.shape[3] and out[1] == x.shape[1] * x.shape[2]:
                    return x.t
                raise VolumeRewriteError(f"{node.name}: view of channels-last {x.shape} to {out}")
            plain = self.mat(x)  # tokens [1, N, C] -> channels-last [1, H, W, C]
            return Tok(plain, out)
        if tgt is aten.permute.default and not isinstance(x, Tok):
            _, c, h, w = x.shape
            return Tok(x.reshape(1, c, h * w).transpose(1, 2), (1, h, w, c))
        if tgt is aten.permute.default and [int(p) % 4 for p in args[1]] == [0, 3, 1, 2]:
            _, h, w, c = x.shape
            return x.t.transpose(1, 2).reshape(1, c, h, w)
        if tgt is aten.native_layer_norm.default:
            out = tgt(x.t, *(self.mat(a) for a in args[1:]))
            return (Tok(out[0], x.shape), None, None)
        if tgt is aten.linear.default:
            out = tgt(x.t, *(self.mat(a) for a in args[1:]))
            return Tok(out, x.shape[:3] + (out.shape[-1],))
        if tgt in POINTWISE:
            return Tok(tgt(x.t, *args[1:], **kwargs), x.shape)
        if tgt in BINARY:
            a, b = args[0], args[1]
            shape = tuple(self._val(node).shape)

            def lift(v, tag):
                if isinstance(v, Tok):
                    return v.t
                if isinstance(v, _Const):
                    real = self.cv[v.name]
                    if real.dim() == 0:
                        return self.mat(v)
                    return self.const(f"{node.name}.tok{tag}", lambda: real.expand(shape).reshape(1, -1, shape[3])
                                      if real.numel() > shape[3] else real.reshape(1, 1, -1))
                return v

            return Tok(tgt(lift(a, "a"), lift(b, "b"), *args[2:], **kwargs), shape)
        raise VolumeRewriteError(f"{node.name}: no token form for {tgt} on a channels-last tensor")

    def _group_matrix(self, key, groups, per, repeat=False):
        """1x1 convolution weight summing (or, `repeat`, replicating) `per` channels in each group."""
        def make():
            m = torch.zeros(groups * per, groups)
            for g in range(groups):
                m[g * per:(g + 1) * per, g] = 1.0
            return (m if repeat else m.t()).reshape(*(m.t().shape if not repeat else m.shape), 1, 1)
        return self.const(f"{key}.{groups}x{per}{'r' if repeat else 's'}", make)

    def _group_sum(self, node, args):
        """sum over the Cg axis of a rank-6 `[1, G, Cg, D, H, W]` value: a 1x1 convolution of ones."""
        x, dims = args[0], args[1]
        keep = args[2] if len(args) > 2 else node.kwargs.get("keepdim", False)
        if len(x.shape) != 6 or list(dims) != [2]:
            raise VolumeRewriteError(f"{node.name}: sum over {dims} of a rank-{len(x.shape)} value")
        g, cg = x.split
        w = self._group_matrix("gsum", g, cg)
        out = aten.convolution.default(x.t, w, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        self._count("group_sum")
        if keep:
            return Vol(out, (1, g, 1) + x.shape[3:], (g, 1))
        return Vol(out, (1, g) + x.shape[3:])

    def _group_norm(self, node, args):
        """L2 norm over the Cg axis (dim 2, keepdim) of a rank-6 value: sqrt of a grouped sum of squares."""
        x, ord_, dims = args[0], args[1], args[2]
        keep = args[3] if len(args) > 3 else node.kwargs.get("keepdim", False)
        if len(x.shape) != 6 or ord_ != 2.0 or list(dims) != [2] or not keep:
            raise VolumeRewriteError(f"{node.name}: vector norm {ord_} over {dims} of a rank-{len(x.shape)} value")
        g, cg = x.split
        w = self._group_matrix("gsum", g, cg)
        sq = aten.convolution.default(x.t * x.t, w, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        self._count("group_norm")
        return Vol(torch.sqrt(sq), (1, g, 1) + x.shape[3:], (g, 1))

    def _expand(self, node, args):
        x = args[0]
        out = tuple(self._val(node).shape)
        if isinstance(x, Vol) and len(x.shape) == 6 and x.split and x.split[1] == 1 and out[2] > 1 and out[1] == x.shape[1]:
            # [1, G, 1, D, H, W] -> [1, G, Cg, D, H, W]: replicate each group's channel Cg times
            g, cg = x.shape[1], out[2]
            w = self._group_matrix("grep", g, cg, repeat=True)
            t = aten.convolution.default(x.t, w, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            return Vol(t, out, (g, cg))
        if not isinstance(x, Vol) or len(out) != len(x.shape) or out[3:] != x.shape[3:] or out[1] != x.shape[1] or out[0] != 1:
            raise VolumeRewriteError(f"{node.name}: expand to {out}")
        return Vol(x.t.expand(1, out[1], out[2], x.t.shape[3]), out, x.split)

    def _permute(self, node, args):
        x, perm = args[0], [int(p) % 5 for p in args[1]]
        if isinstance(x, Vol) and len(x.shape) == 5 and perm == [0, 3, 4, 2, 1]:
            _, c, d, h, w = x.shape
            return Pm(x.t, (1, h, w, d, c), "vol")
        if isinstance(x, Pm) and x.kind == "tok" and perm == [0, 4, 3, 1, 2]:
            _, h, w, d, c = x.shape
            # tokens [H*W, D, C] -> rows (pixel, depth) of C -> [1, C, H*W, D] -> depth rows [1, C, D, H*W]
            t = x.t.reshape(1, h * w * d, c).transpose(1, 2).reshape(1, c, h * w, d).permute(0, 1, 3, 2)
            return Vol(t, (1, c, d, h, w))
        raise VolumeRewriteError(f"{node.name}: permute {perm} of {getattr(x, 'shape', None)}")

    def _view(self, node, args):
        x = args[0]
        out = tuple(self._val(node).shape)
        if isinstance(x, Pm):
            if x.kind == "vol" and len(out) == 3 and out[0] == x.shape[1] * x.shape[2] and out[1:] == x.shape[3:]:
                _, h, w, d, c = x.shape
                # [1, C, D, H*W] -> [1, C, H*W, D] -> rows (pixel, depth) of C -> tokens [H*W, D, C]
                t = x.t.permute(0, 1, 3, 2).reshape(1, c, h * w * d).transpose(1, 2)
                return t.reshape(out)
            raise VolumeRewriteError(f"{node.name}: view of permuted {x.shape} to {out}")
        if not isinstance(x, Vol) and len(out) == 5 and out[0] == 1:
            plain = self.mat(x)
            if plain.dim() == 3 and plain.shape[0] == out[1] * out[2] and tuple(plain.shape[1:]) == tuple(out[3:]):
                return Pm(plain, out, "tok")
        if isinstance(x, Vol):
            if len(out) == 6 and len(x.shape) == 5 and out[0] == 1 and out[1] * out[2] == x.shape[1] and out[3:] == x.shape[2:]:
                return Vol(x.t, out, (out[1], out[2]))
            if len(out) == 5 and len(x.shape) == 6 and out[1] == x.shape[1] * x.shape[2] and out[2:] == x.shape[3:]:
                return Vol(x.t, out)
            if 2 <= len(out) <= 4 and out[0] == 1 and out[1] == x.c:
                return x.t.reshape(out)
            raise VolumeRewriteError(f"{node.name}: view {x.shape} -> {out}")
        x = self.mat(x)
        if len(out) == 5 and out[0] == 1 and x.dim() == 4 and x.shape[0] == 1 and x.shape[1] == out[1]:
            return Vol(x.reshape(1, out[1], out[2], out[3] * out[4]), out)
        raise VolumeRewriteError(f"{node.name}: view of {tuple(x.shape)} to {out}")

    def _index(self, node, args):
        x, indices = args[0], args[1]
        out = tuple(self._val(node).shape)
        if len(x.shape) != 5 or len(indices) != 5 or indices[0] is not None or indices[1] is not None:
            raise VolumeRewriteError(f"{node.name}: only separable (D, H, W) indexing is supported")
        if not all(isinstance(i, _Const) for i in indices[2:]):
            raise VolumeRewriteError(f"{node.name}: indices must be constants")
        idd, idh, idw = (self.cvn[i.name].astype(np.int64) for i in indices[2:])
        idd, idh, idw = (i.reshape((1,) * (3 - i.ndim) + tuple(i.shape)) for i in (idd, idh, idw))
        _, _, do, ho, wo = out
        if tuple(idd.shape) != (do, 1, 1) or tuple(idh.shape) != (1, ho, 1) or tuple(idw.shape) != (1, 1, wo):
            raise VolumeRewriteError(f"{node.name}: indices are not separable ({tuple(idd.shape)}, {tuple(idh.shape)}, {tuple(idw.shape)})")
        t = x.t
        d, h, w = x.shape[2:]
        iv = idd.reshape(-1)
        if not (do == d and np.array_equal(iv, np.arange(d))):
            t = torch.index_select(t, 2, self.const(f"{node.name}.id", lambda: torch.from_numpy(iv)))
        col = (idh.reshape(-1, 1) * w + idw.reshape(1, -1)).reshape(-1)
        if not (ho * wo == h * w and np.array_equal(col, np.arange(h * w))):
            t = torch.index_select(t, 3, self.const(f"{node.name}.ihw", lambda: torch.from_numpy(col)))
        self._count("index3d")
        return Vol(t, out)

    def _index_select(self, node, args):
        """`index_select` of a rank-5 value along D, H or W with a constant index (a separable resize axis)."""
        x, dim, index = args[0], args[1], args[2]
        out = tuple(self._val(node).shape)
        dim = dim + 5 if dim < 0 else dim
        if not isinstance(x, Vol) or len(x.shape) != 5 or dim not in (2, 3, 4) or not isinstance(index, _Const):
            raise VolumeRewriteError(f"{node.name}: index_select on {getattr(x, 'shape', None)} dim {dim}")
        idx = self.cvn[index.name].astype(np.int64).reshape(-1)
        _, _, d, h, w = x.shape
        if dim == 2:
            col = idx
            axis = 2
        else:
            if dim == 3:
                col = (idx[:, None] * w + np.arange(w)[None, :]).reshape(-1)
            else:
                col = (np.arange(h)[:, None] * w + idx[None, :]).reshape(-1)
            axis = 3
        if len(col) == x.t.shape[axis] and np.array_equal(col, np.arange(len(col))):
            return Vol(x.t, out)
        t = torch.index_select(x.t, axis, self.const(f"{node.name}.is", lambda: torch.from_numpy(col)))
        self._count("index_select3d")
        return Vol(t, out)

    # ---- driver ----------------------------------------------------------------------------------
    def forward(self, *inputs):
        env = {}
        view_src = {}  # id(view result) -> the tensor it views, so a view undone by the next view is elided
        outputs = None
        user = iter(inputs)
        for node in self.nodes:
            if node.op == "placeholder":
                env[node] = _Const(node.name) if node.name in self.cv else next(user)
            elif node.op == "call_function":
                if node.target in IGNORED or not node.users:
                    env[node] = None
                elif node.name in self.cv:
                    env[node] = _Const(node.name)
                else:
                    args = _map(node.args, env, self)
                    kwargs = _map(node.kwargs, env, self)
                    if node.target is aten.cat.default and len(args[0]) == 1:
                        env[node] = args[0][0]  # a one-input cat is the input
                        continue
                    if node.target is aten.convolution.default and self._oversized(node, args):
                        x, w, bias, stride, pad, dil, transposed, outpad, groups = (
                            _plain(args[0], self), self.val(args[1]), None if args[2] is None else self.val(args[2]), *args[3:])
                        env[node] = self.conv2d(node.name, x, w, bias, stride, pad, dil, transposed, outpad, groups)
                        continue
                    if node.target in (aten.view.default, aten.reshape.default) and isinstance(args[0], torch.Tensor):
                        src = view_src.get(id(args[0]))
                        shape = tuple(node.meta["val"].shape)
                        if src is not None and tuple(src.shape) == shape:
                            env[node] = src  # a view undone by the next view
                            continue
                    if any(isinstance(a, (Vol, Pm, Tok)) for a in _flat(args)) or self._is_high(node) or self._nhwc_start(node, args):
                        env[node] = self._handle(node, args, kwargs)
                    else:
                        pargs = _plain(args, self)
                        if (node.target is aten.linear.default and pargs[0].dim() == 3 and pargs[0].shape[0] > 1):
                            # rows are independent: [B, L, C] as the [B*L, C] matrix HXFG linears take
                            b3, l3, c3 = pargs[0].shape
                            out = node.target(pargs[0].reshape(b3 * l3, c3), *pargs[1:], **_plain(kwargs, self))
                            env[node] = out.reshape(b3, l3, out.shape[-1])
                        else:
                            env[node] = node.target(*pargs, **_plain(kwargs, self))
                        if node.target in (aten.view.default, aten.reshape.default) and isinstance(pargs[0], torch.Tensor):
                            view_src[id(env[node])] = pargs[0]
            elif node.op == "output":
                outputs = _plain(_map(node.args[0], env, self), self)
        if any(isinstance(o, (Vol, Pm, Tok)) for o in _flat(outputs)):
            raise VolumeRewriteError("a rank-5 value reaches the program output")
        return tuple(outputs) if isinstance(outputs, (tuple, list)) else outputs


_FACTORIES = ("arange", "full", "scalar_tensor", "zeros", "ones", "linspace", "eye")


def _is_factory(node):
    schema = getattr(node.target, "_schema", None)
    return schema is not None and schema.name.split("::")[-1] in _FACTORIES


def _map_const(a, cv):
    if isinstance(a, _Const):
        return cv[a.name]
    if isinstance(a, (list, tuple)):
        return type(a)(_map_const(i, cv) for i in a)
    if isinstance(a, dict):
        return {k: _map_const(i, cv) for k, i in a.items()}
    return a


def _plain(a, rw):
    """Materialise `_Const` markers for an ordinary ATen call."""
    if isinstance(a, _Const):
        return rw.mat(a)
    if isinstance(a, (list, tuple)):
        return type(a)(_plain(i, rw) for i in a)
    if isinstance(a, dict):
        return {k: _plain(i, rw) for k, i in a.items()}
    return a


def _flat(x):
    if isinstance(x, (list, tuple)):
        for i in x:
            yield from _flat(i)
    else:
        yield x


def _map(a, env, rw=None):
    if isinstance(a, torch.fx.Node):
        return env[a]
    if isinstance(a, (list, tuple)):
        return type(a)(_map(i, env, rw) for i in a)
    if isinstance(a, dict):
        return {k: _map(i, env, rw) for k, i in a.items()}
    return a


def _prune_unused_getitem(ep):
    """The export tracer emits a getitem per output of a multi-output operation (layer norm statistics,
    batch-norm state); the unused ones keep the delegate from taking the operation."""
    graph = ep.graph_module.graph
    # Dead nodes: the unused outputs above and the views a peephole made redundant.
    for node in reversed(list(graph.nodes)):
        if node.op == "call_function" and not node.users and (
                node.target is operator.getitem or getattr(node.target, "_schema", None) is not None
                and not node.target._schema.is_mutable):
            graph.erase_node(node)
    ep.graph_module.recompile()


def has_oversized_conv(ep):
    """A dense 2D convolution too large for the HMX scratch budget (it is split by the rewrite)."""
    for n in ep.graph.nodes:
        if n.op == "call_function" and str(n.target) in ("aten.convolution.default", "aten.conv2d.default",
                                                           "aten.conv_transpose2d.input"):
            w = n.args[1].meta.get("val") if isinstance(n.args[1], torch.fx.Node) else None
            if not isinstance(w, torch.Tensor) or w.dim() != 4:
                continue
            groups = n.args[8] if str(n.target).endswith("convolution.default") and len(n.args) > 8 else 1
            transposed = str(n.target).endswith("convolution.default") and len(n.args) > 6 and n.args[6] or "transpose" in str(n.target)
            ci, co = (w.shape[0], w.shape[1]) if transposed else (w.shape[1], w.shape[0])
            if groups == 1 and conv_chunks(ci, co, w.shape[2], w.shape[3]) is not None:
                return True
    return False


def has_nhwc_linear(ep):
    """A `linear` or layer norm over a rank-4 channels-last tensor (a permuted NCHW feature map), or a
    linear over a rank-3 token tensor whose batch is above one."""
    for n in ep.graph.nodes:
        if n.op == "call_function" and str(n.target) in ("aten.linear.default", "aten.native_layer_norm.default", "aten.layer_norm.default"):
            v = n.args[0].meta.get("val") if isinstance(n.args[0], torch.fx.Node) else None
            if isinstance(v, torch.Tensor) and (v.dim() == 4 or (v.dim() == 3 and v.shape[0] > 1 and "linear" in str(n.target))):
                return True
    return False


def has_high_rank(ep):
    for n in ep.graph.nodes:
        if n.op == "placeholder":
            continue  # an unused rank-5 weight does not matter; only values the program computes do
        v = n.meta.get("val")
        if isinstance(v, torch.Tensor) and v.dim() >= 5:
            return True
        if isinstance(v, (tuple, list)) and any(isinstance(x, torch.Tensor) and x.dim() >= 5 for x in v):
            return True
    return False


def rewrite_volumes(ep, example_inputs, decomposition_table=None, check=True, rtol=1e-4, atol=1e-4, force=False):
    """Return an ExportedProgram with no rank-5 tensors, token-form channels-last blocks and split oversized
    convolutions, or the (decomposed) program itself when it needs none of that. `force` always re-traces,
    which also folds constant subgraphs (the weights a custom decomposition builds)."""
    if decomposition_table is not None or not has_high_rank(ep):
        ep = ep.run_decompositions(decomposition_table) if decomposition_table is not None else ep
    if not (force or has_high_rank(ep) or has_nhwc_linear(ep) or has_oversized_conv(ep)):
        return ep
    rw = _Rewriter(ep)
    with torch.no_grad():
        want = ep.module()(*example_inputs)
        got = rw(*example_inputs)
        if check:
            for w, g in zip(_flat(want), _flat(got)):
                if not torch.allclose(w, g, rtol=rtol, atol=atol):
                    raise VolumeRewriteError(
                        f"rewritten program disagrees with the original (max abs {float((w - g).abs().max()):.3e})")
        rw.setup = False
        new = torch.export.export(rw, tuple(example_inputs))
    _prune_unused_getitem(new)
    new.rewrite_stats = dict(rw.stats)
    return new
