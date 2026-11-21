# SPDX-License-Identifier: LGPL-3.0-or-later
"""Offline decompositions of ATen operations that HXFG has no kind for, into operations it has.

`export_program` applies these (through `custom_table`) when a program uses one of the operations in
`CUSTOM_OPS`; everything else is untouched. They are exact up to float re-association.

* `linalg_vector_norm` (order 2 or 1) -> sqrt(sum(x * x)) / sum(|x|). `F.normalize` and `torch.norm`
  decompose to it.
* `grid_sampler_2d` (batch one, bilinear, zeros/border padding) -> four `index` pixel gathers plus weights.
* `select.int` -> `narrow` (a slice) plus `squeeze`, which HXFG lowers as a slice and a reshape.
* `adaptive_max_pool2d` to 1x1 -> `amax` over the spatial dims.
* `im2col` (`F.unfold`, batch one) -> one dense convolution whose weights are one-hot taps, so the
  patch matrix is read as `[1, C*kh*kw, oh, ow]` without index gathers.
"""

import torch
import torch.nn.functional as F

aten = torch.ops.aten


def _vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        x = x.to(dtype)
    if dim is None:
        dim = list(range(x.dim()))
    dim = [dim] if isinstance(dim, int) else list(dim)
    if ord == 2:
        return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim))
    if ord == 1:
        return torch.sum(torch.abs(x), dim=dim, keepdim=keepdim)
    raise NotImplementedError(f"vector norm of order {ord}")


def _im2col(x, kernel_size, dilation, padding, stride):
    n, c, h, w = x.shape
    kh, kw = kernel_size
    if n != 1:
        raise NotImplementedError("im2col needs batch one")
    # one-hot taps: the flat patch index (channel, ky, kx) is exactly the identity matrix
    weight = torch.eye(c * kh * kw).reshape(c * kh * kw, c, kh, kw)
    y = F.conv2d(x, weight, None, stride=list(stride), padding=list(padding), dilation=list(dilation))
    return y.reshape(1, c * kh * kw, -1)


def _grid_sampler_2d(inp, grid, interpolation_mode, padding_mode, align_corners):
    """Bilinear `grid_sample` (zeros or border padding, batch one) as four pixel gathers weighted by the
    interpolation weights, which are zero for corners outside the image."""
    n, c, h, w = inp.shape
    _, ho, wo, _ = grid.shape
    if n != 1 or interpolation_mode != 0 or padding_mode not in (0, 1):
        raise NotImplementedError("grid_sample: batch one, bilinear, zeros/border padding only")
    g = grid.permute(0, 3, 1, 2)                    # [1, 2, Ho, Wo] maps
    gx, gy = g[:, 0:1], g[:, 1:2]
    if align_corners:
        ix, iy = (gx + 1) * ((w - 1) / 2), (gy + 1) * ((h - 1) / 2)
    else:
        ix, iy = ((gx + 1) * w - 1) / 2, ((gy + 1) * h - 1) / 2
    if padding_mode == 1:
        ix, iy = ix.clamp(0, w - 1), iy.clamp(0, h - 1)
    x0, y0 = torch.floor(ix), torch.floor(iy)
    fx, fy = ix - x0, iy - y0
    out = None
    for yc, wy in ((y0, 1 - fy), (y0 + 1, fy)):
        for xc, wx in ((x0, 1 - fx), (x0 + 1, fx)):
            inside = (xc + 1).clamp(0, 1) * (w - xc).clamp(0, 1) * (yc + 1).clamp(0, 1) * (h - yc).clamp(0, 1)
            yi = yc.clamp(0, h - 1).to(torch.int64).reshape(-1)
            xi = xc.clamp(0, w - 1).to(torch.int64).reshape(-1)
            part = inp[:, :, yi, xi] * (wx * wy * inside).reshape(1, 1, -1)   # [1, C, Ho*Wo]
            out = part if out is None else out + part
    return out.transpose(1, 2).transpose(1, 2).reshape(1, c, ho, wo)


def _upsample_trilinear3d(x, output_size, align_corners, scale_factors):
    """Trilinear resize of `[1, C, D, H, W]` as three separable linear interpolations (W, then H, then D): per axis two
    constant-index gathers and a weighted sum. Upstream's decomposition gathers all eight corners at the full
    output size at once (eight live volumes); this keeps one axis (at most two volumes) live."""
    for axis in (4, 3, 2):
        size = x.shape[axis]
        if output_size is not None:
            out = int(output_size[axis - 2])
            scale = None if not scale_factors else float(scale_factors[axis - 2])
        else:
            scale = float(scale_factors[axis - 2])
            out = int(size * scale)
        dst = torch.arange(out, dtype=torch.float32)
        if align_corners:
            src = dst * (((size - 1) / (out - 1)) if out > 1 else 0.0)
        else:
            ratio = (1.0 / scale) if (scale_factors and scale) else size / out
            src = ((dst + 0.5) * ratio - 0.5).clamp_min(0.0)
        i0 = torch.floor(src).clamp(0, size - 1).to(torch.int64)
        i1 = (i0 + 1).clamp(0, size - 1)
        shape = [1] * 5
        shape[axis] = out
        wt = (src - i0.to(torch.float32)).reshape(shape).to(x.dtype)
        a = torch.index_select(x, axis, i0)
        b = torch.index_select(x, axis, i1)
        x = a * (1 - wt) + b * wt
    return x


def _select(x, dim, index):
    index = index + x.shape[dim] if index < 0 else index
    return torch.squeeze(torch.narrow(x, dim, index, 1), dim)


def _adaptive_max_pool2d(x, output_size):
    if list(output_size) != [1, 1]:
        raise NotImplementedError("adaptive max pooling to a size other than 1x1")
    out = torch.amax(x, dim=[2, 3], keepdim=True)
    return out, torch.zeros_like(out, dtype=torch.int64)  # the indices output is unused


CUSTOM = {
    aten.grid_sampler_2d.default: _grid_sampler_2d,
    aten.select.int: _select,
    aten.upsample_trilinear3d.vec: _upsample_trilinear3d,
    aten.adaptive_max_pool2d.default: _adaptive_max_pool2d,
    aten.linalg_vector_norm.default: _vector_norm,
    aten.im2col.default: _im2col,
}
CUSTOM_OPS = tuple(CUSTOM)


def uses_custom_ops(ep):
    """True when the program calls an operation that `custom_table` rewrites (before decomposition)."""
    entry = set(CUSTOM) | {aten.grid_sampler.default}  # F.grid_sample exports as grid_sampler (-> _2d on decomposition)
    return any(node.op == "call_function" and node.target in entry for node in ep.graph.nodes)


def custom_table(base):
    """`base` (a decomposition table) with the custom entries added."""
    table = dict(base)
    table.update(CUSTOM)
    return table
