# SPDX-License-Identifier: LGPL-3.0-or-later
"""AOT tests for the rank-5 volume rewrite, the custom decompositions and the lowering changes that
carry them (offline tooling only).

Run from the repository root with the pinned venv (put the wheel's `flatc` on PATH when the venv's own
launcher is stale):

    hexecutor/.work/executorch-venv/bin/python -m unittest hexecutor.tests.test_executorch_volume -v

Every numeric case exports the model through `export_program`, requires one delegate that holds the whole
model, runs the HXFG image on the portable CPU reference (`build/bin/hexecutor_tensor_run`) and compares
with PyTorch. The cases skip when the runner is not built or ExecuTorch is absent.
"""

import unittest

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # offline tooling dependency
    torch = None

try:
    from hexecutor.offline.executorch import blob
    from hexecutor.offline.executorch import volume
    from hexecutor.offline.executorch.export import export_program
    from hexecutor.tests.test_executorch_aot import RUNNER, delegate_payloads, ops_of, run_reference
    HAVE = torch is not None
except ImportError:
    HAVE = False

if not HAVE:  # keep the module importable
    nn = type("nn", (), {"Module": object})
    F = None


def check_model(case, model, x, rtol=1e-4, atol=1e-4):
    """Whole model in one delegate; the CPU reference matches torch. Returns the HXFG op kinds."""
    model = model.eval()
    with torch.no_grad():
        want = model(x).numpy()
    delegates = delegate_payloads(export_program(model, (x,)))
    case.assertEqual(len(delegates), 1)
    parsed = blob.unpack(delegates[0][1])
    got = run_reference(parsed, x)
    if want.ndim == 4:  # external output is compact NHWC; compare as NCHW
        _, c, h, w = want.shape
        got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
    np.testing.assert_allclose(got.reshape(want.shape), want, rtol=rtol, atol=atol)
    return ops_of(parsed)


class Volume(nn.Module):
    """A feature map made into a `[1, C, D, H, W]` cost volume (an expand plus a per-depth offset), one 3D
    layer under test, batch norm, ReLU and a 1x1x1 head; the output is `[1, D', H', W']`."""

    def __init__(self, layer, cin=8, depth=6):
        super().__init__()
        self.depth = depth
        self.c0 = nn.Conv2d(3, cin, 3, padding=1)
        self.layer = layer
        cout = layer.out_channels if hasattr(layer, "out_channels") else 6
        self.bn = nn.BatchNorm3d(cout)
        self.bn.running_mean.normal_()
        self.bn.running_var.uniform_(0.5, 2.0)
        self.head = nn.Conv3d(cout, 1, 1)

    def forward(self, x):
        y = F.relu(self.c0(x))
        d = self.depth
        v = y.unsqueeze(2).expand(1, y.shape[1], d, y.shape[2], y.shape[3])
        v = v + torch.arange(d, dtype=torch.float32).view(1, 1, d, 1, 1) * 0.1
        return self.head(F.relu(self.bn(self.layer(v)))).squeeze(1)


@unittest.skipUnless(HAVE and RUNNER.exists(), "ExecuTorch or the portable reference runner is missing")
class Conv3dTests(unittest.TestCase):
    def run_layer(self, layer, shape=(1, 3, 8, 12), cin=8, depth=6):
        torch.manual_seed(3)
        return check_model(self, Volume(layer, cin, depth), torch.randn(*shape))

    def test_spatial_depth_and_dense_kernels(self):
        for name, layer in {
            "spatial 1x3x3": nn.Conv3d(8, 6, (1, 3, 3), padding=(0, 1, 1)),
            "depth 5x1x1": nn.Conv3d(8, 6, (5, 1, 1), padding=(2, 0, 0)),
            "depth 9x1x1": nn.Conv3d(8, 6, (9, 1, 1), padding=(4, 0, 0)),
            "dense 3x3x3": nn.Conv3d(8, 6, 3, padding=1),
            "dilated depth": nn.Conv3d(8, 6, (3, 1, 1), padding=(2, 0, 0), dilation=(2, 1, 1)),
        }.items():
            with self.subTest(name):
                self.run_layer(layer)

    def test_strided_and_grouped(self):
        for name, layer in {
            "stride 2": nn.Conv3d(8, 6, 3, stride=2, padding=1),
            "patch 4x4x4 stride 4 groups 2": nn.Conv3d(8, 6, 4, stride=4, groups=2),
        }.items():
            with self.subTest(name):
                self.run_layer(layer)

    def test_depth_kernel_above_the_hxfg_limit_is_split(self):
        # HXFG kernels are at most 15 rows; a 17-tap depth convolution becomes two partial convolutions.
        torch.manual_seed(4)
        layer = nn.Conv3d(8, 6, (17, 1, 1), padding=(8, 0, 0))
        model = Volume(layer, 8, 20)
        x = torch.randn(1, 3, 6, 8)
        check_model(self, model, x)

    def test_transposed_kernels(self):
        for name, layer in {
            "4x4x4 stride 2": nn.ConvTranspose3d(8, 6, 4, stride=2, padding=1),
            "3x3x3 stride 2 output padding 1": nn.ConvTranspose3d(8, 6, 3, stride=2, padding=1, output_padding=1),
        }.items():
            with self.subTest(name):
                self.run_layer(layer, shape=(1, 3, 4, 6), depth=3)

    def test_rewrite_counts(self):
        torch.manual_seed(5)
        model = Volume(nn.Conv3d(8, 6, 3, padding=1)).eval()
        x = torch.randn(1, 3, 8, 12)
        exported = torch.export.export(model, (x,))
        self.assertTrue(volume.has_high_rank(exported))
        table = torch.export.default_decompositions()
        new = volume.rewrite_volumes(exported, (x,), table)
        self.assertFalse(volume.has_high_rank(new))
        self.assertGreaterEqual(new.rewrite_stats["conv3d"], 2)


class TransposedConv2d(nn.Module):
    def __init__(self, k, stride, pad):
        super().__init__()
        self.c0 = nn.Conv2d(3, 6, 1)
        self.t = nn.ConvTranspose2d(6, 5, k, stride=stride, padding=pad)

    def forward(self, x):
        return F.relu(self.t(self.c0(x)))


class TwoStageConv(nn.Module):
    """A 3x3 convolution of 320 channels: more K tiles than the HMX scratch budget holds."""

    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(3, 320, 1)
        self.c1 = nn.Conv2d(320, 320, 3, padding=1)
        self.bn = nn.BatchNorm2d(320)

    def forward(self, x):
        return F.relu(self.bn(self.c1(F.relu(self.c0(x))))).mean(dim=1, keepdim=True)


@unittest.skipUnless(HAVE and RUNNER.exists(), "ExecuTorch or the portable reference runner is missing")
class SpatialTests(unittest.TestCase):
    def test_transposed_2d_convolution(self):
        torch.manual_seed(6)
        for k, stride, pad in ((4, 2, 1), (3, 2, 1), (2, 2, 0)):
            with self.subTest((k, stride, pad)):
                kinds = check_model(self, TransposedConv2d(k, stride, pad), torch.randn(1, 3, 5, 6))
                self.assertIn(17, kinds)  # ZEROINSERT then a dense convolution

    def test_dense_convolution_over_the_hmx_budget_is_split(self):
        self.assertIsNotNone(volume.conv_chunks(320, 320, 3, 3))
        self.assertIsNone(volume.conv_chunks(28, 28, 3, 3))
        torch.manual_seed(7)
        model = TwoStageConv()
        x = torch.randn(1, 3, 4, 4)
        exported = torch.export.export(model.eval(), (x,))
        self.assertTrue(volume.has_oversized_conv(exported))
        check_model(self, model, x, rtol=2e-4, atol=2e-4)


class Correlation(nn.Module):
    """The group-wise correlation of a stereo pair: shifted copies of the right map stacked along depth,
    viewed as `[1, G, Cg, D, H, W]`, L2-normalised over Cg and summed over Cg."""

    def __init__(self, groups=2, per=4, depth=3):
        super().__init__()
        self.g, self.cg, self.d = groups, per, depth
        self.c0 = nn.Conv2d(3, groups * per, 3, padding=1)
        self.c1 = nn.Conv2d(3, groups * per, 3, padding=1)
        self.head = nn.Conv3d(groups, 1, 1)

    def forward(self, x):
        a, b = F.relu(self.c0(x)), F.relu(self.c1(x))
        w = a.shape[3]
        ref = a.unsqueeze(2).expand(1, a.shape[1], self.d, a.shape[2], w)
        tgt = torch.stack([F.pad(b, (d, 0, 0, 0))[:, :, :, :w] for d in range(self.d)], dim=2)
        ref = ref.view(1, self.g, self.cg, self.d, a.shape[2], w)
        tgt = tgt.view(1, self.g, self.cg, self.d, a.shape[2], w)
        corr = (F.normalize(ref, dim=2) * F.normalize(tgt, dim=2)).sum(dim=2)
        return self.head(corr).squeeze(1)


class TrilinearUp(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Conv3d(4, 1, 1)

    def forward(self, x):
        y = F.relu(self.c0(x))
        v = y.unsqueeze(2).expand(1, 4, 3, y.shape[2], y.shape[3]) + torch.arange(3.0).view(1, 1, 3, 1, 1)
        v = F.interpolate(v, scale_factor=2, mode="trilinear", align_corners=False)
        return self.head(v).squeeze(1)


class ChannelsLast(nn.Module):
    """A ConvNeXt block: depthwise convolution, permute to NHWC, layer norm, two linears around a GELU, a
    per-channel scale and the permute back."""

    def __init__(self, dim=8):
        super().__init__()
        self.c0 = nn.Conv2d(3, dim, 1)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1, self.fc2 = nn.Linear(dim, 3 * dim), nn.Linear(3 * dim, dim)
        self.gamma = nn.Parameter(torch.rand(dim) + 0.5)

    def forward(self, x):
        y = self.dw(self.c0(x))
        z = y.permute(0, 2, 3, 1)
        z = self.gamma * self.fc2(F.gelu(self.fc1(self.norm(z))))
        return y + z.permute(0, 3, 1, 2)


@unittest.skipUnless(HAVE and RUNNER.exists(), "ExecuTorch or the portable reference runner is missing")
class VolumeGraphTests(unittest.TestCase):
    def test_group_correlation_volume(self):
        torch.manual_seed(8)
        check_model(self, Correlation(), torch.randn(1, 3, 6, 8), rtol=2e-4, atol=2e-4)

    def test_trilinear_resize_is_two_index_selects(self):
        torch.manual_seed(9)
        kinds = check_model(self, TrilinearUp(), torch.randn(1, 3, 4, 6))
        self.assertIn(32, kinds)  # HX_GATHER (index_select over depth rows and over pixel columns)

    def test_channels_last_block_runs_as_token_matrices(self):
        torch.manual_seed(10)
        kinds = check_model(self, ChannelsLast(), torch.randn(1, 3, 6, 8), rtol=2e-4, atol=2e-4)
        self.assertIn(1, kinds)


class DecompositionNet(nn.Module):
    def __init__(self, which):
        super().__init__()
        self.which = which
        self.c0 = nn.Conv2d(3, 6, 3, padding=1)

    def forward(self, x):
        y = F.relu(self.c0(x))
        if self.which == "normalize":
            return F.normalize(y, dim=1)
        if self.which == "unfold":  # im2col of a feature map, viewed back as maps
            return F.unfold(y[:, :2], 3, 1, 1).reshape(1, -1, y.shape[2], y.shape[3])
        if self.which == "global_max":
            return F.adaptive_max_pool2d(y, 1) + F.adaptive_avg_pool2d(y, 1)
        if self.which == "select":
            return y[:, :, 2].reshape(1, 6, 1, y.shape[3]) * 2.0
        raise ValueError(self.which)


class GridSample(nn.Module):
    def __init__(self, align, padding):
        super().__init__()
        self.align, self.padding = align, padding
        self.c0 = nn.Conv2d(3, 4, 3, padding=1)
        self.cg = nn.Conv2d(3, 2, 3, padding=1)

    def forward(self, x):
        feat = F.relu(self.c0(x))
        grid = torch.tanh(self.cg(x) * 3.0).permute(0, 2, 3, 1)
        grid = F.interpolate(grid.permute(0, 3, 1, 2), size=(5, 7), mode="bilinear").permute(0, 2, 3, 1)
        return F.grid_sample(feat, grid * 1.2, mode="bilinear", padding_mode=self.padding,
                             align_corners=self.align)


class RowSelect(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c0 = nn.Conv2d(3, c, 1)
        self.register_buffer("rows", torch.tensor([0, 0, 1, 3, 5, 5, 2]))

    def forward(self, x):
        return F.relu(torch.index_select(self.c0(x), 2, self.rows))


class ViewKeepsChannels(nn.Module):
    """[1, C, H, W] -> [1, C, 1, H*W] -> [1, C, H, W] keeps each channel vector: one pixel-order copy, even
    when the element count is above the 4096-element NCHW flatten limit."""

    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(3, 6, 3, padding=1)
        self.c1 = nn.Conv2d(6, 4, 1)

    def forward(self, x):
        y = F.relu(self.c0(x))
        flat = y.reshape(1, 6, 1, -1)
        return self.c1(torch.relu(flat + 1.0).reshape(y.shape))


class KeyTransposeAttention(nn.Module):
    """Attention with a batch above one and the key transposed by a permute that feeds the bmm."""

    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8)

    def forward(self, x):
        t = x.reshape(6, 4, 8)  # a batch of six four-token rows
        q, k, v = self.q(t), self.k(t), self.v(t)
        att = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * 0.35, dim=-1)
        return torch.bmm(att, v).reshape(1, 24, 8)


@unittest.skipUnless(HAVE and RUNNER.exists(), "ExecuTorch or the portable reference runner is missing")
class DecompositionTests(unittest.TestCase):
    def test_vector_norm_unfold_global_max_and_select(self):
        torch.manual_seed(11)
        for which in ("normalize", "unfold", "global_max", "select"):
            with self.subTest(which):
                check_model(self, DecompositionNet(which), torch.randn(1, 3, 5, 6), rtol=2e-4, atol=2e-4)

    def test_grid_sample_bilinear_batch_one(self):
        torch.manual_seed(12)
        for align in (True, False):
            for padding in ("zeros", "border"):
                with self.subTest((align, padding)):
                    check_model(self, GridSample(align, padding), torch.randn(1, 3, 6, 8), rtol=2e-4, atol=2e-4)


class StemModel(nn.Module):
    """`examples/fastfoundationstereo` VolumeStem on a stereo pair of feature maps, then a 3D convolution."""

    def __init__(self, stem, cin=8, groups=2):
        super().__init__()
        self.c0 = nn.Conv2d(3, cin, 3, padding=1)
        self.stem = stem
        self.head = nn.Conv3d(groups + 2 * stem.proj.out_channels, 1, 3, padding=1)

    def forward(self, x):
        ref = F.relu(self.c0(x))
        tgt = torch.roll(ref, 1, 3) + 0.1
        return self.head(self.stem(ref, tgt)).squeeze(1)


def _load_ffs_export():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "examples" / "fastfoundationstereo" / "ffs_export.py"
    spec = importlib.util.spec_from_file_location("ffs_export_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upstream_volumes(ref, tgt, proj, depth, groups):
    """Upstream's `_build_gwc_volume_onnx` + `_build_concat_volume_onnx` + channel concatenation."""
    b, c, h, w = ref.shape
    cg = c // groups
    shifted = torch.stack([F.pad(tgt, (d, 0, 0, 0))[:, :, :, :w] for d in range(depth)], dim=2)
    rv = ref.unsqueeze(2).expand(b, c, depth, h, w).reshape(b, groups, cg, depth, h, w)
    tv = shifted.reshape(b, groups, cg, depth, h, w)
    gwc = (F.normalize(rv, dim=2) * F.normalize(tv, dim=2)).sum(dim=2)
    left, right = proj(ref), proj(tgt)
    shifted_r = torch.stack([F.pad(right, (d, 0, 0, 0))[:, :, :, :w] for d in range(depth)], dim=2)
    concat = torch.cat((left.unsqueeze(2).expand(b, left.shape[1], depth, h, w), shifted_r), dim=1)
    return torch.cat([gwc, concat], dim=1)


@unittest.skipUnless(HAVE, "ExecuTorch is missing")
class MemoryTests(unittest.TestCase):
    """The pieces that keep a cost volume's live activations small (no runner needed)."""

    def test_trilinear_decomposition_matches_interpolate(self):
        from hexecutor.offline.executorch import decompositions
        torch.manual_seed(20)
        x = torch.randn(1, 3, 2, 3, 5)
        for kwargs in ({"scale_factor": 4}, {"scale_factor": 2}, {"size": (5, 7, 9)},
                       {"scale_factor": 3, "align_corners": True}, {"size": (4, 6, 11), "align_corners": True}):
            with self.subTest(kwargs):
                align = kwargs.get("align_corners", False)
                want = F.interpolate(x, mode="trilinear", align_corners=align,
                                     **{k: v for k, v in kwargs.items() if k != "align_corners"})
                if "size" in kwargs:
                    got = decompositions._upsample_trilinear3d(x, list(kwargs["size"]), align, None)
                else:
                    got = decompositions._upsample_trilinear3d(x, None, align, [float(kwargs["scale_factor"])] * 3)
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def test_per_disparity_builder_equals_the_upstream_volumes(self):
        module = _load_ffs_export()
        torch.manual_seed(21)
        groups, depth = 4, 3
        model = type("M", (), {})()
        model.cv_group = groups
        model.proj_cmb = nn.Conv2d(24, 6, 1)
        stem = module.VolumeStem(model, depth * 4)
        ref, tgt = torch.randn(1, 24, 5, 7), torch.randn(1, 24, 5, 7)
        got = stem(ref, tgt)
        want = upstream_volumes(ref, tgt, model.proj_cmb, depth, groups)
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def test_demand_order_runs_each_gather_before_its_reader(self):
        from hexecutor.offline import hxfg_writer as hx
        from hexecutor.offline.executorch import lowering
        none = hx.NONE
        # four gathers of one input, then the four activations that read them, summed pairwise
        ops = [[hx.GATHER, 0, none, none, g] + [0] * 11 for g in (1, 2, 3, 4)]
        ops += [[hx.ACT, g, none, none, g + 4] + [0] * 11 for g in (1, 2, 3, 4)]
        ops += [[hx.ADD, 5, 6, none, 9] + [0] * 11, [hx.ADD, 7, 8, none, 10] + [0] * 11, [hx.ADD, 9, 10, none, 11] + [0] * 11]
        size = lambda i: 100
        new = lowering._demand_order(ops, [11], size)
        self.assertEqual(sorted(map(id, new)), sorted(map(id, ops)))
        position = {op[4]: n for n, op in enumerate(new)}
        for op in new:
            for t in lowering._op_inputs(op):
                if t in position:
                    self.assertLess(position[t], position[op[4]])
        self.assertLess(lowering._live_peak(new, size, [11]), lowering._live_peak(ops, size, [11]))
        # state and in-place operations keep the emitted order
        self.assertIs(lowering._demand_order(ops, [11], size, persistent=True), ops)
        with_kv = ops + [[hx.KV_WRITE, 0, none, none, 12] + [0] * 11]
        self.assertIs(lowering._demand_order(with_kv, [11], size), with_kv)


@unittest.skipUnless(HAVE and RUNNER.exists(), "ExecuTorch or the portable reference runner is missing")
class ChunkedVolumeTests(unittest.TestCase):
    def setUp(self):
        self.saved = volume.MAX_TEMP_BYTES

    def tearDown(self):
        volume.MAX_TEMP_BYTES = self.saved

    def run_layer(self, layer, shape=(1, 3, 8, 12), depth=6):
        torch.manual_seed(3)
        return check_model(self, Volume(layer, 8, depth), torch.randn(*shape))

    def test_column_chunks_match_the_unchunked_convolution(self):
        for name, layer, shape, depth, budget in (
            ("dense 3x3x3", nn.Conv3d(8, 6, 3, padding=1), (1, 3, 8, 12), 6, 16384),
            ("stride 2", nn.Conv3d(8, 6, 3, stride=2, padding=1), (1, 3, 8, 12), 6, 16384),
            ("transposed 4x4x4", nn.ConvTranspose3d(8, 6, 4, stride=2, padding=1), (1, 3, 4, 6), 3, 32768),
        ):
            with self.subTest(name):
                volume.MAX_TEMP_BYTES = 0
                whole = self.run_layer(layer, shape, depth).count(32)
                volume.MAX_TEMP_BYTES = budget  # a handful of chunks (the default runner caps a graph at 1024 operations)
                chunked = self.run_layer(layer, shape, depth).count(32)
                self.assertGreater(chunked, whole)

    def test_volume_filling_the_column_limit_needs_no_padded_copy(self):
        # 8 x 12 = 96 pixels: with the limit at 96 a whole-volume copy with an appended zero column would be 97
        # columns wide, so each column chunk reads only its own range plus the zero column (and a chunk is split further when its
        # source range alone would not fit one tensor).
        saved = volume.MAX_COLUMNS
        volume.MAX_COLUMNS = 96
        volume.MAX_TEMP_BYTES = 16384
        try:
            for name, layer in (("dense 3x3x3", nn.Conv3d(8, 6, 3, padding=1)),
                                ("stride 2", nn.Conv3d(8, 6, 3, stride=2, padding=1))):
                with self.subTest(name):
                    self.run_layer(layer)
            volume.MAX_TEMP_BYTES = 0  # no byte budget: chunks are still split until the source range fits
            self.run_layer(nn.Conv3d(8, 6, 3, padding=1))
        finally:
            volume.MAX_COLUMNS = saved

    def test_per_disparity_builder_lowers_and_matches(self):
        module = _load_ffs_export()
        torch.manual_seed(22)
        model = type("M", (), {})()
        model.cv_group = 2
        model.proj_cmb = nn.Conv2d(8, 4, 1)
        check_model(self, StemModel(module.VolumeStem(model, 12), 8, 2), torch.randn(1, 3, 6, 8), rtol=2e-4, atol=2e-4)


@unittest.skipUnless(HAVE and RUNNER.exists(), "ExecuTorch or the portable reference runner is missing")
class LoweringTests(unittest.TestCase):
    def test_index_select_along_rows_reads_pixels(self):
        torch.manual_seed(13)
        for c in (4, 28, 32):  # padded and unpadded channel strides
            with self.subTest(c):
                kinds = check_model(self, RowSelect(c), torch.randn(1, 3, 6, 5))
                self.assertIn(32, kinds)

    def test_channel_preserving_view_is_a_pixel_copy(self):
        torch.manual_seed(14)
        kinds = check_model(self, ViewKeepsChannels(), torch.randn(1, 3, 30, 30))  # 6 * 900 > 4096 elements
        self.assertIn(8, kinds)

    def test_key_transpose_of_a_batched_attention(self):
        torch.manual_seed(15)
        kinds = check_model(self, KeyTransposeAttention(), torch.randn(1, 24, 8), rtol=2e-4, atol=2e-4)
        self.assertIn(21, kinds)  # HX_MATMUL


if __name__ == "__main__":
    unittest.main()
