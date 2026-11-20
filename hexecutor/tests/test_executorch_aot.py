# SPDX-License-Identifier: LGPL-3.0-or-later
"""AOT tests for the ExecuTorch hexecutor delegate (offline tooling only).

Run from the repository root with the pinned venv:

    hexecutor/.work/executorch-venv/bin/python -m unittest hexecutor.tests.test_executorch_aot -v

Blob and HXFG structure checks run without ExecuTorch. The export tests need
the pinned executorch package and skip when it is absent. The numeric check
runs the portable CPU reference (build/bin/hexecutor_tensor_run) only when that
binary exists.
"""

import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # offline tooling dependency
    torch = None

try:
    from hexecutor.offline.executorch import blob
    from hexecutor.offline.executorch import lowering
except ImportError:  # the package needs numpy; torch is optional for blob tests
    blob = None
    lowering = None

try:
    from hexecutor.offline.executorch.export import export_program
    from hexecutor.offline.executorch.partitioner import HexecutorPartitioner
    HAVE_EXECUTORCH = True
except ImportError:
    HAVE_EXECUTORCH = False

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(os.environ.get("HEXECUTOR_TENSOR_RUN", ROOT / "build/bin/hexecutor_tensor_run"))


class Tiny(nn.Module):
    """conv3x3 + ReLU, multiplier-one depthwise, add, sigmoid (1x3x32x32 input)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)

    def forward(self, x):
        y = torch.relu(self.conv(x))
        return torch.sigmoid(y + self.dw(y))


class WideOps(nn.Module):
    """Strided conv, max pool, PReLU, concat, pad, bilinear resize, mean, linear, sigmoid."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1)
        self.prelu = nn.PReLU(16)
        self.lin = nn.Linear(32, 10)

    def forward(self, x):
        y = F.max_pool2d(self.c1(x), 3, 2, 1)
        y = self.prelu(self.c2(y))
        z = F.pad(torch.cat([y, y], dim=1), (1, 1, 1, 1))
        z = F.interpolate(z, size=(20, 20), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.lin(z.mean(dim=(2, 3))))


class ModernOps(nn.Module):
    """Squeeze-excite gate, hardswish/SiLU/GELU/tanh/leaky ReLU, average pools, constant and
    broadcast arithmetic, and a ReLU that has no producer to fuse into."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.se = nn.Conv2d(16, 16, 1)
        self.register_buffer("shift", torch.linspace(-1, 1, 16).reshape(1, 16, 1, 1))

    def forward(self, x):
        y = F.hardswish(self.c1(x))
        y = F.silu(self.c2(y))
        gate = F.hardsigmoid(self.se(y.mean(dim=(2, 3), keepdim=True)))
        y = y * gate                      # 1x1 broadcast over the feature map
        y = y - self.shift                # constant per channel
        y = 2.0 * y + 0.5                 # scalar mul and add
        z = F.avg_pool2d(y, 3, 2, 1, count_include_pad=False)
        z = F.relu(F.leaky_relu(F.gelu(z), 0.1))   # leaky and relu stay standalone
        w = F.avg_pool2d(torch.tanh(y), 2, 2)
        return z - w                      # tensor - tensor, same shape


class CumsumOnly(nn.Module):  # cumsum is not in the HXFG operation set
    def forward(self, x):
        return torch.cumsum(x, dim=1)


class NearestUp(nn.Module):
    """conv3x3 + ReLU, then nearest resize to a non-integer 5x7 (sizes given, no scales)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return F.interpolate(F.relu(self.conv(x)), size=(5, 7), mode="nearest")


class ConvSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return torch.softmax(torch.relu(self.conv(x)), dim=1)


class ConvCumsumTail(nn.Module):  # unsupported tail: stays on the CPU
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return torch.cumsum(torch.relu(self.conv(x)), dim=1)


class ConvTransposeUp(nn.Module):
    """Transposed convolution stride 2 (5x5 -> 10x10) with a fused ReLU."""

    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose2d(4, 6, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        return F.relu(self.up(x))


class PermuteFlatten(nn.Module):
    """conv + ReLU, permute to NHWC, a 1x1 convolution, a flatten of its NCHW output, then a 1x1
    convolution standing in for a linear layer (nn.Linear exports as addmm, which is not
    delegated). A view that directly follows a permute gets a dim-order clone at the delegate
    boundary, so the permute is consumed by a convolution inside the delegate instead."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pw = nn.Conv2d(16, 8, 1)
        self.fc = nn.Conv2d(1024, 10, 1)

    def forward(self, x):
        y = self.pw(F.relu(self.conv(x)).permute(0, 2, 3, 1))  # [1, 8, 16, 8]
        return self.fc(y.reshape(1, 1024, 1, 1))


class ViewMix(nn.Module):
    """conv + ReLU, then a view that moves spatial extent into channels, then a 1x1 convolution."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pw = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        return self.pw(F.relu(self.conv(x)).reshape(1, 16, 8, 16))


class PermuteOut(nn.Module):
    """conv + ReLU, then a permute whose 4-D result is the delegate output."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x)).permute(0, 2, 3, 1)


def ops_of(parsed):
    """Operation kinds of a parsed HXFG image, in order."""
    count = struct.unpack_from("<I", parsed.image, 16)[0]
    tensors = struct.unpack_from("<I", parsed.image, 12)[0]
    return [struct.unpack_from("<I", parsed.image, 256 + 32 * tensors + 64 * i)[0]
            for i in range(count)]


def delegate_payloads(program):
    """(delegate id, processed bytes) for every delegate in the program."""
    prog = program.executorch_program
    out = []
    for plan in prog.execution_plan:
        for delegate in plan.delegates:
            data = prog.backend_delegate_data[delegate.processed.index].data
            out.append((delegate.id, bytes(data)))
    return out


@unittest.skipIf(blob is None, "numpy-backed HXFG tooling unavailable")
class BlobFormatTests(unittest.TestCase):
    def test_round_trip_layout_and_lengths(self):
        image = b"H" * 300
        consts = b"c" * 17
        ins = [blob.TensorSpec.for_shape((1, 3, 32, 32))]
        outs = [blob.TensorSpec.for_shape((1, 8, 32, 32)), blob.TensorSpec.for_shape((1, 10))]
        data = blob.pack(image, consts, ins, outs)
        self.assertEqual(data[:4], b"HXET")
        self.assertEqual(struct.unpack_from("<4I", data, 4), (1, 1, 1, 2))
        parsed = blob.unpack(data)
        self.assertEqual(parsed.inputs[0], blob.TensorSpec(1, (1, 3, 32, 32)))
        self.assertEqual(parsed.outputs[0].layout, 1)
        self.assertEqual(parsed.outputs[1], blob.TensorSpec(0, (1, 10)))
        self.assertEqual((parsed.image, parsed.constants), (image, consts))

    def test_rejects_truncation_and_bad_magic(self):
        data = blob.pack(b"x", b"", [blob.TensorSpec(1, (1, 3, 4, 4))], [blob.TensorSpec(1, (1, 1, 4, 4))])
        with self.assertRaises(ValueError):
            blob.unpack(data[:-1])
        with self.assertRaises(ValueError):
            blob.unpack(b"NOPE" + data[4:])


@unittest.skipIf(not HAVE_EXECUTORCH or torch is None, "executorch/torch not installed")
class ExportTests(unittest.TestCase):
    def test_tiny_model_is_delegated_with_valid_blob(self):
        x = torch.randn(1, 3, 32, 32)
        program = export_program(Tiny(), (x,))
        delegates = delegate_payloads(program)
        self.assertEqual(len(delegates), 1)
        name, payload = delegates[0]
        self.assertEqual(name, "HexecutorBackend")
        parsed = blob.unpack(payload)
        self.assertEqual((parsed.version, parsed.backend), (1, blob.BACKEND_TENSOR_F32))
        self.assertEqual([s.dims for s in parsed.inputs], [(1, 3, 32, 32)])
        self.assertEqual([s.dims for s in parsed.outputs], [(1, 8, 32, 32)])
        self.assertEqual(parsed.inputs[0].layout, blob.LAYOUT_NCHW)
        # The HXFG image is structurally valid and uses compact RGB input storage.
        lowering.check_image(parsed.image, parsed.constants)
        magic, _, size, tensors, ops, n_in, n_out = struct.unpack_from("<7I", parsed.image)
        self.assertEqual((magic, size), (0x47465848, len(parsed.image)))
        self.assertEqual((n_in, n_out), (1, 1))
        self.assertEqual(ops, 4)  # conv(+relu), depthwise, add, logistic

    def test_unsupported_operation_is_left_on_cpu(self):
        program = export_program(CumsumOnly(), (torch.randn(1, 4, 8, 8),))
        self.assertEqual(delegate_payloads(program), [])

    def test_conv_softmax_is_fully_delegated(self):
        delegates = delegate_payloads(export_program(ConvSoftmax(), (torch.randn(1, 3, 16, 16),)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        self.assertEqual([s.dims for s in parsed.outputs], [(1, 8, 16, 16)])
        kinds = struct.unpack_from("<I", parsed.image, 16)[0]
        ops = [struct.unpack_from("<I", parsed.image, 256 + 32 * struct.unpack_from("<I", parsed.image, 12)[0] + 64 * i)[0]
               for i in range(kinds)]
        self.assertEqual(ops.count(lowering.hx.SOFTMAX), 1)

    def test_unsupported_tail_stays_outside_delegate(self):
        program = export_program(ConvCumsumTail(), (torch.randn(1, 3, 16, 16),))
        delegates = delegate_payloads(program)
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        # The cumsum runs on the CPU, so the delegate returns the conv+ReLU tensor.
        self.assertEqual([s.dims for s in parsed.outputs], [(1, 8, 16, 16)])

    def test_bmm_is_fully_delegated(self):
        a = torch.randn(2, 4, 8)
        b_arg = torch.randn(2, 8, 6)
        c = torch.randn(2, 6, 5)
        delegates = delegate_payloads(export_program(BmmModel().eval(), (a, b_arg, c)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        self.assertEqual([s.dims for s in parsed.outputs], [(2, 4, 5)])
        kinds = struct.unpack_from("<I", parsed.image, 16)[0]
        tensors = struct.unpack_from("<I", parsed.image, 12)[0]
        ops = [struct.unpack_from("<I", parsed.image, 256 + 32 * tensors + 64 * i)[0]
               for i in range(kinds)]
        self.assertEqual(ops.count(lowering.hx.MATMUL), 2)

    def test_sdpa_is_fully_delegated(self):
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        delegates = delegate_payloads(export_program(SdpaModel(is_causal=True).eval(), (q, k, v)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        self.assertEqual([s.dims for s in parsed.outputs], [(1, 2, 4, 8)])
        kinds = struct.unpack_from("<I", parsed.image, 16)[0]
        tensors = struct.unpack_from("<I", parsed.image, 12)[0]
        ops = [struct.unpack_from("<I", parsed.image, 256 + 32 * tensors + 64 * i)[0]
               for i in range(kinds)]
        self.assertEqual(ops.count(lowering.hx.SDPA), 1)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_conv_softmax_matches_torch_on_cpu_reference(self):
        torch.manual_seed(3)
        model = ConvSoftmax().eval()
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            want = model(x).numpy()
        parsed = blob.unpack(delegate_payloads(export_program(model, (x,)))[0][1])
        got = run_reference(parsed, x)
        _, c, h, w = want.shape
        got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_hxfg_matches_torch_on_cpu_reference(self):
        for model, x in [(Tiny(), torch.randn(1, 3, 32, 32)), (WideOps(), torch.randn(1, 3, 32, 32))]:
            model.eval()
            with torch.no_grad():
                want = model(x).numpy()
            parsed = blob.unpack(delegate_payloads(export_program(model, (x,)))[0][1])
            got = run_reference(parsed, x)
            self.assertEqual(got.size, want.size)
            if want.ndim == 4:  # external output is compact NHWC; compare as NCHW
                _, c, h, w = want.shape
                got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
            np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-3, atol=1e-3)

    def test_modern_ops_model_is_fully_delegated(self):
        model = ModernOps().eval()
        x = torch.randn(1, 3, 16, 16)
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        kinds = struct.unpack_from("<I", parsed.image, 16)[0]
        # Nothing may fall back to the CPU: the delegate is the whole graph.
        self.assertEqual(kinds, 18)
        self.assertEqual([s.dims for s in parsed.outputs], [(1, 16, 8, 8)])
        ops = [struct.unpack_from("<I", parsed.image, 256 + 32 * struct.unpack_from("<I", parsed.image, 12)[0] + 64 * i)[0]
               for i in range(kinds)]
        for kind in (lowering.hx.ACT, lowering.hx.BINARY, lowering.hx.AVGPOOL):
            self.assertIn(kind, ops)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_modern_ops_match_torch_on_cpu_reference(self):
        torch.manual_seed(1)
        model = ModernOps().eval()
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            want = model(x).numpy()
        parsed = blob.unpack(delegate_payloads(export_program(model, (x,)))[0][1])
        got = run_reference(parsed, x)
        _, c, h, w = want.shape
        got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_nearest_resize_matches_torch_on_cpu_reference(self):
        torch.manual_seed(2)
        model = NearestUp().eval()
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            want = model(x).numpy()
        parsed = blob.unpack(delegate_payloads(export_program(model, (x,)))[0][1])
        got = run_reference(parsed, x)
        _, c, h, w = want.shape
        got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    def test_unsupported_broadcasts_stay_on_cpu(self):
        class Outer(nn.Module):
            def forward(self, x):  # [1,C,H,1] * [1,C,1,W] broadcasts both operands
                a = F.silu(x[:, :, :, :1] * 1.0)
                return a * F.silu(x[:, :, :1, :])
        # slicing is unsupported, so nothing reaches the delegate; the call must still export.
        export_program(Outer(), (torch.randn(1, 4, 8, 8),))

    def test_wide_op_model_is_fully_delegated(self):
        model = WideOps().eval()
        x = torch.randn(1, 3, 32, 32)
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        self.assertEqual([s.dims for s in parsed.outputs], [(1, 10)])
        # conv, maxpool, conv, prelu, concat, pad, resize, mean, linear, logistic
        self.assertEqual(struct.unpack_from("<I", parsed.image, 16)[0], 10)

    def test_conv_transpose_is_fully_delegated(self):
        model = ConvTransposeUp().eval()
        delegates = delegate_payloads(export_program(model, (torch.randn(1, 4, 5, 5),)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        self.assertEqual([s.dims for s in parsed.outputs], [(1, 6, 10, 10)])
        kinds = struct.unpack_from("<I", parsed.image, 16)[0]
        ops = [struct.unpack_from("<I", parsed.image, 256 + 32 * struct.unpack_from("<I", parsed.image, 12)[0] + 64 * i)[0]
               for i in range(kinds)]
        # zero insertion, then a dense convolution; the ReLU is fused into the convolution
        self.assertEqual(ops, [lowering.hx.ZEROINSERT, lowering.hx.CONV])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_conv_transpose_matches_torch_on_cpu_reference(self):
        torch.manual_seed(4)
        for stride_pad, k in ((dict(stride=2, padding=1, output_padding=1), 3),
                              (dict(stride=2, padding=1), 4)):
            with self.subTest(k=k, **stride_pad):
                model = nn.Sequential(nn.ConvTranspose2d(4, 6, k, **stride_pad), nn.ReLU()).eval()
                x = torch.randn(1, 4, 5, 5)
                with torch.no_grad():
                    want = model(x).numpy()
                parsed = blob.unpack(delegate_payloads(export_program(model, (x,)))[0][1])
                got = run_reference(parsed, x)
                _, c, h, w = want.shape
                got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_dilated_convolution_matches_torch_on_cpu_reference(self):
        torch.manual_seed(5)
        cases = [
            (nn.Conv2d(3, 6, 3, dilation=2, padding=2), (1, 3, 10, 10)),
            (nn.Conv2d(3, 6, 3, dilation=(1, 3), stride=2, padding=(1, 3)), (1, 3, 11, 11)),
            (nn.Conv2d(4, 4, 3, dilation=2, padding=2, groups=4), (1, 4, 8, 8)),  # depthwise
        ]
        for conv, shape in cases:
            with self.subTest(conv=str(conv)):
                model = nn.Sequential(conv, nn.ReLU()).eval()
                x = torch.randn(*shape)
                with torch.no_grad():
                    want = model(x).numpy()
                delegates = delegate_payloads(export_program(model, (x,)))
                self.assertEqual(len(delegates), 1)
                parsed = blob.unpack(delegates[0][1])
                got = run_reference(parsed, x)
                _, c, h, w = want.shape
                got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_permute_then_flatten_matches_torch_on_cpu_reference(self):
        torch.manual_seed(5)
        model = PermuteFlatten().eval()
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            want = model(x).numpy()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        kinds = ops_of(parsed)
        self.assertIn(lowering.hx.TRANSPOSE, kinds)  # permute: nothing falls back to the CPU
        self.assertIn(lowering.hx.RESHAPE, kinds)
        got = run_reference(parsed, x)
        np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-3, atol=1e-3)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_view_mixing_spatial_and_channel_order_matches_torch_on_cpu_reference(self):
        torch.manual_seed(6)
        model = ViewMix().eval()
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            want = model(x).numpy()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        self.assertEqual(ops_of(parsed).count(lowering.hx.RESHAPE), 2)  # flatten, unflatten
        got = run_reference(parsed, x)
        _, c, h, w = want.shape
        got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_permute_output_matches_torch_on_cpu_reference(self):
        torch.manual_seed(7)
        model = PermuteOut().eval()
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            want = model(x).numpy()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        self.assertEqual(ops_of(parsed)[-1], lowering.hx.TRANSPOSE)
        got = run_reference(parsed, x)
        _, c, h, w = want.shape  # the permuted output is (1, 16, 16, 8): C=16, H=16, W=8
        got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_reductions_match_torch_on_cpu_reference(self):
        torch.manual_seed(6)
        for mode in ("channel_max", "channel_mean", "sum_hw", "min_hw"):
            with self.subTest(mode=mode):
                model = Reduction(mode).eval()
                x = torch.randn(1, 3, 8, 8)
                with torch.no_grad():
                    want = model(x).numpy()
                delegates = delegate_payloads(export_program(model, (x,)))
                self.assertEqual(len(delegates), 1)
                parsed = blob.unpack(delegates[0][1])
                kinds = struct.unpack_from("<I", parsed.image, 16)[0]
                ops = [struct.unpack_from("<I", parsed.image,
                                          256 + 32 * struct.unpack_from("<I", parsed.image, 12)[0]
                                          + 64 * i)[0] for i in range(kinds)]
                self.assertEqual(ops[-1], lowering.hx.REDUCE, ops)
                got = run_reference(parsed, x)
                if want.ndim == 4:  # external output is compact NHWC; compare as NCHW
                    _, c, h, w = want.shape
                    got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
                np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-4, atol=1e-4)

    def _norm_model_matches_torch(self, model, x):
        """Fully delegated, one HX_NORM, and the portable reference matches torch."""
        torch.manual_seed(7)
        with torch.no_grad():
            want = model(x).numpy()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        kinds = struct.unpack_from("<I", parsed.image, 16)[0]
        ops = [struct.unpack_from("<I", parsed.image,
                                  256 + 32 * struct.unpack_from("<I", parsed.image, 12)[0]
                                  + 64 * i)[0] for i in range(kinds)]
        self.assertIn(lowering.hx.NORM, ops)
        got = run_reference(parsed, x)
        if want.ndim == 4:  # external output is compact NHWC; compare as NCHW
            _, c, h, w = want.shape
            got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_layer_norm_over_chw_matches_torch_on_cpu_reference(self):
        model = NormNet("layer_chw").eval()
        self._norm_model_matches_torch(model, torch.randn(1, 3, 8, 8))

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_group_norm_matches_torch_on_cpu_reference(self):
        # One group (LayerNorm over C, H, W), two groups, and one group per channel.
        for groups in (1, 2, 8):
            with self.subTest(groups=groups):
                model = NormNet("group", groups=groups).eval()
                self._norm_model_matches_torch(model, torch.randn(1, 3, 8, 8))

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_layer_norm_over_channels_matches_torch_on_cpu_reference(self):
        model = NormNet("layer_channels").eval()
        self._norm_model_matches_torch(model, torch.randn(1, 16, 1, 1))

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_instance_norm_matches_torch_on_cpu_reference(self):
        for kind in ("instance_affine", "instance_noaffine"):
            with self.subTest(kind=kind):
                model = NormNet(kind).eval()
                self._norm_model_matches_torch(model, torch.randn(1, 3, 8, 8))

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_rms_norm_matches_torch_on_cpu_reference(self):
        for with_weight in (True, False):
            with self.subTest(with_weight=with_weight):
                class RMSNormNet(nn.Module):
                    def __init__(self, with_w):
                        super().__init__()
                        self.norm1 = nn.RMSNorm(16, elementwise_affine=with_w)
                        self.norm2 = nn.RMSNorm(16, elementwise_affine=with_w)
                    def forward(self, x):
                        return self.norm2(self.norm1(x))
                model = RMSNormNet(with_weight).eval()
                x = torch.randn(1, 8, 16)
                pte = export_program(model, (x,))
                delegates = delegate_payloads(pte)
                self.assertTrue(len(delegates) > 0)
                parsed = blob.unpack(delegates[0][1])
                got = run_reference(parsed, x)
                want = model(x).detach().numpy().flatten()
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_token_layer_norm_matches_torch_on_cpu_reference(self):
        class TokenLNNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(16)
                self.norm2 = nn.LayerNorm(16)
            def forward(self, x):
                return self.norm2(self.norm1(x))
        model = TokenLNNet().eval()
        x = torch.randn(1, 8, 16)
        pte = export_program(model, (x,))
        delegates = delegate_payloads(pte)
        self.assertTrue(len(delegates) > 0)
        parsed = blob.unpack(delegates[0][1])
        got = run_reference(parsed, x)
        want = model(x).detach().numpy().flatten()
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    def test_layer_norm_over_a_spatial_axis_stays_on_cpu(self):
        # Normalising over W alone is not one of the HX_NORM modes; the convolution is delegated.
        model = NormNet("layer_w").eval()
        delegates = delegate_payloads(export_program(model, (torch.randn(1, 3, 8, 8),)))
        if delegates:
            parsed = blob.unpack(delegates[0][1])
            kinds = struct.unpack_from("<I", parsed.image, 16)[0]
            ops = [struct.unpack_from("<I", parsed.image,
                                      256 + 32 * struct.unpack_from("<I", parsed.image, 12)[0]
                                      + 64 * i)[0] for i in range(kinds)]
            self.assertNotIn(lowering.hx.NORM, ops)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_embedding_matches_torch_on_cpu_reference(self):
        torch.manual_seed(10)
        model = EmbeddingNet().eval()
        x = torch.tensor([[1, 3, 5, 7]], dtype=torch.int32)
        with torch.no_grad():
            want = model(x).numpy()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        ops = ops_of(parsed)
        self.assertIn(lowering.hx.GATHER, ops)
        got = run_reference(parsed, x)
        np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_index_select_matches_torch_on_cpu_reference(self):
        torch.manual_seed(11)
        model = IndexSelectNet().eval()
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            want = model(x).numpy()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        ops = ops_of(parsed)
        self.assertIn(lowering.hx.GATHER, ops)
        got = run_reference(parsed, x)
        _, c, h, w = want.shape
        got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_index_tensor_matches_torch_on_cpu_reference(self):
        torch.manual_seed(12)
        model = IndexTensorNet().eval()
        x = torch.tensor([[1, 3, 5]], dtype=torch.int32)
        with torch.no_grad():
            want = model(x).numpy()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        ops = ops_of(parsed)
        self.assertIn(lowering.hx.GATHER, ops)
        got = run_reference(parsed, x)
        np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-4, atol=1e-4)


class NormNet(nn.Module):
    """Conv, then one normalisation (kind selects it), then ReLU."""

    def __init__(self, kind, groups=1):
        super().__init__()
        self.kind = kind
        self.conv = nn.Conv2d(3, 8 if kind == "group" else 4, 3, padding=1)
        if kind == "layer_chw":
            self.norm = nn.LayerNorm([4, 8, 8])
        elif kind == "group":
            self.norm = nn.GroupNorm(groups, 8)
        elif kind == "layer_w":
            self.norm = nn.LayerNorm([8])
        elif kind == "instance_affine":
            self.norm = nn.InstanceNorm2d(4, affine=True)
        elif kind == "instance_noaffine":
            self.norm = nn.InstanceNorm2d(4, affine=False)
        else:  # a [1, 16] tensor normalised over its channels: flatten, then LayerNorm
            self.linear = nn.Linear(16, 12)
            self.norm = nn.LayerNorm(12)

    def forward(self, x):
        if self.kind == "layer_channels":
            return self.norm(self.linear(x.flatten(1)))
        return F.relu(self.norm(self.conv(x)))


class BmmModel(nn.Module):
    def forward(self, a, b, c):
        return torch.bmm(torch.bmm(a, b), c)


class SdpaModel(nn.Module):
    def __init__(self, is_causal=False):
        super().__init__()
        self.is_causal = is_causal

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal) + q


class Reduction(nn.Module):
    """One ATen reduction after a convolution with a fused ReLU; mode selects the call."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv(x))
        if self.mode == "channel_max":  # spatial-attention style channel map
            return y.amax(dim=1, keepdim=True)
        if self.mode == "channel_mean":
            return y.mean(dim=1, keepdim=True)
        if self.mode == "sum_hw":  # keepdim=False: [1, C]
            return y.sum(dim=(2, 3))
        return y.amin(dim=(2, 3), keepdim=True)


class EmbeddingNet(nn.Module):
    """Embedding lookup followed by activation."""

    def __init__(self, num_embeddings=16, embedding_dim=8):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.emb(x))


class IndexSelectNet(nn.Module):
    """Conv2d followed by index_select along channel axis."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.register_buffer("idx", torch.tensor([1, 3, 5], dtype=torch.int64))

    def forward(self, x):
        y = self.conv(x)
        return torch.index_select(y, 1, self.idx)


class IndexTensorNet(nn.Module):
    """aten.index lookup from constant table buffer."""

    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.randn(16, 8))
        self.relu = nn.ReLU()

    def forward(self, idx):
        y = self.table[idx]
        return self.relu(y)


def run_reference(parsed, x):
    """Run the HXFG image on the portable CPU reference; returns the first output, flat."""
    import subprocess
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        (root / "g.hxfg").write_bytes(parsed.image)
        (root / "g.bin").write_bytes(parsed.constants)
        arr = x[0].numpy()
        data = arr.transpose(1, 2, 0) if arr.ndim == 3 else arr
        dt = np.int32 if np.issubdtype(arr.dtype, np.integer) else np.float32
        (root / "in.f32").write_bytes(np.ascontiguousarray(data, dtype=dt).tobytes())
        subprocess.run(
            [str(RUNNER), str(root / "g.hxfg"), str(root / "g.bin"),
             str(root / "in.f32"), str(root / "out_")],
            check=True, capture_output=True,
        )
        return np.fromfile(root / "out_0.f32", np.float32)


if __name__ == "__main__":
    unittest.main()
