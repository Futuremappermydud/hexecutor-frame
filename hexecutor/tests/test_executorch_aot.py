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
    def test_spatial_attention_is_delegated_and_matches_torch(self):
        torch.manual_seed(5)
        model = SpatialAttention().eval()
        x = torch.randn(1, 3, 4, 4)  # 16 tokens < 32 channels: a scalar on [1, L, C] must cover C
        with torch.no_grad():
            want = model(x).numpy()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        tensors, kinds = struct.unpack_from("<II", parsed.image, 12)
        ops = [struct.unpack_from("<I", parsed.image, 256 + 32 * tensors + 64 * i)[0] for i in range(kinds)]
        # One unmasked head: Q.K^T, softmax and P.V instead of the scalar HX_SDPA.
        self.assertEqual((ops.count(lowering.hx.SDPA), ops.count(lowering.hx.MATMUL)), (0, 2))
        got = run_reference(parsed, x).reshape(4, 4, 32).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_attention_score_scale_matches_torch_on_cpu_reference(self):
        torch.manual_seed(5)
        model = SpatialAttention().eval()
        x = torch.randn(1, 3, 4, 4)
        with torch.no_grad():
            want = model(x).numpy()
        lowering.set_attention_score_scale(1 / 1024)
        try:
            parsed = blob.unpack(delegate_payloads(export_program(model, (x,)))[0][1])
        finally:
            lowering.set_attention_score_scale(1.0)
        soft = [p for k, p in op_params(parsed) if k == lowering.hx.ATTN_SOFTMAX]
        self.assertEqual([p[:4] for p in soft], [[2, 0, 0, lowering.hx.f32_bits(1024.0)]])
        got = run_reference(parsed, x).reshape(4, 4, 32).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
        with self.assertRaises(ValueError):
            lowering.set_attention_score_scale(0.3)

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
    def test_composite_activations_match_torch_on_cpu_reference(self):
        class Acts(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(3, 6, 3, padding=1)

            def forward(self, x):
                y = self.c(x)
                return (F.elu(y, alpha=0.7) + F.selu(y) + F.celu(y, alpha=1.3) + F.softplus(y)
                        + F.mish(y) + torch.sinh(y) + torch.cosh(y) + torch.sign(y))

        torch.manual_seed(4)
        model, x = Acts().eval(), torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            want = model(x).numpy()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)  # the whole graph, no CPU fallback
        got = run_reference(blob.unpack(delegates[0][1]), x)
        _, c, h, w = want.shape
        got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_softplus_with_other_beta_stays_on_cpu(self):
        class Sp(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(3, 4, 1)

            def forward(self, x):
                return F.softplus(self.c(x), beta=2.0)

        torch.manual_seed(5)
        model, x = Sp().eval(), torch.randn(1, 3, 4, 4)
        with torch.no_grad():
            conv_only = model.c(x).numpy()  # beta != 1 is not expressible: softplus stays outside
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        got = run_reference(blob.unpack(delegates[0][1]), x)
        _, c, h, w = conv_only.shape
        np.testing.assert_allclose(got.reshape(h, w, c).transpose(2, 0, 1)[None], conv_only,
                                   rtol=1e-3, atol=1e-3)

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

    def _check_delegated(self, model, x, required, rtol=1e-4, atol=1e-4):
        """The whole model is one delegate holding the HXFG kinds in `required`, and the portable
        reference run matches torch. Returns the delegate's kinds."""
        with torch.no_grad():
            want = model(x).numpy()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        kinds = ops_of(parsed)
        for kind in required:
            self.assertIn(kind, kinds)
        got = run_reference(parsed, x)
        if want.ndim == 4:  # external output is compact NHWC; compare as NCHW
            _, c, h, w = want.shape
            got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got.reshape(want.shape), want, rtol=rtol, atol=atol)
        return kinds

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_where_and_comparisons_match_torch_on_cpu_reference(self):
        torch.manual_seed(20)
        x = torch.randn(1, 3, 8, 8)
        cases = {
            "tensor_branches": lambda y: torch.where(y > 0.25, y, -y),          # gt.Scalar, where.self
            "scalar_branch": lambda y: torch.where(y >= 0.5, y, 0.5),          # scalar operand folded
            "bitwise_and": lambda y: torch.where((y < 0.3) & (y > -0.2), y, 2 * y),
            "le_eq_or": lambda y: torch.where((y <= 0.1) | (y.round() == 0.0), y, -y),  # le.Scalar, eq.Scalar
        }
        for name, fn in cases.items():
            with self.subTest(name):
                self._check_delegated(OpNet(fn).eval(), x, [lowering.hx.WHERE, lowering.hx.BINARY])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_ne_and_bitwise_not_match_torch_on_cpu_reference(self):
        torch.manual_seed(21)
        x = torch.randn(1, 3, 8, 8)
        # ne.Scalar is EQUAL then NOT; bitwise_not on the bool result is a second NOT.
        model = OpNet(lambda y: torch.where(~(y != 0.1), y, -y)).eval()
        self._check_delegated(model, x, [lowering.hx.BINARY, lowering.hx.ACT, lowering.hx.WHERE])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_argmax_argmin_match_torch_on_cpu_reference(self):
        torch.manual_seed(22)
        x = torch.randn(1, 3, 8, 8)
        cases = {
            "argmax": lambda y: torch.where(torch.argmax(y, dim=1, keepdim=True) == 2, y, -y),
            "argmin": lambda y: torch.where(torch.argmin(y, dim=1, keepdim=True) == 1, y, 0.5 * y),
        }
        for name, fn in cases.items():
            with self.subTest(name):
                # The indices feed only comparisons, so they stay inside the delegate (int32 in HXFG).
                kinds = self._check_delegated(OpNet(fn).eval(), x, [lowering.hx.WHERE])
                self.assertIn(lowering.hx.ARGMAX if name == "argmax" else lowering.hx.ARGMIN, kinds)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_topk_matches_torch_on_cpu_reference(self):
        torch.manual_seed(23)
        x = torch.randn(1, 3, 8, 8)
        cases = {
            "largest": lambda y: (lambda v, i: torch.where(i == 0, v, -v) + v)(*torch.topk(y, 2, dim=1)),
            "smallest": lambda y: (lambda v, i: torch.where(i == 1, v, 2 * v) * 0.5)(
                *torch.topk(y, 3, dim=1, largest=False)),
        }
        for name, fn in cases.items():
            with self.subTest(name):
                kinds = self._check_delegated(OpNet(fn).eval(), x, [lowering.hx.TOPK, lowering.hx.WHERE])
                self.assertEqual(kinds.count(lowering.hx.TOPK), 1)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_max_and_min_dim_match_torch_on_cpu_reference(self):
        torch.manual_seed(24)
        x = torch.randn(1, 3, 8, 8)
        cases = {
            "max": (lambda y: (lambda v, i: torch.where(i == 3, v, -v) + v)(*y.max(dim=1, keepdim=True)),
                    lowering.hx.REDUCE, lowering.hx.ARGMAX),
            "min": (lambda y: (lambda v, i: torch.where(i == 1, v, -v))(*y.min(dim=1, keepdim=True)),
                    lowering.hx.REDUCE, lowering.hx.ARGMIN),
        }
        for name, (fn, *required) in cases.items():
            with self.subTest(name):
                self._check_delegated(OpNet(fn).eval(), x, [lowering.hx.WHERE] + required)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_expand_matches_torch_on_cpu_reference(self):
        torch.manual_seed(25)
        x = torch.randn(1, 3, 8, 8)
        model = OpNet(lambda y: y * y.mean(dim=(2, 3), keepdim=True).expand(1, 8, 8, 8)).eval()
        self._check_delegated(model, x, [lowering.hx.TILE, lowering.hx.BINARY])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_rsqrt_matches_torch_on_cpu_reference(self):
        torch.manual_seed(26)
        x = torch.randn(1, 3, 8, 8)
        model = OpNet(lambda y: torch.rsqrt(F.relu(y) + 0.5)).eval()
        self._check_delegated(model, x, [lowering.hx.ACT], rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_log_sigmoid_matches_torch_on_cpu_reference(self):
        torch.manual_seed(27)
        x = torch.randn(1, 3, 8, 8)
        model = OpNet(lambda y: F.logsigmoid(y) * 2).eval()
        self._check_delegated(model, x, [lowering.hx.ACT])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_general_clamp_and_hardtanh_match_torch_on_cpu_reference(self):
        torch.manual_seed(28)
        x = torch.randn(1, 3, 8, 8)
        cases = {
            "clamp_both": lambda y: torch.clamp(y, -0.3, 0.6),
            "clamp_min": lambda y: torch.clamp(y, min=0.1),
            "clamp_max": lambda y: torch.clamp(y, max=0.2),
            "hardtanh": lambda y: F.hardtanh(y, -0.5, 0.5),
        }
        for name, fn in cases.items():
            with self.subTest(name):
                self._check_delegated(OpNet(fn).eval(), x, [lowering.hx.BINARY])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_three_input_cat_matches_torch_on_cpu_reference(self):
        torch.manual_seed(29)
        x = torch.randn(1, 3, 8, 8)
        cases = {
            "channels": lambda y: torch.cat([y, -y, 2 * y], dim=1),
            "width": lambda y: torch.cat([y, -y, 2 * y], dim=3),
        }
        for name, fn in cases.items():
            with self.subTest(name):
                kinds = self._check_delegated(OpNet(fn).eval(), x, [lowering.hx.CONCAT])
                self.assertEqual(kinds.count(lowering.hx.CONCAT), 2)  # a chain of two-input concatenations

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_var_and_std_match_torch_on_cpu_reference(self):
        torch.manual_seed(30)
        x = torch.randn(1, 3, 8, 8)
        cases = {  # name: (function, required kinds); spatial means are HX_MEAN, channel means HX_REDUCE
            "var_spatial": (lambda y: y.var(dim=(2, 3)), [lowering.hx.MEAN]),  # keepdim False, [1, C]
            "var_channel_scaled": (lambda y: y.var(dim=1, keepdim=True, correction=0) * y,
                                   [lowering.hx.REDUCE, lowering.hx.BINARY]),
            "std_channel": (lambda y: y.std(dim=1, keepdim=True, correction=0) * y,  # var then sqrt
                            [lowering.hx.REDUCE, lowering.hx.ACT]),
        }
        for name, (fn, required) in cases.items():
            with self.subTest(name):
                self._check_delegated(OpNet(fn).eval(), x, required)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_constant_factories_fold_into_consumers(self):
        torch.manual_seed(31)
        x = torch.randn(1, 3, 8, 8)
        cases = {  # name: (function, required kinds); the factories themselves emit nothing
            "full": (lambda y: y + torch.full((1, 8, 8, 8), 0.5), [lowering.hx.BINARY]),
            "arange_per_channel": (lambda y: y * torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1),
                                   [lowering.hx.BINARY]),
            "arange_int_cast": (lambda y: y * torch.arange(8).float().reshape(1, 8, 1, 1), [lowering.hx.BINARY]),
            "full_like_where": (lambda y: torch.where(y > 0.25, y, torch.zeros_like(y)),
                                [lowering.hx.WHERE, lowering.hx.BINARY]),
        }
        for name, (fn, required) in cases.items():
            with self.subTest(name):
                self._check_delegated(OpNet(fn).eval(), x, required)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_conv1d_matches_torch_on_cpu_reference(self):
        torch.manual_seed(32)
        cases = [
            (nn.Conv1d(3, 8, 3, padding=1), (1, 3, 16)),
            (nn.Conv1d(3, 6, 5, stride=2, padding=2), (1, 3, 17)),
            (nn.Conv1d(4, 4, 3, padding=2, dilation=2, groups=4), (1, 4, 16)),  # depthwise, dilated
        ]
        for conv, shape in cases:
            with self.subTest(conv=str(conv)):
                model = nn.Sequential(conv, nn.ReLU()).eval()
                kinds = self._check_delegated(model, torch.randn(*shape),
                                              [lowering.hx.TRANSPOSE])
                self.assertTrue(lowering.hx.CONV in kinds or lowering.hx.DEPTHWISE in kinds)

    def test_relu6_still_fuses_into_the_convolution(self):
        # clamp(0, 6) is a ReLU6 fusion, not the general clamp: one CONV operation with activation 3.
        class Relu6Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)

            def forward(self, x):
                return F.relu6(self.conv(x))

        delegates = delegate_payloads(export_program(Relu6Net().eval(), (torch.randn(1, 3, 8, 8),)))
        parsed = blob.unpack(delegates[0][1])
        tensors = struct.unpack_from("<I", parsed.image, 12)[0]
        op = struct.unpack_from("<16I", parsed.image, 256 + 32 * tensors)
        self.assertEqual(ops_of(parsed), [lowering.hx.CONV])
        self.assertEqual(op[5 + 8], lowering.hx.ACT_RELU6)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_argmax_outside_the_channel_axis_stays_on_cpu(self):
        torch.manual_seed(33)
        x = torch.randn(1, 3, 8, 8)
        # argmax over H is not an HWC channel reduction: it and the comparison and where after it stay
        # outside, so the delegate returns relu(conv(x)) and the reference is that tensor.
        model = OpNet(lambda y: torch.where(torch.argmax(y, dim=2, keepdim=True) == 1, y, -y)).eval()
        with torch.no_grad():
            want = F.relu(model.conv(x)).numpy()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        kinds = ops_of(parsed)
        self.assertIn(lowering.hx.CONV, kinds)
        self.assertNotIn(lowering.hx.ARGMAX, kinds)
        self.assertNotIn(lowering.hx.WHERE, kinds)
        got = run_reference(parsed, x)
        _, c, h, w = want.shape
        got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_integer_output_is_not_delegated(self):
        torch.manual_seed(34)
        x = torch.randn(1, 3, 8, 8)
        # The indices are a graph output; delegate outputs are float32, so argmax stays on the CPU.
        model = OpNet(lambda y: torch.argmax(y, dim=1, keepdim=True)).eval()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        self.assertNotIn(lowering.hx.ARGMAX, ops_of(parsed))
        self.assertEqual([s.dims for s in parsed.outputs], [(1, 8, 8, 8)])

    def _check_tokens(self, model, x, required, absent=(), rtol=1e-4, atol=1e-4):
        """Like _check_delegated for token models: a rank-2 input is fed as [1, L, C] (the reference strips the
        batch dimension), and `absent` kinds must not appear."""
        with torch.no_grad():
            want = model(x).numpy()
        delegates = delegate_payloads(export_program(model, (x,)))
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        kinds = ops_of(parsed)
        for kind in required:
            self.assertIn(kind, kinds)
        for kind in absent:
            self.assertNotIn(kind, kinds)
        got = run_reference(parsed, x[None] if x.dim() == 2 else x)
        if want.ndim == 4:
            _, c, h, w = want.shape
            got = got.reshape(h, w, c).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(got.reshape(want.shape), want, rtol=rtol, atol=atol)
        return parsed

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_cast_of_runtime_tensors_matches_torch_on_cpu_reference(self):
        torch.manual_seed(40)
        self._check_delegated(CastNet().eval(), torch.randn(1, 3, 8, 8), [lowering.hx.CAST])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_cast_dtypes_use_the_expected_modes(self):
        torch.manual_seed(41)
        parsed = self._check_tokens(
            Tokens(lambda s, y: ((y * 3).to(torch.int32).to(torch.float32)) + (y > 0.1).to(torch.int32).to(torch.float32)).eval(),
            torch.randn(1, 6, 8), [lowering.hx.CAST])
        modes = [p[0] for kind, p in op_params(parsed) if kind == lowering.hx.CAST]
        self.assertEqual(sorted(modes), [0, 0, 1, 1])  # float -> int32 twice (one via bool), int32 -> float twice

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_gather_matches_torch_on_cpu_reference(self):
        torch.manual_seed(42)
        gather = lowering.hx.GATHER_ELEMENTS
        cases = {
            "runtime_index_rank3": (lambda s, y: torch.gather(y, -1, torch.argmax(y, dim=-1, keepdim=True)),
                                    torch.randn(1, 6, 8)),
            "constant_index_channels": (lambda s, y: torch.gather(y, 2, s.cidx), torch.randn(1, 6, 8)),
            "constant_index_tokens": (lambda s, y: torch.gather(y, 1, s.gidx), torch.randn(1, 6, 8)),
            "rank2": (lambda s, y: torch.gather(y, 0, s.gidx2), torch.randn(6, 8)),
        }
        for name, (fn, x) in cases.items():
            with self.subTest(name):
                self._check_tokens(Tokens(fn).eval(), x, [gather])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_gather_along_a_spatial_axis_of_a_feature_map(self):
        class SpatialGather(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 3, padding=1)
                self.register_buffer("idx", torch.randint(0, 8, (1, 4, 8, 5)))

            def forward(self, x):
                return torch.gather(F.relu(self.conv(x)), 3, self.idx)

        torch.manual_seed(43)
        self._check_delegated(SpatialGather().eval(), torch.randn(1, 3, 8, 8), [lowering.hx.GATHER_ELEMENTS])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_advanced_index_matches_torch_on_cpu_reference(self):
        torch.manual_seed(44)
        for runtime in (False, True):
            with self.subTest(runtime=runtime):
                kinds = self._check_delegated(PixelGather(runtime).eval(), torch.randn(1, 3, 8, 8),
                                              [lowering.hx.GATHER_ND, lowering.hx.TRANSPOSE])
                self.assertEqual(lowering.hx.CONCAT in kinds, runtime)  # runtime pairs are concatenated
        cases = {  # x[idx] on the token axis, x[:, idx], x[:, :, idx] stay HX_GATHER
            "rows": (lambda s, y: y[s.idx3], torch.randn(6, 8)),
            "tokens": (lambda s, y: y[:, s.idx3], torch.randn(1, 6, 8)),
            "channels": (lambda s, y: y[:, :, s.idx3], torch.randn(1, 6, 8)),
        }
        for name, (fn, x) in cases.items():
            with self.subTest(name):
                self._check_tokens(Tokens(fn).eval(), x, [lowering.hx.GATHER])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_scatter_matches_torch_on_cpu_reference(self):
        torch.manual_seed(45)
        scatter = lowering.hx.SCATTER_ND
        cases = {
            "index_put": lambda s, y: y.index_put((s.idx3,), y[:3] * 2),
            "index_put_accumulate": lambda s, y: y.index_put((s.idx3,), y[:3] * 2, accumulate=True),
            "index_put_constant_rows": lambda s, y: y.index_put((s.idx3,), s.rows),
            "index_put_runtime_index": lambda s, y: y.index_put(
                (torch.argmax(y, dim=1, keepdim=True).reshape(8),), y * 2),
            "select_scatter": lambda s, y: torch.select_scatter(y, (y[2:3] * 3).reshape(8), 0, 4),
            "slice_scatter": lambda s, y: torch.slice_scatter(y, y[:2] * 3, 0, 1, 3),
        }
        for name, fn in cases.items():
            with self.subTest(name):
                x = torch.randn(8, 8) if name == "index_put_runtime_index" else torch.randn(6, 8)
                parsed = self._check_tokens(Tokens(fn).eval(), x, [scatter])
                reductions = [p[4] for kind, p in op_params(parsed) if kind == scatter]
                self.assertEqual(reductions, [1 if name.endswith("accumulate") else 0])
        with self.subTest("rank3_tokens"):
            def put_tokens(s, y):
                z = y.clone()
                z[:, s.idx3] = y[:, :3] * 2
                return z

            self._check_tokens(Tokens(put_tokens).eval(), torch.randn(1, 6, 8), [scatter])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_softmax_over_every_axis_matches_torch_on_cpu_reference(self):
        torch.manual_seed(46)
        hx_ = lowering.hx
        cases = [  # (dim, input, kind)
            (1, torch.randn(6, 8), hx_.SOFTMAX), (-1, torch.randn(6, 8), hx_.SOFTMAX),
            (0, torch.randn(6, 8), hx_.ATTN_SOFTMAX),
            (2, torch.randn(1, 6, 8), hx_.SOFTMAX), (1, torch.randn(1, 6, 8), hx_.ATTN_SOFTMAX),
            (1, torch.randn(1, 3, 8, 8), hx_.SOFTMAX),
            (2, torch.randn(1, 3, 8, 8), hx_.ATTN_SOFTMAX), (3, torch.randn(1, 3, 8, 8), hx_.ATTN_SOFTMAX),
        ]
        for dim, x, kind in cases:
            with self.subTest(dim=dim, shape=tuple(x.shape)):
                model = SoftmaxAxis(dim, x.dim()).eval()
                if x.dim() == 4:
                    self._check_delegated(model, x, [kind])
                else:
                    self._check_tokens(model, x, [kind])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_softmax_masks_match_torch_on_cpu_reference(self):
        torch.manual_seed(47)
        hx_ = lowering.hx
        for kind in ("additive", "masked_fill"):  # a causal mask folds into HX_ATTN_SOFTMAX's causal flag
            for x in (torch.randn(1, 6, 8), torch.randn(6, 8)):
                with self.subTest(kind=kind, rank=x.dim()):
                    parsed = self._check_tokens(MaskedSoftmax(kind).eval(), x, [hx_.ATTN_SOFTMAX],
                                                absent=[hx_.WHERE])
                    ops = op_params(parsed)
                    soft = [p for k, p in ops if k == hx_.ATTN_SOFTMAX]
                    self.assertEqual([p[:2] for p in soft], [[2, 1]])  # key axis, causal
                    self.assertEqual([k for k, _ in ops].count(hx_.BINARY), 1)  # the 0.5 scale only
        with self.subTest("general constant mask"):  # not causal: a per-element multiply and add, then softmax
            parsed = self._check_tokens(MaskedSoftmax("general").eval(), torch.randn(1, 6, 8), [hx_.SOFTMAX])
            self.assertEqual([k for k, _ in op_params(parsed)].count(hx_.BINARY), 3)
        with self.subTest("runtime mask"):  # the mask is a tensor input of HX_ATTN_SOFTMAX
            parsed = self._check_tokens(MaskedSoftmax("runtime").eval(), torch.randn(1, 6, 8), [hx_.ATTN_SOFTMAX])
            soft = next(op for op in struct_ops(parsed) if op[0] == hx_.ATTN_SOFTMAX)
            self.assertNotEqual(soft[2], lowering.NONE)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_attention_blocks_match_torch_on_cpu_reference(self):
        torch.manual_seed(48)
        hx_ = lowering.hx
        for causal in (True, False):
            with self.subTest(projected=True, causal=causal):
                parsed = self._check_tokens(ProjectedAttention(causal).eval(), torch.randn(1, 5, 8),
                                            [hx_.MATMUL, hx_.TRANSPOSE])
                flags = [p[1] for k, p in op_params(parsed) if k == hx_.ATTN_SOFTMAX]
                self.assertEqual(flags, [1] if causal else [])
        with self.subTest("spatial"):
            parsed = self._check_delegated(SpatialSelfAttention().eval(), torch.randn(1, 3, 4, 4),
                                           [hx_.MATMUL, hx_.ATTN_SOFTMAX])
        with self.subTest("batched multi-head layout"):
            model, x = BatchedAttention().eval(), torch.randn(2, 4, 4)
            with torch.no_grad():
                want = model(x).numpy()
            parsed = blob.unpack(delegate_payloads(export_program(model, (x,)))[0][1])
            self.assertIn(hx_.ATTN_SOFTMAX, ops_of(parsed))
            self.assertEqual([p[:2] for k, p in op_params(parsed) if k == hx_.ATTN_SOFTMAX], [[2, 1]])
            got = run_reference(parsed, x, raw=True)  # [B, L, K] is the HXFG layout of attention tensors
            np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_token_permute_and_token_to_rank4_view_match_torch_on_cpu_reference(self):
        torch.manual_seed(49)
        self._check_tokens(Tokens(lambda s, y: y.transpose(1, 2)).eval(), torch.randn(1, 5, 8), [lowering.hx.TRANSPOSE])
        self._check_tokens(Tokens(lambda s, y: y.t()).eval(), torch.randn(5, 8), [lowering.hx.TRANSPOSE])
        # A rank-3 token tensor viewed as [1, C, H, W]: tokens are row-major, the NCHW side is unflattened.
        self._check_tokens(Tokens(lambda s, y: F.relu(y.view(1, 4, 2, 4))).eval(), torch.randn(1, 4, 8), [lowering.hx.RESHAPE])

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_batch_norm_inference_folds_into_the_convolution(self):
        torch.manual_seed(50)
        x = torch.randn(1, 3, 8, 8)
        kinds = self._check_delegated(ConvBatchNorm(True).eval(), x, [lowering.hx.CONV], rtol=1e-3, atol=1e-4)
        self.assertEqual(kinds, [lowering.hx.CONV])  # the norm and the ReLU both fold into one convolution
        kinds = self._check_delegated(ConvBatchNorm(False).eval(), x, [lowering.hx.BINARY], rtol=1e-3, atol=1e-4)
        self.assertEqual(kinds.count(lowering.hx.BINARY), 2)  # scale and shift as per-channel multiply and add

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_grouped_convolution_and_linear_after_view_match_torch_on_cpu_reference(self):
        class Grouped(nn.Module):
            def __init__(self):
                super().__init__()
                self.c0 = nn.Conv2d(3, 8, 1)
                self.c1 = nn.Conv2d(8, 16, 3, padding=1, groups=2)
                self.c2 = nn.Conv2d(16, 16, 3, padding=1, stride=2, groups=4)

            def forward(self, x):
                return F.relu(self.c2(F.relu(self.c1(F.relu(self.c0(x))))))

        class LinearAfterView(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(192, 10)
                self.fc2 = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc2(F.relu(self.fc(x.reshape(1, 192))))

        torch.manual_seed(51)
        x = torch.randn(1, 3, 8, 8)
        kinds = self._check_delegated(Grouped().eval(), x, [lowering.hx.SLICE, lowering.hx.CONCAT])
        self._check_delegated(LinearAfterView().eval(), x, [lowering.hx.CONV])

    def test_index_select_unsupported_combos_rejected(self):
        class IndexSelectDim0Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("idx", torch.tensor([0], dtype=torch.int64))
            def forward(self, x):
                return torch.index_select(x, 0, self.idx)

        m = IndexSelectDim0Net().eval()
        x = torch.randn(1, 4, 8, 8)
        with self.assertRaises(Exception):
            lowering.lower_exported_program(export_program(m, (x,)))


class OpNet(nn.Module):
    """conv3x3 + ReLU, then `fn` applied to the feature map (one new operation per test)."""

    def __init__(self, fn):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fn = fn

    def forward(self, x):
        return self.fn(F.relu(self.conv(x)))


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


class SpatialAttention(nn.Module):
    """A diffusion-UNet attention block: group norm, [1, C, H, W] -> [1, HW, C] token view, rank-3 linears,
    single-head SDPA, the reverse view, a residual and a scalar rescale (a rank-0 constant)."""

    def __init__(self, c=32):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.q, self.k, self.v, self.o = (nn.Linear(c, c) for _ in range(4))
        self.inp = nn.Conv2d(3, c, 3, padding=1)

    def forward(self, x):
        h = self.inp(x)
        b, c, hh, ww = h.shape
        t = self.norm(h).view(b, c, hh * ww).transpose(1, 2)
        q, k, v = (f(t).view(b, 1, hh * ww, c) for f in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(q, k, v).view(b, hh * ww, c)
        a = (self.o(a) * 0.5).transpose(1, 2).reshape(b, c, hh, ww)
        return (a + h) / torch.tensor(2.0)


def struct_ops(parsed):
    """Raw (kind, x, y, z, out, p...) tuples of a parsed HXFG image."""
    tensors = struct.unpack_from("<I", parsed.image, 12)[0]
    count = struct.unpack_from("<I", parsed.image, 16)[0]
    return [struct.unpack_from("<16I", parsed.image, 256 + 32 * tensors + 64 * i) for i in range(count)]


def op_params(parsed):
    """(kind, p[0..10]) of every operation of a parsed HXFG image, in order."""
    tensors = struct.unpack_from("<I", parsed.image, 12)[0]
    count = struct.unpack_from("<I", parsed.image, 16)[0]
    out = []
    for i in range(count):
        op = struct.unpack_from("<16I", parsed.image, 256 + 32 * tensors + 64 * i)
        out.append((op[0], list(op[5:16])))
    return out


class Tokens(nn.Module):
    """A linear layer with ReLU on [L, 8] or [1, L, 8] tokens, then `fn(self, tokens)`."""

    def __init__(self, fn, features=8):
        super().__init__()
        self.lin = nn.Linear(8, features)
        self.fn = fn
        self.register_buffer("idx3", torch.tensor([5, 0, 2]))
        self.register_buffer("rows", torch.randn(3, features))
        self.register_buffer("gidx", torch.randint(0, 6, (1, 4, features)))
        self.register_buffer("gidx2", torch.randint(0, 6, (4, features)))
        self.register_buffer("cidx", torch.randint(0, features, (1, 6, 3)))
        self.register_buffer("mask", torch.triu(torch.full((6, 6), float("-inf")), 1))
        self.register_buffer("bmask", torch.triu(torch.ones(6, 6, dtype=torch.bool), 1))

    def forward(self, x):
        return self.fn(self, F.relu(self.lin(x)))


class CastNet(nn.Module):
    """Casts of runtime tensors: float -> int32 (truncation) -> float, int32 -> bool, bool -> float and int32."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        y = self.conv(x)
        i = (y * 3).to(torch.int32)
        flag = (y * 2).to(torch.int32).to(torch.bool)
        return (i.to(torch.float32) + (y > 0).to(torch.float32) + flag.to(torch.float32)
                + (y > 0.2).to(torch.int32).to(torch.float32))


class PixelGather(nn.Module):
    """y[:, :, iy, ix] of a feature map: a GatherND over (h, w) pairs. Indices are constants or computed."""

    def __init__(self, runtime):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.runtime = runtime
        self.register_buffer("iy", torch.tensor([0, 3, 7, 2, 5]))
        self.register_buffer("ix", torch.tensor([1, 6, 2, 2, 7]))

    def forward(self, x):
        y = F.relu(self.conv(x))
        if not self.runtime:
            return y[:, :, self.iy, self.ix]
        a = torch.sigmoid(y.sum(dim=(2, 3)))  # [1, 8]
        iy = (a * 7.99).to(torch.int32).reshape(8)
        ix = ((1 - a) * 7.99).to(torch.int32).reshape(8)
        return y[:, :, iy, ix]


class SoftmaxAxis(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.dim = dim
        self.net = nn.Conv2d(3, 8, 3, padding=1) if rank == 4 else nn.Linear(8, 8)

    def forward(self, x):
        return torch.softmax(F.relu(self.net(x)), dim=self.dim)


class MaskedSoftmax(nn.Module):
    """Softmax over the key axis of token scores with a mask: `kind` selects an additive causal buffer, a
    masked_fill with a bool buffer, a general constant bool mask, or a runtime additive mask."""

    def __init__(self, kind, length=6):
        super().__init__()
        self.kind = kind
        self.lin = nn.Linear(8, length)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool), 1)
        self.register_buffer("additive", torch.where(causal, float("-inf"), 0.0))
        self.register_buffer("upper", causal)
        general = torch.rand(length, length) > 0.5
        general.fill_diagonal_(False)
        self.register_buffer("general", general)

    def forward(self, x):
        scores = self.lin(x) * 0.5
        if self.kind == "additive":
            scores = scores + self.additive
        elif self.kind == "masked_fill":
            scores = scores.masked_fill(self.upper, float("-inf"))
        elif self.kind == "general":
            scores = scores.masked_fill(self.general, -1e9)
        else:  # runtime mask: another function of the input
            scores = scores + x[:, :, :6] * 3
        return torch.softmax(scores, dim=-1)


class ProjectedAttention(nn.Module):
    """Single-head attention from linear projections: scores = (Q K^T) * scale + causal mask, softmax, P V."""

    def __init__(self, causal=True):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8)
        self.causal = causal
        self.register_buffer("mask", torch.triu(torch.full((5, 5), float("-inf")), 1))

    def forward(self, x):
        scores = torch.bmm(self.q(x), self.k(x).transpose(1, 2)) * 0.35
        if self.causal:
            scores = scores + self.mask
        return torch.bmm(torch.softmax(scores, dim=-1), self.v(x))


class SpatialSelfAttention(nn.Module):
    """Attention over the pixels of a feature map through views of [1, C, H, W]; the keys enter as a view."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.register_buffer("mask", torch.triu(torch.full((16, 16), float("-inf")), 1))

    def forward(self, x):
        y = F.relu(self.conv(x))
        b, c, h, w = y.shape
        keys = y.view(b, c, h * w)                    # [1, C, HW] = K^T
        tokens = y.view(b, c, h * w).transpose(1, 2)  # [1, HW, C]
        probs = torch.softmax(torch.bmm(tokens, keys) * 0.35 + self.mask, dim=-1)
        return torch.bmm(probs, tokens).transpose(1, 2).reshape(b, c, h, w)


class BatchedAttention(nn.Module):
    """Two attention heads' worth of batch (B = 2): the multi-head layout with a causal mask."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mask", torch.triu(torch.full((4, 4), float("-inf")), 1))

    def forward(self, x):
        return torch.bmm(torch.softmax(torch.bmm(x, x) * 0.4 + self.mask, dim=-1), x)


class ConvBatchNorm(nn.Module):
    """conv -> inference BatchNorm -> ReLU; `fold` False puts a PReLU between conv and BN so it cannot fold."""

    def __init__(self, fold):
        super().__init__()
        self.fold = fold
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.prelu = nn.PReLU(8)
        self.bn = nn.BatchNorm2d(8)
        self.bn.running_mean.normal_()
        self.bn.running_var.uniform_(0.5, 2.0)
        self.bn.weight.data.normal_()
        self.bn.bias.data.normal_()

    def forward(self, x):
        if self.fold:
            return F.relu(self.bn(self.conv(x)))
        return self.bn(self.prelu(self.conv(x)))


def run_reference(parsed, x, raw=False):
    """Run the HXFG image on the portable CPU reference; returns the first output, flat.

    `x` carries a leading batch dimension that is stripped, and a rank-3 remainder is read as [C, H, W]
    and fed as HWC; `raw` feeds the whole array as it is."""
    import subprocess
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        (root / "g.hxfg").write_bytes(parsed.image)
        (root / "g.bin").write_bytes(parsed.constants)
        arr = x.numpy() if raw else x[0].numpy()
        data = arr.transpose(1, 2, 0) if arr.ndim == 3 and not raw else arr
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
