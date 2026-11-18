"""AOT tests for the ExecuTorch hexecutor delegate (offline tooling only).

Run from the repository root with the pinned venv:

    hexecutor/.work/executorch-venv/bin/python -m unittest hexecutor.tests.test_executorch_aot -v

Blob and HXFG structure checks run without ExecuTorch. The export tests need
the pinned executorch package and skip when it is absent. The numeric check
runs the portable CPU reference (build/bin/hexecutor_tensor_run) only when that
binary exists.
"""

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
RUNNER = ROOT / "build/bin/hexecutor_tensor_run"


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


class SoftmaxOnly(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=1)


class ConvSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return torch.softmax(torch.relu(self.conv(x)), dim=1)


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
        program = export_program(SoftmaxOnly(), (torch.randn(1, 4, 8, 8),))
        self.assertEqual(delegate_payloads(program), [])

    def test_unsupported_tail_stays_outside_delegate(self):
        program = export_program(ConvSoftmax(), (torch.randn(1, 3, 16, 16),))
        delegates = delegate_payloads(program)
        self.assertEqual(len(delegates), 1)
        parsed = blob.unpack(delegates[0][1])
        # The softmax runs on the CPU, so the delegate returns the conv+ReLU tensor.
        self.assertEqual([s.dims for s in parsed.outputs], [(1, 8, 16, 16)])

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


def run_reference(parsed, x):
    """Run the HXFG image on the portable CPU reference; returns the first output, flat."""
    import subprocess
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        (root / "g.hxfg").write_bytes(parsed.image)
        (root / "g.bin").write_bytes(parsed.constants)
        (root / "in.f32").write_bytes(x[0].numpy().transpose(1, 2, 0).astype(np.float32).tobytes())
        subprocess.run(
            [str(RUNNER), str(root / "g.hxfg"), str(root / "g.bin"),
             str(root / "in.f32"), str(root / "out_")],
            check=True, capture_output=True,
        )
        return np.fromfile(root / "out_0.f32", np.float32)


if __name__ == "__main__":
    unittest.main()
