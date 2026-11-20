"""Export Mercury's two ONNX models to ExecuTorch .pte files delegated to HexecutorBackend (the Frame CDSP).

    <executorch venv>/bin/python export_executorch.py --models-dir <Steam-Frame-Hand-Tracking>/models

The ahead-of-time half of HexecutorBackend lives in the frame-spatial-runtime repository
(hexecutor/offline/executorch); --hexecutor-root overrides this checkout, and the interpreter must be the venv its
README pins (executorch 1.0.0, torch 2.9.1, plus onnx and onnxruntime for the checks here). Nothing is taken
from a default location.

Each model is read from models/fp32/ (checked against its SHA-256 pin), rebuilt as a PyTorch module, compared
with ONNX Runtime, exported, and refused unless the whole graph is one HexecutorBackend delegate: the Frame
build of the ExecuTorch runtime has no CPU kernels. The one rewrite: the keypoint model's 42 -> 8192
lastKeypoints Linear is split into 4096-row parts (the HXFG channel limit), each reshaped to whole channels of
the [32, 16, 16] map before the concatenation, which is the same arithmetic.

Next to each .pte it writes check/<model>.in<i>.f32 and .out<j>.f32 (float32, ONNX Runtime outputs on a
seeded input) for fhm_nn_check, and manifest.json with hashes and the check results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = {
    'grayscale_detection_160x160': '1f1a039a266e13dc186bb884430ebd9c8216bdda680ab08a533d4c671f27ed36',
    'grayscale_keypoint_jan18': '40c0daa598cedb993b54fff17685231b7465d6db342656c401b01c2029efd1d5',
}
MAX_CHANNELS = 4096  # hexecutor/offline/hxfg_writer.py MAX_CHANNELS
# Both hands in one DSP call: <source>_x2 runs the source model on two input sets (hand 0 inputs, then hand 1)
# and returns hand 0's outputs, then hand 1's. The lowering's constant pool stores the shared weights once.
PAIRED = {'grayscale_keypoint_jan18_x2': 'grayscale_keypoint_jan18'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mercury_module(path):
    """The ONNX graph as a torch.nn.Module, for exactly the operators Mercury's two models use."""
    import onnx
    import torch
    import torch.nn.functional as F
    from onnx import numpy_helper

    class Parts(list):
        """A [1, N] tensor held as row blocks of at most MAX_CHANNELS."""

    def each(x, fn):
        return Parts(fn(p) for p in x) if isinstance(x, Parts) else fn(x)

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = onnx.load(str(path)).graph
            self.inputs = [i.name for i in g.input]
            self.outputs = [o.name for o in g.output]
            self.consts = {t.name: numpy_helper.to_array(t) for t in g.initializer}
            self.nodes = []
            for n in g.node:
                if n.op_type == 'Constant':
                    self.consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
                else:
                    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
                    self.nodes.append((n.op_type, list(n.input), list(n.output), attrs))
            split = {name for op, ins, _, _ in self.nodes
                     if op == 'Gemm' and self.consts[ins[1]].shape[0] > MAX_CHANNELS for name in ins[1:3]}
            self.params = torch.nn.ParameterDict()
            self.key = {}
            for name, arr in self.consts.items():
                if arr.dtype not in (np.float32, np.float64) or arr.size <= 1:
                    continue
                self.key[name] = key = f'c{len(self.key)}'
                arr = arr.astype(np.float32)
                parts = [(f'{key}_{i}', arr[i:i + MAX_CHANNELS]) for i in range(0, len(arr), MAX_CHANNELS)] \
                    if name in split else [(key, arr)]
                for k, a in parts:  # split weights are stored only as their parts, so each stays a constant
                    self.params[k] = torch.nn.Parameter(torch.from_numpy(a.copy()), requires_grad=False)

        def t(self, env, name):
            return env[name] if name in env else self.params[self.key[name]]

        def ints(self, name):
            return [int(v) for v in np.asarray(self.consts[name]).reshape(-1)]

        def forward(self, *args):
            env = dict(zip(self.inputs, args))
            for op, ins, outs, a in self.nodes:
                if op not in ('Gemm', 'Clip', 'Reshape'):
                    assert not any(isinstance(env.get(i), Parts) for i in ins), op
                x = None if op == 'Concat' else self.t(env, ins[0])
                if op == 'Conv':
                    pads = a.get('pads', [0, 0, 0, 0])
                    assert pads[0] == pads[2] and pads[1] == pads[3]
                    y = F.conv2d(x, self.t(env, ins[1]), self.t(env, ins[2]) if len(ins) > 2 else None,
                                 stride=a.get('strides', [1, 1]), padding=pads[:2],
                                 dilation=a.get('dilations', [1, 1]), groups=a.get('group', 1))
                elif op == 'ConvTranspose':
                    assert a.get('pads', [0] * 4) == [0] * 4 and a.get('group', 1) == 1
                    assert a.get('dilations', [1, 1]) == [1, 1] and not a.get('output_padding')
                    y = F.conv_transpose2d(x, self.t(env, ins[1]), self.t(env, ins[2]), stride=a['strides'])
                elif op == 'Clip':
                    lo, hi = float(self.consts[ins[1]]), float(self.consts[ins[2]])
                    y = each(x, lambda v: F.hardtanh(v, lo, hi))
                elif op == 'Add':
                    y = x + self.t(env, ins[1])
                elif op == 'Mul':
                    y = x * self.t(env, ins[1])
                elif op == 'Sigmoid':
                    y = torch.sigmoid(x)
                elif op == 'Flatten':
                    assert a.get('axis', 1) == 1
                    y = torch.flatten(x, 1)
                elif op == 'Gemm':
                    assert a.get('transB') == 1 and a.get('alpha', 1.0) == 1.0 and a.get('beta', 1.0) == 1.0
                    w, b = self.key[ins[1]], self.key[ins[2]]
                    if w in self.params:
                        y = F.linear(x, self.params[w], self.params[b])
                    else:
                        rows = self.consts[ins[1]].shape[0]
                        y = Parts(F.linear(x, self.params[f'{w}_{i}'], self.params[f'{b}_{i}'])
                                  for i in range(0, rows, MAX_CHANNELS))
                elif op == 'BatchNormalization':
                    y = F.batch_norm(x, self.t(env, ins[3]), self.t(env, ins[4]), self.t(env, ins[1]),
                                     self.t(env, ins[2]), False, 0.0, a.get('epsilon', 1e-5))
                elif op == 'Slice':
                    starts, ends, axes = self.ints(ins[1]), self.ints(ins[2]), self.ints(ins[3])
                    steps = self.ints(ins[4]) if len(ins) > 4 else [1] * len(axes)
                    y = x
                    for s, e, ax, st in zip(starts, ends, axes, steps):
                        assert st == 1
                        y = y.narrow(ax, s, e - s)
                elif op == 'Reshape':
                    shape = self.ints(ins[1])
                    if isinstance(x, Parts):
                        n, _, h, w = shape
                        assert n == -1 and all(p.shape[1] % (h * w) == 0 for p in x)
                        y = torch.cat([p.reshape(1, p.shape[1] // (h * w), h, w) for p in x], dim=1)
                    else:
                        y = x.reshape(shape)
                elif op == 'Unsqueeze':
                    y = x
                    for ax in self.ints(ins[1]):
                        y = y.unsqueeze(ax)
                elif op == 'Concat':
                    y = torch.cat([self.t(env, i) for i in ins], dim=a['axis'])
                elif op == 'Pad':
                    assert all(v == 0 for v in self.ints(ins[1])), 'only the no-op pads Mercury exports'
                    y = x
                elif op == 'AveragePool':
                    assert a.get('ceil_mode', 0) == 0 and a.get('pads', [0] * 4) == [0] * 4
                    y = F.avg_pool2d(x, a['kernel_shape'], a['strides'])
                else:
                    raise NotImplementedError(f'{path.name}: ONNX operator {op}')
                env[outs[0]] = y
            return tuple(env[o] for o in self.outputs)

    return Module().eval()


def paired_module(module):
    """`module` on two input sets in one forward: (hand 0 inputs, hand 1 inputs) -> hand 0 outputs, hand 1's."""
    import torch

    class Pair(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = module
            self.n = len(module.inputs)

        def forward(self, *args):
            return tuple(self.net(*args[:self.n])) + tuple(self.net(*args[self.n:]))

    return Pair().eval()


def example_inputs(session, seed):
    """Inputs like Mercury's: images normalized to mean 0.5, std 0.25; keypoints in [-1, 1]; the flag set."""
    rng = np.random.default_rng(seed)
    out = []
    for i in session.get_inputs():
        shape = [int(d) for d in i.shape]
        if i.name == 'inputImg':
            out.append(np.clip(rng.normal(0.5, 0.25, shape), -1, 2).astype(np.float32))
        elif i.name == 'useLastKeypoints':
            out.append(np.ones(shape, np.float32))
        else:
            out.append(rng.uniform(-1, 1, shape).astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hexecutor-root', type=Path, default=ROOT.parents[2],
                    help='frame-spatial-runtime checkout that holds hexecutor/offline/executorch (default: this one)')
    ap.add_argument('--models-dir', required=True, type=Path,
                    help='Steam-Frame-Hand-Tracking/models (holds fp32/, from fetch_models.py)')
    ap.add_argument('--out', type=Path, help='default: <models-dir>/executorch')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    root = a.hexecutor_root.resolve()
    if not (root / 'hexecutor' / 'offline' / 'executorch' / 'export.py').is_file():
        raise SystemExit(f'{root} has no hexecutor/offline/executorch/export.py')
    sys.path.insert(0, str(root))

    import onnxruntime as ort
    import torch
    from hexecutor.offline.executorch.export import export_program

    out = (a.out or a.models_dir / 'executorch').resolve()
    (out / 'check').mkdir(parents=True, exist_ok=True)
    manifest = {'hexecutor_root_revision': None, 'models': {}}
    head = root / '.git'
    if head.exists():
        import subprocess
        manifest['hexecutor_root_revision'] = subprocess.run(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
    for name in list(MODELS) + list(PAIRED):
        source = PAIRED.get(name, name)
        digest = MODELS[source]
        onnx_path = a.models_dir / 'fp32' / f'{source}.onnx'
        if sha256(onnx_path) != digest:
            raise SystemExit(f'{onnx_path}: SHA-256 does not match the pin (run models/fetch_models.py)')
        module = mercury_module(onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        inputs = example_inputs(session, a.seed)
        expected = session.run(None, {i.name: v for i, v in zip(session.get_inputs(), inputs)})
        if name in PAIRED:  # hand 1 gets different inputs, so a swapped or shared output shows in the check
            second = example_inputs(session, a.seed + 1)
            expected += session.run(None, {i.name: v for i, v in zip(session.get_inputs(), second)})
            inputs += second
            module = paired_module(module)
        with torch.no_grad():
            got = module(*[torch.from_numpy(v) for v in inputs])
        torch_err = max(float(np.abs(e - g.numpy()).max()) for e, g in zip(expected, got))
        if torch_err > 1e-4:
            raise SystemExit(f'{name}: PyTorch rebuild differs from ONNX Runtime by {torch_err}')

        program = export_program(module, tuple(torch.from_numpy(v) for v in inputs))
        graph = program.exported_program().graph_module.graph
        calls = [str(n.target) for n in graph.nodes if n.op == 'call_function']
        delegates = calls.count('executorch_call_delegate')
        cpu = sorted({c for c in calls if c not in ('executorch_call_delegate', '<built-in function getitem>')})
        if delegates != 1 or cpu:
            raise SystemExit(f'{name}: not one whole delegate ({delegates} delegates, CPU operators {cpu})')
        pte = out / f'{name}.pte'
        pte.write_bytes(program.buffer)
        for i, v in enumerate(inputs):
            v.tofile(out / 'check' / f'{name}.in{i}.f32')
        for j, v in enumerate(expected):
            np.asarray(v, np.float32).tofile(out / 'check' / f'{name}.out{j}.f32')
        manifest['models'][name] = {
            'onnx_sha256': digest, 'pte': pte.name, 'pte_sha256': sha256(pte), 'pte_bytes': pte.stat().st_size,
            'inputs': [list(v.shape) for v in inputs], 'outputs': [list(np.shape(v)) for v in expected],
            'torch_vs_onnxruntime_max_abs': torch_err, 'seed': a.seed}
        print(f'{name}: {pte} ({pte.stat().st_size} bytes), one delegate, torch vs ONNX Runtime {torch_err:.2e}')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
