#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture sequence-1022 Xwb/FI after the CPU QDRX prefix and build a C oracle header."""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "depth-model/python"),
                             str(_source_root / "depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = (PATHS.repo)
FRAME = (PATHS.native)
sys.path[:0] = [str(FRAME), str((PATHS.model_tools))]
from check_kernels import load_inputs
from depth_model.adapters import NativeGraph
from depth_model.lowering.resident_image import unpack
from htp_emu import Memory, conv_accumulate, conv_requant

image = FRAME / 'build/resident-primary.qdrx'
constants = (PATHS.model_data / 'recovered/primary/constants/constant_pool_5.bin')
inputs_dir = FRAME / 'build/depth-inputs/random'
img = unpack(image.read_bytes())
idx = next(i for i, c in enumerate(img['commands']) if c['sequence'] == 1022)
cmd = img['commands'][idx]
tensors = {t['id']: t for t in img['tensors']}
input_tid, weights_tid, fi_tid = cmd['inputs'][:3]

def direct_bytes(host, tid):
    t = tensors[tid]
    ref = t['storage']['direct']
    n = t['element_bytes'] * int(np.prod(t['shape']['dimensions']))
    b = host.pools[ref['pool']][ref['offset']:ref['offset'] + n].tobytes()
    return b, ref

# Capture activation pages before the target op in case input/output tensors alias pool pages.
host = NativeGraph(image, constants)
try:
    host.use_cpu()
    host.run(load_inputs(inputs_dir), last=idx)
    input_tensor = tensors[input_tid]
    input_pages = b''.join(bytes(host.pools[e['pool']][e['offset']:e['offset'] + 2048])
                           for e in input_tensor['storage']['blocks'])
    mem = Memory(img, constants.read_bytes())
    mem.pools[1] = host.pools[1]
    actual_activation = mem.read(input_tid)[0]
finally:
    host.close()

# Capture target weights/FI and the exact CPU output after the target op.
host = NativeGraph(image, constants)
try:
    host.use_cpu()
    host.run(load_inputs(inputs_dir), last=idx + 1)
    weights_raw, weights_ref = direct_bytes(host, weights_tid)
    fi_raw, fi_ref = direct_bytes(host, fi_tid)
    mem = Memory(img, constants.read_bytes())
    mem.pools[1] = host.pools[1]
    cpu_output = mem.read(cmd['outputs'][0]).tobytes()
finally:
    host.close()

# Xwb element address from export_constants.py; channels are [I/4, O, I%4].
def xwb_offset(ky, kx, ci, co):
    return ((((co // 32 * 1 + ci // 32) * 3 + ky) * 3 + kx) * 1024
            + ((ci // 4) % 8) * 128 + (co % 32) * 4 + ci % 4)
weights = np.empty((3, 3, 32, 32), np.int8)
for y in range(3):
    for x in range(3):
        for i in range(32):
            for o in range(32):
                weights[y, x, i, o] = np.frombuffer(weights_raw, np.int8)[xwb_offset(y, x, i, o)]
weights = weights[:, ::-1]  # Xwb stride-1 semantic: reverse stored width.
fi = np.frombuffer(fi_raw, dtype='<i4').reshape(2, 32)

# The simulator fixture uses this distinct deterministic logical activation.
y, x, c = np.indices((10, 258, 32), dtype=np.uint32)
activation = (1 + ((y * 29 + x * 13 + c * 7) % 250)).astype(np.uint8)
acc = conv_accumulate(activation, weights, 0)
acc += fi[1].astype(np.int64)[None, None, :]
oracle = conv_requant(acc, fi[0], False).astype(np.uint8)
simple_mult = np.full(32, 0x406000, np.int32)
simple_acc = conv_accumulate(activation, weights, 0)
simple_oracle = conv_requant(simple_acc, simple_mult, False).astype(np.uint8)
actual_simple_acc = conv_accumulate(actual_activation, weights, 0)
actual_simple_oracle = conv_requant(actual_simple_acc, simple_mult, False).astype(np.uint8)
bias_only = np.broadcast_to(conv_requant(fi[1][None, None, :].astype(np.int64), fi[0], False), (8, 256, 32)).astype(np.uint8)
dense_mult = np.full(32, 0x403c00, np.int32)
dense_bias = np.full(32, 65536, np.int32)
dense_acc = conv_accumulate(np.ones((10, 258, 32), np.uint8), weights, 0)
dense_acc += dense_bias[None, None, :]
dense_safe_oracle = conv_requant(dense_acc, dense_mult, False).astype(np.uint8)
actual_low_acc = conv_accumulate(actual_activation, weights, 0)
actual_low_acc += dense_bias[None, None, :]
actual_input_lowfi_oracle = conv_requant(actual_low_acc, dense_mult, False).astype(np.uint8)
ones_actual_acc = conv_accumulate(np.ones((10, 258, 32), np.uint8), weights, 0)
ones_actual_acc += fi[1][None, None, :]
ones_actualfi_oracle = conv_requant(ones_actual_acc, fi[0], False).astype(np.uint8)
assert oracle.shape == (8, 256, 32)

outdir = (PATHS.native / 'kernels/convolution')
((PATHS.native / 'kernels/convolution/xwb1022_weights.bin')).write_bytes(weights_raw)
((PATHS.native / 'kernels/convolution/xwb1022_fi.bin')).write_bytes(fi_raw)
((PATHS.native / 'kernels/convolution/xwb1022_input_pages.bin')).write_bytes(input_pages)
((PATHS.native / 'kernels/convolution/xwb1022_cpu_output.bin')).write_bytes(cpu_output)
((PATHS.native / 'kernels/convolution/xwb1022_oracle.bin')).write_bytes(oracle.tobytes())
def fnv64(data):
    h=0xcbf29ce484222325
    for v in data: h=((h ^ int(v))*0x100000001b3)&0xffffffffffffffff
    return h
with ((PATHS.native / 'kernels/convolution/xwb1022_actual_data.h')).open('w') as f:
    f.write('#include <stdint.h>\n')
    f.write('/* Generated by make_xwb1022_oracle.py; QDRX sequence 1022 CPU prefix. */\n')
    f.write(f'#define ACTUAL_INPUT_FNV64 UINT64_C(0x{fnv64(input_pages):016x})\n')
    f.write(f'#define ACTUAL_WEIGHTS_FNV64 UINT64_C(0x{fnv64(weights_raw):016x})\n')
    f.write(f'#define ACTUAL_FI_FNV64 UINT64_C(0x{fnv64(fi_raw):016x})\n')
    for name, arr, ctype in [('actual_input_pages', np.frombuffer(input_pages, np.uint8), 'uint8_t'),
                             ('actual_weights', np.frombuffer(weights_raw, np.uint8), 'uint8_t'),
                             ('actual_fi', np.frombuffer(fi_raw, dtype='<i4'), 'int32_t'),
                             ('oracle', np.frombuffer(cpu_output, np.uint8), 'uint8_t'),
                             ('synthetic_oracle', oracle.reshape(-1), 'uint8_t'),
                             ('simple_oracle', simple_oracle.reshape(-1), 'uint8_t'),
                             ('actual_simple_oracle', actual_simple_oracle.reshape(-1), 'uint8_t'),
                             ('bias_only_oracle', bias_only.reshape(-1), 'uint8_t'),
                             ('dense_safe_oracle', dense_safe_oracle.reshape(-1), 'uint8_t'),
                             ('actual_input_lowfi_oracle', actual_input_lowfi_oracle.reshape(-1), 'uint8_t'),
                             ('ones_actualfi_oracle', ones_actualfi_oracle.reshape(-1), 'uint8_t')]:
        f.write(f'static const {ctype} {name}[{arr.size}] = {{\n')
        for i in range(0, arr.size, 24):
            vals = arr[i:i+24]
            if ctype == 'int32_t': text = ','.join(str(int(v)) for v in vals)
            else: text = ','.join(f'0x{int(v):02x}' for v in vals)
            f.write('  ' + text + ',\n')
        f.write('};\n')
report = {
    'image': str(image.relative_to(ROOT)), 'sequence': 1022, 'command_index': idx,
    'variant': cmd['variant'], 'input_tensor_id': input_tid,
    'weights_tensor_id': weights_tid, 'weights_ref': weights_ref, 'weights_bytes': len(weights_raw),
    'weights_sha256': hashlib.sha256(weights_raw).hexdigest(), 'fi_tensor_id': fi_tid,
    'fi_ref': fi_ref, 'fi_bytes': len(fi_raw), 'fi_sha256': hashlib.sha256(fi_raw).hexdigest(),
    'synthetic_oracle_sha256': hashlib.sha256(oracle.tobytes()).hexdigest(),
    'cpu_output_sha256': hashlib.sha256(cpu_output).hexdigest(),
    'actual_activation_sha256': hashlib.sha256(actual_activation.tobytes()).hexdigest(),
    'actual_simple_oracle_sha256': hashlib.sha256(actual_simple_oracle.tobytes()).hexdigest(),
    'actual_input_lowfi_oracle_sha256': hashlib.sha256(actual_input_lowfi_oracle.tobytes()).hexdigest(),
    'ones_actualfi_oracle_sha256': hashlib.sha256(ones_actualfi_oracle.tobytes()).hexdigest(),
    'actual_fi_first_multiplier_hex': hex(int(fi[0,0])), 'actual_fi_channel4_bias': int(fi[1,4]),
    'raw_acc_actual_input_minmax': [int(actual_low_acc.min()), int(actual_low_acc.max())],
    'raw_acc_ones_actualfi_minmax': [int(ones_actual_acc.min()), int(ones_actual_acc.max())],
    'cpu_output_method': 'NativeGraph CPU QDRX prefix through command index 836, then htp_emu.Memory logical read of output tensor',
    'input_pages_bytes': len(input_pages), 'input_pages_sha256': hashlib.sha256(input_pages).hexdigest(),
    'execution': 'NativeGraph CPU QDRX prefix through command index 836; no DSP/device'
}
((PATHS.native / 'kernels/convolution/xwb1022_oracle.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
