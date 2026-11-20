#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture a QDRX low-CI command and zero-expand its Xpb filter to CI32."""

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

import argparse
import struct
import sys
from pathlib import Path

FRAME = (PATHS.native)
ROOT = (PATHS.repo)
sys.path[:0] = [str(FRAME), str(ROOT / 'tools')]
from depth_model.adapters import NativeGraph
from depth_model.lowering.resident_image import unpack
from check_kernels import load_inputs

MAGIC = 0x31585151
SUPPORTED = {115: 12, 3416: 20}


def block_bytes(tensor, pools):
    return b''.join(bytes(pools[entry['pool']][entry['offset']:entry['offset'] + 2048])
                    for entry in tensor['storage']['blocks'])


def direct_bytes(tensor, pools):
    storage = tensor['storage']['direct']
    size = tensor['element_bytes']
    for dim in tensor['shape']['dimensions']:
        size *= dim
    start = storage['offset']
    return bytes(pools[storage['pool']][start:start + size])


def xpb_expand_ci32(raw, ci, co):
    if ci not in SUPPORTED.values() or len(raw) != ci * co or co % 32:
        raise ValueError('unexpected original Xpb shape')
    expanded = bytearray(32 * co)
    for out_ch in range(co):
        for in_ch in range(ci):
            source = ((out_ch // 32) * (ci // 4) + in_ch // 4) * 128
            source += (out_ch % 32) * 4 + in_ch % 4
            target = ((out_ch // 32) * (32 // 4) + in_ch // 4) * 128
            target += (out_ch % 32) * 4 + in_ch % 4
            expanded[target] = raw[source]
    return bytes(expanded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, default=(PATHS.native / 'build/resident-primary.qdrx'))
    parser.add_argument('--inputs', type=Path, default=(PATHS.native / 'build/depth-inputs/random'))
    parser.add_argument('--sequence', type=int, choices=SUPPORTED, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    image = unpack(args.image.read_bytes())
    command = next(c for c in image['commands'] if c['sequence'] == args.sequence)
    command_index = image['commands'].index(command)
    tensors = {t['id']: t for t in image['tensors']}
    x, weight, fi = (tensors[i] for i in command['inputs'][:3])
    y = tensors[command['outputs'][0]]
    original_ci = weight['shape']['dimensions'][2]
    co = weight['shape']['dimensions'][3]
    if (original_ci != SUPPORTED[args.sequence] or weight['class_name'] != 'Xpb' or
            weight['shape']['dimensions'][:2] != [1, 1] or
            x['shape']['dimensions'][3] >= 32 or x['shape']['max_dimensions'][3] != 32):
        raise ValueError('command does not match the low-CI padded experiment')

    inputs = load_inputs(args.inputs)
    graph = NativeGraph(args.image, ROOT / 'recovered/primary/constants/constant_pool_5.bin')
    try:
        graph.use_cpu()
        graph.run(inputs, last=command_index)
        x_blocks = block_bytes(x, graph.pools)
        y_before = block_bytes(y, graph.pools)
        original_weights = direct_bytes(weight, graph.pools)
        fi_bytes = direct_bytes(fi, graph.pools)
        expanded_weights = xpb_expand_ci32(original_weights, original_ci, co)
        graph.run(inputs, first=command_index, last=command_index + 1)
        y_reference = block_bytes(y, graph.pools)
    finally:
        graph.close()

    if len(fi_bytes) != co * 8 or len(y_before) != len(y_reference):
        raise ValueError('invalid FI or output block data')
    vals = [MAGIC, args.sequence,
            x['shape']['dimensions'][1], x['shape']['dimensions'][2],
            x['shape']['max_dimensions'][1], x['shape']['max_dimensions'][2],
            x['storage']['slice_offset'], len(x['storage']['blocks']),
            y['shape']['dimensions'][1], y['shape']['dimensions'][2],
            y['shape']['max_dimensions'][1], y['shape']['max_dimensions'][2],
            y['storage']['slice_offset'], len(y['storage']['blocks']),
            32, co, 1, len(expanded_weights), len(fi_bytes),
            x['shape']['padding'][1], x['shape']['padding'][2],
            y['shape']['padding'][1], y['shape']['padding'][2],
            x.get('quantization', {}).get('zero_point', 0),
            original_ci, len(original_weights), len(y_reference)]
    vals.append(x['shape']['dimensions'][3])
    vals += [0] * (32 - len(vals))
    payload = (struct.pack('<32I', *vals) + x_blocks + y_before + expanded_weights +
               fi_bytes + original_weights + y_reference)
    destination = args.output or (PATHS.native / f'kernels/adapters/padded-ci-{args.sequence}.bin')
    destination.write_bytes(payload)
    print(f'wrote sequence {args.sequence}: input={len(x_blocks)} output={len(y_before)} '
          f'Xpb{original_ci}->{32}={len(expanded_weights)} FI={len(fi_bytes)} '
          f'originalW={len(original_weights)} cpuOutput={len(y_reference)} bytes')


if __name__ == '__main__':
    main()
