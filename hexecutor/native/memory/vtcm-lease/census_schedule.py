#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Offline byte census for one serialized QDRX schedule.

Usage: python3 frame/vtcm_lease/census_schedule.py [path/to/image.qdrx]
The default is frame/build/resident-primary.qdrx. This reads no hardware.
"""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import hashlib
import json
import math
from pathlib import Path
import sys

FRAME = (PATHS.native)
sys.path.insert(0, str(FRAME))
import depth_model.lowering.resident_image as resident_image
DDR_POOLS = {3, 4, 5, 6}  # spill pool 6 is private DDR too
VTCM_POOL = 1
GBPS = 10_000_000_000  # optimistic measured bulk-copy envelope, decimal bytes/s


def pool_tags(t):
    storage = t.get('storage') or {}
    if storage.get('direct'):
        return {storage['direct']['pool']}
    return {b['pool'] for b in storage.get('blocks', [])}


def max_bytes(t):
    shape = t.get('shape')
    if not shape:
        return 0
    return math.prod(shape['max_dimensions']) * t['element_bytes']


def transfer_direction(src, dst):
    """Return a boundary direction only for an unambiguous pool crossing."""
    if VTCM_POOL in src and src <= {VTCM_POOL} and dst and dst <= DDR_POOLS:
        return 'VTCM_to_DDR'
    if VTCM_POOL in dst and dst <= {VTCM_POOL} and src and src <= DDR_POOLS:
        return 'DDR_to_VTCM'
    return None


def dense_window_bytes(command, tensors):
    """Mirror k_hmx_conv's rounded input slice dimensions for a dense path.

    This is staging volume, not scheduled boundary traffic. A direct blocked
    HMX path may bypass this materialization. Once pool 1 is VTCM, it is local
    VTCM movement and must not be counted as DDR traffic.
    """
    if len(command['inputs']) < 2 or not command['outputs']:
        return 0
    x, w, out = (tensors[command['inputs'][0]], tensors[command['inputs'][1]],
                 tensors[command['outputs'][0]])
    if not all('shape' in t for t in (x, w, out)):
        return 0
    wd = w['shape']['max_dimensions']
    od = out['shape']['dimensions']
    wm = out['shape']['max_dimensions']
    kh, kw, ci = wd[:3]
    xpad = x['shape']['padding']
    xextent = x['shape']['max_dimensions']
    oh = min((od[1] + 7) // 8 * 8, wm[1], xextent[1] - xpad[1] - kh + 1)
    sixteen = x['class_name'] in ('CH', 'cH')
    cw = 4 if sixteen else 8
    ow = min((od[2] + cw - 1) // cw * cw, wm[2], xextent[2] - xpad[2] - kw + 1)
    if oh <= 0 or ow <= 0:
        return 0
    h, width = oh + kh - 1, ow + kw - 1
    if kh == 1 and kw == 1:
        h, width = 1, h * width
    return h * width * ci * x['element_bytes']


def census(image):
    raw = image.read_bytes()
    graph = resident_image.unpack(raw)
    tensors = {t['id']: t for t in graph['tensors']}
    totals = {'weights_to_vtcm': {'DDR_to_VTCM': 0, 'VTCM_to_DDR': 0, 'commands': 0, 'edges': {}},
              'spill_fill': {'DDR_to_VTCM': 0, 'VTCM_to_DDR': 0, 'fill_commands': 0, 'spill_commands': 0},
              'activation_copy': {'DDR_to_VTCM': 0, 'VTCM_to_DDR': 0, 'commands': 0, 'edges': {}}}
    unclassified = []
    dense_staging = 0
    dense_commands = 0

    for c in graph['commands']:
        if c['opcode'] == 4 and c['inputs'] and c['outputs']:
            src, dst = tensors[c['inputs'][0]], tensors[c['outputs'][0]]
            n = min(max_bytes(src), max_bytes(dst))
            direction = transfer_direction(pool_tags(src), pool_tags(dst))
            if direction:
                row = totals['weights_to_vtcm']
                row[direction] += n
                row['commands'] += 1
                edge = f"{','.join(map(str, sorted(pool_tags(src))))}->{','.join(map(str, sorted(pool_tags(dst))))}"
                row['edges'][edge] = row['edges'].get(edge, 0) + n
            else:
                unclassified.append({'sequence': c['sequence'], 'kind': 'weights_to_vtcm',
                                     'src_pools': sorted(pool_tags(src)), 'dst_pools': sorted(pool_tags(dst)), 'bytes': n})

        elif c['opcode'] == 17 and c['inputs'] and c['outputs']:
            src, dst = tensors[c['inputs'][0]], tensors[c['outputs'][0]]
            n = min(max_bytes(src), max_bytes(dst))
            direction = transfer_direction(pool_tags(src), pool_tags(dst))
            if direction:
                row = totals['activation_copy']
                row[direction] += n
                row['commands'] += 1
                edge = f"{','.join(map(str, sorted(pool_tags(src))))}->{','.join(map(str, sorted(pool_tags(dst))))}"
                row['edges'][edge] = row['edges'].get(edge, 0) + n
            else:
                unclassified.append({'sequence': c['sequence'], 'kind': 'activation_copy',
                                     'src_pools': sorted(pool_tags(src)), 'dst_pools': sorted(pool_tags(dst)), 'bytes': n})

        elif c['opcode'] == 26:
            fv = c['framework_vector']
            if len(fv) < 4 or (len(fv) - 4) % 2:
                raise ValueError(f"malformed spill/fill vector at command {c['sequence']}")
            spill = c['variant'] == '@Spill'
            row = totals['spill_fill']
            touched = False
            for i in range(4, len(fv), 2):
                p, n = fv[i], fv[i + 1]
                pool = p & 0x7ff
                if pool == VTCM_POOL:
                    direction = 'VTCM_to_DDR' if spill else 'DDR_to_VTCM'
                    row[direction] += n
                    touched = True
            if touched:
                row['spill_commands' if spill else 'fill_commands'] += 1

        elif c['opcode'] == 6:
            n = dense_window_bytes(c, tensors)
            dense_staging += n
            dense_commands += 1

    vtcm_bytes = sum(totals[k]['DDR_to_VTCM'] + totals[k]['VTCM_to_DDR']
                     for k in ('weights_to_vtcm', 'spill_fill', 'activation_copy'))
    inputs = sum(max_bytes(tensors[i]) for i in graph['graph_inputs'])
    outputs = sum(max_bytes(tensors[i]) for i in graph['graph_outputs'])
    combined = vtcm_bytes + inputs + outputs
    return {
        'image': str(image), 'image_sha256': hashlib.sha256(raw).hexdigest(),
        'commands': len(graph['commands']), 'pool_bytes': graph['pool_bytes'],
        'transfers': totals,
        'required_ddr_vtcm_bytes': vtcm_bytes,
        'graph_input_bytes_host_to_ddr': inputs,
        'graph_output_bytes_ddr_to_host': outputs,
        'host_io_bytes_separate_from_vtcm': inputs + outputs,
        'all_scheduled_boundary_bytes_including_host_io': combined,
        'optimistic_10GBps_ms': {
            'ddr_vtcm_only': vtcm_bytes / GBPS * 1000,
            'including_host_io': combined / GBPS * 1000,
            'remaining_of_20ms_after_ddr_vtcm': 20 - vtcm_bytes / GBPS * 1000,
        },
        'standard_conv_dense_staging': {
            'conv_commands': dense_commands,
            'window_bytes_if_all_use_dense_path': dense_staging,
            'interpretation': 'upper-bound activation materialization volume; direct blocked HMX may bypass it. With pool 1 native VTCM, this is local VTCM traffic, not DDR traffic.'
        },
        'unclassified_op4_or_op17': unclassified,
        'limitations': '10 GB/s is an optimistic bulk-copy envelope only. This omits latency, DMA setup, barriers, graph compute, HMX contention, cache effects, eye-client contention, and overlap. It is not a 20 ms performance guarantee.'
    }


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (PATHS.native / 'build/resident-primary.qdrx')
    print(json.dumps(census(path), indent=2))
