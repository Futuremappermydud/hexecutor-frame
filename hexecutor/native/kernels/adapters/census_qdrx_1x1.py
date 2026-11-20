#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Offline census of actual compiled QDRX 1x1 HMX command descriptors."""

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
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

FRAME = (PATHS.native)
sys.path.insert(0, str(FRAME))
import depth_model.lowering.resident_image as resident_image
WS_LIMIT = 512 * 1024
def active_block_indices(t, h, w, channels):
    shape = t['shape']
    maxd = shape['max_dimensions']
    nbd = (maxd[3] + 31) // 32
    nbw = (maxd[2] + 7) // 8
    slice_offset = t['storage'].get('slice_offset', 0)
    out = []
    for ty in range((h + 7) // 8):
        for tx in range((w + 7) // 8):
            for cb in range(channels // 32):
                out.append(((ty * nbw + tx) * nbd) + slice_offset + cb)
    return out


def inspect(path):
    raw = path.read_bytes()
    graph = resident_image.unpack(raw)
    tensors = {t['id']: t for t in graph['tensors']}
    convs = [c for c in graph['commands'] if 'ConvLayer_s1.opt' in c['variant'] and
             len(c['inputs']) >= 3 and c['outputs']]
    reasons = Counter()
    eligible = []
    one_by_one = 0
    for c in convs:
        x, wt, fi = (tensors[i] for i in c['inputs'][:3])
        y = tensors[c['outputs'][0]]
        xd = x.get('shape', {}).get('dimensions', [])
        xm = x.get('shape', {}).get('max_dimensions', [])
        xp = x.get('shape', {}).get('padding', [])
        wd = wt.get('shape', {}).get('dimensions', [])
        yd = y.get('shape', {}).get('dimensions', [])
        ym = y.get('shape', {}).get('max_dimensions', [])
        yp = y.get('shape', {}).get('padding', [])
        fd = fi.get('shape', {}).get('dimensions', [])
        reason = None
        if len(wd) != 4 or wd[:2] != [1, 1]:
            continue
        one_by_one += 1
        if c['variant'] not in ('ConvLayer_s1.opt@CB*2.Xpb.Fi.t',
                                'ConvLayer_s1.opt@CB*2.Xwb.Fi.t'):
            reason = 'variant'
        elif (x['class_name'], x['dtype'], y['class_name'], y['dtype']) != ('CB', 'uint8', 'CB', 'uint8'):
            reason = 'activation_dtype_or_layout'
        elif (wt['class_name'], wt['dtype'], fi['class_name'], fi['dtype']) not in (
                ('Xpb', 'int8', 'Fi', 'int32'), ('Xwb', 'int8', 'Fi', 'int32')):
            reason = 'filter_or_fi_type'
        elif not (len(xd) == len(xm) == len(xp) == len(yd) == len(ym) == len(yp) == 4 and len(fd) == 4):
            reason = 'shape'
        elif (xd[0] != 1 or yd[0] != 1 or wd[2] != xd[3] or wd[3] != yd[3] or
              xm[3] != wd[2] or ym[3] != wd[3]):
            reason = 'batch_or_channels'
        elif wd[2] % 32 or wd[3] % 32:
            reason = 'channels_not_32_aligned'
        elif fd[0] * fd[1] * fd[2] * fd[3] * 4 != wd[3] * 8:
            reason = 'fi_footprint'
        elif any(xp[i] % 8 or yp[i] % 8 for i in (1, 2)):
            reason = 'unaligned_crop_or_padding'
        elif 'blocks' not in x.get('storage', {}) or 'blocks' not in y.get('storage', {}):
            reason = 'not_blocked'
        elif 'direct' not in wt.get('storage', {}) or 'direct' not in fi.get('storage', {}):
            reason = 'filter_fi_not_direct'
        if reason:
            reasons[reason] += 1
            continue
        oh = min((yd[1] + 7) // 8 * 8, ym[1], xm[1] - xp[1])
        ow = min((yd[2] + 7) // 8 * 8, ym[2], xm[2] - xp[2])
        if oh <= 0 or ow <= 0:
            reasons['empty_effective_geometry'] += 1
            continue
        ph, pw = (oh + 7) // 8 * 8, (ow + 7) // 8 * 8
        ib = (ph // 8) * (pw // 8) * (wd[2] // 32)
        ob = (ph // 8) * (pw // 8) * (wd[3] // 32)
        if ib > 256 or ob > 256:
            reasons['pointer_table_capacity'] += 1
            continue
        inp = active_block_indices(x, ph, pw, wd[2])
        out = active_block_indices(y, ph, pw, wd[3])
        if max(inp, default=-1) >= len(x['storage']['blocks']) or max(out, default=-1) >= len(y['storage']['blocks']):
            reasons['physical_table_short'] += 1
            continue
        output_refs = [y['storage']['blocks'][i] for i in out]
        if len(set((b['pool'], b['offset']) for b in output_refs)) != len(output_refs):
            reasons['output_block_alias'] += 1
            continue
        # Native leaf ABI uses the proven fixed partition through two distinct
        # guard croutons (X table +0x56000, Y table +0x54000).
        used = 0x56800
        row = {'sequence': c['sequence'], 'variant': c['variant'],
               'input_id': x['id'], 'weight_id': wt['id'], 'fi_id': fi['id'],
               'output_id': y['id'], 'input_hw': xd[1:3], 'input_max_hw': xm[1:3],
               'output_hw': yd[1:3], 'output_max_hw': ym[1:3],
               'ci': wd[2], 'co': wd[3], 'effective_hw': [oh, ow],
               'physical_hw': [ph, pw], 'slice_offsets': [x['storage']['slice_offset'], y['storage']['slice_offset']],
               'input_blocks': ib, 'output_blocks': ob, 'weights_bytes': wd[2] * wd[3],
               'fi_bytes': wd[3] * 8, 'workspace_bytes': used}
        if used > WS_LIMIT:
            reasons['workspace_over_512k'] += 1
            continue
        eligible.append(row)
    return {'image': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'command_count': len(graph['commands']), 'conv_s1_commands': len(convs),
            'one_by_one_commands': one_by_one,
            'eligible_commands': len(eligible), 'excluded': dict(sorted(reasons.items())),
            'workspace_max_bytes': max((x['workspace_bytes'] for x in eligible), default=0),
            'signatures': {f'{k[0]} CI{k[1]} CO{k[2]}': n for k, n in
                            sorted(Counter((x['variant'], x['ci'], x['co']) for x in eligible).items())},
            'commands': eligible}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='*', type=Path,
                    default=[(PATHS.native / 'build/resident-primary.qdrx'), (PATHS.native / 'build/resident-alternate.qdrx')])
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    reports = [inspect(p) for p in args.images]
    result = {'offline_only': True, 'workspace_limit_bytes': WS_LIMIT, 'images': reports}
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    for report in reports:
        print(f"{Path(report['image']).name}: {report['eligible_commands']}/"
              f"{report['one_by_one_commands']} 1x1 candidates; "
              f"workspace max {report['workspace_max_bytes']} B; "
              f"{report['signatures']}")


if __name__ == '__main__':
    main()
