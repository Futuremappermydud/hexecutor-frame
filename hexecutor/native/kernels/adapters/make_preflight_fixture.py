#!/usr/bin/env python3
"""Encode actual primary QDRX candidate metadata for the host C prepare gate."""

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
import json
import struct
from pathlib import Path
import sys

FRAME = (PATHS.native)
sys.path.insert(0, str(FRAME))
import depth_model.lowering.resident_image as resident_image
MAGIC = 0x31505151  # QQP1


def tensor_meta(t):
    s = t['shape']
    return s['dimensions'] + s['max_dimensions'] + s['padding'] + [
        t['storage'].get('slice_offset', 0), len(t['storage']['blocks'])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=Path, default=(PATHS.native / 'build/resident-primary.qdrx'))
    ap.add_argument('--census', type=Path, default=(PATHS.native / 'kernels/adapters/qdrx-1x1-census.json'))
    ap.add_argument('--output', type=Path, default=(PATHS.native / 'kernels/adapters/preflight-primary.bin'))
    args = ap.parse_args()
    graph = resident_image.unpack(args.image.read_bytes())
    tensors = {t['id']: t for t in graph['tensors']}
    census = json.loads(args.census.read_text())['images'][0]
    pools = graph['pool_bytes']
    pool_sizes = [0] * 7
    for k, v in pools.items():
        pool_sizes[6 if k == 'spill' else int(k)] = v
    out = bytearray(struct.pack('<9I', MAGIC, len(census['commands']), *pool_sizes))
    for row in census['commands']:
        x, wt, fi, y = (tensors[row[k]] for k in ('input_id', 'weight_id', 'fi_id', 'output_id'))
        wref, fref = wt['storage']['direct'], fi['storage']['direct']
        wt_dims = wt['shape']['dimensions']
        word = [row['sequence']] + tensor_meta(x) + tensor_meta(y) + [
            wt_dims[2], wt_dims[3], wt_dims[0], wt_dims[1],
            1, 1, 2, 5, 1 if wt['class_name'] == 'Xpb' else 2,
            wref['pool'], wref['offset'], row['weights_bytes'],
            fref['pool'], fref['offset'], row['fi_bytes']]
        assert len(word) == 44
        out.extend(struct.pack('<44I', *word))
        out.extend(struct.pack(f'<{len(x["storage"]["blocks"])}I', *[
            (b['offset'] | b['pool']) for b in x['storage']['blocks']]))
        out.extend(struct.pack(f'<{len(y["storage"]["blocks"])}I', *[
            (b['offset'] | b['pool']) for b in y['storage']['blocks']]))
    args.output.write_bytes(out)
    print(f'wrote {args.output}: {len(census["commands"])} actual primary commands')


if __name__ == '__main__':
    main()
