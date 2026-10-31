#!/usr/bin/env python3
"""Make a deterministic input fixture with command 1195's exact raw Xpb/FI."""
import argparse
import struct
from pathlib import Path
import sys

FRAME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAME))
import resident_image

MAGIC = 0x31585151  # QQX1
SEQ = 1195


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=Path, default=FRAME / 'build/resident-primary.qdrx')
    ap.add_argument('--constants', type=Path, default=FRAME.parent / 'recovered/primary/constants/constant_pool_5.bin')
    ap.add_argument('--output', type=Path, default=Path(__file__).with_name('fixture-qdrx-1195.bin'))
    args = ap.parse_args()
    graph = resident_image.unpack(args.image.read_bytes())
    tensors = {t['id']: t for t in graph['tensors']}
    command = next(c for c in graph['commands'] if c['sequence'] == SEQ)
    x, weight, fi = (tensors[i] for i in command['inputs'][:3])
    y = tensors[command['outputs'][0]]
    assert command['variant'] == 'ConvLayer_s1.opt@CB*2.Xpb.Fi.t'
    assert x['shape']['dimensions'] == [1, 8, 64, 128]
    assert weight['shape']['dimensions'] == [1, 1, 128, 128]
    assert fi['shape']['dimensions'] == [1, 4, 1, 64]
    producers = {tid: c for c in graph['commands'] for tid in c['outputs']}
    weight_source = producers[weight['id']]['inputs'][0]
    fi_source = producers[fi['id']]['inputs'][0]
    weight_raw, fi_tensor = tensors[weight_source], tensors[fi_source]
    constants = args.constants.read_bytes()

    def constant_bytes(t):
        ref = t['storage']['direct']
        assert ref['pool'] == 5
        size = 1
        for d in t['shape']['dimensions']:
            size *= d
        size *= t['element_bytes']
        return constants[ref['offset']:ref['offset'] + size]

    weights, fi_raw = constant_bytes(weight_raw), constant_bytes(fi_tensor)
    h, w, ci, co = 8, 64, 128, 128
    blocks = bytearray((h // 8) * (w // 8) * (ci // 32) * 2048)
    for ty in range(h // 8):
        for tx in range(w // 8):
            for cb in range(ci // 32):
                b = ((ty * (w // 8) + tx) * (ci // 32) + cb) * 2048
                for ry in range(8):
                    for rx in range(8):
                        for lane in range(32):
                            y0, x0, c0 = ty * 8 + ry, tx * 8 + rx, cb * 32 + lane
                            # Deterministic, nontrivial byte data; test oracle uses
                            # the exact fixture bytes rather than this formula.
                            blocks[b + ry * 256 + rx * 32 + lane] = (y0 * 29 + x0 * 17 + c0 * 13 + 7) & 255
    header = struct.pack('<9I', MAGIC, SEQ, h, w, ci, co, len(blocks) // 2048,
                         len(weights), len(fi_raw))
    args.output.write_bytes(header + blocks + weights + fi_raw)
    print(f'wrote {args.output}: actual command {SEQ} Xpb={len(weights)}B Fi={len(fi_raw)}B; synthetic blocked X')


if __name__ == '__main__':
    main()
