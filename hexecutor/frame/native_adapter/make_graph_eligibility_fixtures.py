#!/usr/bin/env python3
"""Serialize actual compiled-QDRX tensor metadata for the C bridge predicate."""
import argparse
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAME = HERE.parent
sys.path.insert(0, str(FRAME))
from resident_image import unpack

MAGIC = 0x31454e47  # GNE1
DTYPE = {'uint8': 1, 'int8': 2, 'int32': 5}
DIRECT, BLOCKED, CROUTON = 4, 2, 64


def tensor_words(t, qdr_flags):
    shape = t['shape']
    storage = t['storage']
    flags = qdr_flags & (DIRECT | BLOCKED | CROUTON)
    cls = t['class_name'].lower()
    dims, maximum, padding = shape['dimensions'], shape['max_dimensions'], shape['padding']
    d = dims + [0] * (4-len(dims))
    m = maximum + [0] * (4-len(maximum))
    p = padding + [0] * (4-len(padding))
    values = [flags, DTYPE[t['dtype']], t['element_bytes'], bool(flags & CROUTON),
              cls == 'cb', cls == 'fi', cls == 'xpb', cls == 'xwb', *d, *m, *p]
    return [int(v) for v in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, default=FRAME / 'build/resident-primary.qdrx')
    parser.add_argument('--sequences', nargs='+', type=int, default=[115, 3416, 1195])
    args = parser.parse_args()
    image = unpack(args.image.read_bytes())
    raw = args.image.read_bytes()
    header = struct.unpack_from('<64I', raw)
    tensor_section_offset = header[37]
    tensors = {t['id']: t for t in image['tensors']}
    for sequence in args.sequences:
        command = next(c for c in image['commands'] if c['sequence'] == sequence)
        x, w, fi = (tensors[i] for i in command['inputs'][:3])
        y = tensors[command['outputs'][0]]
        records = [tensor_words(t, struct.unpack_from('<I', raw,
                    tensor_section_offset + (t['id'] - 1) * 128 + 12)[0])
                   for t in (x, w, fi, y)]
        packet = struct.pack('<2I', MAGIC, sequence)
        packet += b''.join(struct.pack('<20I', *words) for words in records)
        destination = HERE / f'graph-eligibility-{sequence}.bin'
        destination.write_bytes(packet)
        print(f'{sequence}: wrote actual compiled QDRX tensors to {destination.name}')


if __name__ == '__main__':
    main()
