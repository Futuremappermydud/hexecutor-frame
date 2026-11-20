#!/usr/bin/env python3
"""Export decoded HTP constants and packed convolution weights as NumPy arrays.

No NumPy dependency. The int8 HWIO exports retain the compiled quantization:
they are not a reconstructed floating-point training checkpoint.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import zipfile


NP_DTYPES = {1: '|u1', 2: '<u2', 3: '<i2', 4: '<f4', 5: '<i4', 6: '<i4', 7: '|i1'}


def npy_bytes(raw, shape, dtype):
    header = repr(dict(descr=dtype, fortran_order=False, shape=tuple(shape))).encode('ascii')
    header += b' ' * ((64 - ((10 + len(header) + 1) % 64)) % 64) + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + raw


def weight_offset(coords, dimensions, packed):
    """SDK SWeights/pkSWeights element_addr, into contiguous signed-byte data.

    Derived from QAIRT 2.50 ARM64 routines at VAs 0x136d180 / 0x136d4b4,
    reached through each tensor's RTTI/vtable element_addr method.
    """
    h, w, i, o = coords
    H, W, I, O = dimensions
    if packed:
        # [O/32,H,W,I/4,O%32,I%4]
        return ((((o // 32 * H + h) * W + w) * (I // 4) + i // 4) * 128
                + (o % 32) * 4 + i % 4)
    # [O/32,I/32,H,W,(I%32)/4,O%32,I%4]
    outer = (((o // 32 * (I // 32) + i // 32) * H + h) * W + w)
    return outer * 1024 + ((i // 4) % 8) * 128 + (o % 32) * 4 + i % 4


def constant_pools(data, output):
    pos = data.find(struct.pack('<I', 0x71c43c9b))
    if pos < 0:
        raise ValueError('missing constant extent descriptor')
    _, word, extent_word, pool_word = struct.unpack_from('<4I', data, pos)
    header_bytes, desc_bytes = (word >> 24) * 16, (word & 0xffffff) * 64
    n_extent, n_pool = extent_word & 65535, pool_word & 0xffffff
    if header_bytes < 16 or desc_bytes < header_bytes + 16 * (n_extent + n_pool):
        raise ValueError('invalid extent descriptor')
    extents = []
    for i in range(n_extent):
        flags_align, off, length, reserved = struct.unpack_from('<4I', data, pos + header_bytes + i * 16)
        start, length = pos + off * 64, length * 64
        if start < pos + desc_bytes or start + length > len(data):
            raise ValueError('constant extent outside context')
        extents.append(dict(offset=start, bytes=length, alignment=1 << (flags_align & 255)))
    pools, metadata = {}, []
    for i in range(n_pool):
        pid, extent_id, off, length = struct.unpack_from('<4I', data, pos + header_bytes + (n_extent + i) * 16)
        extent = extents[extent_id - 1]
        start, length = extent['offset'] + off * 64, length * 64
        if start < extent['offset'] or start + length > extent['offset'] + extent['bytes']:
            raise ValueError('pool outside constant extent')
        blob = data[start:start + length]
        filename = f'constant_pool_{pid}.bin'
        (output / filename).write_bytes(blob)
        pools[pid] = blob
        metadata.append(dict(id=pid, context_offset=start, bytes=length,
                             file=filename, sha256=hashlib.sha256(blob).hexdigest()))
    return pools, dict(descriptor_offset=pos, extents=extents, pools=metadata)


def extract_tensor(tensor, pools):
    code = tensor['format']
    mode, dtype = code >> 28, NP_DTYPES.get(code & 15)
    if mode == 3:
        return struct.pack('<I', tensor['scalar_word']), [], dtype, 'scalar'
    if mode != 1:
        raise ValueError('unsupported constant tensor mode')
    shape = tensor['shape']
    dims, maximum, pad = shape['dimensions'], shape['max_dimensions'], shape['padding']
    if any(pad):
        raise ValueError('nonzero constant coordinate padding not implemented')
    storage, name, elbytes = tensor['storage'], tensor['class_name'], tensor['element_bytes']
    raw = bytearray(math.prod(dims) * elbytes)
    direct = storage.get('direct')
    if 'direct' in storage:
        if direct is None or direct['pool'] not in pools:
            raise ValueError('constant has no constant-pool storage')
        blob, start = pools[direct['pool']], direct['offset']
        size = math.prod(maximum) * elbytes
        if start + size > len(blob):
            raise ValueError('constant exceeds pool')
        source = blob[start:start + size]
        if name in ['xwb', 'xpb']:
            if elbytes != 1 or maximum[3] % 32 or maximum[2] % (4 if name == 'xpb' else 32):
                raise ValueError('unsupported packed-weight shape')
            for j, coords in enumerate(itertools.product(*(range(d) for d in dims))):
                raw[j] = source[weight_offset(coords, maximum, name == 'xpb')]
            return bytes(raw), dims, dtype, 'HWIO unpacked from ' + name
        if code & (1 << 20):
            raise ValueError(f'unknown contiguous non-flat layout {name}')
        for j, coords in enumerate(itertools.product(*(range(d) for d in dims))):
            index = 0
            for c, d in zip(coords, maximum):
                index = index * d + c
            raw[j * elbytes:(j + 1) * elbytes] = source[index * elbytes:(index + 1) * elbytes]
        return bytes(raw), dims, dtype, 'flat logical dimensions'
    if name not in ['cB', 'cH']:
        raise ValueError(f'unknown indirect constant layout {name}')
    # Public R4CroutonLayout (8-bit) and R4Crouton2Layout (16-bit).
    block_shape = [1, 8, 8 if elbytes == 1 else 4, 32]
    chunks = [m // c for m, c in zip(maximum, block_shape)]
    blocks = storage['blocks']
    if blocks is None:
        raise ValueError('unresolved constant blocktable')
    for j, coords in enumerate(itertools.product(*(range(d) for d in dims))):
        block_idx = 0
        for c, size, n in zip(coords, block_shape, chunks):
            block_idx = block_idx * n + c // size
        block_idx += storage['slice_offset']
        pointer = blocks[block_idx]
        if pointer is None or pointer.get('pool') not in pools:
            raise ValueError('constant points outside constant pool')
        _, h, w, c = [v % size for v, size in zip(coords, block_shape)]
        offset = ((h * 8 + w) * 32 + c) if elbytes == 1 else (((h * 2 + w // 2) * 32 + c) * 2 + w % 2)
        start = pointer['offset'] + offset * elbytes
        source = pools[pointer['pool']]
        if start + elbytes > len(source):
            raise ValueError('constant block outside pool')
        raw[j * elbytes:(j + 1) * elbytes] = source[start:start + elbytes]
    return bytes(raw), dims, dtype, 'logical dimensions from ' + name


def export(context, graph_file, output):
    data = context.read_bytes()
    graph = json.loads(graph_file.read_text())
    if hashlib.sha256(data).hexdigest() != graph['context_sha256']:
        raise ValueError('context/graph mismatch')
    output.mkdir(parents=True, exist_ok=True)
    arrays = output / 'arrays'
    arrays.mkdir(exist_ok=True)
    pools, extent_metadata = constant_pools(data, output)
    nodes = {n['sequence']: n for n in graph['nodes']}
    exports = []
    for tensor in graph['tensors']:
        producer = nodes.get(tensor['producer'])
        if producer is None or producer['type_name'] != 'Const':
            continue
        raw, shape, dtype, layout = extract_tensor(tensor, pools)
        filename = f'arrays/tensor_{tensor["id"]:05d}.npy'
        blob = npy_bytes(raw, shape, dtype)
        (output / filename).write_bytes(blob)
        entry = dict(tensor_id=tensor['id'], node_sequence=producer['sequence'],
                     source_id=producer['source_id'], original_class=tensor['class_name'],
                     file=filename, shape=shape, numpy_dtype=dtype, layout=layout,
                     data_bytes=len(raw), sha256=hashlib.sha256(blob).hexdigest())
        if 'source_name' in producer:
            entry['source_name'] = producer['source_name']
        if 'quantization' in tensor:
            entry['quantization'] = tensor['quantization']
        exports.append(entry)
    weights = [x for x in exports if x['original_class'] in ['xwb', 'xpb']]
    with zipfile.ZipFile(output / 'convolution_weights.npz', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in weights:
            name = entry.get('source_name', f'source_{entry["source_id"]}')
            archive.write(output / entry['file'], name + '.npy')
    by_source = {}
    for node in graph['nodes']:
        by_source.setdefault(node['source_id'], []).append(node)
    tensor_by_id = {t['id']: t for t in graph['tensors']}
    layers = []
    for entry in weights:
        group = by_source[entry['source_id']]
        compute = [n for n in group if n['type_name'].startswith(('ConvLayer_s1.', 'ConvLayer_s2.'))]
        params = [e for e in exports if e['source_id'] == entry['source_id'] and e['original_class'] == 'fi']
        inputs, outputs = [], []
        for node in compute:
            for ids, dest in [(node['inputs'][:1], inputs), (node['outputs'], outputs)]:
                for tid in ids:
                    q = tensor_by_id[tid].get('quantization')
                    if q and q not in dest:
                        dest.append(q)
        layers.append(dict(source_id=entry['source_id'], name=entry.get('source_name'),
                           weight_tensor=entry['tensor_id'], hwio_shape=entry['shape'],
                           weight_quantization=entry.get('quantization'),
                           weight_file=entry['file'], npz_key=entry.get('source_name'),
                           compiled_parameter_tensors=[e['tensor_id'] for e in params],
                           tiled_compute_kernel_count=len(compute),
                           input_quantizations=inputs, output_quantizations=outputs))
    (output / 'layers.json').write_text(json.dumps(layers, indent=2) + '\n')
    with (output / 'layers.csv').open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['source_id', 'name', 'HWIO', 'weight_scale', 'weight_zero_point',
                         'weight_tensor', 'compiled_parameter_tensors', 'tiled_compute_kernels', 'npy_file'])
        for layer, entry in zip(layers, weights):
            q = entry.get('quantization', {})
            writer.writerow([entry['source_id'], entry.get('source_name'), 'x'.join(map(str, entry['shape'])),
                             q.get('scale'), q.get('zero_point'), entry['tensor_id'],
                             ' '.join(map(str, layer['compiled_parameter_tensors'])),
                             layer['tiled_compute_kernel_count'], entry['file']])
    manifest = dict(context_sha256=graph['context_sha256'], **extent_metadata,
                    constants=exports, convolution_weight_tensor_ids=[w['tensor_id'] for w in weights],
                    status='Decoded compiled constants; operator semantics and end-to-end inference remain unvalidated.')
    (output / 'constants.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{len(exports)} constants, including {len(weights)} HWIO int8 convolution weights -> {output}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('context', type=Path)
    parser.add_argument('graph', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.context, args.graph, args.output)
