#!/usr/bin/env python3
"""Lower audited QDRP into QDRX, the relative-address native decoder format.

This is an upload image, not a runnable inference backend. It assigns no DSP
addresses and makes no claim about ownership of the reference pools. Constants
remain a separate, SHA256-bound file. No Python or source JSON is needed to
decode QDRX. See RESIDENT_FORMAT.md for the wire ABI and execution gates.
"""

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
import struct
import zlib

import depth_model.lowering.resident_program as rp
import hexecutor.program.resident_schema as schema
VERSION = 1
NONE = 0xffffffff
HEADER_BYTES = 256
STRIDES = (128, 4, 96, 4, 4, 88)
SECTIONS = ('tensors', 'blocks', 'commands', 'refs', 'words', 'calls')
MAX_BYTES = 8 << 20
SHAPE, BLOCKED, DIRECT, SHAPE_VALUE, SCALAR, QUANT, CROUTON = (1 << i for i in range(7))
VARIANTS = {name: i + 1 for i, name in enumerate(schema.VARIANTS)}
CALLS = {name: i + 1 for i, name in enumerate(schema.CALL_NAMES)}
DTYPES = {name: i for i, name in enumerate(schema.DTYPES)}
ABI = zlib.crc32(rp.encode([list(schema.VARIANTS),
                           list(schema.VARIANT_HANDLERS), list(schema.CALL_NAMES),
                           list(schema.CLASSES), list(schema.DTYPES), rp.OPCODES]))


def u32(value):
    rp.require(type(value) is int and 0 <= value <= NONE, 'value does not fit u32')
    return value


def fourcc(name):
    rp.require(name in schema.CLASSES, f'unknown tensor class {name}')
    return int.from_bytes(name.encode().ljust(4, b'\0'), 'little')


def intern(table, cache, values):
    values = tuple(values)
    if not values:
        return 0, 0
    if values not in cache:
        cache[values] = len(table)
        table.extend(values)
    return cache[values], len(values)


def semantic_tensors(program):
    """Only execution metadata; omit serialized offsets and producer debug IDs."""
    result = []
    for tensor in program['graph']['tensors']:
        t = {k: tensor[k] for k in ('id', 'class_name', 'dtype', 'element_bytes')}
        for key in ('quantization', 'shape', 'shape_values', 'scalar_word', 'scalar'):
            if key in tensor:
                t[key] = tensor[key]
        store = tensor.get('storage') or {}
        if store.get('direct'):
            t['storage'] = {'direct': store['direct']}
        elif store.get('blocks'):
            t['storage'] = {'blocks': store['blocks'], 'slice_offset': store.get('slice_offset', 0)}
        elif 'shape' in tensor:
            t['storage'] = {'direct': None}
        result.append(t)
    return result


def semantic_commands(program):
    result = []
    for c in program['commands']:
        node = c['node']
        result.append({
            'sequence': node['sequence'], 'opcode': c['opcode'],
            'variant': node['type_name'], 'duplicate_input_pad': c['duplicate_input_pad'],
            'after': c['after'], 'io_index': node.get('graph_input_index', node.get('graph_output_index', NONE)),
            **{key: node.get(key, []) for key in ('inputs', 'outputs', 'checkpoints', 'framework_words', 'framework_vector')},
            'calls': c['calls'],
        })
    return result


def pack(program):
    # Recheck the audited IR, including serial dependencies and memory bounds.
    rp.unpack(rp.pack(program))
    ts = semantic_tensors(program)
    ids = {t['id']: i for i, t in enumerate(ts)}
    rp.require(list(ids) == sorted(ids), 'tensor IDs must be increasing')
    tensors, blocks, commands, refs, words, calls = [], [], [], [], [], []
    bc, rc, wc, cc = {}, {}, {}, {}
    graph = program['graph']
    input_ref = intern(refs, rc, [ids[i] for i in graph['graph_inputs']])
    output_ref = intern(refs, rc, [ids[i] for i in graph['graph_outputs']])
    for t in ts:
        rp.require(t['dtype'] in DTYPES, f'unknown dtype {t["dtype"]}')
        v = [0] * 32
        v[:3] = [t['id'], DTYPES[t['dtype']], fourcc(t['class_name'])]
        shape, store = t.get('shape'), t.get('storage', {})
        if shape:
            v[3] |= SHAPE
            v[4:16] = shape['dimensions'] + shape['max_dimensions'] + shape['padding']
            v[21] = shape['flags']
        direct = store.get('direct')
        if direct:
            v[3] |= DIRECT
            v[16:18] = [direct['pool'], direct['offset']]
        if store.get('blocks'):
            v[3] |= BLOCKED
            if t['class_name'][0] in 'Cc': v[3] |= CROUTON
            entries = []
            for b in store['blocks']:
                rp.require(isinstance(b, dict) and set(b) == {'pool', 'offset'}, 'unknown block descriptor')
                rp.require(b['pool'] in (1, 3, 4, 5) and b['offset'] & 127 == 0, 'unencodable block')
                entries.append(u32(b['offset']) | b['pool'])
            v[18:20] = intern(blocks, bc, entries)
            v[20] = store.get('slice_offset', 0)
        if 'shape_values' in t:
            v[3] |= SHAPE_VALUE
            rp.require(len(t['shape_values']) == 4, 'expected four shape values')
            v[26:30] = t['shape_values']
        if 'scalar_word' in t:
            v[3] |= SCALAR
            v[25] = t['scalar_word']
        if 'quantization' in t:
            v[3] |= QUANT
            q = t['quantization']
            v[22] = struct.unpack('<I', struct.pack('<i', q['zero_point']))[0]
            v[23:25] = struct.unpack('<2I', struct.pack('<d', q['scale']))
        tensors.append(struct.pack('<32I', *map(u32, v)))
    for c in semantic_commands(program):
        v = [0] * 24
        rp.require(c['variant'] in VARIANTS, 'unknown kernel variant')
        v[:6] = [c['sequence'], c['opcode'], VARIANTS[c['variant']], int(c['duplicate_input_pad']),
                 c['after'] & NONE, c['io_index']]
        for at, name in ((6, 'inputs'), (8, 'outputs')):
            v[at:at+2] = intern(refs, rc, [ids[i] for i in c[name]])
        for at, name in ((10, 'checkpoints'), (12, 'framework_words'), (14, 'framework_vector')):
            v[at:at+2] = intern(words, wc, map(u32, c[name]))
        encoded = []
        for name, regs, stack in c['calls']:
            rp.require(name in CALLS and len(regs) == 6 and len(stack) == 4, 'unknown kernel call/signature')
            mask, values = 0, []
            for j, arg in enumerate(regs + stack):
                if arg == 'P':
                    mask |= 1 << j
                    values.append(0)
                else:
                    rp.require(type(arg) is int and 0 <= arg < 1 << 64, 'unencodable kernel immediate')
                    values.append(arg)
            encoded.append(struct.pack('<II10Q', CALLS[name], mask, *values))
        v[16:18] = intern(calls, cc, encoded)
        commands.append(struct.pack('<24I', *map(u32, v)))
    sections = [b''.join(tensors), struct.pack(f'<{len(blocks)}I', *blocks), b''.join(commands),
                struct.pack(f'<{len(refs)}I', *refs), struct.pack(f'<{len(words)}I', *words), b''.join(calls)]
    header = [0] * 64
    header[:6] = [int.from_bytes(b'QDRX', 'little'), VERSION, HEADER_BYTES, 0, 0, ABI]
    header[8:16] = struct.unpack('<8I', bytes.fromhex(program['context_sha256']))
    header[16:24] = struct.unpack('<8I', bytes.fromhex(program['constants_sha256']))
    pools = program['memory']['pool_bytes']
    header[24:31] = [0, pools['1'], 0, pools['3'], pools['4'], pools['5'], pools['spill']]
    header[31:37] = [program['memory']['hmx_scratch_bytes'], program['memory']['verified_arena_bytes'],
                     *input_ref, *output_ref]
    body = bytearray(HEADER_BYTES)
    for i, section in enumerate(sections):
        body.extend(bytes((-len(body)) % 8))
        header[37 + i*2:39 + i*2] = [len(body), len(section) // STRIDES[i]]
        body.extend(section)
    rp.require(len(body) <= MAX_BYTES, 'image exceeds decoder byte budget')
    header[3] = len(body)
    body[:HEADER_BYTES] = struct.pack('<64I', *header)
    struct.pack_into('<I', body, 16, zlib.crc32(body))
    return bytes(body)


def unpack(raw):
    """Independent semantic reconstruction; native qdr_open is the wire validator."""
    rp.require(HEADER_BYTES <= len(raw) <= MAX_BYTES, 'bad image size')
    h = struct.unpack_from('<64I', raw)
    rp.require(h[:4] == (int.from_bytes(b'QDRX', 'little'), VERSION, HEADER_BYTES, len(raw)), 'bad header')
    rp.require(h[5] == ABI and h[6:8] == (0, 0) and not any(h[49:]), 'unsupported ABI/flags')
    zeroed = raw[:16] + bytes(4) + raw[20:]
    rp.require(zlib.crc32(zeroed) == h[4], 'image CRC mismatch')
    tables = {}
    end = HEADER_BYTES
    for i, (name, stride) in enumerate(zip(SECTIONS, STRIDES)):
        off, n = h[37+i*2:39+i*2]
        rp.require(off == (end + 7) & ~7 and not any(raw[end:off]), 'noncanonical section')
        end = off + n*stride
        rp.require(end <= len(raw), 'section outside image')
        tables[name] = [raw[j:j+stride] for j in range(off, end, stride)]
    rp.require(end == len(raw), 'trailing bytes')
    def uints(name): return [struct.unpack('<I', b)[0] for b in tables[name]]
    blocks, refs, words = map(uints, ('blocks', 'refs', 'words'))
    def span(values, start, count):
        rp.require(start <= len(values) and count <= len(values)-start, 'bad table span')
        return values[start:start+count]
    tensors = []
    for record in tables['tensors']:
        v = struct.unpack('<32I', record)
        rp.require(v[1] < len(schema.DTYPES), 'bad dtype')
        dtype = schema.DTYPES[v[1]]
        t = {'id': v[0], 'dtype': dtype, 'class_name': v[2].to_bytes(4, 'little').rstrip(b'\0').decode(),
             'element_bytes': rp.DTYPES.get(dtype, 1)}
        f = v[3]
        if f & SHAPE:
            t['shape'] = dict(zip(('dimensions', 'max_dimensions', 'padding'),
                                  [list(v[j:j+4]) for j in (4, 8, 12)]), flags=v[21])
            t['storage'] = {'direct': None}
        if f & DIRECT: t['storage'] = {'direct': {'pool': v[16], 'offset': v[17]}}
        if f & BLOCKED:
            t['storage'] = {'blocks': [{'pool': b & 127, 'offset': b & ~127} for b in span(blocks, v[18], v[19])],
                            'slice_offset': v[20]}
        if f & QUANT:
            t['quantization'] = {'zero_point': struct.unpack_from('<i', record, 88)[0],
                                 'scale': struct.unpack_from('<d', record, 92)[0]}
        if f & SHAPE_VALUE: t['shape_values'] = list(v[26:30])
        if f & SCALAR:
            t['scalar_word'] = v[25]
            t['scalar'] = struct.unpack_from('<f' if dtype == 'float32' else '<i', record, 100)[0]
        tensors.append(t)
    def tids(start, count):
        indices = span(refs, start, count)
        rp.require(all(i < len(tensors) for i in indices), 'bad tensor index')
        return [tensors[i]['id'] for i in indices]
    calls = []
    for record in tables['calls']:
        name, mask, *args = struct.unpack('<II10Q', record)
        rp.require(1 <= name <= len(schema.CALL_NAMES) and mask < 1024, 'bad call')
        args = ['P' if mask & (1 << j) else a for j, a in enumerate(args)]
        calls.append([schema.CALL_NAMES[name-1], args[:6], args[6:]])
    commands = []
    for record in tables['commands']:
        v = struct.unpack('<24I', record)
        rp.require(1 <= v[2] <= len(schema.VARIANTS), 'bad variant')
        commands.append({
            'sequence': v[0], 'opcode': v[1], 'variant': schema.VARIANTS[v[2]-1],
            'duplicate_input_pad': bool(v[3]), 'after': -1 if v[4] == NONE else v[4], 'io_index': v[5],
            'inputs': tids(v[6], v[7]), 'outputs': tids(v[8], v[9]),
            **{name: span(words, v[at], v[at+1]) for name, at in
               (('checkpoints', 10), ('framework_words', 12), ('framework_vector', 14))},
            'calls': span(calls, v[16], v[17]),
        })
    return {'context_sha256': raw[32:64].hex(), 'constants_sha256': raw[64:96].hex(),
            'tensors': tensors, 'commands': commands,
            'graph_inputs': tids(h[33], h[34]), 'graph_outputs': tids(h[35], h[36]),
            'pool_bytes': dict(zip(('0', '1', '2', '3', '4', '5', 'spill'), h[24:31]))}


def verify_sources(raw, program):
    decoded = unpack(raw)
    rp.require(decoded['context_sha256'] == program['context_sha256'] and
               decoded['constants_sha256'] == program['constants_sha256'], 'image identity mismatch')
    rp.require(decoded['tensors'] == semantic_tensors(program), 'tensor semantic mismatch')
    rp.require(decoded['commands'] == semantic_commands(program), 'command semantic mismatch')
    for key in ('graph_inputs', 'graph_outputs'):
        rp.require(decoded[key] == program['graph'][key], 'graph I/O mismatch')
    rp.require(decoded['pool_bytes'] == {'0': 0, '2': 0, **program['memory']['pool_bytes']}, 'pool mismatch')


def c_registry():
    lines = ['/* Generated by resident_image.py --write-schema-header. Do not edit. */',
             '#ifndef QD_RESIDENT_SCHEMA_H', '#define QD_RESIDENT_SCHEMA_H',
             f'#define QDR_ABI 0x{ABI:08x}u', f'#define QDR_VARIANTS {len(VARIANTS)}u',
             f'#define QDR_CALLS {len(CALLS)}u', f'#define QDR_OPCODES {len(rp.HANDLERS)}u']
    for name, opcode in rp.OPCODES.items(): lines.append(f'#define QDR_{name[2:].upper()} {opcode}u')
    for name, value in CALLS.items(): lines.append(f'#define QDR_CALL_{name.upper()} {value}u')
    lines.append('static const unsigned char qdr_variant_ops[] = {0,')
    lines.extend(f'    {rp.OPCODES[handler]}, /* {name} */' for name, handler in zip(schema.VARIANTS, schema.VARIANT_HANDLERS))
    lines.append('};')
    lines.append('static const unsigned int qdr_classes[] = {' + ','.join(f'0x{fourcc(c):08x}u' for c in schema.CLASSES) + '};')
    lines.extend(['#endif', ''])
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recovered', type=Path, nargs='?')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--write-schema-header', type=Path)
    args = ap.parse_args()
    if args.write_schema_header:
        args.write_schema_header.write_text(c_registry())
    if not args.recovered:
        rp.require(args.write_schema_header is not None, 'recovered directory required')
        return
    rp.require(args.output is not None, '--output required')
    root = (PATHS.repo)
    registry = rp.kernel_registry(((PATHS.model_tools / 'htp_emu.py')).read_text())
    rp.require(registry == dict(zip(schema.VARIANTS, schema.VARIANT_HANDLERS)), 'kernel registry changed; revise ABI')
    program = rp.compile_program(*rp.read_sources(args.recovered), registry)
    raw = pack(program)
    verify_sources(raw, program)
    h = struct.unpack_from('<64I', raw)
    report = {'format': 'QDRX', 'version': VERSION, 'abi': f'{ABI:08x}',
              'status': 'native decoder image; no hardware executor or verified pool mapping',
              'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
              'context_sha256': program['context_sha256'], 'constants_sha256': program['constants_sha256'],
              'sections': {n: {'offset': h[37+2*i], 'count': h[38+2*i], 'stride': STRIDES[i]}
                           for i, n in enumerate(SECTIONS)},
              'source_block_entries': sum(len((t.get('storage') or {}).get('blocks', [])) for t in program['graph']['tensors']),
              'opcode_counts': dict(sorted(Counter(str(c['opcode']) for c in program['commands']).items())),
              'memory': program['memory'], 'source_semantics_verified': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
