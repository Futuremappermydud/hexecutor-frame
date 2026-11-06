#!/usr/bin/env python3
"""Offline resident-graph IR compiler. This is NOT a DSP execution backend.

QDRP v1: little-endian header <4sIII32s> (magic, version, body bytes,
command count, SHA256(body)), followed by a typed value body. Every length,
integer and reference is checked before publication. Pool addresses remain
relative: no DSP address is assigned without a separately verified mapping.
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
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

VERSION = 1
HEADER = struct.Struct('<4sIII32s')
# Stable opcodes. Unsupported DSP kernels must fail dispatch, never run on ARM.
HANDLERS = '''nop input_slice_pad input_slice weights_to_vtcm im2col hmx_conv
space_to_depth requant_u8 requant_u16 add_u8 add_u16 addsub_u16 add1111_u16
mul_u16 mul1111_u16 copy_values copy_activation transpose concat gather slice
slice_pad decimate resize depth_to_space spill_fill output_slice quantize
dequantize sign mean weight_scales invscale combine_scales convert_weights
scale_convert dynamic_bias bias_shuffle linearclip abs mul_inline addsub_flat
addsub_inline block_zap'''.split()
OPCODES = {f'k_{name}': i + 1 for i, name in enumerate(HANDLERS)}
REQUIRED_CALLS = {
    'k_requant_u8': 'requantize_crouton_u8',
    'k_requant_u16': 'requantize_crouton_u16',
    'k_add_u8': 'elementwise_add_crouton_u8',
    'k_add_u16': 'elementwise_add_crouton_u16',
    'k_addsub_u16': 'elementwise_addsub_crouton_u16',
    'k_add1111_u16': 'elementwise_add1111_crouton_u16',
    'k_mul_u16': 'elementwise_mul_crouton_u16',
    'k_mul1111_u16': 'elementwise_mul1111_crouton_u16',
}
DTYPES = {'uint8': 1, 'int8': 1, 'uint16': 2, 'int16': 2,
          'int32': 4, 'float32': 4, 'int32_quantized': 4}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    """Canonical typed IR: n/f/t, i=u64 signed magnitude, d=f64,
    s=UTF8, b=bytes, l=list, m=map. Counts and lengths are u32.
    Map keys are sorted UTF8 strings. Floats are stored bit-exact as f64.
    """
    def size(n):
        require(0 <= n <= 0xffffffff, 'IR length exceeds u32')
        return struct.pack('<I', n)
    if value is None: return b'n'
    if value is False: return b'f'
    if value is True: return b't'
    if isinstance(value, int):
        require(abs(value) < 1 << 64, 'IR integer exceeds u64')
        return b'i' + struct.pack('<BQ', value < 0, abs(value))
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite IR float')
        return b'd' + struct.pack('<d', value)
    if isinstance(value, (str, bytes)):
        raw = value.encode() if isinstance(value, str) else value
        return (b's' if isinstance(value, str) else b'b') + size(len(raw)) + raw
    if isinstance(value, list):
        return b'l' + size(len(value)) + b''.join(map(encode, value))
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), 'IR map key is not string')
        return b'm' + size(len(value)) + b''.join(encode(k) + encode(value[k]) for k in sorted(value))
    raise ValueError(f'unsupported IR value {type(value)}')


def decode(data):
    pos = 0
    def take(n):
        nonlocal pos
        require(n <= len(data) - pos, 'truncated IR')
        out = data[pos:pos+n]; pos += n
        return out
    def count(): return struct.unpack('<I', take(4))[0]
    def item(depth=0):
        require(depth < 64, 'IR nesting limit')
        tag = take(1)
        if tag == b'n': return None
        if tag == b'f': return False
        if tag == b't': return True
        if tag == b'i':
            sign, value = struct.unpack('<BQ', take(9))
            require(sign in (0, 1) and not (sign and value == 0), 'invalid integer sign')
            return -value if sign else value
        if tag == b'd':
            value = struct.unpack('<d', take(8))[0]
            require(math.isfinite(value), 'nonfinite IR float')
            return value
        if tag in (b's', b'b'):
            raw = take(count())
            return raw.decode() if tag == b's' else raw
        if tag in (b'l', b'm'):
            n = count()
            require(n <= len(data) - pos, 'impossible IR count')
            if tag == b'l': return [item(depth+1) for _ in range(n)]
            result = {}; previous = None
            for _ in range(n):
                key = item(depth+1)
                require(isinstance(key, str) and (previous is None or previous < key), 'unordered/duplicate map key')
                previous = key; result[key] = item(depth+1)
            return result
        raise ValueError('unknown IR tag')
    result = item()
    require(pos == len(data), 'trailing IR data')
    return result


def kernel_registry(source):
    registry = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef): continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == 'kernel':
                require(node.name in OPCODES, f'unknown operation handler {node.name}')
                for arg in decorator.args:
                    name = ast.literal_eval(arg)
                    require(name not in registry, f'duplicate kernel {name}')
                    registry[name] = node.name
    return registry


def read_sources(recovered):
    p = recovered / 'graph.json.gz'
    graph = json.loads(gzip.decompress(p.read_bytes())) if p.exists() else json.loads((recovered / 'graph.json').read_text())
    return (graph, json.loads((recovered / 'execution_order.json').read_text()),
            json.loads((recovered / 'kernel_calls.json').read_text()),
            (recovered / 'constants/constant_pool_5.bin').read_bytes())


def compile_program(graph, order, calls, constants, registry):
    context = graph['context_sha256']
    require(order['context_sha256'] == calls['context_sha256'] == context, 'source context mismatch')
    require(len(bytes.fromhex(context)) == 32, 'invalid context hash')
    tensors = {t['id']: t for t in graph['tensors']}
    nodes = {n['sequence']: n for n in graph['nodes']}
    require(len(tensors) == len(graph['tensors']) and len(nodes) == len(graph['nodes']), 'duplicate source ID')
    commands = []; seen = set()
    for index, sequence in enumerate(order['nodes']):
        require(sequence in nodes, f'missing scheduled node {sequence}')
        node = nodes[sequence]
        require(node['type_name'] in registry, f'unknown operation {node["type_name"]}')
        handler = registry[node['type_name']]
        require(handler in OPCODES, f'unsupported handler {handler}')
        require(all(tid in tensors for tid in node['inputs'] + node['outputs']), f'missing tensor at {sequence}')
        recorded = calls['calls'].get(str(sequence), [])
        if handler in REQUIRED_CALLS:
            expected = REQUIRED_CALLS[handler]
            matches = [c for c in recorded if c[0] == expected]
            require(len(matches) == 1 and len(matches[0]) == 3 and len(matches[0][1]) == 6
                    and len(matches[0][2]) == 4, f'missing/invalid {expected} arguments at {sequence}')
        if handler == 'k_addsub_flat':
            require(len(recorded) == 1 and recorded[0][0] in
                    ('elementwise_add_flat_u16', 'elementwise_sub_flat_u16'),
                    f'missing/unknown flat add/sub arguments at {sequence}')
        duplicate_pad = handler == 'k_input_slice_pad' and sequence in seen
        # Serial dependencies deliberately include DMA/checkpoints and aliases.
        # No dependency is removed based only on logical tensor producer IDs.
        commands.append({'index': index, 'after': index - 1,
                         'opcode': OPCODES['k_nop' if duplicate_pad else handler],
                         'duplicate_input_pad': duplicate_pad, 'node': node,
                         'calls': recorded})
        seen.add(sequence)
    scheduled = set(order['nodes'])
    require(all(n['flags'] == 1 or n['sequence'] in scheduled for n in graph['nodes']), 'unscheduled executable node')
    program = {'version': VERSION, 'context_sha256': context,
               'constants_sha256': hashlib.sha256(constants).hexdigest(),
               'constant_bytes': len(constants), 'opcodes': OPCODES,
               'graph': graph, 'commands': commands}
    program['memory'] = memory_requirements(program)
    return program


def memory_requirements(program):
    # Preserve the reference pools and full alias tables; pool 0 is discard/zero.
    pools = {'1': 8 << 20, '3': 0, '4': 0, '5': program['constant_bytes'], 'spill': 1769472}
    def span(pool, off, size):
        key = str(pool)
        require(key in pools and isinstance(off, int) and off >= 0 and size >= 0, f'invalid pool span {pool}:{off}+{size}')
        end = off + size
        require(end <= 0xffffffff, 'pool offset overflow')
        if key in ('3', '4'): pools[key] = max(pools[key], end)
        else: require(end <= pools[key], f'pool {key} span out of bounds: {end} > {pools[key]}')
    for t in program['graph']['tensors']:
        store = t.get('storage') or {}; shape = t.get('shape')
        if not shape: continue
        require(t['dtype'] in DTYPES, f'unknown dtype {t["dtype"]}')
        require(t['element_bytes'] == DTYPES[t['dtype']], 'element size mismatch')
        for key in ('dimensions', 'max_dimensions', 'padding'):
            require(len(shape[key]) == 4 and all(isinstance(v, int) and v >= 0 for v in shape[key]), 'invalid shape')
        direct = store.get('direct')
        if direct: span(direct['pool'], direct['offset'], math.prod(shape['max_dimensions']) * DTYPES[t['dtype']])
        for block in store.get('blocks', []):
            if block and isinstance(block, dict) and block.get('pool', 0):
                # Constant croutons can start at 128-byte offsets; retain them.
                require(block['offset'] % (128 if block['pool'] == 5 else 2048) == 0, 'unaligned block')
                span(block['pool'], block['offset'], 2048)
    for command in program['commands']:
        node = command['node']; name = node['type_name']; v = node.get('framework_vector', [])
        if name in ('@Spill', '@Fill'):
            require(len(v) >= 4 and len(v) % 2 == 0, 'invalid spill/fill vector')
            off = v[3]
            for addr, n in zip(v[4::2], v[5::2]):
                span(addr & 0x7ff, addr & ~0x7ff, n); span('spill', off, n); off += n
        if name == '@BlockZap':
            require(len(v) >= 2, 'invalid block-zap vector')
            count = v[1] & 0x7fff
            require(v[1] & 0x8000 and v[1] & 0xc00000 == 0xc00000, 'unknown block-zap mode')
            require(len(v) >= 2 + 2 * count and not any(v[2+2*count:]), 'invalid block-zap count/tail')
            for off in v[2:2+2*count]: span(1, off, 2048)
    total = sum((n + 127) & ~127 for n in pools.values())
    return {'pool_bytes': pools, 'pool_total_aligned_bytes': total,
            'verified_arena_bytes': 8 << 20, 'hmx_scratch_bytes': 0x2d0000,
            'naive_pool_shortfall_bytes': max(0, total - (8 << 20)),
            'hardware_layout_verified': False}


def pack(program):
    body = encode(program)
    return HEADER.pack(b'QDRP', VERSION, len(body), len(program['commands']), hashlib.sha256(body).digest()) + body


def unpack(raw):
    require(len(raw) >= HEADER.size, 'truncated program header')
    magic, version, size, count, digest = HEADER.unpack_from(raw)
    require(magic == b'QDRP' and version == VERSION, 'unsupported program format')
    body = raw[HEADER.size:]
    require(len(body) == size and hashlib.sha256(body).digest() == digest, 'program length/digest mismatch')
    program = decode(body)
    require(program['version'] == VERSION and len(program['commands']) == count, 'program count/version mismatch')
    require(program['opcodes'] == OPCODES, 'unknown opcode registry')
    for i, c in enumerate(program['commands']):
        require(c['index'] == i and c['after'] == i-1 and c['opcode'] in OPCODES.values(), 'invalid command/dependency')
    require(program['memory'] == memory_requirements(program), 'memory descriptor mismatch')
    return program


def verify_sources(program, graph, order, calls, constants, registry):
    require(program == compile_program(graph, order, calls, constants, registry), 'program differs from recovered graph')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recovered', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--kernel-source', type=Path, default=(PATHS.model_tools / 'htp_emu.py'))
    args = ap.parse_args()
    sources = read_sources(args.recovered); registry = kernel_registry(args.kernel_source.read_text())
    program = compile_program(*sources, registry)
    raw = pack(program); restored = unpack(raw)
    verify_sources(restored, *sources, registry)
    require(raw == pack(restored), 'noncanonical binary')
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(raw)
    report = {'format': 'QDRP', 'version': VERSION, 'status': 'offline IR only; DSP executor unavailable',
              'context_sha256': program['context_sha256'], 'sha256': hashlib.sha256(raw).hexdigest(),
              'bytes': len(raw), 'commands': len(program['commands']),
              'operation_types': len({c['node']['type_name'] for c in program['commands']}),
              'opcode_counts': dict(sorted(Counter(str(c['opcode']) for c in program['commands']).items())),
              'duplicate_input_pad_noops': sum(c['duplicate_input_pad'] for c in program['commands']),
              'memory': program['memory'], 'source_roundtrip_verified': True}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
