#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build and validate a compact, model-bound NPRG table from a QDRX image."""
from __future__ import annotations

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
from dataclasses import dataclass
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

ROOT = (PATHS.native)
sys.path.insert(0, str(ROOT))
import depth_model.lowering.resident_image as qdrx  # noqa: E402
import hexecutor.program.resident_schema as schema  # noqa: E402

MAGIC = b'NPRG'
VERSION = 1
HEADER = 240
TENSOR_BYTES = 56
COMMAND_BYTES = 20
BLOCK_BYTES = 3
NONE16 = 0xffff
SECTION_NAMES = ('tensors', 'blocks', 'commands', 'refs', 'vectors', 'calls', 'io')
# 1 MiB nominal L2 less three 64 KiB worker stacks and 64 KiB shared state.
L2_CEILING = 768 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def put_u24(n):
    require(0 <= n < (1 << 24), 'u24 overflow')
    return n.to_bytes(3, 'little')


def intern(table, cache, values):
    values = tuple(values)
    if values not in cache:
        cache[values] = len(table)
        table.extend(values)
    return cache[values], len(values)


def lower(raw):
    decoded = qdrx.unpack(raw)
    ts = decoded['tensors']
    tensors, blocks, refs, vectors, calls = [], [], [], [], []
    bc, rc, vc, cc = {}, {}, {}, {}
    tids = {t['id']: i for i, t in enumerate(ts)}
    call_id = {name: i + 1 for i, name in enumerate(schema.CALL_NAMES)}

    for t in ts:
        flags = 0
        st = t.get('storage') or {}
        shape = t.get('shape')
        qflags = (qdrx.SHAPE if shape else 0)
        qflags |= qdrx.DIRECT if st.get('direct') else 0
        qflags |= qdrx.BLOCKED if st.get('blocks') else 0
        qflags |= qdrx.CROUTON if st.get('blocks') and t['class_name'][0] in 'Cc' else 0
        qflags |= qdrx.SHAPE_VALUE if 'shape_values' in t else 0
        qflags |= qdrx.SCALAR if 'scalar_word' in t else 0
        qflags |= qdrx.QUANT if 'quantization' in t else 0
        flags = qflags
        cls = t['class_name'].encode().ljust(4, b'\0')[:4]
        dtype = qdrx.DTYPES[t['dtype']]
        payload = bytes(24)
        if shape:
            vals = shape['dimensions'] + shape['max_dimensions'] + shape['padding']
            require(all(-32768 <= int(x) <= 32767 for x in vals),
                    f"tensor {t['id']} shape field does not fit i16")
            payload = struct.pack('<12h', *vals)
        elif 'shape_values' in t:
            vals = t['shape_values']
            require(len(vals) == 4 and all(-2147483648 <= int(x) < 2147483648 for x in vals),
                    f"tensor {t['id']} bad shape values")
            payload = struct.pack('<4i', *vals) + bytes(8)
        elif 'scalar_word' in t:
            payload = struct.pack('<I', t['scalar_word']) + bytes(20)
        else:
            require(t.get('class_name') == 'None',
                    f"tensor {t['id']} has no recognized runtime payload")

        storage = bytes(12)
        if st.get('direct'):
            d = st['direct']
            require(0 <= d['pool'] <= 255 and 0 <= d['offset'] <= 0xffffffff,
                    f"tensor {t['id']} direct storage overflow")
            storage = struct.pack('<BI', d['pool'], d['offset']) + bytes(7)
        elif st.get('blocks'):
            encoded = []
            for b in st['blocks']:
                pool, offset = b['pool'], b['offset']
                require(0 <= pool < 8 and offset % 128 == 0 and offset // 128 < (1 << 21),
                        f"tensor {t['id']} block cannot be narrowed")
                encoded.append(put_u24((pool << 21) | (offset // 128)))
            # Intern only identical ordered block lists. Tensor views and their
            # slice origins remain separate descriptors even when lists match.
            key = tuple(encoded)
            if key not in bc:
                bc[key] = len(blocks)
                blocks.extend(encoded)
            first, count = bc[key], len(encoded)
            require(first < (1 << 24) and count < (1 << 16), 'block span overflow')
            sl = st.get('slice_offset', 0)
            require(0 <= sl <= 0xffffffff, 'slice offset overflow')
            storage = put_u24(first) + struct.pack('<H', count) + struct.pack('<I', sl) + bytes(3)

        quant = t.get('quantization', {})
        zp = quant.get('zero_point', 0)
        scale = quant.get('scale', 0.0)
        qbytes = struct.pack('<id', zp, scale)
        tensors.append(struct.pack('<BB4s', dtype, flags, cls) + payload + storage + qbytes + bytes(2))
        require(len(tensors[-1]) == TENSOR_BYTES, 'internal tensor stride error')

    graph_io = []
    for key in ('graph_inputs', 'graph_outputs'):
        vals = [tids[i] for i in decoded[key]]
        graph_io.append(intern(refs, rc, vals))

    commands = []
    for c in decoded['commands']:
        ins = [tids[i] for i in c['inputs']]
        outs = [tids[i] for i in c['outputs']]
        vf, vn = intern(vectors, vc, c['framework_vector'])
        encoded_calls = []
        for name, regs, stack in c['calls']:
            mask, args = 0, []
            for j, arg in enumerate(regs + stack):
                if arg == 'P':
                    mask |= 1 << j
                    args.append(0)
                else:
                    args.append(arg)
            encoded_calls.append(struct.pack('<II10Q', call_id[name], mask, *args))
        cf, cn = intern(calls, cc, encoded_calls)
        inf, ni = intern(refs, rc, ins)
        outf, no = intern(refs, rc, outs)
        seq, op, var, ioidx = c['sequence'], c['opcode'], qdrx.VARIANTS[c['variant']], c['io_index']
        require(seq < (1 << 16) and op < 256 and var < 256 and (ioidx == qdrx.NONE or ioidx <= NONE16),
                f"command {seq} header fields do not fit")
        require(all(n < 256 for n in (ni, no, vn, cn)), f"command {seq} arity overflow")
        require(all(x < (1 << 16) for x in (inf, outf, vf, cf)), f"command {seq} table offset overflow")
        commands.append(struct.pack('<HBBHHHHHBBBBBB', seq, op, var,
                                    NONE16 if ioidx == qdrx.NONE else ioidx,
                                    inf, outf, vf, cf, ni, no, vn, cn,
                                    int(c['duplicate_input_pad']), 0))
        require(len(commands[-1]) == COMMAND_BYTES, 'internal command stride error')

    sections = [b''.join(tensors), b''.join(blocks), b''.join(commands),
                struct.pack(f'<{len(refs)}H', *refs),
                struct.pack(f'<{len(vectors)}I', *vectors), b''.join(calls),
                struct.pack('<4H', graph_io[0][0], graph_io[0][1], graph_io[1][0], graph_io[1][1])]
    counts = [len(ts), len(blocks), len(commands), len(refs), len(vectors), len(calls), 4]
    header = bytearray(HEADER)
    # Stable fixed header: magic/version/header/total/crc/ABI then counts.
    struct.pack_into('<4sHHIIIIII', header, 0, MAGIC, VERSION, HEADER, 0, 0,
                     qdrx.ABI, *counts[:3])
    struct.pack_into('<4I', header, 32, *counts[3:])
    header[48:80] = bytes.fromhex(decoded['context_sha256'])
    header[80:112] = bytes.fromhex(decoded['constants_sha256'])
    header[112:144] = hashlib.sha256(raw).digest()
    pool_order = ('0', '1', '2', '3', '4', '5', 'spill')
    struct.pack_into('<7I', header, 200, *(decoded['pool_bytes'][k] for k in pool_order))
    out = bytearray(header)
    for i, section in enumerate(sections):
        out.extend(bytes((-len(out)) % 8))
        struct.pack_into('<II', out, 144 + i * 8, len(out), len(section))
        out.extend(section)
    struct.pack_into('<I', out, 8, len(out))
    struct.pack_into('<I', out, 12, zlib.crc32(out))
    return bytes(out)


def decode(raw):
    require(len(raw) >= HEADER, 'truncated NPRG header')
    magic, ver, hbytes, total, crc, abi, nt, nb, nc = struct.unpack_from('<4sHHIIIIII', raw)
    require((magic, ver, hbytes, total) == (MAGIC, VERSION, HEADER, len(raw)), 'bad NPRG header')
    require(abi == qdrx.ABI, 'NPRG ABI mismatch')
    crcbuf = bytearray(raw)
    struct.pack_into('<I', crcbuf, 12, 0)
    require(zlib.crc32(crcbuf) == crc, 'NPRG CRC mismatch')
    nrefs, nvec, ncall, nio = struct.unpack_from('<4I', raw, 32)
    counts = [nt, nb, nc, nrefs, nvec, ncall, nio]
    data = {}
    end = HEADER
    for i, name in enumerate(SECTION_NAMES):
        off, size = struct.unpack_from('<II', raw, 144 + i * 8)
        require(off == (end + 7) & ~7 and not any(raw[end:off]), f'noncanonical {name} offset')
        require(off + size <= len(raw), f'{name} out of bounds')
        data[name] = raw[off:off + size]
        end = off + size
    require(end == len(raw), 'trailing NPRG bytes')
    strides = (TENSOR_BYTES, BLOCK_BYTES, COMMAND_BYTES, 2, 4, 88, 2)
    for name, size, stride in zip(SECTION_NAMES, (len(x) for x in data.values()), strides):
        require(size % stride == 0, f'bad {name} section length')
    require([len(data[n]) // s for n, s in zip(SECTION_NAMES, strides)] == counts,
            'NPRG section count mismatch')
    require(nio == 4, 'bad graph I/O section')
    pool_bytes = struct.unpack_from('<7I', raw, 200)
    refs = struct.unpack(f'<{nrefs}H', data['refs'])
    vectors = struct.unpack(f'<{nvec}I', data['vectors'])
    for i in range(ncall):
        name, mask, *args = struct.unpack_from('<II10Q', data['calls'], i*88)
        require(1 <= name <= len(schema.CALL_NAMES) and mask < 1024, f'bad call record {i}')
        require(all(args[j] == 0 for j in range(10) if mask & (1 << j)), f'bad pointer placeholder in call {i}')
    cmds = []
    for i in range(nc):
        v = struct.unpack_from('<HBBHHHHHBBBBBB', data['commands'], i * COMMAND_BYTES)
        seq, op, variant, ioidx, inf, outf, vf, cf, ni, no, nv, ncall_i, dup, reserved = v
        # Array order is the serialized schedule order; source sequence values
        # are provenance identifiers and need not be monotonic.
        require(op > 0 and variant > 0 and not reserved and dup < 2, 'bad command record')
        require(inf + ni <= nrefs and outf + no <= nrefs and vf + nv <= nvec and cf + ncall_i <= ncall,
                f'command {i} table range')
        require(all(x < nt for x in refs[inf:inf + ni] + refs[outf:outf + no]), 'bad tensor ref')
        cmds.append((seq, op, variant, ioidx, tuple(refs[inf:inf+ni]), tuple(refs[outf:outf+no]),
                     tuple(vectors[vf:vf+nv]), cf, ncall_i, dup))
    # Validate all compressed blocks and tensor descriptors while retaining the
    # exact block span/slice tuple of every descriptor.
    block_values = [int.from_bytes(data['blocks'][i:i+3], 'little') for i in range(0, len(data['blocks']), 3)]
    for i in range(nt):
        r = data['tensors'][i*TENSOR_BYTES:(i+1)*TENSOR_BYTES]
        dtype, flags, cls = struct.unpack_from('<BB4s', r)
        require(dtype < len(schema.DTYPES) and not flags & ~0x7f, f'bad tensor {i} header')
        require(not any(r[54:56]), f'nonzero tensor {i} reserved bytes')
        if flags & qdrx.BLOCKED:
            first = int.from_bytes(r[30:33], 'little')
            nblocks = struct.unpack_from('<H', r, 33)[0]
            require(first + nblocks <= nb, f'bad tensor {i} block span')
            for ent in block_values[first:first+nblocks]:
                pool, off = ent >> 21, (ent & ((1 << 21)-1)) * 128
                require(pool in (1, 3, 4, 5) and off + 2048 <= pool_bytes[pool],
                        f'bad tensor {i} block location')
        elif flags & qdrx.DIRECT:
            pool, off = struct.unpack_from('<BI', r, 30)
            require(pool in (1, 3, 4, 5), f'bad tensor {i} direct pool')
            if flags & qdrx.SHAPE:
                maxdims = struct.unpack_from('<4h', r, 14)
                esize = qdrx.rp.DTYPES[schema.DTYPES[dtype]]
                nbytes = esize
                for d in maxdims:
                    nbytes *= d
                require(off + nbytes <= pool_bytes[pool], f'bad tensor {i} direct range')
            else:
                require(off < pool_bytes[pool], f'bad tensor {i} direct location')
    io = struct.unpack('<4H', data['io'])
    require(io[0] + io[1] <= nrefs and io[2] + io[3] <= nrefs, 'bad graph I/O refs')
    require(all(x < nt for x in refs[io[0]:io[0]+io[1]] + refs[io[2]:io[2]+io[3]]), 'bad graph I/O tensor')
    return {'bytes': len(raw), 'counts': counts, 'sections': data, 'commands': cmds, 'block_values': block_values,
            'context_sha256': raw[48:80].hex(), 'constants_sha256': raw[80:112].hex(),
            'source_sha256': raw[112:144].hex(), 'pool_bytes': pool_bytes}


def tensor_features(program, index):
    """Decode the descriptor fields a per-command kernel resolver must inspect."""
    raw = program['sections']['tensors'][index*TENSOR_BYTES:(index+1)*TENSOR_BYTES]
    dtype, flags, cls = struct.unpack_from('<BB4s', raw)
    result = {'dtype': schema.DTYPES[dtype], 'flags': flags,
              'class': cls.rstrip(b'\0').decode('ascii'), 'storage': None,
              'quantization': None}
    if flags & qdrx.SHAPE:
        vals = struct.unpack_from('<12h', raw, 6)
        result['shape'] = {'dimensions': vals[:4], 'max_dimensions': vals[4:8], 'padding': vals[8:]}
    if flags & qdrx.SHAPE_VALUE:
        result['shape_values'] = struct.unpack_from('<4i', raw, 6)
    if flags & qdrx.SCALAR:
        result['scalar_word'] = struct.unpack_from('<I', raw, 6)[0]
    if flags & qdrx.DIRECT:
        result['storage'] = {'direct': struct.unpack_from('<BI', raw, 30)}
    elif flags & qdrx.BLOCKED:
        first = int.from_bytes(raw[30:33], 'little')
        count = struct.unpack_from('<H', raw, 33)[0]
        slice_origin = struct.unpack_from('<I', raw, 35)[0]
        blocks = program['block_values'][first:first+count]
        result['storage'] = {'blocks': tuple(((v >> 21), (v & ((1 << 21)-1))*128) for v in blocks),
                             'slice_origin': slice_origin}
    if flags & qdrx.QUANT:
        result['quantization'] = struct.unpack_from('<id', raw, 42)
    return result


def command_features(program, index):
    seq, op, variant, ioidx, ins, outs, vectors, cf, cn, dup = program['commands'][index]
    if 'tensor_features' in program:
        input_features = tuple(program['tensor_features'][i] for i in ins)
        output_features = tuple(program['tensor_features'][i] for i in outs)
        call_records = [{'id': name, 'pointer_mask': mask, 'args': args}
                        for name, mask, args in program['calls'][cf:cf+cn]]
    else:
        input_features = tuple(tensor_features(program, i) for i in ins)
        output_features = tuple(tensor_features(program, i) for i in outs)
        call_records = []
        for j in range(cn):
            name, mask, *args = struct.unpack_from('<II10Q', program['sections']['calls'], (cf+j)*88)
            call_records.append({'id': name, 'pointer_mask': mask, 'args': tuple(args)})
    return {'index': index, 'sequence': seq, 'opcode': op, 'variant': variant,
            'io_index': ioidx, 'duplicate_input_pad': bool(dup),
            'inputs': input_features, 'outputs': output_features,
            'framework_vector': vectors, 'calls': tuple(call_records),
            'pool_bytes': program['pool_bytes']}


@dataclass(frozen=True)
class Capability:
    supported: bool = False
    scratch_bytes: int = 0
    concurrent_scratch_bytes: int = 0
    may_fallback: bool = True
    reason: str = 'no exact kernel proof supplied'


def exact_feature_capability(approved_fingerprints, scratch_bytes=0,
                             concurrent_scratch_bytes=0):
    """Create a resolver that accepts only fully enumerated proof fingerprints."""
    approved = frozenset(approved_fingerprints)
    def resolve(features):
        fp = hashlib.sha256(json.dumps(features, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if fp not in approved:
            return Capability(reason=f'no numerical proof for feature fingerprint {fp}')
        return Capability(True, scratch_bytes, concurrent_scratch_bytes, False, 'exact proof fingerprint')
    return resolve


def preflight_native_program(program, resolvers=None, scratch_budget=0):
    """Fail closed per command; opcode registration alone cannot authorize a kernel."""
    if not resolvers:
        raise ValueError('native mode refused before execution; no per-command kernel resolvers registered')
    planned_free = L2_CEILING - program['bytes']
    if scratch_budget < 0 or scratch_budget > planned_free:
        raise ValueError(f'native mode refused before execution; claimed scratch budget {scratch_budget} exceeds planned {planned_free}')
    peak = 0
    decisions = []
    for i, cmd in enumerate(program['commands']):
        resolver = resolvers.get(cmd[1])
        if resolver is None:
            raise ValueError(f'native mode refused before execution; no resolver for command {i} opcode {cmd[1]}')
        features = command_features(program, i)
        decision = resolver(features)
        if not isinstance(decision, Capability) or not decision.supported:
            reason = getattr(decision, 'reason', 'resolver returned no explicit capability')
            raise ValueError(f'native mode refused before execution; command {i}: {reason}')
        if decision.may_fallback:
            raise ValueError(f'native mode refused before execution; command {i} permits legacy fallback')
        if min(decision.scratch_bytes, decision.concurrent_scratch_bytes) < 0:
            raise ValueError(f'native mode refused before execution; command {i} has invalid scratch budget')
        live = decision.scratch_bytes + decision.concurrent_scratch_bytes
        peak = max(peak, live)
        if live > scratch_budget:
            raise ValueError(f'native mode refused before execution; command {i} needs {live} scratch bytes, budget is {scratch_budget}')
        decisions.append({'index': i, 'opcode': cmd[1], 'variant': cmd[2], 'scratch_bytes': live})
    return {'commands': decisions, 'peak_scratch_bytes': peak}


def feature_fingerprint(features):
    return hashlib.sha256(json.dumps(features, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_equivalence(qdrx_raw, prepared_raw):
    src = qdrx.unpack(qdrx_raw)
    got = decode(prepared_raw)
    require(got['source_sha256'] == hashlib.sha256(qdrx_raw).hexdigest(), 'source QDRX binding mismatch')
    require(got['context_sha256'] == src['context_sha256'] and
            got['constants_sha256'] == src['constants_sha256'], 'model hash binding mismatch')
    ts = src['tensors']
    # Compare every narrowed runtime field against the source descriptors.
    tensor_blob = got['sections']['tensors']
    for i, t in enumerate(ts):
        r = tensor_blob[i*TENSOR_BYTES:(i+1)*TENSOR_BYTES]
        st = t.get('storage') or {}
        dtype, flags, cls = struct.unpack_from('<BB4s', r)
        expected_flags = 0
        expected_flags |= qdrx.SHAPE if 'shape' in t else 0
        expected_flags |= qdrx.DIRECT if st.get('direct') else 0
        expected_flags |= qdrx.BLOCKED if st.get('blocks') else 0
        expected_flags |= qdrx.SHAPE_VALUE if 'shape_values' in t else 0
        expected_flags |= qdrx.SCALAR if 'scalar_word' in t else 0
        expected_flags |= qdrx.QUANT if 'quantization' in t else 0
        expected_flags |= qdrx.CROUTON if t['class_name'][0] in 'Cc' and st.get('blocks') else 0
        require(dtype == qdrx.DTYPES[t['dtype']] and cls == t['class_name'].encode().ljust(4,b'\0')[:4],
                f'tensor {i} type/class mismatch')
        require(flags == expected_flags, f'tensor {i} runtime flags mismatch')
        shape = t.get('shape')
        if shape:
            vals = list(struct.unpack_from('<12h', r, 6))
            expected = shape['dimensions'] + shape['max_dimensions'] + shape['padding']
            require(vals == expected, f'tensor {i} shape mismatch')
        if 'shape_values' in t:
            require(struct.unpack_from('<4i', r, 6) == tuple(t['shape_values']),
                    f'tensor {i} shape-value mismatch')
        if 'scalar_word' in t:
            require(struct.unpack_from('<I', r, 6)[0] == t['scalar_word'],
                    f'tensor {i} scalar mismatch')
        if st.get('direct'):
            require(struct.unpack_from('<BI', r, 30) == (st['direct']['pool'], st['direct']['offset']),
                    f'tensor {i} direct storage mismatch')
        elif st.get('blocks'):
            first = int.from_bytes(r[30:33], 'little'); count = struct.unpack_from('<H', r, 33)[0]
            sl = struct.unpack_from('<I', r, 35)[0]
            require(count == len(st['blocks']) and sl == st.get('slice_offset', 0), f'tensor {i} block span mismatch')
            entries = got['block_values'][first:first+count]
            expected = [(b['pool'] << 21) | (b['offset'] // 128) for b in st['blocks']]
            require(entries == expected, f'tensor {i} block table mismatch')
        if 'quantization' in t:
            zp, scale = struct.unpack_from('<id', r, 42)
            require(zp == t['quantization']['zero_point'] and scale == t['quantization']['scale'],
                    f'tensor {i} quantization mismatch')
    require(len(got['commands']) == len(src['commands']), 'command count mismatch')
    for i, (p, c) in enumerate(zip(got['commands'], src['commands'])):
        seq, op, var, ioidx, ins, outs, fv, cf, cn, dup = p
        require((seq, op, var, ioidx, dup) == (c['sequence'], c['opcode'], qdrx.VARIANTS[c['variant']],
                NONE16 if c['io_index'] == qdrx.NONE else c['io_index'], int(c['duplicate_input_pad'])),
                f'command {i} header mismatch')
        tidx = {t['id']: j for j, t in enumerate(ts)}
        require(ins == tuple(tidx[x] for x in c['inputs']) and outs == tuple(tidx[x] for x in c['outputs']),
                f'command {i} refs mismatch')
        require(fv == tuple(c['framework_vector']), f'command {i} framework vector mismatch')
        call_bytes = got['sections']['calls'][cf*88:(cf+cn)*88]
        call_id = {name: j + 1 for j, name in enumerate(schema.CALL_NAMES)}
        expected_calls = []
        for name, regs, stack in c['calls']:
            mask, args = 0, []
            for j, arg in enumerate(regs + stack):
                if arg == 'P':
                    mask |= 1 << j
                    args.append(0)
                else:
                    args.append(arg)
            expected_calls.append(struct.pack('<II10Q', call_id[name], mask, *args))
        require(call_bytes == b''.join(expected_calls), f'command {i} call metadata mismatch')
    return {'tensors': len(ts), 'commands': len(src['commands']), 'blocks': len(got['block_values']),
            'bytes': len(prepared_raw)}


def traffic_census(qdrx_raw, prepared_raw):
    """Count current DSP metadata-copy bytes under declared-ref access."""
    src = qdrx.unpack(qdrx_raw)
    got = decode(prepared_raw)
    byid = {t['id']: t for t in src['tensors']}
    copied = Counter()
    capacity = 0x300ef000 - 0x300b8000
    for c in src['commands']:
        copied['command_record'] += 48  # DSP sizeof(C): 8*u32 + 4 32-bit pointers
        cursor = 0
        for kind, nbytes in (('input_refs', 4*len(c['inputs'])),
                             ('output_refs', 4*len(c['outputs'])),
                             ('framework_vectors', 4*len(c['framework_vector'])),
                             ('calls', 88*len(c['calls']))):
            if not nbytes:
                continue
            start = (cursor + 127) & ~127
            if start <= capacity and nbytes <= capacity - start:
                cursor = start + nbytes
                copied[kind] += nbytes
        seen = set()
        for tid in c['inputs'] + c['outputs']:
            if tid in seen:
                continue
            seen.add(tid)
            if len(seen) > 32:
                continue
            copied['tensor_records'] += 128  # DSP sizeof(T), including 32-bit block pointer
            nbytes = 4*len((byid[tid].get('storage') or {}).get('blocks', []))
            if nbytes:
                start = (cursor + 127) & ~127
                if start <= capacity and nbytes <= capacity - start:
                    cursor = start + nbytes
                    copied['block_lists'] += nbytes
    section_bytes = {name: len(data) for name, data in got['sections'].items()}
    block_sequences = []
    ref_sequences = []
    tids = {t['id']: i for i, t in enumerate(src['tensors'])}
    ref_sequences.extend(tuple(tids[i] for i in src[k]) for k in ('graph_inputs', 'graph_outputs'))
    for t in src['tensors']:
        block_sequences.append(tuple((b['pool'], b['offset'])
            for b in (t.get('storage', {}).get('blocks') or ())))
    for c in src['commands']:
        ref_sequences.extend((tuple(tids[i] for i in c['inputs']), tuple(tids[i] for i in c['outputs'])))
    # Count only exact contiguous reuse opportunities; partial overlap merging
    # would require indirect refs and a separate alias/view proof.
    def span_reuse(sequences):
        unique = set(s for s in sequences if s)
        byfirst = {}
        for big in unique:
            byfirst.setdefault(big[0], []).append(big)
        embedded = [s for s in unique if any(len(big) > len(s) and s in big
                    for big in byfirst.get(s[0], ()))]
        return {'unique_ordered_sequences': len(unique), 'unique_entries': sum(map(len, unique)),
                'strict_subspan_sequences': len(embedded), 'strict_subspan_entries': sum(map(len, embedded))}
    return {'current_declared_ref_copy_bytes': dict(copied),
            'current_declared_ref_copy_total': sum(copied.values()),
            'prepared_section_bytes': section_bytes,
            'prepared_one_time_bytes': len(prepared_raw),
            'prepared_per_command_record_bytes': COMMAND_BYTES * len(src['commands']),
            'exact_subspan_dedup_opportunity': {'block_lists': span_reuse(block_sequences),
                                                'ref_lists': span_reuse(ref_sequences)},
            'assumption': 'upper structural estimate: every unique declared input/output tensor is fetched once per command; dispatcher local C is 48 B and DSP T is 128 B'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('image', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--report', type=Path)
    args = ap.parse_args()
    raw = args.image.read_bytes()
    packed = lower(raw)
    check = verify_equivalence(raw, packed)
    if args.output:
        args.output.write_bytes(packed)
    rep = {'source': str(args.image), 'source_bytes': len(raw), 'prepared_bytes': len(packed),
           'saved_bytes': len(raw)-len(packed), 'l2_planning_ceiling_bytes': L2_CEILING,
           'within_planning_budget': len(packed) <= L2_CEILING,
           'l2_placement_proven': False,
           'source_sha256': hashlib.sha256(raw).hexdigest(),
           'metadata_traffic': traffic_census(raw, packed), **check}
    out = json.dumps(rep, indent=2) + '\n'
    if args.report:
        args.report.write_text(out)
    print(out, end='')


if __name__ == '__main__':
    main()
