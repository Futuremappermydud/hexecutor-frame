#!/usr/bin/env python3
"""Read the observed QNN 2.19 HTP graph serialization without executing it.

Format evidence: the installed Qualcomm HTP serialization headers and static
inspection of libQnnHtpPrepare.so. This is a model-specific research decoder,
not a supported general QNN converter. Fail on unrecognized structures.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

from pte import FB

MAGIC = 0x7309F72B
SEQ_XOR = 0x1303EE71
DTYPES = {0: 'unknown', 1: 'uint8', 2: 'uint16', 3: 'int16',
          4: 'float32', 5: 'int32', 6: 'int32_quantized', 7: 'int8'}


class DecodeError(ValueError):
    pass


class Reader:
    def __init__(self, data, pos=0, end=None):
        self.data, self.pos = data, pos
        self.end = len(data) if end is None else end

    def word(self):
        if self.pos + 4 > self.end:
            raise DecodeError(f'word outside record at {self.pos:#x}')
        v = struct.unpack_from('<I', self.data, self.pos)[0]
        self.pos += 4
        return v

    def words(self, count):
        return [self.word() for _ in range(count)]


def dictionary(data, pos, magic):
    r = Reader(data, pos)
    if r.word() != magic:
        raise DecodeError(f'dictionary magic at {pos:#x}')
    nwords, count, textwords = r.words(3)
    start = r.pos
    text = data[start:start + textwords * 4]
    names = text.split(b'\0')[:count]
    if len(names) != count or not all(names):
        raise DecodeError('invalid dictionary names')
    r.pos += textwords * 4
    codes = r.words(count)
    if r.pos != pos + 8 + nwords * 4:
        raise DecodeError('dictionary length mismatch')
    return [dict(index=i, name=n.decode(), format=c) for i, (n, c) in enumerate(zip(names, codes))], r.pos


def source_names(data, limit):
    """Locate the backend graph-info table and read its ID -> debug-name map."""
    f = FB(data)
    for t in range(0x480, limit, 4):
        try:
            vt = t - f.i32(t)
            if not 0 <= vt < limit - 4:
                continue
            vl, size = struct.unpack_from('<HH', data, vt)
            if vl != 14 or not 20 <= size <= 64:
                continue
            q = f.field(t, 3)
            if q is None:
                continue
            pos = f.ref(q)
            if not 0 <= pos < limit - 4:
                continue
            length = f.u32(pos)
            if not 1 <= length <= 80 or pos + 4 + length > limit:
                continue
            if f.string(t, 3) != 'zelda_qvga_boltnn_delegate - delegate_0':
                continue
            if len(f.vec_tables(t, 1)) != 4 or len(f.vec_tables(t, 2)) != 2:
                continue
            names = {f.scalar(row, 0, 'I'): f.string(row, 1) for row in f.vec_tables(t, 0)}
            if len(names) < 100 or not all(isinstance(v, str) and v for v in names.values()):
                continue
            return names
        except (struct.error, IndexError, UnicodeError):
            pass
    raise DecodeError('backend debug-name table not found')


class Tensors:
    def __init__(self, classes):
        self.classes = classes
        self.shared = {}
        self.blocktables = {}
        self.tensors = []

    def shared_obj(self, r, kind, read):
        code = r.word()
        idx = code & 0x7fffffff
        if code & 0x80000000:
            if idx in self.shared:
                raise DecodeError(f'duplicate shared object {idx}')
            obj = read()
            self.shared[idx] = kind, obj
        else:
            if idx not in self.shared:
                raise DecodeError(f'unknown shared {kind} object {idx} at {r.pos - 4:#x}')
            k, obj = self.shared[idx]
            if k != kind:
                raise DecodeError(f'shared object type mismatch: {kind} vs {k}')
        return obj

    def shape(self, r, rank):
        def read():
            code = r.word()
            flags = 0
            if code >> 16 == 0xcccc:
                flags, code = code & 0xffff, r.word()
            dims, maximum, pad = [], [], []
            for i in range(rank):
                nib = (code >> (4 * i)) & 15
                how = nib & 3
                d, m, p = 1, 1, 0
                if how:
                    w = r.word()
                    if how == 1:
                        d = w & 65535
                        m = d + ((w >> 16) & 255)
                        p = w >> 24
                    elif how == 2:
                        d = m = w & 0xffffff
                        p = w >> 24
                    else:
                        d = m = w
                if nib & 4:
                    m = r.word()
                if nib & 8:
                    p = r.word()
                dims.append(d)
                maximum.append(m)
                pad.append(p)
            return dict(dimensions=dims, max_dimensions=maximum, padding=pad, flags=flags)
        return self.shared_obj(r, 'shape', read)

    @staticmethod
    def direct_pointer(r):
        code = r.word()
        if code == 0:
            return None
        if code >> 16 <= 0x3fff:
            return dict(pool=code >> 16, offset=(code & 65535) * 64)
        return dict(pool=code & 0xfffffff, offset=r.word())

    @staticmethod
    def pointer_list(r, count):
        out, pool, offset = [], None, None
        while len(out) < count:
            word = r.word()
            if not word & 1:
                if pool is None:
                    raise DecodeError('relative pointer without pool')
                offset = word
                out.append(dict(pool=pool, offset=offset))
            elif not word & 2:
                p, off = word >> 22, (word << 6) & 0xfffff00
                if p:
                    pool, offset = p, off
                    out.append(dict(pool=pool, offset=offset))
                elif off:
                    out.extend([dict(unallocated=True)] * (off >> 8))
                    pool, offset = None, None
                else:
                    out.append(None)
                    pool, offset = None, None
            elif not word & 4:
                p, off = word >> 3, r.word()
                if p:
                    pool, offset = p, off
                    out.append(dict(pool=pool, offset=offset))
                else:
                    n, delta = off >> 16, struct.unpack('<h', struct.pack('<H', off & 65535))[0]
                    if pool is None or n < 3:
                        raise DecodeError('invalid pointer stride run')
                    for _ in range(n):
                        offset += delta
                        out.append(dict(pool=pool, offset=offset))
            elif not word & 8:
                pool = 1
                vals = [(word >> 8) & 4095, word >> 20]
                nwords = (word >> 4) & 15
                packed = sum(v << (32 * i) for i, v in enumerate(r.words(nwords)))
                vals += [(packed >> (12 * i)) & 4095 for i in range(nwords * 8 // 3)]
                for v in vals:
                    offset = v * 2048
                    out.append(dict(pool=1, offset=offset))
            else:
                raise DecodeError(f'unsupported pointer word {word:#x}')
        if len(out) != count:
            raise DecodeError(f'pointer run overruns table: {len(out)} > {count}')
        return out

    def indirect_pointer(self, r):
        code = r.word()
        count = (code >> 14) & 16383
        if code & 16383 == 16383:
            r.word()  # extended table slice offset
        if count == 16383:
            count = r.word()
        link, offset = None, 0
        if code & 0x40000000:
            w = r.word()
            if w & 0x80000000:
                link, offset = w & 0x7fffffff, r.word()
            else:
                link, offset = w >> 16, w & 65535
        blocks = self.pointer_list(r, count)
        if count and link is not None:
            # Link slots are reused after the prior table's consumers.
            self.blocktables[link] = blocks
        if not count and link is not None:
            blocks = self.blocktables.get(link)
        return dict(blocktable_link=link, slice_offset=offset,
                    serialized_block_count=count, blocks=blocks)

    def tensor(self, r, producer):
        start = r.pos
        index = r.word()
        if index >= len(self.classes):
            raise DecodeError(f'unknown tensor type {index} at {start:#x}')
        cls = self.classes[index]
        code = cls['format']
        mode, rank = code >> 28, (code >> 8) & 15
        t = dict(id=len(self.tensors) + 1, producer=producer, offset=start,
                 class_index=index, class_name=cls['name'], format=code,
                 dtype=DTYPES.get(code & 15, code & 15), element_bytes=1 << ((code >> 6) & 3))
        if mode == 2:
            t['shape_values'] = r.words(rank)
        elif mode == 3:
            value = r.word()
            t['scalar_word'] = value
            t['scalar'] = struct.unpack('<f' if code & 15 == 4 else '<i', struct.pack('<I', value))[0]
        elif mode == 1:
            if code & (1 << 17):
                def read_quant():
                    zero, scale = r.words(2)
                    return dict(zero_point=struct.unpack('<i', struct.pack('<I', zero))[0],
                                scale=struct.unpack('<f', struct.pack('<I', scale))[0])
                t['quantization'] = self.shared_obj(r, 'interface', read_quant)
            t['shape'] = self.shape(r, rank)
            if code & (1 << 18):
                t['storage'] = self.indirect_pointer(r)
            else:
                t['storage'] = dict(direct=self.direct_pointer(r))
        else:
            raise DecodeError(f'unknown tensor mode {mode}')
        t['serialized_bytes'] = r.pos - start
        self.tensors.append(t)
        return t['id']


def decode(path, output):
    data = path.read_bytes()
    base = data.find(struct.pack('<I', MAGIC))
    if base < 0:
        raise DecodeError('no HTP pickle header')
    hwords = struct.unpack_from('<I', data, base + 4)[0] & 65535
    end_header = base + hwords * 4
    tags, p = {}, base + 8
    while p < end_header:
        desc = struct.unpack_from('<I', data, p)[0]
        n, tag = desc & 65535, (desc >> 16).to_bytes(2, 'little').decode('ascii')
        if not n or p + n * 4 > end_header:
            raise DecodeError('invalid HTP header record')
        tags[tag] = dict(offset=p, words=list(struct.unpack_from('<' + 'I' * (n - 1), data, p + 4)))
        p += n * 4
    ops, p = dictionary(data, end_header, 0x6f4390bc)
    classes, p = dictionary(data, p, 0x74438bbc)
    names = source_names(data, base)
    first = data.find(struct.pack('<I', SEQ_XOR), p)
    if first < 0:
        raise DecodeError('no operation sequence')
    count = struct.unpack_from('<I', data, first - 12)[0]
    markers = []
    for pos in range(first, len(data) - 4, 4):
        value = struct.unpack_from('<I', data, pos)[0]
        if ((value ^ SEQ_XOR) & 0x1fffffff) == len(markers):
            markers.append(pos)
            if len(markers) == count:
                break
    if len(markers) != count:
        raise DecodeError('operation marker count mismatch')
    sentinel = data.find(struct.pack('<I', 0xbeeff00d), markers[-1])
    if sentinel < 0:
        raise DecodeError('missing graph end sentinel')
    # Deserializer::resolve_fixups consumes the final zero identity after ops.
    if struct.unpack_from('<I', data, sentinel - 4)[0] != 0:
        raise DecodeError('missing final tensor fixup terminator')
    markers.append(sentinel - 4)
    tensors = Tensors(classes)
    io = data.find(struct.pack('<6I', 4, 0, 0x80000001, 0, 0x3b808081, 0x80000002), p, first)
    if io < 0:
        raise DecodeError('observed four-input graph preamble not found')
    r = Reader(data, io, first)
    inputs = [tensors.tensor(r, 'input') for _ in range(r.word())]
    outputs = [tensors.tensor(r, 'output') for _ in range(r.word())]
    nodes = []
    for seq, (start, end) in enumerate(zip(markers, markers[1:])):
        r = Reader(data, start, end)
        marker, typeword, source, site = r.words(4)
        kind = typeword & 0xfffffff
        if kind >= len(ops):
            raise DecodeError('unknown operation type')
        flags, fmt = typeword >> 28, ops[kind]['format']
        node = dict(sequence=seq, offset=start, serialized_bytes=end - start,
                    type_index=kind, type_name=ops[kind]['name'], flags=flags,
                    source_id=source, source_site=site, inputs=[], outputs=[])
        if source in names:
            node['source_name'] = names[source]
        if not flags & 1:
            node['checkpoints'] = r.words(1 if flags & 2 else 2)
        try:
            if fmt == 0xffffffff:
                counts = r.word()
                ni, no = counts & 65535, counts >> 16
            else:
                ni, no = fmt & 65535, (fmt >> 16) & 0x7fff
            if fmt & 0x80000000 and fmt != 0xffffffff:
                extra = (fmt >> 4) & 63
                if extra:
                    node['framework_words'] = r.words(extra)
                # Framework pointer counts use separate bitfields.
                ni = (fmt >> 8) & 15
                no = (fmt >> 16) & 7
                if fmt & 0x8000:
                    length = r.word()
                    if length % 4:
                        raise DecodeError('unaligned framework vector')
                    node['framework_vector'] = r.words(length // 4)
            node['inputs'] = r.words(ni)
            node['outputs'] = [tensors.tensor(r, seq) for _ in range(no)]
            if r.pos != end:
                raise DecodeError(f'{end-r.pos} trailing bytes')
        except Exception as exc:
            raise DecodeError(f'operation {seq} {node["type_name"]} at {start:#x}, cursor {r.pos:#x}: {exc}') from exc
        nodes.append(node)
    by_id = {t['id']: t for t in tensors.tensors}
    for n in nodes:
        if n['type_name'].startswith('*InputSlice'):
            parameter = 4 if n['type_name'].startswith('*InputSlicePad') else 3
            n['graph_input_index'] = by_id[n['inputs'][parameter]]['shape_values'][3]
        elif n['type_name'].startswith('*OutputSlice'):
            n['graph_output_index'] = by_id[n['inputs'][3]]['shape_values'][3]
    # Every reference must resolve to an earlier tensor definition.
    for n in nodes:
        for i in n['inputs']:
            if i not in by_id:
                raise DecodeError(f'undefined tensor reference {i}')
            producer = by_id[i]['producer']
            if isinstance(producer, int) and producer >= n['sequence']:
                raise DecodeError(f'non-causal tensor reference {i}')
    result = dict(context_sha256=hashlib.sha256(data).hexdigest(), pickle_offset=base,
                  header=tags, op_types=ops, tensor_classes=classes,
                  graph_inputs=inputs, graph_outputs=outputs, source_names=names,
                  nodes=nodes, tensors=tensors.tensors)
    output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, separators=(',', ':'), allow_nan=False).encode() + b'\n'
    with gzip.GzipFile(filename=str(output / 'graph.json.gz'), mode='wb', mtime=0) as out:
        out.write(serialized)
    # Keep a compact uncompressed graph for tools which do not read gzip.
    (output / 'graph.json').write_bytes(serialized)
    with (output / 'operations.csv').open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['sequence', 'kernel', 'source_id', 'source_name', 'input_tensor_ids', 'output_tensor_ids'])
        for n in nodes:
            writer.writerow([n['sequence'], n['type_name'], n['source_id'], n.get('source_name', ''),
                             ' '.join(map(str, n['inputs'])), ' '.join(map(str, n['outputs']))])
    summary = dict(context_sha256=result['context_sha256'], operation_count=len(nodes),
                   tensor_count=len(tensors.tensors), source_name_count=len(names),
                   kernel_counts=dict(Counter(n['type_name'] for n in nodes)),
                   checks=['All operation records consumed exactly', 'All tensor references resolve without forward references',
                           'All shared shape/interface references resolve'],
                   status='Structural decoding only; numerical inference unverified.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{len(nodes)} operations, {len(tensors.tensors)} tensors -> {output}')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('context', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    decode(args.context, args.output)
