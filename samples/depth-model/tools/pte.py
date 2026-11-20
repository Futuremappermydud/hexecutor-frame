# Minimal ExecuTorch .pte reader (no executorch/torch dependency).
# Parses the Program flatbuffer: execution plans, value table, inputs/outputs,
# delegates with compile specs, and segment table. Reads files as data only.
import struct
import sys

SCALAR = {0: 'uint8', 1: 'int8', 2: 'int16', 3: 'int32', 4: 'int64', 5: 'float16',
          6: 'float32', 7: 'float64', 11: 'bool', 12: 'qint8', 13: 'quint8',
          14: 'qint32', 15: 'bfloat16', 27: 'uint16'}
KTYPE = {1: 'Null', 2: 'Int', 3: 'Bool', 4: 'Double', 5: 'Tensor', 6: 'String',
         7: 'IntList', 8: 'DoubleList', 9: 'BoolList', 10: 'TensorList',
         11: 'OptionalTensorList'}


class FB:
    def __init__(self, buf):
        self.b = buf

    def u8(self, o): return self.b[o]
    def u32(self, o): return struct.unpack_from('<I', self.b, o)[0]
    def i32(self, o): return struct.unpack_from('<i', self.b, o)[0]
    def u64(self, o): return struct.unpack_from('<Q', self.b, o)[0]
    def i64(self, o): return struct.unpack_from('<q', self.b, o)[0]

    def root(self): return self.u32(0)

    def field(self, t, idx):
        vt = t - self.i32(t)
        vlen = struct.unpack_from('<H', self.b, vt)[0]
        off_pos = 4 + 2 * idx
        if off_pos >= vlen:
            return None
        off = struct.unpack_from('<H', self.b, vt + off_pos)[0]
        return t + off if off else None

    def ref(self, p): return p + self.u32(p)

    def table(self, t, idx):
        p = self.field(t, idx)
        return self.ref(p) if p is not None else None

    def string(self, t, idx):
        p = self.field(t, idx)
        if p is None:
            return None
        s = self.ref(p)
        n = self.u32(s)
        return self.b[s + 4:s + 4 + n].decode('utf8', 'replace')

    def vec(self, t, idx):
        p = self.field(t, idx)
        if p is None:
            return None, 0
        v = self.ref(p)
        return v + 4, self.u32(v)

    def vec_tables(self, t, idx):
        start, n = self.vec(t, idx)
        return [self.ref(start + 4 * i) for i in range(n)] if start else []

    def vec_scalars(self, t, idx, fmt, size):
        start, n = self.vec(t, idx)
        return [struct.unpack_from('<' + fmt, self.b, start + size * i)[0] for i in range(n)] if start else []

    def vec_bytes(self, t, idx):
        start, n = self.vec(t, idx)
        return bytes(self.b[start:start + n]) if start else b''

    def scalar(self, t, idx, fmt, default=0):
        p = self.field(t, idx)
        return struct.unpack_from('<' + fmt, self.b, p)[0] if p is not None else default


def parse(data):
    assert data[4:8] == b'ET12', 'not an ExecuTorch program'
    eh = data.find(b'eh00', 8, 64)
    prog_size, seg_base = struct.unpack_from('<QQ', data, eh + 8) if eh >= 0 else (len(data), 0)
    fb = FB(data)
    prog = fb.root()
    out = {'program_size': prog_size, 'segment_base': seg_base, 'plans': [], 'segments': []}
    for seg in fb.vec_tables(prog, 4):
        out['segments'].append((fb.scalar(seg, 0, 'Q'), fb.scalar(seg, 1, 'Q')))
    for plan in fb.vec_tables(prog, 1):
        values = []
        for ev in fb.vec_tables(plan, 2):
            kt = fb.scalar(ev, 0, 'B')
            v = fb.table(ev, 1)
            ent = {'kind': KTYPE.get(kt, kt)}
            if kt == 5 and v is not None:
                ent['dtype'] = SCALAR.get(fb.scalar(v, 0, 'b'), fb.scalar(v, 0, 'b'))
                ent['sizes'] = fb.vec_scalars(v, 2, 'i', 4)
                ent['dim_order'] = list(fb.vec_bytes(v, 3))
                ent['data_buffer_idx'] = fb.scalar(v, 5, 'I')
            elif kt == 2 and v is not None:
                ent['int'] = fb.scalar(v, 0, 'q')
            elif kt == 4 and v is not None:
                ent['double'] = fb.scalar(v, 0, 'd')
            elif kt == 6 and v is not None:
                ent['str'] = fb.string(v, 0)
            values.append(ent)
        delegates = []
        for d in fb.vec_tables(plan, 7):
            ref = fb.table(d, 1)
            specs = {}
            for cs in fb.vec_tables(d, 2):
                specs[fb.string(cs, 0)] = fb.vec_bytes(cs, 1)
            delegates.append({'id': fb.string(d, 0),
                              'location': fb.scalar(ref, 0, 'b') if ref else None,
                              'index': fb.scalar(ref, 1, 'I') if ref else None,
                              'compile_specs': specs})
        out['plans'].append({'name': fb.string(plan, 0), 'values': values,
                             'inputs': fb.vec_scalars(plan, 3, 'i', 4),
                             'outputs': fb.vec_scalars(plan, 4, 'i', 4),
                             'delegates': delegates})
    return out


def delegate_bytes(data, info, plan=0, which=0):
    d = info['plans'][plan]['delegates'][which]
    assert d['location'] == 1, 'only segment-stored delegates supported'
    off, size = info['segments'][d['index']]
    base = info['segment_base'] + off
    return data[base:base + size]


def show(info, label):
    print(f'== {label}: program={info["program_size"]} segment_base={info["segment_base"]} segments={info["segments"]}')
    for p in info['plans']:
        print(f'  plan {p["name"]!r}: {len(p["values"])} values')
        for io, idxs in (('input', p['inputs']), ('output', p['outputs'])):
            for i in idxs:
                print(f'    {io} v{i}: {p["values"][i]}')
        for d in p['delegates']:
            specs = {k: (v.decode('latin1') if all(32 <= c < 127 for c in v) else v.hex()) for k, v in d['compile_specs'].items()}
            print(f'    delegate {d["id"]} loc={d["location"]} idx={d["index"]} specs={specs}')


if __name__ == '__main__':
    data = open(sys.argv[1], 'rb').read()
    info = parse(data)
    show(info, 'outer')
    for di in range(len(info['plans'][0]['delegates'])):
        blob = delegate_bytes(data, info, 0, di)
        if blob[4:8] == b'ET12':
            inner = parse(blob)
            show(inner, f'inner (delegate {di})')
            if len(sys.argv) > 2:
                payload = delegate_bytes(blob, inner)
                open(sys.argv[2], 'wb').write(payload)
                print(f'  wrote innermost delegate payload: {len(payload)} bytes -> {sys.argv[2]}')
