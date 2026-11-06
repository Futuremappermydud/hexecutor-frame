#!/usr/bin/env python3
"""NPRG v2: deduplicated immutable geometry plus delta-coded block spans."""
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
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str((PATHS.native)))
import depth_model.lowering.resident_image as qdrx
import hexecutor.program.resident_schema as schema
import prepared_program as v1

MAGIC = b'NPRG'
VERSION = 2
HEADER = 256
GEOMETRY_BYTES = 44
TENSOR_BYTES = 12
COMMAND_BYTES = 20
SECTIONS = ('geometries', 'tensors', 'block_stream', 'commands', 'refs', 'vectors', 'calls')
NONE16 = 0xffff


def require(ok, message):
    if not ok:
        raise ValueError(message)


def uleb(value):
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def zigzag(n):
    return 2*n if n >= 0 else -2*n-1


def unzigzag(n):
    return (n >> 1) if not (n & 1) else -((n >> 1)+1)


def intern(table, cache, values):
    values = tuple(values)
    if not values:
        return 0, 0
    if values not in cache:
        cache[values] = len(table)
        table.extend(values)
    return cache[values], len(values)


def _geometry(t):
    storage = t.get('storage') or {}
    flags = 0
    flags |= qdrx.SHAPE if 'shape' in t else 0
    flags |= qdrx.CROUTON if storage.get('blocks') and t['class_name'][0] in 'Cc' else 0
    flags |= qdrx.SHAPE_VALUE if 'shape_values' in t else 0
    flags |= qdrx.SCALAR if 'scalar_word' in t else 0
    flags |= qdrx.QUANT if 'quantization' in t else 0
    payload = bytes(24)
    if 'shape' in t:
        shape = t['shape']
        dims = shape['dimensions'] + shape['max_dimensions'] + shape['padding']
        require(all(-32768 <= int(v) <= 32767 for v in dims), 'shape outside compact i16 range')
        payload = struct.pack('<12h', *dims)
    elif 'shape_values' in t:
        payload = struct.pack('<4i', *t['shape_values']) + bytes(8)
    elif 'scalar_word' in t:
        payload = struct.pack('<I', t['scalar_word']) + bytes(20)
    else:
        require(t['class_name'] == 'None', f"tensor {t['id']} has no value geometry")
    q = t.get('quantization', {})
    row = struct.pack('<BB4s', qdrx.DTYPES[t['dtype']], flags,
                      t['class_name'].encode().ljust(4, b'\0')[:4])
    row += payload + struct.pack('<id', q.get('zero_point', 0), q.get('scale', 0.0)) + bytes(2)
    require(len(row) == GEOMETRY_BYTES, 'geometry record stride mismatch')
    return row


def _encode_blocks(blocks):
    require(blocks, 'empty blocked tensor list')
    pools = {b['pool'] for b in blocks}
    require(len(pools) == 1, 'compact v2 requires a single pool per tensor block span')
    pool = blocks[0]['pool']
    require(pool in (1, 3, 4, 5), 'unsupported block pool')
    offsets = [b['offset'] // 128 for b in blocks]
    require(all(b['offset'] % 128 == 0 for b in blocks), 'unaligned block descriptor')
    stream = bytearray(struct.pack('<I', (pool << 21) | offsets[0])[:3])
    for previous, current in zip(offsets, offsets[1:]):
        stream.extend(uleb(zigzag(current-previous)))
    return bytes(stream)


def lower(raw):
    src = qdrx.unpack(raw)
    tids = {t['id']: i for i, t in enumerate(src['tensors'])}
    geos, geo_cache, tensors, block_stream = [], {}, [], bytearray()
    block_cache = {}
    for t in src['tensors']:
        geometry = _geometry(t)
        if geometry not in geo_cache:
            geo_cache[geometry] = len(geos)
            geos.append(geometry)
        gi = geo_cache[geometry]
        st = t.get('storage') or {}
        if st.get('direct'):
            tag = 1
            payload = struct.pack('<BI', st['direct']['pool'], st['direct']['offset']) + bytes(4)
        elif st.get('blocks'):
            tag = 2
            key = tuple((b['pool'], b['offset']) for b in st['blocks'])
            if key not in block_cache:
                block_cache[key] = len(block_stream)
                block_stream.extend(_encode_blocks(st['blocks']))
            first = block_cache[key]
            require(first < (1 << 24) and len(st['blocks']) < (1 << 16), 'block span fields overflow')
            payload = first.to_bytes(3, 'little') + struct.pack('<HI', len(st['blocks']), st.get('slice_offset', 0))
        else:
            tag, payload = 0, bytes(9)
        require(gi < (1 << 16), 'geometry index overflow')
        tensors.append(struct.pack('<HB', gi, tag) + payload)
        require(len(tensors[-1]) == TENSOR_BYTES, 'tensor location record stride mismatch')

    refs, vectors, calls = [], [], []
    rc, vc, cc = {}, {}, {}
    graph_io = []
    for key in ('graph_inputs', 'graph_outputs'):
        graph_io.append(intern(refs, rc, [tids[i] for i in src[key]]))
    call_id = {name: j+1 for j, name in enumerate(schema.CALL_NAMES)}
    commands = []
    for c in src['commands']:
        ins = [tids[i] for i in c['inputs']]
        outs = [tids[i] for i in c['outputs']]
        inf, ni = intern(refs, rc, ins)
        outf, no = intern(refs, rc, outs)
        vf, nv = intern(vectors, vc, c['framework_vector'])
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
        seq, op, variant, ioidx = c['sequence'], c['opcode'], qdrx.VARIANTS[c['variant']], c['io_index']
        require(seq < (1 << 16) and op < 256 and variant < 256 and (ioidx == qdrx.NONE or ioidx <= NONE16),
                f'command {seq} narrow-field overflow')
        require(all(x < (1 << 16) for x in (inf, outf, vf, cf)) and all(x < 256 for x in (ni, no, nv, cn)),
                f'command {seq} table-span overflow')
        commands.append(struct.pack('<HBBHHHHHBBBBBB', seq, op, variant,
            NONE16 if ioidx == qdrx.NONE else ioidx, inf, outf, vf, cf, ni, no, nv, cn,
            int(c['duplicate_input_pad']), 0))

    sections = [b''.join(geos), b''.join(tensors), bytes(block_stream), b''.join(commands),
                struct.pack(f'<{len(refs)}H', *refs), struct.pack(f'<{len(vectors)}I', *vectors), b''.join(calls)]
    header = bytearray(HEADER)
    struct.pack_into('<4sHHIIIIIII', header, 0, MAGIC, VERSION, HEADER, 0, 0, qdrx.ABI,
                     len(geos), len(tensors), len(block_stream), len(commands))
    struct.pack_into('<3I', header, 36, len(refs), len(vectors), len(calls))
    header[48:80] = bytes.fromhex(src['context_sha256'])
    header[80:112] = bytes.fromhex(src['constants_sha256'])
    header[112:144] = hashlib.sha256(raw).digest()
    struct.pack_into('<7I', header, 144, *(src['pool_bytes'][k] for k in ('0','1','2','3','4','5','spill')))
    struct.pack_into('<4H', header, 172, graph_io[0][0], graph_io[0][1], graph_io[1][0], graph_io[1][1])
    out = bytearray(header)
    for i, section in enumerate(sections):
        out.extend(bytes((-len(out)) % 8))
        struct.pack_into('<II', out, 180 + i*8, len(out), len(section))
        out.extend(section)
    struct.pack_into('<I', out, 8, len(out))
    struct.pack_into('<I', out, 12, zlib.crc32(out))
    return bytes(out)


def _read_uleb(data, at):
    value = shift = 0
    while True:
        require(at < len(data) and shift < 35, 'bad delta varint')
        b = data[at]; at += 1
        value |= (b & 127) << shift
        if not b & 128:
            return value, at
        shift += 7


def _decode_blocks(stream, first, count, pool_bytes):
    require(count > 0 and first + 3 <= len(stream), 'bad compressed block span')
    code = int.from_bytes(stream[first:first+3], 'little')
    pool, offset = code >> 21, (code & ((1 << 21)-1))
    require(pool in (1,3,4,5), 'bad compressed block pool')
    result = [(pool, offset*128)]
    at = first+3
    for _ in range(count-1):
        value, at = _read_uleb(stream, at)
        offset += unzigzag(value)
        require(0 <= offset < (1 << 21), 'compressed block delta overflow')
        result.append((pool, offset*128))
    require(all(o+2048 <= pool_bytes[p] for p,o in result), 'compressed block outside pool')
    return tuple(result), at


def decode(raw):
    require(len(raw) >= HEADER, 'truncated NPRG2')
    magic, version, hbytes, total, crc, abi, ng, nt, nblock_bytes, nc = struct.unpack_from('<4sHHIIIIIII', raw)
    require((magic,version,hbytes,total)==(MAGIC,VERSION,HEADER,len(raw)), 'bad NPRG2 header')
    require(abi == qdrx.ABI, 'NPRG2 ABI mismatch')
    cbuf=bytearray(raw);struct.pack_into('<I',cbuf,12,0)
    require(zlib.crc32(cbuf)==crc, 'NPRG2 CRC mismatch')
    nr,nv,ncals=struct.unpack_from('<3I',raw,36)
    require(not any(raw[236:HEADER]), 'nonzero NPRG2 reserved header')
    pool_bytes=struct.unpack_from('<7I',raw,144)
    io=struct.unpack_from('<4H',raw,172)
    sections={};end=HEADER
    for i,name in enumerate(SECTIONS):
        off,size=struct.unpack_from('<II',raw,180+i*8)
        require(off==(end+7)&~7 and not any(raw[end:off]),f'bad {name} section offset')
        require(off+size<=len(raw),f'{name} section bounds')
        sections[name]=raw[off:off+size];end=off+size
    require(end==len(raw),'trailing NPRG2 bytes')
    require(len(sections['geometries'])==ng*GEOMETRY_BYTES and len(sections['tensors'])==nt*TENSOR_BYTES and
            len(sections['block_stream'])==nblock_bytes and len(sections['commands'])==nc*COMMAND_BYTES and
            len(sections['refs'])==nr*2 and len(sections['vectors'])==nv*4 and len(sections['calls'])==ncals*88,
            'NPRG2 section counts mismatch')
    # Check every compact storage pointer, including duplicate geometry/view entries.
    block_cache={}; tensor_features=[]
    for i in range(nt):
        gi,tag=struct.unpack_from('<HB',sections['tensors'],i*TENSOR_BYTES)
        payload=sections['tensors'][i*TENSOR_BYTES+3:(i+1)*TENSOR_BYTES]
        require(gi<ng and tag<=2, f'bad tensor location {i}')
        if tag==0:
            require(not any(payload), f'bad absent storage {i}')
            storage=None
        elif tag==1:
            pool=payload[0];offset=struct.unpack_from('<I',payload,1)[0]
            require(pool in (1,3,4,5), f'bad direct pool {i}')
            require(not any(payload[5:]), f'nonzero direct location reserved bytes {i}')
            storage={'direct':(pool,offset)}
        else:
            first=int.from_bytes(payload[:3],'little');count=struct.unpack_from('<H',payload,3)[0]
            slice_origin=struct.unpack_from('<I',payload,5)[0]
            if (first,count) not in block_cache:
                block_cache[(first,count)],_=_decode_blocks(sections['block_stream'],first,count,pool_bytes)
            storage={'blocks':block_cache[(first,count)],'slice_origin':slice_origin}
        grow=sections['geometries'][gi*GEOMETRY_BYTES:(gi+1)*GEOMETRY_BYTES]
        dtype,flags,cls=struct.unpack_from('<BB4s',grow)
        require(dtype<len(schema.DTYPES) and not flags & ~(qdrx.SHAPE|qdrx.CROUTON|qdrx.SHAPE_VALUE|qdrx.SCALAR|qdrx.QUANT),
                f'bad geometry {gi}')
        feat={'dtype':schema.DTYPES[dtype],'flags':flags,'class':cls.rstrip(b'\0').decode('ascii'),
              'storage':storage,'quantization':None}
        require(not any(grow[42:44]),f'nonzero geometry reserved bytes {gi}')
        if flags&qdrx.SHAPE:
            dims=struct.unpack_from('<12h',grow,6)
            feat['shape']={'dimensions':dims[:4],'max_dimensions':dims[4:8],'padding':dims[8:]}
        if flags&qdrx.SHAPE_VALUE:feat['shape_values']=struct.unpack_from('<4i',grow,6)
        if flags&qdrx.SCALAR:feat['scalar_word']=struct.unpack_from('<I',grow,6)[0]
        if flags&qdrx.QUANT:feat['quantization']=struct.unpack_from('<id',grow,30)
        if tag==1 and flags&qdrx.SHAPE:
            maxdims=struct.unpack_from('<4h',grow,14)
            esize=qdrx.rp.DTYPES[schema.DTYPES[dtype]]
            size=esize
            for dim in maxdims:size*=dim
            require(storage['direct'][1]+size<=pool_bytes[pool],f'direct tensor {i} outside pool')
        tensor_features.append(feat)
    refs=struct.unpack(f'<{nr}H',sections['refs'])
    vectors=struct.unpack(f'<{nv}I',sections['vectors'])
    calls=[]
    for j in range(ncals):
        name,mask,*args=struct.unpack_from('<II10Q',sections['calls'],j*88)
        require(1<=name<=len(schema.CALL_NAMES) and mask<1024 and
                all(args[k]==0 for k in range(10) if mask&(1<<k)),f'bad call record {j}')
        calls.append((name,mask,tuple(args)))
    commands=[]
    for i in range(nc):
        v=struct.unpack_from('<HBBHHHHHBBBBBB',sections['commands'],i*COMMAND_BYTES)
        seq,op,var,ioidx,inf,outf,vf,cf,ni,no,nv_i,nc_i,dup,res=v
        require(op>0 and var>0 and dup<2 and res==0 and inf+ni<=nr and outf+no<=nr and vf+nv_i<=nv and cf+nc_i<=ncals,
                f'bad command {i}')
        ins=refs[inf:inf+ni];outs=refs[outf:outf+no]
        require(all(x<nt for x in ins+outs),f'bad command tensor refs {i}')
        commands.append((seq,op,var,ioidx,ins,outs,tuple(vectors[vf:vf+nv_i]),cf,nc_i,dup))
    require(io[0]+io[1]<=nr and io[2]+io[3]<=nr and
            all(x<nt for x in refs[io[0]:io[0]+io[1]]+refs[io[2]:io[2]+io[3]]),'bad graph I/O refs')
    return {'bytes':len(raw),'sections':sections,'counts':(ng,nt,nblock_bytes,nc,nr,nv,ncals),
            'tensor_features':tensor_features,'commands':commands,'block_cache':block_cache,
            'calls':calls,'pool_bytes':pool_bytes,'io':io,'context_sha256':raw[48:80].hex(),
            'constants_sha256':raw[80:112].hex(),'source_sha256':raw[112:144].hex()}


def verify_equivalence(source_raw, compact_raw):
    src=qdrx.unpack(source_raw);got=decode(compact_raw)
    require(got['source_sha256']==hashlib.sha256(source_raw).hexdigest(),'NPRG2 source hash mismatch')
    require(got['context_sha256']==src['context_sha256'] and got['constants_sha256']==src['constants_sha256'],
            'NPRG2 model binding mismatch')
    tidx={t['id']:i for i,t in enumerate(src['tensors'])}
    require(len(src['tensors'])==len(got['tensor_features']),'tensor count mismatch')
    for i,(t,f) in enumerate(zip(src['tensors'],got['tensor_features'])):
        require(f['dtype']==t['dtype'] and f['class']==t['class_name'],f'tensor {i} type/class mismatch')
        st=t.get('storage') or {}
        expected_flags=0
        expected_flags |= qdrx.SHAPE if 'shape' in t else 0
        expected_flags |= qdrx.CROUTON if st.get('blocks') and t['class_name'][0] in 'Cc' else 0
        expected_flags |= qdrx.SHAPE_VALUE if 'shape_values' in t else 0
        expected_flags |= qdrx.SCALAR if 'scalar_word' in t else 0
        expected_flags |= qdrx.QUANT if 'quantization' in t else 0
        require(f['flags']==expected_flags,f'tensor {i} geometry flags mismatch')
        if 'shape' in t:require(f['shape']['dimensions']==tuple(t['shape']['dimensions']) and f['shape']['max_dimensions']==tuple(t['shape']['max_dimensions']) and f['shape']['padding']==tuple(t['shape']['padding']),f'tensor {i} shape mismatch')
        if 'shape_values' in t:require(f['shape_values']==tuple(t['shape_values']),f'tensor {i} shape-values mismatch')
        if 'scalar_word' in t:require(f['scalar_word']==t['scalar_word'],f'tensor {i} scalar mismatch')
        if st.get('direct'):require(f['storage']['direct']==(st['direct']['pool'],st['direct']['offset']),f'tensor {i} direct mismatch')
        elif st.get('blocks'):
            expected=tuple((b['pool'],b['offset']) for b in st['blocks'])
            require(f['storage']['blocks']==expected and f['storage']['slice_origin']==st.get('slice_offset',0),f'tensor {i} blocked view mismatch')
        else:require(f['storage'] is None,f'tensor {i} unexpected storage')
        if 'quantization' in t:require(f['quantization']==(t['quantization']['zero_point'],t['quantization']['scale']),f'tensor {i} quantization mismatch')
    require(len(got['commands'])==len(src['commands']),'command count mismatch')
    for i,(p,c) in enumerate(zip(got['commands'],src['commands'])):
        seq,op,var,ioidx,ins,outs,fv,cf,cn,dup=p
        require((seq,op,var,ioidx,dup)==(c['sequence'],c['opcode'],qdrx.VARIANTS[c['variant']],
          NONE16 if c['io_index']==qdrx.NONE else c['io_index'],int(c['duplicate_input_pad'])),f'command {i} fields mismatch')
        require(ins==tuple(tidx[x] for x in c['inputs']) and outs==tuple(tidx[x] for x in c['outputs']),f'command {i} refs mismatch')
        require(fv==tuple(c['framework_vector']),f'command {i} vector mismatch')
        def enc(call):
            name,regs,stack=call;mask=0;args=[]
            for j,a in enumerate(regs+stack):
                if a=='P':mask|=1<<j;args.append(0)
                else:args.append(a)
            return (schema.CALL_NAMES.index(name)+1,mask,tuple(args))
        require(got['calls'][cf:cf+cn]==[enc(x) for x in c['calls']],f'command {i} calls mismatch')
    return {'bytes':len(compact_raw),'tensors':len(src['tensors']),'geometries':got['counts'][0],
            'commands':len(src['commands']),'compressed_block_stream_bytes':len(got['sections']['block_stream'])}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('image',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--report',type=Path)
    args=ap.parse_args();source=args.image.read_bytes();packed=lower(source);check=verify_equivalence(source,packed)
    old_size=len(v1.lower(source));decoded=decode(packed)
    hot_bytes=len(packed)-len(decoded['sections']['block_stream'])
    stack_bytes=3*64*1024;state_bytes=64*1024;physical_l2=1024*1024
    planning_ceiling=physical_l2-stack_bytes-state_bytes
    current_scratch_peak=491968
    sections={k:len(v) for k,v in decoded['sections'].items()}
    if args.output:args.output.write_bytes(packed)
    report={'source':str(args.image),'source_bytes':len(source),'v1_bytes':old_size,
      'prepared_v2_bytes':len(packed),'saved_vs_v1_bytes':old_size-len(packed),
      'section_bytes':sections,'l2_physical_bytes':physical_l2,
      'stack_reserve_bytes':stack_bytes,'graph_worker_state_reserve_bytes':state_bytes,
      'l2_planning_ceiling_bytes':planning_ceiling,
      'program_hot_bytes':hot_bytes,'compressed_block_stream_cold_bytes':sections['block_stream'],
      'hot_plus_current_scratch_bytes':hot_bytes+current_scratch_peak,
      'hot_plan_remaining_scratch_bytes':planning_ceiling-hot_bytes,
      'current_generic_scratch_peak_bytes':current_scratch_peak,
      'fits_hot_plus_current_scratch_planning_budget':hot_bytes+current_scratch_peak<=planning_ceiling,
      'full_prepared_program_plus_scratch_bytes':len(packed)+current_scratch_peak,
      'full_program_plus_current_scratch_fits_planning_budget':len(packed)+current_scratch_peak<=planning_ceiling,
      'l2_placement_proven':False,
      'prepared_sha256':hashlib.sha256(packed).hexdigest(),
      'source_qdrx_sha256':hashlib.sha256(source).hexdigest(),
      'context_sha256':decoded['context_sha256'],'constants_sha256':decoded['constants_sha256'],
      'pool_bytes':list(decoded['pool_bytes']),**check}
    rendered=json.dumps(report,indent=2)+'\n'
    if args.report:args.report.write_text(rendered)
    print(rendered,end='')


if __name__=='__main__':main()
