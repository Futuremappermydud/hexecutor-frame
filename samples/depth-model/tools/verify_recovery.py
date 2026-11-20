#!/usr/bin/env python3
"""Check graph recovery and compare weight addressing to original SDK ARM64 code.

The optional machine-code check requires Capstone, already installed on the
analysis Mac. Model extraction and constant export themselves use stdlib only.
"""
import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import struct
import zipfile

from export_constants import weight_offset

PREPARE_SHA256 = '7c9a8a59ce090bb0766f5ff660809ef58e0a5e65d1c6ba46c7fa182f8d032682'


def sdk_address_functions(path):
    from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM
    from capstone.arm64 import ARM64_OP_IMM, ARM64_OP_REG
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != PREPARE_SHA256:
        raise ValueError('address oracle requires the inspected QAIRT 2.50 ARM64 library')
    cs = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
    cs.detail = True

    def make(start, end):
        # For these text symbols in this ELF, file offset == virtual address.
        instructions = list(cs.disasm(blob[start:end], start))
        if not instructions or instructions[-1].mnemonic != 'ret':
            raise ValueError('unexpected address routine')

        def execute(coords, dimensions):
            regs = {'x0': 0x10000, 'x1': 0x20000}
            mem = {0x10000 + 8 * i: c for i, c in enumerate(coords)}
            mem.update({0x20000 + 8 * i: d for i, d in enumerate(dimensions)})
            def value(op):
                if op.type == ARM64_OP_IMM:
                    return op.imm
                if op.type != ARM64_OP_REG:
                    raise ValueError('unexpected operand')
                return regs.get(cs.reg_name(op.reg), 0) << op.shift.value
            for ins in instructions:
                ops, kind = ins.operands, ins.mnemonic
                if kind in ['bti', 'nop']:
                    continue
                if kind == 'ret':
                    return regs['x0']
                dest = cs.reg_name(ops[0].reg)
                if kind in ['ldp', 'ldr']:
                    addr = regs[cs.reg_name(ops[-1].mem.base)] + ops[-1].mem.disp
                    regs[dest] = mem[addr]
                    if kind == 'ldp':
                        regs[cs.reg_name(ops[1].reg)] = mem[addr + 8]
                elif kind == 'lsr':
                    regs[dest] = value(ops[1]) >> value(ops[2])
                elif kind == 'lsl':
                    regs[dest] = value(ops[1]) << value(ops[2])
                elif kind == 'madd':
                    regs[dest] = value(ops[1]) * value(ops[2]) + value(ops[3])
                elif kind == 'add':
                    regs[dest] = value(ops[1]) + value(ops[2])
                elif kind == 'and':
                    regs[dest] = value(ops[1]) & value(ops[2])
                elif kind == 'ubfx':
                    regs[dest] = (value(ops[1]) >> value(ops[2])) & ((1 << value(ops[3])) - 1)
                elif kind == 'ubfiz':
                    regs[dest] = (value(ops[1]) & ((1 << value(ops[3])) - 1)) << value(ops[2])
                else:
                    raise ValueError(f'unsupported address-oracle instruction {kind}')
            raise ValueError('address routine did not return')
        return execute
    return {False: make(0x136d180, 0x136d1cc), True: make(0x136d4b4, 0x136d4fc)}


def verify(model, qnn_file, prepare):
    graph = json.loads((model / 'graph.json').read_text())
    constants = json.loads((model / 'constants/constants.json').read_text())
    qnn = json.loads(qnn_file.read_text())['info']['graphs'][0]['info']
    tensors = {t['id']: t for t in graph['tensors']}
    nodes = {n['sequence']: n for n in graph['nodes']}
    for ids, field in [(graph['graph_inputs'], 'graphInputs'), (graph['graph_outputs'], 'graphOutputs')]:
        for i, sdk in zip(ids, qnn[field]):
            tensor, sdk = tensors[i], sdk['info']
            assert tensor['id'] == sdk['id']
            assert tensor['shape']['dimensions'] == sdk['dimensions']
            assert tensor['dtype'] == ('uint8' if sdk['dataType'] == 'QNN_DATATYPE_UFIXED_POINT_8' else 'float32')
            if 'quantization' in tensor:
                assert tensor['quantization']['scale'] == sdk['quantizeParams']['scaleOffset']['scale']
                assert tensor['quantization']['zero_point'] == -sdk['quantizeParams']['scaleOffset']['offset']
    for n in nodes.values():
        for i in n['inputs']:
            t = tensors[i]
            assert not isinstance(t['producer'], int) or t['producer'] < n['sequence']
        for i in n['outputs']:
            assert tensors[i]['producer'] == n['sequence']
    oracle = sdk_address_functions(prepare) if prepare else None
    rng, cases = random.Random(219), 0
    for entry in constants['constants']:
        data = (model / 'constants' / entry['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
        assert data[:8] == b'\x93NUMPY\x01\x00'
        length = struct.unpack_from('<H', data, 8)[0]
        header = ast.literal_eval(data[10:10 + length].decode().strip())
        assert list(header['shape']) == entry['shape']
        assert header['descr'] == entry['numpy_dtype']
        assert len(data) - 10 - length == entry['data_bytes']
        if entry['tensor_id'] not in constants['convolution_weight_tensor_ids']:
            continue
        t = tensors[entry['tensor_id']]
        shape = t['shape']['max_dimensions']
        packed = entry['original_class'] == 'xpb'
        if oracle:
            coords = list(itertools.product(*[[0, d - 1] for d in shape]))
            coords += [tuple(rng.randrange(d) for d in shape) for _ in range(64)]
            for c in coords:
                sdk_offset = oracle[packed](c, shape)
                recovered_offset = weight_offset(c, shape, packed)
                assert sdk_offset == recovered_offset, (entry['tensor_id'], c, sdk_offset, recovered_offset)
                assert 0 <= recovered_offset < math.prod(shape)
                cases += 1
    with zipfile.ZipFile(model / 'constants/convolution_weights.npz') as archive:
        weights = [e for e in constants['constants'] if e['tensor_id'] in constants['convolution_weight_tensor_ids']]
        assert len(archive.namelist()) == len(weights)
        for entry in weights:
            assert archive.read(entry['source_name'] + '.npy') == (model / 'constants' / entry['file']).read_bytes()
    result = dict(context_sha256=graph['context_sha256'],
                  operations=len(nodes), tensors=len(tensors),
                  constants=len(constants['constants']),
                  stored_convolution_weights=len(constants['convolution_weight_tensor_ids']),
                  sdk_io_metadata_matches=True, all_tensor_references_causal=True,
                  all_numpy_headers_sizes_hashes_match=True,
                  named_weight_archive_matches_arrays=True,
                  sdk_arm64_weight_address_cases=cases,
                  sdk_prepare_sha256=PREPARE_SHA256 if oracle else None,
                  inference_verified=False)
    if prepare:
        runtime = prepare.parents[2] / 'lib/hexagon-v75/unsigned/libQnnHtpV75.so'
        if runtime.exists():
            blob = runtime.read_bytes()
            missing = [t['name'] for t in graph['op_types'] if t['name'].encode() + b'\0' not in blob]
            result['v75_kernel_name_matches'] = len(graph['op_types']) - len(missing)
            result['v75_missing_kernel_names'] = missing
            result['v75_library_sha256'] = hashlib.sha256(blob).hexdigest()
    (model / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('qnn_metadata', type=Path)
    parser.add_argument('--sdk-prepare', type=Path)
    args = parser.parse_args()
    verify(args.model, args.qnn_metadata, args.sdk_prepare)
