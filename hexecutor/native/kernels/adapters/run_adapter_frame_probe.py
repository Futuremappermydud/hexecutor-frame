#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run the adapter + 1x1 leaf Frame payload; never used by offline checks."""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
FRAME = (PATHS.native)
sys.path.insert(0, str(FRAME))
from hexecutor.offline.frame_transport import FrameTransport
from bench.verify_resident_decoder import snapshot

HERE = (PATHS.native / 'kernels/adapters')
MAGIC = 0x31514151
FIXTURE_MAGIC = 0x31585151
ARENA_BYTES = 0x800000
FIXTURE_OFFSET = 0x10000
OUTPUT_OFFSET = 0x90000
ECHO_OFFSET = 0xA0000
EXPECTED_TLB = [0xF7062020, 0xC0031000, 0xF7062820, 0xC0031400]

def dense(blocks, h, w, channels):
    return (np.frombuffer(blocks, np.uint8)
            .reshape(h // 8, w // 8, channels // 32, 8, 8, 32)
            .transpose(0, 3, 1, 4, 2, 5).reshape(h, w, channels).astype(np.int64))

def check_oracle(fixture, output):
    magic, seq, h, w, ci, co, block_count, weight_bytes, fi_bytes = struct.unpack_from('<9I', fixture)
    pos = 36
    input_blocks = fixture[pos:pos + block_count * 2048]
    pos += block_count * 2048
    packed_weights = fixture[pos:pos + weight_bytes]
    pos += weight_bytes
    fi = np.frombuffer(fixture[pos:pos + fi_bytes], '<i4').reshape(co // 32, 64).astype(np.int64)
    weights = (np.frombuffer(packed_weights, np.int8)
               .reshape(co // 32, ci // 32, 8, 32, 4)
               .transpose(1, 2, 4, 0, 3).reshape(ci, co).astype(np.int64))
    acc = dense(input_blocks, h, w, ci) @ weights + fi[:, 32:].reshape(co)
    mult = fi[:, :32].reshape(co) & 0xFFFF
    preshift = 34 - ((mult >> 10) & 31) - 14
    expected = np.clip((((acc >> preshift) * (1024 + (mult & 1023)) + 8192) >> 14), 0, 255)
    actual = dense(output, h, w, co)
    delta = actual - expected
    return int(delta.min()), int(delta.max()), int(np.count_nonzero(delta))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture', type=Path, default=(PATHS.native / 'kernels/adapters/fixture-qdrx-1195.bin'))
    ap.add_argument('--loader', type=Path, default=(PATHS.native / 'build/qd_loader.elf'))
    ap.add_argument('--payload', type=Path, default=(PATHS.native / 'kernels/adapters/adapter_frame_probe.elf'))
    ap.add_argument('--iterations', type=int, default=1)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    fixture = args.fixture.read_bytes()
    magic, seq, h, w, ci, co, blocks, weights, fi = struct.unpack_from('<9I', fixture)
    if magic != FIXTURE_MAGIC or seq != 1195 or (h, w, ci, co) != (8, 64, 128, 128):
        raise ValueError('Frame probe currently requires the reviewed command-1195 fixture')
    if len(fixture) != 36 + blocks * 2048 + weights + fi or blocks != 32 or weights != ci * co or fi != co * 8:
        raise ValueError('inconsistent fixture lengths')
    if not 1 <= args.iterations <= 100:
        raise ValueError('iterations must be in 1..100')
    if FIXTURE_OFFSET + len(fixture) >= 0x30000 or OUTPUT_OFFSET + blocks * 2048 > ECHO_OFFSET:
        raise ValueError('fixture/output ranges overlap probe reserved regions')
    report = {
        'offline_only': False,
        'fixture_sha256': hashlib.sha256(fixture).hexdigest(),
        'before': snapshot(),
        'vtcm_offset': 0x700000,
        'iterations': args.iterations,
        'run': None,
    }
    with FrameTransport(args.loader, args.payload, 10) as transport:
        report['payload_sha256'] = transport.payload_sha256

        def prepare(sequence):
            transport.arena[:4096] = bytes(4096)
            packet = struct.pack('<10I', MAGIC, 1, sequence, h, w, ci, co,
                                 args.iterations, 0, 0x700000)
            transport.arena[:len(packet)] = packet
            transport.arena[FIXTURE_OFFSET:FIXTURE_OFFSET + len(fixture)] = fixture
            transport.fence()

        def collect():
            words = struct.unpack_from('<32I', transport.arena, 2048)
            before_tlb = list(words[19:23])
            restore_tlb = list(words[23:27])
            if words[10] or words[11] or (words[18] and (not words[17] or before_tlb != EXPECTED_TLB or restore_tlb != EXPECTED_TLB)):
                transport.retain_session(
                    'adapter probe could not verify VTCM/power restore; retain backup and callback for recovery')
            output = bytes(transport.arena[OUTPUT_OFFSET:OUTPUT_OFFSET + blocks * 2048])
            echo = bytes(transport.arena[ECHO_OFFSET:ECHO_OFFSET + blocks * 2048])
            return words, output, echo

        started = time.monotonic()
        words, output, echo = transport.invoke(
            prepare, lambda: transport.workers_done() and transport.workers_exited(), collect)
        lo, hi, different = check_oracle(fixture, output)
        expected_tlb = EXPECTED_TLB
        before_tlb = list(words[19:23])
        restore_tlb = list(words[23:27])
        row = {
            'tlb_before_backup': before_tlb,
            'tlb_before_restore': restore_tlb,
            'expected_tlb': expected_tlb,
            'seconds': time.monotonic() - started,
            'words': list(words),
            'native_minus_integer_bound': [lo, hi],
            'different_outputs': different,
            'input_unchanged': echo == fixture[36:36 + blocks * 2048],
            'vtcm_restored': bool(words[17]),
            'output_sha256': hashlib.sha256(output).hexdigest(),
        }
        report['run'] = row
        if words[10] or words[11] or (words[18] and (not words[17] or before_tlb != expected_tlb or restore_tlb != expected_tlb)):
            raise RuntimeError('restore/power verification failed; FrameTransport retained session and backup')
        if words[0] != MAGIC or words[1] or before_tlb != expected_tlb or restore_tlb != expected_tlb or not row['input_unchanged'] or not words[17] or lo < 0 or hi > 1:
            raise RuntimeError(f'adapter Frame probe failed: {row}')
    report['after'] = snapshot()
    report['passed'] = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['run'], indent=2))

if __name__ == '__main__':
    main()
