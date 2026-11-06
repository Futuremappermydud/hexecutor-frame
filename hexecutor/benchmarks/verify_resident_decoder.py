#!/usr/bin/env python3
"""Hardware structural-decoder probe; does NOT execute model inference."""

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
import fcntl
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib

FRAME = (PATHS.native)
sys.path.insert(0, str(FRAME))
from hexecutor.offline.frame_transport import FrameTransport
from hexecutor.offline.hvx_protocol import STATUS_OFFSET

LOADER_SHA256 = '828ca0638750a6d8b69b3aaa536cb6c23508ce821b2333e270b455c321414572'
IMAGE_OFFSET = 0x10000


def change(raw, offset, value):
    data = bytearray(raw)
    struct.pack_into('<I', data, offset, value)
    struct.pack_into('<I', data, 16, 0)
    struct.pack_into('<I', data, 16, zlib.crc32(data))
    return bytes(data)


def cases(raw):
    h = struct.unpack_from('<64I', raw)
    corrupt = bytearray(raw);corrupt[-1] ^= 1
    return [('valid', raw, 0, 65536, 0), ('missing-kernel', raw, 1, 65536, 10),
            ('budget', raw, 1, 0, 12), ('crc', bytes(corrupt), 0, 0, 2),
            ('ref', change(raw, h[43], 0xffffffff), 0, 0, 7),
            ('block', change(raw, h[39], 0x800001), 0, 0, 5),
            ('truncated', raw[:-1], 0, 0, 1)]


def snapshot():
    # State snapshots use the same transport lock, independently of sessions.
    with open('/tmp/depth-model-cdsp.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proc = subprocess.run([sys.executable, str((PATHS.benchmarks / 'depth_state.py'))],
                              check=True, capture_output=True, text=True)
        state = json.loads(proc.stdout)
    if state['control'] != [1, 0, 1] or state['callback'] != '0x0' or any(
            not r['zero'] for r in state['regions'].values()) or any(
            p['state'] in ('T', 'Z') for p in state['eye_clients']):
        raise RuntimeError(f'DSP is not idle/restored: {state}')
    return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', type=Path, action='append', required=True)
    ap.add_argument('--sessions', type=int, default=2)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.sessions <= 10: raise ValueError('sessions must be 1..10')
    build = (PATHS.native / 'build')
    loader = build/'qd_loader.elf'
    if hashlib.sha256(loader.read_bytes()).hexdigest() != LOADER_SHA256: raise ValueError('wrong loader ELF')
    images = []
    for path in args.image:
        raw = path.read_bytes()
        manifest = json.loads(path.with_suffix('.json').read_text())
        if hashlib.sha256(raw).hexdigest() != manifest['sha256']: raise ValueError(f'image digest mismatch: {path}')
        if manifest['format'] != 'QDRX': raise ValueError('QDRX required')
        images.append((path.name, raw))
    report = {'status': 'decoder-only; no graph kernels or physical pool mapping', 'sessions': [],
              'loader_sha256': LOADER_SHA256,
              'payload_sha256': hashlib.sha256((build/'qd_resident_probe.elf').read_bytes()).hexdigest(),
              'image_sha256': {n: hashlib.sha256(r).hexdigest() for n, r in images}, 'before': snapshot()}
    try:
        for session in range(args.sessions):
            rows = []
            with FrameTransport(loader, build/'qd_resident_probe.elf', 30, raise_clock=False) as t:
                for name, raw in images:
                    for case, data, mode, budget, expected in cases(raw):
                        def prepare(seq):
                            t.arena[STATUS_OFFSET:STATUS_OFFSET+48] = bytes(48)
                            t.arena[2048:2240] = bytes(192)
                            t.arena[IMAGE_OFFSET:IMAGE_OFFSET+len(data)] = data
                            struct.pack_into('<8I', t.arena, 0, 0x31505251, 1, seq,
                                             IMAGE_OFFSET, len(data), mode, budget, 0)
                            t.fence()
                        start = time.monotonic()
                        out = t.invoke(prepare, t.workers_done,
                                       lambda: [struct.unpack_from('<16I', t.arena, 2048+64*i) for i in range(3)])
                        if any(w[0] != 0x31505251 or w[10] != i for i, w in enumerate(out)):
                            raise RuntimeError('worker diagnostic mismatch')
                        w = out[0]
                        row = {'image': name, 'case': case, 'code': w[1], 'command': w[2],
                               'tensor': w[3], 'detail': w[4], 'cycles': w[5] | w[6]<<32,
                               'tensors': w[7], 'commands': w[8], 'bytes': w[9],
                               'wall_seconds': time.monotonic()-start}
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        if w[1] != expected: raise RuntimeError(f'expected {expected}: {row}')
                        if case == 'valid':
                            h = struct.unpack_from('<64I', raw)
                            if (w[7], w[8]) != (h[38], h[42]): raise RuntimeError('decoded count mismatch')
            report['sessions'].append({'index': session, 'cases': rows, 'after': snapshot()})
        report['passed'] = True
    except BaseException as error:
        report['passed'] = False
        report['error'] = str(error)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
