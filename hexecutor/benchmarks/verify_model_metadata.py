#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run preserved Model DSP programs through the signed vendor runtime over ADB.

Only stages files under /data/local/tmp/qd-qnn-20261030. Does not change vendor
files, properties, services, firmware, or the Frame. A fresh private DSP user PD
is opened per case. Timing excludes creation and input/output file operations,
and includes the libhexagon RPC call. Profiling changes execution overhead.
The host native reference is checked against recorded simulator hashes first.
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
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

HERE = (PATHS.benchmarks)
ROOT = (PATHS.repo)
sys.path[:0] = [str(HERE), str(ROOT / 'frame')]
from verify_native_graph import NativeGraph, cases, ensure_image, load_inputs
from model_metadata_profile import decode, summary

VENDOR_SHA = {
    '/vendor/lib64/libhexagon.so': '9750f275055a5c3f187ef28882c4d5a5f92f6182fcbab8c46a52d4bf3ed33a37',
    '/vendor/lib/rfsa/adsp/libhexagon_skel.so': '2594841ca8fd412cb7ca9780f190b69bde8a18b787fc3cb5e95770a47078e76b',
    '/vendor/lib/rfsa/adsp/libQnnHtpV69.so': '9b687661f8a2b5c1511a21b324803aa96692a2c4f4a0f73ad18151562eb94d22',
}
DEVICE_DIR = '/data/local/tmp/qd-qnn-20261030'
RUN = re.compile(r'run index=(\d+) rc=0x([0-9a-f]+) status=(-?\d+) ms=([\d.]+).*fnv64=([0-9a-f]+),([0-9a-f]+)')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(got, expected):
    delta = got.astype(np.float64) - expected.astype(np.float64)
    absdelta = abs(delta)
    return {'equal': got.tobytes() == expected.tobytes(),
            'values': got.size, 'different': int(np.count_nonzero(got != expected)),
            'finite': bool(np.isfinite(got).all()), 'min_delta': float(delta.min()),
            'max_delta': float(delta.max()), 'mae': float(absdelta.mean()),
            'p95': float(np.quantile(absdelta, .95)), 'max_abs': float(absdelta.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--serial', default='2G0YC1ZF860CBG')
    ap.add_argument('--model', action='append', choices=('primary', 'alternate'))
    ap.add_argument('--case', action='append')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--profile', action='store_true', help='vendor mode 1: per-op cycles/wait/overlap')
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.repeat <= 1000:
        ap.error('--repeat must be 1..1000')
    report_path = args.report.resolve()
    evidence = report_path.parent / report_path.stem
    evidence.mkdir(parents=True, exist_ok=True)
    adb = ['adb', '-s', args.serial]

    def command(cmd):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError(f'{shlex.join(cmd)}\n{result.stdout}\n{result.stderr}')
        return result.stdout

    def shell(cmd):
        return command([*adb, 'shell', cmd])

    # Refuse to use an inferred ABI on another firmware build.
    checks = shell('sha256sum ' + ' '.join(shlex.quote(p) for p in VENDOR_SHA))
    observed = {line.split()[1]: line.split()[0] for line in checks.splitlines()}
    if observed != VENDOR_SHA:
        raise RuntimeError(f'vendor ABI fingerprint changed: {observed}')
    subprocess.run(['sh', str(ROOT / 'frame/build_model_metadata_probe.sh')], check=True)
    probe = ROOT / 'frame/build/model_metadata_qnn_probe'
    shell(f'mkdir -p {DEVICE_DIR}/meta-inputs {DEVICE_DIR}/meta-outputs')
    command([*adb, 'push', str(probe), f'{DEVICE_DIR}/model_metadata_qnn_probe'])
    report = {'device_serial': args.serial, 'vendor_sha256': observed, 'probe_sha256': sha(probe),
              'source_sha256': sha((PATHS.benchmarks / 'model_metadata_qnn_probe.c')),
              'profiling_mode': 1 if args.profile else 0,
              'timing_scope': 'libhexagon RPC call, excluding model creation and file I/O',
              'models': {}}

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    save()
    for model in args.model or ('primary', 'alternate'):
        pte = ROOT / 'models' / model / 'dsp.pte'
        remote_pte = f'{DEVICE_DIR}/meta-{model}.pte'
        command([*adb, 'push', str(pte), remote_pte])
        names, todo = cases(model, args.case)
        if not todo:
            raise ValueError(f'no cases selected for {model}')
        rows = []
        report['models'][model] = {'pte_sha256': sha(pte), 'runs': rows}
        graph = NativeGraph(ensure_image(model), ROOT / 'recovered' / model / 'constants/constant_pool_5.bin')
        try:
            for name, directory, expected_hashes in todo:
                local_out = evidence / model / name
                local_out.mkdir(parents=True, exist_ok=True)
                # Generate numerical reference once and enforce its established hashes.
                wanted = graph.run(load_inputs(directory))
                hashes = {n + '.raw': hashlib.sha256(a.tobytes()).hexdigest() for n, a in zip(names, wanted)}
                if hashes != expected_hashes:
                    raise RuntimeError(f'{model}/{name}: host reference differs from simulator hashes')
                for n, a in zip(('u8', 'f32'), wanted):
                    (local_out / f'cpu-{n}.raw').write_bytes(a.tobytes())
                remote_inputs = f'{DEVICE_DIR}/meta-inputs/{model}-{name}'
                remote_out = f'{DEVICE_DIR}/meta-outputs/{model}-{name}'
                shell(f'mkdir -p {remote_inputs} {remote_out}')
                command([*adb, 'push', *[str(directory / (n + '.raw')) for n in ('x', 'y', 'hidden', 't')], remote_inputs + '/'])
                run_command = (
                    f'cd {DEVICE_DIR} && LD_LIBRARY_PATH=/vendor/lib64 '
                    'DSP_LIBRARY_PATH="/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp" '
                    'ADSP_LIBRARY_PATH="/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp" '
                    f'QD_QNN_PROFILE={int(args.profile)} timeout 60 ./model_metadata_qnn_probe '
                    f'{remote_pte} {remote_inputs} {remote_out} {args.repeat}')
                (local_out / 'command.txt').write_text(run_command + '\n')
                # Save failures as well as successful calls.
                result = subprocess.run([*adb, 'shell', run_command], capture_output=True, text=True, timeout=75)
                log = result.stdout + result.stderr
                (local_out / 'run.log').write_text(log)
                runs = [{'index': int(i), 'rc': int(rc, 16), 'status': int(s), 'ms': float(ms),
                         'fnv64': [u, f]} for i, rc, s, ms, u, f in RUN.findall(log)]
                if result.returncode or len(runs) != args.repeat or any(r['rc'] or r['status'] for r in runs):
                    raise RuntimeError(f'{model}/{name}: RPC run failed; see {local_out / "run.log"}')
                command([*adb, 'pull', remote_out + '/.', str(local_out)])
                outputs = [np.fromfile(local_out / (n + '.raw'), dtype=d) for n, d in (('u8', np.uint8), ('f32', np.float32))]
                for a, b in zip(outputs, wanted):
                    if a.size != b.size:
                        raise RuntimeError(f'{model}/{name}: output size mismatch')
                row = {'case': name, 'calls': runs,
                       'deterministic': len({tuple(r['fnv64']) for r in runs}) == 1,
                       'input_sha256': {n: sha(directory / (n + '.raw')) for n in ('x', 'y', 'hidden', 't')},
                       'output_sha256': {n: sha(local_out / (n + '.raw')) for n in ('u8', 'f32')},
                       'simulator_sha256': hashes,
                       'u8': differences(outputs[0], wanted[0].reshape(-1)),
                       'f32': differences(outputs[1], wanted[1].reshape(-1))}
                if args.profile:
                    events = decode((local_out / 'profile.bin').read_bytes())
                    prof = {'summary': summary(events), 'events': events}
                    (local_out / 'profile.json').write_text(json.dumps(prof, indent=2) + '\n')
                    row['profile'] = prof['summary']
                rows.append(row)
                warm = [r['ms'] for r in runs[1:] or runs]
                print(f'{model} {name}: {statistics.median(warm):.3f} ms, '
                      f'u8 different={row["u8"]["different"]}, f32 max={row["f32"]["max_abs"]:.6g}, '
                      f'deterministic={row["deterministic"]}', flush=True)
                save()
        finally:
            graph.close()
        warm = [c['ms'] for r in rows for c in r['calls'][1:] or r['calls']]
        report['models'][model]['summary'] = {
            'cases_completed': len(rows), 'calls_completed': sum(len(r['calls']) for r in rows),
            'warm_median_ms': statistics.median(warm),
            'u8_equal_cases': sum(r['u8']['equal'] for r in rows),
            'f32_equal_cases': sum(r['f32']['equal'] for r in rows),
            'f32_worst_max_abs': max(r['f32']['max_abs'] for r in rows),
            'all_finite': all(r['f32']['finite'] for r in rows),
            'all_deterministic': all(r['deterministic'] for r in rows)}
        save()
    print(f'Completed. Numerical differences are recorded, not treated as exact passes. Report: {report_path}')


if __name__ == '__main__':
    main()
