#!/usr/bin/env python3
"""Run exported HTP simulator kernels directly via sim/build/kcall (Docker).

    from kcall import Job, run
    j = Job('requantize_crouton_u8')
    j.buf('x', data_bytes); j.buf('y', nbytes)
    j.ptrs('xp', 'x', 2048, n); j.ptrs('yp', 'y', 2048, n)
    j.args('@yp', '@xp', n, 0x1004114, 0, 0)
    j.save('y')
    out = run([j], sdk)[0]['y']

All jobs in one run() share a container start. Research tool.
"""

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

from pathlib import Path
import shutil
import subprocess
import tempfile

from run_qnn_sim import IMAGE

ROOT = (PATHS.model_data)


class Job:
    def __init__(self, func, rel=False):
        """func: exported name, or a load-base offset (int) with rel=True."""
        self.func = hex(func) if rel else func
        self.verb = 'callrel' if rel else 'call'
        self.lines, self.files, self.saves = [], {}, []
        self.int_args, self.stack_args, self.xmm_args = [], [], []

    def buf(self, name, data):
        if isinstance(data, int):
            self.lines.append(f'buf {name} {data}')
        else:
            data = bytes(data)
            self.files[name] = data
            self.lines.append(f'buf {name} {len(data)} FILE:{name}')
        return self

    def ptrs(self, name, base, stride, count):
        self.lines.append(f'ptrs {name} {base} {stride} {count}')
        return self

    def args(self, *ints, stack=(), xmm=()):
        self.int_args, self.stack_args, self.xmm_args = ints, stack, xmm
        return self

    def save(self, name):
        self.saves.append(name)
        return self


def run(jobs, sdk, keep=None):
    work = Path(keep) if keep else Path(tempfile.mkdtemp(prefix='kcall-', dir=(PATHS.model_data / 'sim/build')))
    work.mkdir(parents=True, exist_ok=True)
    script = []
    for i, j in enumerate(jobs):
        d = work / f'job{i}'
        d.mkdir(exist_ok=True)
        lines = []
        for ln in j.lines:
            if 'FILE:' in ln:
                name = ln.split('FILE:')[1]
                (d / f'{name}.in').write_bytes(j.files[name])
                ln = ln.replace(f'FILE:{name}', f'/w/job{i}/{name}.in')
            lines.append(ln)
        fmt = lambda v: str(v) if isinstance(v, str) else hex(v & 0xffffffffffffffff)
        lines.append(j.verb + ' ' + j.func + ' ' + ' '.join(map(fmt, j.int_args)) + ' | '
                     + ' '.join(map(fmt, j.stack_args)) + ' | ' + ' '.join(map(fmt, j.xmm_args)))
        lines += [f'save {n} /w/job{i}/{n}.out' for n in j.saves]
        (d / 'spec.txt').write_text('\n'.join(lines) + '\n')
        script.append(f'/k/kcall /sdk/lib/x86_64-linux-clang/libQnnHtp.so /w/job{i}/spec.txt > /w/job{i}/log 2>&1')
    subprocess.run(['docker', 'run', '--rm', '--platform', 'linux/amd64',
                    '-e', 'LD_LIBRARY_PATH=/sdk/lib/x86_64-linux-clang:/usr/lib/llvm-14/lib',
                    '-v', f'{Path(sdk).resolve()}:/sdk:ro', '-v', f'{(PATHS.model_data / 'sim/build')}:/k:ro',
                    '-v', f'{work}:/w', IMAGE, 'bash', '-c', '; '.join(script)], check=True)
    results = []
    for i, j in enumerate(jobs):
        d = work / f'job{i}'
        r = {n: (d / f'{n}.out').read_bytes() for n in j.saves if (d / f'{n}.out').exists()}
        r['log'] = (d / 'log').read_text()
        results.append(r)
    if not keep:
        shutil.rmtree(work)
    return results
