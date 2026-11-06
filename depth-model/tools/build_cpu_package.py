#!/usr/bin/env python3
"""Build a CPU-only distribution without importing any device transport."""

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

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = (PATHS.model_data)


def main():
    files = {}
    for name in ('htp_emu.py', 'sim_trace.py', 'check_kernels.py', 'run_htp_emu.py',
                 'model_depth.py', 'prepare_inputs.py', 'stereo_probes.py',
                 'verify_htp_emu.py', 'test_cpu_adapter.py'):
        files['tools/' + name] = ((PATHS.model_tools) / name).read_bytes()
    for model in ('primary', 'alternate'):
        for name in ('graph.json.gz', 'execution_order.json', 'kernel_calls.json',
                     'constants/constant_pool_5.bin', 'emulator_smoke.json', 'emulator_random.json'):
            path = f'recovered/{model}/{name}'
            files[path] = (ROOT / path).read_bytes()
        path = f'metadata/{model}.qnn.json'
        files[path] = (ROOT / path).read_bytes()
    files['reference/simulator_output_hashes.json'] = ((PATHS.model_data / 'reference/simulator_output_hashes.json')).read_bytes()
    files['README.md'] = ((PATHS.model_data / 'portable/README.md')).read_bytes()
    files['requirements.txt'] = ((PATHS.model_data / 'portable/requirements.txt')).read_bytes()
    validation = (PATHS.model_data / 'portable/validation.json')
    if validation.exists():
        files['validation.json'] = validation.read_bytes()
    files['run.py'] = b'import runpy\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))\nrunpy.run_module("model_depth", run_name="__main__")\n'
    manifest = {'format': 'depth-model-cpu-v1', 'backend': 'numpy-cpu',
                'files': {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}}
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    out = (PATHS.model_data / 'portable/dist/depth-model-cpu.zip')
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo('depth-model-cpu/' + name, date_time=(2026, 10, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    print(f'{out}: {out.stat().st_size} bytes; sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
