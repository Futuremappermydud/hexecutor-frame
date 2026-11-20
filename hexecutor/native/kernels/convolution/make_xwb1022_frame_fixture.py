#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pack the simulator-proven QDRX 1022 operands and expected outputs."""

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

import hashlib, json, struct
from pathlib import Path

HERE = (PATHS.native / 'kernels/convolution')
FILES = ["xwb1022_input_pages.bin", "xwb1022_weights.bin", "xwb1022_fi.bin",
         "xwb1022_hmx_output.bin", "xwb1022_cpu_output.bin"]
EXPECTED = ["247be4851d5804d80e254055530dac59f333574057c07ca30459dae00d0016b8",
            "5119e1c781aad96bfdaad620cc10813d10a9c15873ab37e8a2159ceaa46cf102",
            "8ff7ffedc5ab78ce2faee5d91bcd17cc79928fc34700a979165a6620ca00739d",
            "865cee388e871029fca5f6310d92f977b966e72db3a320195f000790e86b11fa",
            "7c4b4e4a705f3d771564612f0be77ae3cb3f21ca8906014f7e80166a531fd1b5"]
MAGIC = 0x334b4851

def main():
    blobs = [ (HERE / name).read_bytes() for name in FILES ]
    for name, data, expected in zip(FILES, blobs, EXPECTED):
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected: raise SystemExit(f"{name} SHA mismatch: {actual}")
    if list(map(len, blobs)) != [208896, 9216, 256, 65536, 65536]:
        raise SystemExit("unexpected operand/output lengths")
    # Fixed LE header: magic, schema, sequence, command index, input pages,
    # output pages, input/W/FI/HMX/CPU byte lengths, then three SHA256 digests.
    header = struct.pack('<11I', MAGIC, 1, 1022, 836, 102, 32,
                         *(len(x) for x in blobs)) + b''.join(bytes.fromhex(x) for x in EXPECTED)
    dest = (PATHS.native / 'kernels/convolution/xwb1022_frame_fixture.bin')
    dest.write_bytes(header + b''.join(blobs))
    ((PATHS.native / 'kernels/convolution/xwb1022_frame_fixture.json')).write_text(json.dumps({
        'fixture': dest.name, 'bytes': dest.stat().st_size,
        'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
        'sources': dict(zip(FILES, EXPECTED)),
        'note': 'Simulator-proven CI32/CO32 QDRX 1022 generic 3x3 fixture; not a device result.'
    }, indent=2) + '\n')
    print(f"{dest.name}: {dest.stat().st_size} bytes sha256={hashlib.sha256(dest.read_bytes()).hexdigest()}")

if __name__ == '__main__': main()
