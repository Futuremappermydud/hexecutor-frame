# SPDX-License-Identifier: LGPL-3.0-or-later
"""Calibrate direct u8 HMX conversion against the recovered two-stage epilogue."""

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

from pathlib import Path
import struct
import numpy as np
ROOT=(PATHS.repo);rng=np.random.default_rng(750)
rows=[];expected=[];meta=[]
a=rng.integers(0,256,(64,32),dtype=np.uint8);w=rng.integers(-8,9,(32,32),dtype=np.int8)
acc=a.astype(np.int64)@w.astype(np.int64)
packed=w.reshape(8,4,32).transpose(0,2,1).copy().tobytes()
for pre in range(16):
 for M in (1024,1025,1536,2047):
  target=rng.integers(1,255,32,dtype=np.int64)
  bias=target*16384*(1<<pre)//M-acc[0]+rng.integers(0,1<<pre,32,dtype=np.int64)
  fi=np.concatenate((np.full(32,0x400000|((20-pre)<<10)|(M-1024),np.int32),bias.astype(np.int32)))
  want=np.clip((((acc+bias)>>pre)*M+8192)>>14,0,255).astype(np.uint8)
  for convert in (0,1,2,3,512,513,1024,1025):
   rows.append(struct.pack('<2I',convert,1)+a.reshape(16,4,32).transpose(0,2,1).copy().tobytes()+packed+fi.tobytes());expected.append(want);meta.append((pre,M,convert))
((PATHS.native / 'build/native-hmx-input.bin')).write_bytes(struct.pack('<I',len(rows))+b''.join(rows))
np.savez((PATHS.native / 'build/native-hmx-expected.npz'),output=np.stack(expected),meta=np.asarray(meta))
print(f'{len(rows)} cases; {len(rows)*2048} output bytes')
