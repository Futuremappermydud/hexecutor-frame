#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare the native graph engine with every recorded simulator stereo probe."""

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
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=(PATHS.repo)
sys.path[:0]=[str((PATHS.model_tools)),str((PATHS.native))]
import sim_trace
from depth_model.reference.native_reference import NativeEmulator
from check_kernels import load_inputs
from run_htp_emu import execute,output_names
from stereo_probes import make


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args();directory=(PATHS.native / 'build/native-probes');make(directory.resolve())
    refs=json.loads(((PATHS.model_data / 'reference/simulator_output_hashes.json')).read_text())['models']
    report={}
    for model in ('primary','alternate'):
        recovered=(PATHS.model_data / 'recovered')/model;graph=sim_trace.load_graph(recovered);nodes=sim_trace.execution_nodes(graph)
        emu=NativeEmulator(graph,(recovered/'constants/constant_pool_5.bin').read_bytes(),{})
        report[model]={}
        for case,expected in refs[model].items():
            begin=time.monotonic();out=execute(emu,nodes,load_inputs(directory/case))
            hashes={name+'.raw':hashlib.sha256(out[i].tobytes()).hexdigest() for i,name in enumerate(output_names(recovered))}
            assert hashes==expected,(model,case,hashes,expected)
            report[model][case]={'equal':True,'seconds':time.monotonic()-begin}
            print(model,case,'PASS',flush=True)
        rng=np.random.default_rng(1234)
        inputs={0:rng.integers(0,256,(320,256),dtype=np.uint8),1:rng.integers(0,256,(320,256),dtype=np.uint8),
                2:rng.uniform(-5,95,(1,2,320,256)).astype(np.float32),3:rng.normal(size=(1,2,4,4)).astype(np.float32)}
        out=execute(emu,nodes,inputs);expected=json.loads((recovered/'emulator_random.json').read_text())
        for i,name in enumerate(output_names(recovered)):
            assert hashlib.sha256(out[i].tobytes()).hexdigest()==expected[name]['sha256'],(model,'random',name)
        report[model]['random']={'equal':True}
        del emu;gc.collect()
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
