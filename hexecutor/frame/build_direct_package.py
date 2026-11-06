#!/usr/bin/env python3
"""Package the validated native ARM/HMX depth path, models and offline image tools."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    files={}
    for name in ('htp_emu.py','sim_trace.py','check_kernels.py','run_htp_emu.py','model_depth.py','prepare_inputs.py','stereo_probes.py','verify_htp_emu.py'):
        files['tools/'+name]=(ROOT/'tools'/name).read_bytes()
    for model in ('primary','alternate'):
        for name in ('graph.json.gz','execution_order.json','kernel_calls.json','constants/constant_pool_5.bin','emulator_smoke.json','emulator_random.json'):
            path=f'recovered/{model}/{name}';files[path]=(ROOT/path).read_bytes()
        path=f'metadata/{model}.qnn.json';files[path]=(ROOT/path).read_bytes()
    for name in ('run_frame.py','run_stereo.py','frame_hvx.py','frame_transport.py','conv_protocol.py','hvx_protocol.py','local_transport.py',
                 'fast_depth.py','qd_memory.c','qd_hmx_layout.h','qd_hmx_kernel.h','qd_hmx_power.h','qd_hmx_ops.S','qd_hmx_ops_encoded.S',
                 'qd_conv.c','qd_conv.h','qd_axim.h','qd_fence.c','qd_protocol.h','qd_loader.S','qd_link.ld','qd_payload.ld','Makefile',
                 'test_conv_local.py','tests/test_conv_publication.py','tests/test_quantized_conv.py','tests/test_native_packets.py','tests/test_native_memory.py','tests/test_native_elementwise.py','bench/hmx_conv_sim.c','bench/make_hmx_conv_cases.py','bench/check_hmx_conv_sim.py','bench/verify_optimized.py','bench/probe_depth.py','bench/depth_state.py',
                 'build/qd_loader.elf','build/qd_conv.elf'):
        files['frame/'+name]=(ROOT/'frame'/name).read_bytes()
    for path in (ROOT/'frame/build').glob('depth-*.json'):
        files['validation/'+path.name]=path.read_bytes()
    for name in ('depth-primary-smoke','depth-alternate-smoke','depth-primary-random','depth-alternate-random','depth-stereo-pair'):
        path=ROOT/'frame/build'/name/'report.json'
        if path.exists():files[f'validation/{name}.json']=path.read_bytes()
        path=ROOT/'frame/build'/name/'hardware-report.json'
        if path.exists():files[f'validation/{name}-hardware.json']=path.read_bytes()
    for name in ('left.npy','right.npy'):
        files['examples/'+name]=(ROOT/'frame/build/depth-inputs'/name).read_bytes()
    files['README.md']=(ROOT/'frame/DIRECT.md').read_bytes()
    files['requirements.txt']=b'numpy>=1.24\nPillow>=10\n'
    files['run.py']=b'import runpy\nfrom pathlib import Path\nrunpy.run_path(str(Path(__file__).resolve().parent / "frame/run_stereo.py"), run_name="__main__")\n'
    files['reference/simulator_output_hashes.json']=(ROOT/'reference/simulator_output_hashes.json').read_bytes()
    manifest={'format':'depth-model-direct-v5','backend':'Native ARM graph + integer HMX/HVX convolution; 750 MHz AXI master',
              'files':{k:hashlib.sha256(v).hexdigest() for k,v in sorted(files.items())}}
    files['depth-package.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    output=ROOT/'frame/dist/depth-model-direct.zip'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo('depth-model-direct/'+name,date_time=(2026,10,29,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            archive.writestr(info,data)
    print(f'{output}: {output.stat().st_size} bytes, sha256 {hashlib.sha256(output.read_bytes()).hexdigest()}')


if __name__=='__main__':main()
