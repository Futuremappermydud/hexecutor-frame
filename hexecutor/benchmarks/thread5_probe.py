"""Focused HT5 callback/vector proof, with fifth-thread-aware retention."""

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

import json
import struct
import sys
from pathlib import Path
sys.path.insert(0,str((PATHS.native)))
from hexecutor.offline.frame_transport import FrameTransport
from hexecutor.offline.hvx_protocol import STATUS, STATUS_OFFSET

FRAME=(PATHS.native)
class ProbeTransport(FrameTransport):
    def workers_done(self):
        if not super().workers_done():return False
        out=struct.unpack_from('<6I',self.arena,2048)
        if out[2] and (out[5] or not out[3]&(1<<21)):
            self.retain_session('HT5 launch did not prove return to firmware Wait')
            raise RuntimeError('HT5 incomplete; callback/arena retained')
        return not out[2] or STATUS.unpack_from(self.arena,STATUS_OFFSET+5*STATUS.size)[0]==self.sequence
    def workers_exited(self):
        return self.workers_done() and super().workers_exited()

reports=[]
vector='--vector' in sys.argv
with ProbeTransport((PATHS.native / 'build/qd_loader.elf'),(PATHS.native / 'build/qd_thread5_probe.elf'),5,raise_clock=False) as t:
    for _ in range(3):
        def prepare(seq):
            t.arena[:8192]=bytes(8192)
            struct.pack_into('<I',t.arena,8,seq)
            struct.pack_into('<I',t.arena,16,int(vector))
            t.arena[4480:4608]=bytes((3*i+seq)&255 for i in range(128))
            t.arena[4608:4736]=bytes((5*i+7)&255 for i in range(128))
            t.fence()
        def collect():
            raw=struct.unpack_from('<44I',t.arena,2048)
            expected=bytes((8*i+t.sequence+7)&255 for i in range(128))
            return {'sequence':t.sequence,'magic':hex(raw[0]),'mode_before':hex(raw[1]),
                    'launched':raw[2],'mode_after':hex(raw[3]),'saved_opcode':raw[4],'error':raw[5],
                    'vector':vector,'vector_passed':bool(raw[42]) and bytes(t.arena[4736:4864])==expected,
                    'ssr_enabled':hex(raw[40]),'ssr_restored':hex(raw[41]),
                    'full_context_check':raw[43],
                    'threads':{str(i):{'tid':raw[16+4*i],'sp':hex(raw[17+4*i]),'ssr':hex(raw[18+4*i]),
                                      'state':raw[19+4*i]} for i in (0,1,2,5)}}
        r=t.invoke(prepare,t.workers_done,collect);reports.append(r)
        if r['magic']!='0x35544851' or r['error'] or not r['launched']:
            raise RuntimeError(f'probe failed: {r}')
        if vector and (not r['vector_passed'] or r['ssr_restored']!=r['threads']['5']['ssr']):
            raise RuntimeError(f'vector probe failed: {r}')
        if r['full_context_check']==2:raise RuntimeError(f'HVX context restoration failed: {r}')
print(json.dumps({'passed':True,'runs':reports},indent=2))
