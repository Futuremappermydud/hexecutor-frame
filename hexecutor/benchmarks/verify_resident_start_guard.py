# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reject unprepared graph starts without mapping DDR or borrowing VTCM."""

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
from pathlib import Path
import struct
import sys

FRAME=(PATHS.native);sys.path.insert(0,str(FRAME))
from hexecutor.offline.frame_transport import FrameTransport
from verify_resident_decoder import snapshot


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bootstrap',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    if b'QDENTRY1\x01\0\0\0' not in args.bootstrap.read_bytes():
        raise ValueError('bootstrap does not advertise the required entry guard')
    report={'before':snapshot(),'bootstrap_sha256':hashlib.sha256(args.bootstrap.read_bytes()).hexdigest(),'runs':[],'passed':False}
    for label in ('unpublished-zero-header','invalid-config'):
        with FrameTransport((PATHS.native / 'build/qd_loader.elf'),args.bootstrap,10) as transport:
            def prepare(sequence):
                transport.arena[:32768]=bytes(32768)
                if label=='invalid-config':
                    struct.pack_into('<7I',transport.arena,0,0x32564451,6,sequence,3,0,1,0)
                transport.fence()
            def collect():
                statuses=[list(struct.unpack_from('<4I',transport.arena,128+i*16)) for i in range(3)]
                return {'case':label,'statuses':statuses,
                        'result':list(struct.unpack_from('<2I',transport.arena,2048)),
                        'mapping_active':struct.unpack_from('<I',transport.arena,3072)[0],
                        'lease':list(struct.unpack_from('<3I',transport.arena,3328)),
                        'clock_active':struct.unpack_from('<I',transport.arena,256)[0],
                        'hmx_active':struct.unpack_from('<I',transport.arena,512)[0]}
            row=transport.invoke(prepare,transport.workers_exited,collect)
        row['passed']=row['result']==[0x31534451,1] and all(s[1]==1 and s[3]==0x54495845 for s in row['statuses']) and not row['mapping_active'] and row['lease']==[0,0,0] and not row['clock_active'] and not row['hmx_active']
        report['runs'].append(row)
        if not row['passed']:raise RuntimeError(f'unprepared startup was not rejected safely: {row}')
    report['after']=snapshot();report['passed']=all(r['passed'] for r in report['runs'])
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
