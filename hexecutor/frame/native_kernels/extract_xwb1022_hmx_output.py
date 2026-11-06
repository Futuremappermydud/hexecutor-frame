#!/usr/bin/env python3
"""Extract the physical crouton output byte dump from a v75 simulator log."""
import hashlib
import json
from pathlib import Path
here = Path(__file__).resolve().parent
log = (here / 'xwb1022-actual-prefix-hmx-output-dump.log').read_text()
start, end = 'RAW_OUTPUT_PHYSICAL_HEX_BEGIN', 'RAW_OUTPUT_PHYSICAL_HEX_END'
if log.count(start) != 1 or log.count(end) != 1:
    raise SystemExit('missing or duplicate output dump markers')
hexdata = ''.join(log.split(start,1)[1].split(end,1)[0].split())
raw = bytes.fromhex(hexdata)
if len(raw) != 32 * 2048:
    raise SystemExit(f'wrong output size: {len(raw)}')
out = here / 'xwb1022_hmx_output.bin'
out.write_bytes(raw)
report = {'qdrx_sequence':1022,'command_index':836,'bytes':len(raw),
          'layout':'physical 32-page crouton output, row-major page table',
          'sha256':hashlib.sha256(raw).hexdigest(),
          'sim_log_sha256':hashlib.sha256((here/'xwb1022-actual-prefix-hmx-output-dump.log').read_bytes()).hexdigest()}
(here/'xwb1022_hmx_output.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
