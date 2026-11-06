#!/usr/bin/env python3
"""Write a reproducibility manifest for the isolated adapter evidence."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
FILES=('qd_native_1x1_adapter.h','qd_native_1x1_adapter.c','qd_native_1x1_graph_bridge.h','qd_native_1x1_graph_eligibility.h','test_graph_eligibility.c','make_graph_eligibility_fixtures.py','graph-eligibility-115.bin','graph-eligibility-3416.bin','graph-eligibility-1195.bin','census_qdrx_1x1.py','qdrx-1x1-census.json',
 'make_preflight_fixture.py','preflight-primary.bin','preflight_qdrx.c','audit_representative_sources.py','representative-source-audit.json','evaluate_dynamic_fi.py','dynamic-fi-2380.json','make_dynamic_fixture.py','fixture-qdrx-2380.bin','qnn_hmx_1x1_dynamic_sim.c','qnn_hmx_1x1_dynamic_sim.elf','output-qdrx-2380.bin','check_dynamic_2380_sim.py','qdrx-2380-adapter-sim.json','qdrx-1195-adapter-sim.json','qdrx-4136-adapter-sim.json','qdrx-4137-adapter-sim.json','qdrx-4151-adapter-sim.json','qdrx-4154-adapter-sim.json','qdrx-4169-adapter-sim.json','qdrx-4173-adapter-sim.json','qdrx-4269-adapter-sim.json','make_qdrx_1195_fixture.py',
 'fixture-qdrx-1195.bin','qnn_hmx_1x1_adapter_sim.c','qnn_hmx_1x1_adapter_sim.elf',
 'output-qdrx-1195.bin','check_qdrx_1195_sim.py','qdrx-1195-adapter-sim.json',
 'masked_scatter_sim.c','masked_scatter_sim.elf','make_padded_ci_fixture.py','padded_ci_leaf_sim.c','padded_ci_leaf_sim.elf','padded_ci_leaf_poison_sim.elf','check_padded_ci_sim.py','padded-ci-115.bin','padded-ci-3416.bin','padded-ci-115-sim.json','padded-ci-3416-sim.json','padded-ci-115-out.bin','padded-ci-115-poison-out.bin','padded-ci-3416-out.bin','padded-ci-3416-poison-out.bin','adapter_frame_probe.c','adapter_frame_probe.elf','run_adapter_frame_probe.py','validate_frame_probe.py','frame-probe-preflight.json','build_adapter_frame_probe.sh','README.md')
rows=[]
for name in FILES:
 p=HERE/name
 if p.exists(): rows.append({'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
checkpoint=ROOT/'.work/frame-direct-kernels-20261030/native-adapter-1195-first.json'
checkpoint_record=None
if checkpoint.exists():
 raw=checkpoint.read_bytes();data=json.loads(raw)
 checkpoint_record={'path':str(checkpoint.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'adapter_source_sha256_at_run':'d1d0b2fb9512ba71ebbfa5696b552fbc927cc1db5e9b53fb11282d49589fd102','payload_sha256':data.get('payload_sha256'),'passed':data.get('passed'),'run':data.get('run'),'adapter_optimization_applied_after_checkpoint':True}
report={'scope':'native 1x1 adapter and source-included graph bridge offline evidence plus first Frame checkpoint','files':rows,'frame_checkpoint':checkpoint_record,
 'checks':['host adapter C -Wall -Wextra -Werror compile with QDS_NATIVE_PADDED_1X1=0','actual-primary QDRX C prepare 140/140','source-included DSP translation-unit bridge compile with QDS_NATIVE_1X1 and QDS_PRESERVE_VTCM','Hexagon Simulator nine actual CPU-prefix QDRX commands with workspace initialized to 0xA5; each native delta [0,+1] and inactive output tables preserved','Hexagon Simulator detects first and last byte corruption in all 16 128-byte lanes of both guard blocks','Hexagon Simulator effective-H2 masked-scatter preservation','Host C graph eligibility predicate tests use actual compiled QDRX metadata for commands 115, 3416 and 1195, with padded feature on and off','Hexagon Simulator actual raw-Xpb adapter expansion for CI12 command 115 and CI20 command 3416 matches independent expansion and integer oracle [0,+1]; poisoned input tails are byte-identical; unsupported prepare leaves workspace unchanged','Frame command 1195 first controlled run passed; report hash recorded above'],
 'not_performed':['Frame hardware run of the source-included graph bridge','full-graph exact arithmetic (native leaf bound is [0,+1])']}
(HERE/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'wrote manifest for {len(rows)} files')
