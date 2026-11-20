#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run the Q4 stage modules eagerly (hexecutor::lfm_* reference implementations) against the HF hidden states.

Usage: eager_check.py MODEL_DIR REF_DIR [LMAX] [N_POSITIONS]
"""
import json, pathlib, sys, time
import numpy as np
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lfm3_stages as L
from hexecutor.offline.executorch import ops

model_dir, ref_dir = sys.argv[1], pathlib.Path(sys.argv[2])
lmax = int(sys.argv[3]) if len(sys.argv) > 3 else 64
ref = json.loads((ref_dir / "ref.json").read_text())
hidden = np.fromfile(ref_dir / "ref_hidden.f32", "<f4").reshape(-1, L.D)
ids = (ref["prompt_ids"] + ref["generated"])[: hidden.shape[0]]
n = int(sys.argv[4]) if len(sys.argv) > 4 else len(ids)
ck = L.Checkpoint(model_dir)
import os
quant = L.q4_best_blocks if os.environ.get('BEST') else L.q4_blocks
emb = ck.f.get_tensor("model.embed_tokens.weight").float()
t0 = time.time()
stages = [L.Stage(ck, lo, hi, lmax, final=(hi == 24), log=lambda *_: None, quant=quant, q8_names=eval(os.environ.get('Q8N','[]')), q4_names=[f'l{i}_w3' for i in eval(os.environ.get('Q4W3','[]'))], q8=__import__('os').environ.get('Q8','').split(), exact=bool(int(__import__('os').environ.get('EXACT','0')))) for lo, hi in L.stage_ranges()]
print("built", time.time() - t0, flush=True)
ops.reset_state()
errs = []
agree = 0
E = emb.numpy()
kl = []
with torch.no_grad():
    for p in range(n):
        x = emb[ids[p]].reshape(1, 1, L.D)
        mf, ms = (torch.from_numpy(m) for m in L.masks(p, lmax))
        pos = torch.full((1, 1), float(p))
        for si, s in enumerate(stages):
            x = s(x, pos, mf, ms)
            if os.environ.get('DUMP'):
                open(os.environ['DUMP'] + f'_stage{si}.f32', 'ab').write(x.numpy().astype('<f4').tobytes())
        e = np.abs(x.numpy().ravel() - hidden[p]).max() / np.abs(hidden[p]).max()
        cos = float(np.dot(x.numpy().ravel(), hidden[p]) / np.linalg.norm(x.numpy()) / np.linalg.norm(hidden[p]))
        errs.append(e)
        lg = E @ x.numpy().ravel(); lr = E @ hidden[p]
        agree += int(lg.argmax() == lr.argmax())
        pr = np.exp(lr - lr.max()); pr /= pr.sum(); pq = np.exp(lg - lg.max()); pq /= pq.sum()
        kl.append(float((pr * (np.log(pr + 1e-30) - np.log(pq + 1e-30))).sum()))
        print(p, "max rel err %.4f cos %.5f" % (e, cos), flush=True)

print('top1 agree %d/%d  mean KL %.4f  mean cos-err %.4f' % (agree, n, np.mean(kl), np.mean(errs)))
