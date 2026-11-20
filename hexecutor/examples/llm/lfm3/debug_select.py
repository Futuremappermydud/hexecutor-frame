#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Why is a node not delegated? Exports layers [lo, hi) and prints every node the partitioner refuses, with the reason.

Usage: debug_select.py MODEL_DIR LO HI [LMAX]
"""
import pathlib, sys
import torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[3]))
import lfm3_stages as L
from hexecutor.offline.executorch import partitioner as P, lowering as LW
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

model, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lmax = int(sys.argv[4]) if len(sys.argv) > 4 else 64


class Dbg(P.HexecutorPartitioner):
    def partition(self, ep):
        chosen = set(LW.select_nodes(ep))
        for n in ep.graph.nodes:
            if n.op != "call_function" or n in chosen:
                continue
            try:
                kind = LW.kind_of(n, ep)
                why = f"kind {kind} (refused by ordering/cycle rule)"
            except LW.Unsupported as e:
                why = f"Unsupported: {e}"
            except Exception as e:  # noqa
                why = f"{type(e).__name__}: {e}"
            print("REFUSED", n.name, LW.op_name(n.target), "->", why)
        print("selected", len(chosen))
        return super().partition(ep)


ck = L.Checkpoint(model)
st = L.Stage(ck, lo, hi, lmax, final=(hi == 24), log=lambda *_: None).eval()
ex = (torch.zeros(1, 1, L.D), torch.zeros(1, 1), torch.zeros(1, 1, 1, lmax), torch.zeros(1, 1, 1, lmax))
with torch.no_grad():
    ep = torch.export.export(st, ex)
try:
    to_edge_transform_and_lower(ep, partitioner=[Dbg()], compile_config=EdgeCompileConfig())
except Exception as e:
    print("lower error:", str(e)[:300])
