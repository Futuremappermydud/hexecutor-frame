#!/usr/bin/env python3
"""Capture multi-command CPU reference prefixes for prepared runtime validation."""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frame'))
sys.path.insert(0, str(ROOT / 'tools'))
from native_graph import NativeGraph
import resident_image
from check_kernels import load_inputs
from bench.verify_native_graph import cases

PREFIX_TARGETS = {
    'primary': 132,   # commands 0..131 (stops before cmd 132: ConvLayer_SpaceToDepth)
    'alternate': 100, # commands 0..99 (stops before cmd 100: addsub_op_flat)
}

def capture(model: str, case: str, outdir: Path):
    target_count = PREFIX_TARGETS[model]
    image_path = ROOT / f'frame/build/resident-{model}.qdrx'
    constants = ROOT / f'recovered/{model}/constants/constant_pool_5.bin'
    case_dir = next(row[1] for row in cases(model, [case])[1] if row[0] == case)
    inputs = load_inputs(case_dir)

    outdir.mkdir(parents=True, exist_ok=True)
    graph = NativeGraph(image_path, constants)
    try:
        graph.use_cpu()
        graph.set_hoist(False)
        graph.set_block_output(False)

        # Capture initial pools before any execution
        pools_before = [(graph.pools[p].copy() if p in graph.pools else np.zeros(0, dtype=np.uint8)) for p in range(7)]

        # Run commands [0, target_count)
        print(f"Running {model}/{case} commands 0..{target_count - 1} ({target_count} commands)...", flush=True)
        graph.run(inputs, 0, target_count)

        pools_after = [(graph.pools[p].copy() if p in graph.pools else np.zeros(0, dtype=np.uint8)) for p in range(7)]
        pool_sizes = [len(p) for p in pools_after]

        # Extract camera inputs (inputs is a dict 0..3)
        in_bytes_list = [inputs[i].tobytes() if i in inputs else b'' for i in range(4)]
        in_sizes = [len(b) for b in in_bytes_list]

        model_code = 0 if model == 'primary' else 1
        fixture_path = outdir / f'{model}-{case}-prefix-{target_count}.mpfx'

        with fixture_path.open('wb') as f:
            # Header: magic, ver, model_code, target_count, num_inputs, in_sizes[4], pool_sizes[7]
            hdr = struct.pack('<4s4I4I7I',
                              b'MPFX', 1, model_code, target_count, len(in_sizes),
                              *in_sizes,
                              *pool_sizes)
            f.write(hdr)
            for b in in_bytes_list:
                f.write(b)
            # Initial pools
            for p in range(7):
                f.write(pools_before[p].tobytes())
            # Output pools after prefix
            for p in range(7):
                f.write(pools_after[p].tobytes())

        print(f"Captured {fixture_path.name}: {fixture_path.stat().st_size} bytes (pools: {pool_sizes})", flush=True)
    finally:
        graph.close()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=('smoke', 'random'), default='smoke')
    ap.add_argument('--model', choices=('primary', 'alternate'), action='append')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'fixtures')
    args = ap.parse_args()
    for model in args.model or ('primary', 'alternate'):
        capture(model, args.case, args.out)

if __name__ == '__main__':
    main()
