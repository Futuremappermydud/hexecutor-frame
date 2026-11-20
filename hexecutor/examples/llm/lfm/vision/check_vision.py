# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check LFM3-VL vision encoder and multimodal projector coverage and numerical accuracy.

Workflow:
(1) Export component to float ONNX (batch-one static shape).
(2) Lower to HXFG using hexecutor.offline.lower_onnx_float.
    Reports the first failing node, op type, and reason when it fails.
(3) Run lowered graph on build/bin/hexecutor_tensor_run and compare numerically to PyTorch.
"""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import onnx
import torch

from hexecutor.offline import lower_onnx_float as lowering
from hexecutor.examples.llm.lfm.vision.export_vision import COMPONENTS, export_component

RUNNER = Path("build/bin/hexecutor_tensor_run")


def extract_failing_node_details(error_msg, model_path):
    """Inspect ONNX graph to identify the failing node name, op type, and details from error message."""
    node_name = "unknown"
    op_type = "unknown"
    try:
        model = onnx.load(str(model_path))
        for node in model.graph.node:
            if node.name and (node.name in error_msg or f"'{node.name}'" in error_msg):
                node_name = node.name
                op_type = node.op_type
                break
            if node.output and any(out in error_msg for out in node.output):
                node_name = node.name or node.output[0]
                op_type = node.op_type
                break
            if node.op_type in error_msg:
                node_name = node.name or node.output[0]
                op_type = node.op_type
    except Exception:
        pass
    return node_name, op_type


def run_hexecutor_runner(hxfg_path, bin_path, in_data, tmp_dir):
    """Execute hexecutor_tensor_run on input data."""
    tmp = Path(tmp_dir)
    in_file = tmp / "in.f32"
    out_prefix = tmp / "out_"

    # Save input tensor: token layout [1, L, K] -> [L, K] flat float32
    flat_data = in_data[0].transpose(1, 2, 0) if in_data.ndim == 4 else in_data[0]
    in_file.write_bytes(np.ascontiguousarray(flat_data, np.float32).tobytes())

    cmd = [str(RUNNER), str(hxfg_path), str(bin_path), str(in_file), str(out_prefix)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Runner exited with code {res.returncode}:\n{res.stderr.strip() or res.stdout.strip()}")

    outputs = []
    i = 0
    while (tmp / f"out_{i}.f32").exists():
        outputs.append(np.fromfile(tmp / f"out_{i}.f32", np.float32))
        i += 1
    return outputs


def check_component(name, tolerance=1e-4, raise_pool_limit=False):
    """Test full pipeline: Export -> Lower -> Run -> Compare for a single component."""
    print(f"\n=======================================================")
    print(f" Checking Component: {name}")
    print(f"=======================================================")

    builder, input_shape = COMPONENTS[name]
    model = builder()
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        onnx_file = tmp_path / f"{name}.onnx"
        hxfg_prefix = tmp_path / "graph"

        # 1. Export
        print(f"1. Exporting {name} to ONNX (shape: {input_shape})...")
        try:
            _, model = export_component(name, str(onnx_file), model=model)
        except Exception as e:

            print(f"   [EXPORT FAILED]: {e}")
            return {
                "component": name,
                "status": "EXPORT_FAILED",
                "node": "N/A",
                "op_type": "N/A",
                "reason": str(e),
            }

        # 2. Lower
        print(f"2. Lowering {name} to HXFG...")
        orig_limit = lowering.CONSTANT_POOL_LIMIT
        if raise_pool_limit:
            lowering.CONSTANT_POOL_LIMIT = 256 * 1024 * 1024

        try:
            manifest = lowering.lower(onnx_file, hxfg_prefix)
            pool_bytes = manifest.get("constant_bytes", 0)
            arena_bytes = manifest.get("arena_bytes", 0)
            num_ops = len(manifest.get("ops", []))
            print(f"   [LOWER SUCCESS]: {num_ops} ops, {pool_bytes / (1024*1024):.2f} MiB pool, {arena_bytes / 1024:.1f} KiB arena")
        except Exception as e:
            err_msg = str(e)
            node_name, op_type = extract_failing_node_details(err_msg, onnx_file)
            print(f"   [LOWER FAILED]:")
            print(f"     Node:    {node_name}")
            print(f"     OpType:  {op_type}")
            print(f"     Reason:  {err_msg}")
            return {
                "component": name,
                "status": "LOWER_FAILED",
                "node": node_name,
                "op_type": op_type,
                "reason": err_msg,
            }
        finally:
            lowering.CONSTANT_POOL_LIMIT = orig_limit

        # 3. Run on runner and compare numerically
        if not RUNNER.exists():
            print(f"   [RUNNER SKIP]: Runner binary {RUNNER} not found. Rebuild with cmake --build build --target hexecutor_tensor_run")
            return {
                "component": name,
                "status": "RUNNER_MISSING",
                "node": "-",
                "op_type": "-",
                "reason": f"Binary {RUNNER} missing",
            }

        print(f"3. Running on hexecutor_tensor_run and comparing to PyTorch...")
        dummy_in = torch.randn(*input_shape)
        with torch.no_grad():
            expected = model(dummy_in).numpy()

        hxfg_file = tmp_path / "graph.hxfg"
        bin_file = tmp_path / "graph.bin"

        try:
            got_outputs = run_hexecutor_runner(hxfg_file, bin_file, dummy_in.numpy(), tmp_dir)
            parts = manifest["output_tensors"]
            merged, idx = [], 0
            for o in [expected]:
                flat = o[0].transpose(1, 2, 0) if o.ndim == 4 else o[0]
                width = flat.shape[-1]
                chunks = []
                while sum(c.shape[-1] for c in chunks) < width:
                    c = got_outputs[idx].reshape(-1, parts[idx]["channels"][1] - parts[idx]["channels"][0])
                    chunks.append(c)
                    idx += 1
                merged.append(np.concatenate(chunks, axis=-1).reshape(flat.shape))

            max_diff = float(np.max(np.abs(merged[0] - expected[0])))
            mean_diff = float(np.mean(np.abs(merged[0] - expected[0])))
            passed = max_diff <= tolerance
            print(f"   [NUMERIC RESULT]: Max Abs Diff: {max_diff:.2e}, Mean Diff: {mean_diff:.2e} (Tol: {tolerance})")
            print(f"   [VERIFICATION]: {'PASS' if passed else 'FAIL'}")

            return {
                "component": name,
                "status": "PASS" if passed else "NUMERIC_FAIL",
                "node": "-",
                "op_type": "-",
                "reason": f"MaxDiff: {max_diff:.2e}",
                "pool_bytes": pool_bytes,
                "arena_bytes": arena_bytes,
                "ops": num_ops,
                "max_diff": max_diff,
            }
        except Exception as e:
            print(f"   [EXECUTION FAILED]: {e}")
            return {
                "component": name,
                "status": "RUN_FAILED",
                "node": "-",
                "op_type": "-",
                "reason": str(e),
                "pool_bytes": pool_bytes,
                "arena_bytes": arena_bytes,
                "ops": num_ops,
            }


def main():
    parser = argparse.ArgumentParser(description="Check LFM3-VL vision tower coverage and accuracy.")
    parser.add_argument("--component", choices=list(COMPONENTS.keys()) + ["all"], default="all",
                        help="Component to test (default: all)")
    parser.add_argument("--raise-pool-limit", action="store_true",
                        help="Temporarily raise per-graph constant pool limit to 256 MiB to analyze large subgraphs")
    args = parser.parse_args()

    targets = list(COMPONENTS.keys()) if args.component == "all" else [args.component]
    results = []
    for comp in targets:
        res = check_component(comp, raise_pool_limit=args.raise_pool_limit)
        results.append(res)

    print("\n" + "=" * 80)
    print(" SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Component':<12} | {'Status':<14} | {'Failing Node':<20} | {'OpType':<12} | {'Details':<30}"
    print(header)
    print("-" * len(header))
    for r in results:
        node = r.get("node", "-")[:20]
        op = r.get("op_type", "-")[:12]
        reason = r.get("reason", "-")[:30]
        print(f"{r['component']:<12} | {r['status']:<14} | {node:<20} | {op:<12} | {reason:<30}")
    print("=" * 80)


if __name__ == "__main__":
    main()
