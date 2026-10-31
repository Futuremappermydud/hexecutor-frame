#!/usr/bin/env python3
"""Run one inference in the x86 HTP simulator with sim/qnn_trace_hook.so preloaded.

Writes OUTPUT/trace.log (one line per executed op), OUTPUT/calls.log (exported
HVX kernel calls with their arguments) and, with --writes,
OUTPUT/writes.bin (every VTCM block changed by each op). Build the hook once:

  docker run --rm --platform linux/amd64 -v "$PWD/depth-model/sim":/src \
    depth-model-qnn-sim g++ -O2 -std=c++17 -shared -fPIC \
    -o /src/build/qnn_trace_hook.so /src/qnn_trace_hook.cpp -ldl -lpthread

Inputs are a directory holding x.raw, y.raw, hidden.raw and t.raw (native).
"""
import argparse
from pathlib import Path
import subprocess

import sim_trace

from run_qnn_sim import IMAGE

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--writes", action="store_true")
    parser.add_argument("--recovered", type=Path,
                        help="also snapshot direct DDR outputs (requires a prior trace for run-list mapping)")
    args = parser.parse_args()
    out, inputs, ctx = args.output.resolve(), args.inputs.resolve(), args.context.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "list.txt").write_text(" ".join(f"{n}:=/in/{n}.raw" for n in ("x", "y", "hidden", "t")) + "\n")
    target = "x86_64-linux-clang"
    env = {"LD_LIBRARY_PATH": f"/sdk/lib/{target}:/usr/lib/llvm-14/lib",
           "LD_PRELOAD": "/hook/qnn_trace_hook.so", "QD_TRACE_LOG": "/out/trace.log",
           "QD_CALLS": "/out/calls.log"}
    if args.writes:
        env["QD_WRITES"] = "/out/writes.bin"
    if args.recovered:
        from htp_emu import tensor_bytes
        graph = sim_trace.load_graph(args.recovered)
        mapping = sim_trace.map_entries(graph, sim_trace.read_trace(out))
        tensors = {t['id']: t for t in graph['tensors']}
        config = []
        for (lst, idx), node in mapping.items():
            for j, tid in enumerate(node['outputs']):
                t = tensors[tid]
                if (t.get('storage') or {}).get('direct', {}).get('pool') in (3, 4):
                    config.append(f"{lst} {idx} {1 + len(node['inputs']) + 2*j} {tid} {tensor_bytes(t)}")
        (out / 'tensor_config.txt').write_text('\n'.join(config) + '\n')
        env.update(QD_TENSORS='/out/tensors.bin', QD_TENSOR_CONFIG='/out/tensor_config.txt')
    cmd = ["docker", "run", "--rm", "--platform", "linux/amd64"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["-v", f"{args.sdk.resolve()}:/sdk:ro", "-v", f"{ctx.parent}:/ctx:ro",
            "-v", f"{ROOT / 'sim' / 'build'}:/hook:ro", "-v", f"{inputs}:/in:ro", "-v", f"{out}:/out",
            IMAGE, f"/sdk/bin/{target}/qnn-net-run", "--backend", f"/sdk/lib/{target}/libQnnHtp.so",
            "--retrieve_context", f"/ctx/{ctx.name}", "--input_list", "/out/list.txt",
            "--use_native_input_files", "--use_native_output_files", "--output_dir", "/out",
            "--log_level", "warn"]
    with (out / "run.log").open("w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"exit {rc}; {out}")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
