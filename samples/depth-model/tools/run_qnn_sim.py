#!/usr/bin/env python3
"""Run an extracted context in QAIRT's x86_64 HTP simulator through Docker.

This executes Qualcomm's own HTP graph executor and kernels on the host CPU.
It needs no NPU, Frame, FastRPC or QuRT. On Apple Silicon Docker emulates
amd64, so expect roughly 8-9 s per inference.

The input list uses host paths, one inference per line, as written by
prepare_inputs.py. Outputs are native files under OUTPUT/Result_N/.
"""

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
from pathlib import Path
import shlex
import subprocess

IMAGE = "depth-model-qnn-sim"
DOCKERFILE_DIR = (PATHS.model_data / 'sim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--input-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log-level", default="warn")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sdk, output = args.sdk.resolve(), args.output.resolve()
    context = args.context.resolve()

    # Mount every input directory read-only and rewrite the list to container paths.
    mounts, rewritten = {}, []
    for line in args.input_list.read_text().split("\n"):
        if not line.strip():
            continue
        fields = []
        for item in line.split():
            name, path = item.split(":=", 1)
            path = Path(path).resolve()
            mount = mounts.setdefault(path.parent, f"/in/{len(mounts)}")
            fields.append(f"{name}:={mount}/{path.name}")
        rewritten.append(" ".join(fields))
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_list.container.txt").write_text("\n".join(rewritten) + "\n")

    target = "x86_64-linux-clang"
    net_run = [f"/sdk/bin/{target}/qnn-net-run",
               "--backend", f"/sdk/lib/{target}/libQnnHtp.so",
               "--retrieve_context", f"/ctx/{context.name}",
               "--input_list", "/out/input_list.container.txt",
               "--use_native_input_files", "--use_native_output_files",
               "--output_dir", "/out", "--log_level", args.log_level]
    cmd = ["docker", "run", "--rm", "--platform", "linux/amd64",
           "-e", f"LD_LIBRARY_PATH=/sdk/lib/{target}:/usr/lib/llvm-14/lib",
           "-v", f"{sdk}:/sdk:ro", "-v", f"{context.parent}:/ctx:ro", "-v", f"{output}:/out"]
    for host, mount in mounts.items():
        cmd += ["-v", f"{host}:{mount}:ro"]
    cmd += [IMAGE] + net_run
    (output / "command.txt").write_text(shlex.join(cmd) + "\n")
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode:
        subprocess.run(["docker", "build", "--platform", "linux/amd64", "-t", IMAGE,
                        str(DOCKERFILE_DIR)], check=True)
    with (output / "run.log").open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"exit {result.returncode}; log: {output / 'run.log'}")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
