#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Offline QNN TFLite/ONNX/DLC conversion and HTP preparation with explicit SDK/target.

Run in a Linux QAIRT development environment with its tested dependencies and
clang++ installed. This tool never starts a device session. Generated source,
weights, libraries, context binaries and schematics must remain local artifacts.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from inspect_qnn import inspect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--graph-name", help="DLC only: graph name reported by qairt-dlc-info")
    p.add_argument("--context-only", action="store_true",
                   help="skip optrace schematic generation and inspection")
    p.add_argument("--arch", default="v75")
    p.add_argument("--soc-model", type=int, default=57, help="QNN SM8650 is 57")
    p.add_argument("--vtcm-mb", type=int, default=8)
    p.add_argument("--hvx-threads", type=int, default=4)
    p.add_argument("--optimization", type=int, choices=(2, 3), default=3)
    p.add_argument("--copyright-file", type=Path)
    p.add_argument(
        "--input-dim",
        action="append",
        default=[],
        metavar="NAME=D0,D1,...",
        help="ONNX only: fix an input shape for the converter",
    )
    p.add_argument(
        "--converter-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra converter argument, repeatable (e.g. --converter-arg=--input_list --converter-arg=LIST)",
    )
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", args.name):
        p.error("name must be a C identifier")
    sdk, out, model = args.sdk.resolve(), args.output.resolve(), args.model.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = "x86_64-linux-clang"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(sdk / "lib/python") + os.pathsep + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = (
        str(sdk / "lib" / target) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )
    env["QNN_SDK_ROOT"] = env["QAIRT_SDK_ROOT"] = str(sdk)
    commands = []

    def run(command, log):
        command = list(map(str, command))
        commands.append(command)
        with (out / log).open("w") as f:
            subprocess.run(
                command,
                env=env,
                cwd=out,
                check=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

    tools = sdk / "bin" / target
    cpp, weights = out / (args.name + ".cpp"), out / (args.name + ".bin")
    suffix = model.suffix.lower()
    if suffix not in (".tflite", ".onnx", ".dlc"):
        p.error("model must be .tflite, .onnx, or .dlc")
    if args.input_dim and suffix != ".onnx":
        p.error("--input-dim applies to ONNX models only")
    graph_name = args.name
    if suffix == ".dlc":
        if not args.graph_name:
            p.error("--graph-name is required for DLC models")
        if args.converter_arg or args.copyright_file:
            p.error("converter options do not apply to preconverted DLC models")
        graph_name = args.graph_name
        model_args = ["--model", sdk / "lib" / target / "libQnnModelDlc.so",
                      "--dlc_path", model]
    else:
        if args.graph_name:
            p.error("--graph-name applies to DLC models only")
        converter = [
            sys.executable,
            tools / ("qnn-onnx-converter" if suffix == ".onnx" else "qnn-tflite-converter"),
            "--input_network",
            model,
            "--output_path",
            cpp,
        ]
        for spec in args.input_dim:
            name, sep, dims = spec.partition("=")
            if not sep or not re.fullmatch(r"\d+(,\d+)*", dims):
                p.error(f"invalid --input-dim {spec!r}")
            converter += ["--input_dim", name, dims]
        if args.copyright_file:
            # QAIRT requires a .txt extension even for plain-text NOTICE files.
            notice = out / "copyright.txt"
            notice.write_text(args.copyright_file.read_text())
            converter += ["--copyright_file", notice]
        converter += args.converter_arg
        run(converter, "convert.log")
        run(
            [
                sys.executable,
                tools / "qnn-model-lib-generator",
                "-c",
                cpp,
                "-b",
                weights,
                "-t",
                target,
                "-o",
                out / "lib",
            ],
            "library.log",
        )
        model_args = ["--model", out / "lib" / target / ("lib" + args.name + ".so")]
    backend = {
        "graphs": [
            {
                "graph_names": [graph_name],
                "O": args.optimization,
                "vtcm_mb": args.vtcm_mb,
                "hvx_threads": args.hvx_threads,
            }
        ],
        "devices": [
            {
                "device_id": 0,
                "soc_model": args.soc_model,
                "dsp_arch": args.arch,
                "cores": [{"core_id": 0, "perf_profile": "burst"}],
            }
        ],
    }
    (out / "htp.json").write_text(json.dumps(backend, indent=2) + "\n")
    ext = {
        "backend_extensions": {
            "shared_library_path": str(
                sdk / "lib" / target / "libQnnHtpNetRunExtensions.so"
            ),
            "config_file_path": str(out / "htp.json"),
        }
    }
    (out / "extensions.json").write_text(json.dumps(ext, indent=2) + "\n")
    optrace_args = [] if args.context_only else [
        "--profiling_level", "detailed", "--profiling_option", "optrace",
    ]
    binary_name = args.name + "-" + args.arch
    run(
        [
            tools / "qnn-context-binary-generator",
            *model_args,
            "--backend",
            sdk / "lib" / target / "libQnnHtp.so",
            "--config_file",
            out / "extensions.json",
            "--binary_file",
            binary_name,
            "--output_dir",
            out,
            *optrace_args,
            "--log_level",
            "info",
        ],
        "prepare.log",
    )
    context = out / (binary_name + ".bin")
    if not context.is_file() or not context.stat().st_size:
        raise RuntimeError(f"QAIRT did not produce {context}; see {out / 'prepare.log'}")
    report = {"context_binary": str(context)}
    if not args.context_only:
        # QAIRT replaces punctuation in graph names when naming schematic files.
        schematic_name = re.sub(r"[^a-zA-Z0-9_]", "_", graph_name)
        report.update(inspect(out / (schematic_name + "_schematic.bin")))
    report.update(
        {
            "source_model": str(model),
            "source_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "target_config": backend,
            "sdk": str(sdk),
            "commands": commands,
        }
    )
    for file in [
        sdk / "lib" / target / "libQnnHtp.so",
        sdk / "lib" / target / "libHtpPrepare.so",
    ]:
        report.setdefault("sdk_hashes", {})[str(file)] = hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
    (out / "compiled-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = out / "compiled-summary.json"
    if args.context_only:
        print(f"{args.name}: {context}; {summary}")
    else:
        print(f'{args.name}: {report["after_nodes"]} lowered nodes; {summary}')


if __name__ == "__main__":
    main()
