#!/usr/bin/env python3
"""Offline LiteRT comparison against the C++ reference or Frame DSP runner.

All inputs, raw outputs, and reports go to an explicit ignored work directory.
The live inference process is C++ in both modes. Requires numpy, ai-edge-litert,
and (with --image) Pillow on the development machine.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import numpy as np
from ai_edge_litert.interpreter import Interpreter


def run(argv):
    result = subprocess.run(list(map(str, argv)), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(list(map(str, argv)))}\n{result.stderr[-4096:]}"
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--runner", required=True)
    p.add_argument("--host", help="optional SSH target for a DSP run")
    p.add_argument(
        "--remote-work", help="explicit remote working directory; required with --host"
    )
    p.add_argument("--loader")
    p.add_argument("--engine")
    p.add_argument("--bootstrap")
    p.add_argument("--chunk-probe")
    p.add_argument("--precise", action="store_true")
    p.add_argument("--model", action="append", choices=("palm", "landmark"))
    p.add_argument(
        "--case",
        action="append",
        choices=("zero", "random", "gradient", "image", "handcrop"),
    )
    p.add_argument("--image", type=Path)
    p.add_argument(
        "--landmark-input",
        type=Path,
        help="optional real hand crop captured by --dump-inputs",
    )
    args = p.parse_args()
    dsp = bool(args.engine)
    if args.host and not (args.remote_work and dsp):
        p.error("--host requires --remote-work and DSP artifacts")
    artifacts = [args.loader, args.engine, args.bootstrap, args.chunk_probe]
    if dsp and not all(artifacts):
        p.error("provide all four DSP artifact paths")
    args.work.mkdir(parents=True, exist_ok=True)
    if args.host:
        run(["ssh", args.host, shlex.join(["mkdir", "-p", args.remote_work])])
    report = {
        "mode": (
            "dsp-float" if args.precise else ("dsp-hmx" if dsp else "cpu-reference")
        ),
        "cases": [],
        "selection": {"models": args.model, "cases": args.case},
        "artifacts": {},
    }
    if args.host:
        result = run(["ssh", args.host, shlex.join(["sha256sum", *artifacts])])
        report["artifacts"] = {
            line.split(maxsplit=1)[1]: line.split()[0]
            for line in result.stdout.splitlines()
        }
    elif dsp:
        report["artifacts"] = {
            str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in artifacts
        }
    failed = False
    for name, tflite, side in [
        ("palm", "hand_detector", 192),
        ("landmark", "hand_landmarks_detector", 224),
    ]:
        if args.model and name not in args.model:
            continue
        interpreter = Interpreter(
            model_path=str(args.assets / (tflite + ".tflite")), num_threads=1
        )
        interpreter.allocate_tensors()
        shape = interpreter.get_input_details()[0]["shape"]
        cases = {
            "zero": np.zeros(shape, dtype=np.float32),
            "random": np.random.default_rng(13).uniform(0, 1, shape).astype(np.float32),
            "gradient": np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(
                shape
            ),
        }
        if args.image:
            from PIL import Image, ImageOps

            im = Image.open(args.image).convert("RGB")
            im = ImageOps.pad(
                im, (side, side), method=Image.Resampling.BILINEAR, color=(0, 0, 0)
            )
            cases["image"] = np.asarray(im, dtype=np.float32)[None] / 255
        if name == "landmark" and args.landmark_input:
            cases["handcrop"] = np.fromfile(
                args.landmark_input, dtype=np.float32
            ).reshape(shape)
        files = [args.models / (name + ".hxfg"), args.models / (name + ".bin")]
        report["artifacts"][str(files[0])] = hashlib.sha256(
            files[0].read_bytes()
        ).hexdigest()
        report["artifacts"][str(files[1])] = hashlib.sha256(
            files[1].read_bytes()
        ).hexdigest()
        report["artifacts"][str(args.assets / (tflite + ".tflite"))] = hashlib.sha256(
            (args.assets / (tflite + ".tflite")).read_bytes()
        ).hexdigest()
        if args.host:
            run(["rsync", "-az", *files, args.host + ":" + args.remote_work + "/"])
        for case, input_data in cases.items():
            if args.case and case not in args.case:
                continue
            stem = name + "-" + case
            input_path = args.work / (stem + "-input.f32")
            input_data.tofile(input_path)
            interpreter.set_tensor(
                int(interpreter.get_input_details()[0]["index"]), input_data
            )
            interpreter.invoke()
            prefix = args.work / (stem + "-output-")
            if args.host:
                run(
                    [
                        "rsync",
                        "-az",
                        input_path,
                        args.host + ":" + args.remote_work + "/",
                    ]
                )
                remote = Path(args.remote_work)
                cmd = [
                    args.runner,
                    remote / files[0].name,
                    remote / files[1].name,
                    remote / input_path.name,
                    remote / prefix.name,
                    *artifacts,
                ]
                if args.precise:
                    cmd = ["env", "HEXECUTOR_TENSOR_PRECISE=1", *cmd]
                result = run(["ssh", args.host, shlex.join(list(map(str, cmd)))])
                names = [
                    str(remote / (prefix.name + str(i) + ".f32"))
                    for i in range(len(interpreter.get_output_details()))
                ]
                for f in names:
                    run(["rsync", "-az", args.host + ":" + f, str(args.work) + "/"])
                run(
                    [
                        "rsync",
                        "-az",
                        args.host + ":" + str(remote / (prefix.name + "profile.json")),
                        str(args.work) + "/",
                    ]
                )
            else:
                cmd = [args.runner, *files, input_path, prefix]
                if dsp:
                    cmd.extend(artifacts)
                if args.precise:
                    cmd = ["env", "HEXECUTOR_TENSOR_PRECISE=1", *cmd]
                result = run(cmd)
            row = {
                "model": name,
                "case": case,
                "outputs": [],
                "runner_stderr": result.stderr,
            }
            for i, detail in enumerate(interpreter.get_output_details()):
                expected = interpreter.get_tensor(int(detail["index"])).reshape(-1)
                actual = np.fromfile(str(prefix) + str(i) + ".f32", dtype=np.float32)
                if actual.shape != expected.shape:
                    raise ValueError("runner returned incorrect output shape")
                error = np.abs(actual - expected)
                # FP16 budget: 1% of network input extent, 2 percentage points
                # of probability, and 2 mm for model-relative world coordinates.
                budget = (
                    (1.92 if i == 0 else 0.02)
                    if name == "palm"
                    else [2.24, 0.02, 0.02, 0.002][i]
                )
                metric_error = error
                if name == "palm" and i == 1:
                    sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100)))
                    metric_error = np.abs(sigmoid(actual) - sigmoid(expected))
                if not dsp or args.precise:
                    budget = 0.002
                passed = bool(
                    np.isfinite(actual).all() and metric_error.max() <= budget
                )
                failed |= not passed
                row["outputs"].append(
                    {
                        "index": i,
                        "max_abs_error": float(error.max()),
                        "mean_abs_error": float(error.mean()),
                        "metric_max_error": float(metric_error.max()),
                        "budget": budget,
                        "passed": passed,
                    }
                )
            if dsp:
                row["profile"] = json.loads(
                    Path(str(prefix) + "profile.json").read_text()
                )
            report["cases"].append(row)
            (args.work / "validation.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                stem,
                "PASS" if all(o["passed"] for o in row["outputs"]) else "FAIL",
                [o["max_abs_error"] for o in row["outputs"]],
                flush=True,
            )
    report["passed"] = not failed
    (args.work / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        raise SystemExit("model comparison exceeded an error budget")


if __name__ == "__main__":
    main()
