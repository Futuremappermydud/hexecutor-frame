#!/usr/bin/env python3
"""Run isolated composite-unary sweeps; evaluation requires a frozen manifest."""
import argparse
import os
from pathlib import Path
import platform
import subprocess

from stereo_composite_common import (ARMS, DEVELOPMENT_SEEDS, EVALUATION_SEEDS, FRAMES,
    REGRESSION_SEEDS, SOURCES, ROOT, config, config_id, digest, grid, identity, load_phase, read_json, write_json)


def make_plan(phase, root, inputs, binary):
    selected = None
    references = {arm: config() for arm in ARMS}
    configurations, seeds, recorded = [], [], []
    if phase in ["smoke", "regression"]:
        configurations = [config(), config("composite-raw", .25), config("composite-normalized", .25),
                          config("composite-raw"), config("composite-normalized")]
        seeds = [1709] if phase == "smoke" else REGRESSION_SEEDS
        recorded = FRAMES if inputs else []
    elif phase == "development":
        configurations = grid("baseline")
        configurations += [config("composite-raw"), config("composite-normalized")]
        for method in ["composite-raw", "composite-normalized"]:
            for alpha in [.25, .5]:
                for beta in [0, .5, 1]:
                    configurations += grid(method, alpha, beta)
        seeds, recorded = DEVELOPMENT_SEEDS, FRAMES
    elif phase == "sensitivity":
        load_phase(root, "development")
        selected = read_json(root / "development-selection.json")
        if selected["identity"] != identity(binary):
            raise ValueError("Development selection belongs to different source/binary")
        references = selected["baseline"]
        for c in selected["finalists"]:
            for tl, tg in [(8, 4), (16, 8), (32, 16)]:
                configurations += grid(c["method"], c["alpha"], c["beta"], tl, tg)
        if not configurations:
            raise ValueError("No development finalists passed; do not run sensitivity or evaluation")
        seeds = DEVELOPMENT_SEEDS
    elif phase == "evaluation":
        selected = read_json(root / "frozen.json")
        if selected["identity"] != identity(binary):
            raise ValueError("Frozen manifest belongs to different source/binary")
        for source, checksum in selected["development_evidence_sha256"].items():
            if digest(root / source) != checksum:
                raise ValueError(f"Frozen development evidence changed: {source}")
        load_phase(root, "development")
        load_phase(root, "sensitivity")
        references = selected["baseline"]
        configurations = [entry["config"] for entry in selected["finalists"]]
        if not configurations:
            raise ValueError("No frozen finalists; evaluation is unnecessary")
        seeds, recorded = EVALUATION_SEEDS, FRAMES
    else:
        raise ValueError(phase)
    if recorded and inputs is None:
        raise ValueError("--inputs with all nine recorded stereo pairs is required for this phase")
    input_hashes = {}
    for frame in recorded:
        for side in ["left", "right"]:
            path = inputs / f"frame-{frame}-{side}.pgm"
            input_hashes[str(path)] = digest(path)
    jobs = []
    for c in configurations:
        for seed in seeds:
            case = f"synthetic-{seed}"
            save = (phase in ["smoke", "regression"] and seed == seeds[0]) or (phase == "evaluation" and seed == seeds[0])
            jobs.append(dict(id=config_id(c) + "/" + case, config=c, case=case, save=save))
        # Recorded coverage is descriptive. Sweep confidence on synthetic GT;
        # replay each initial cost at the default operating point on all nine
        # frames, rather than spending recorded runs calibrating accuracy.
        replay = recorded if phase != "development" or (c["cutoff"] == 120 and c["margin"] == 3) else []
        for frame in replay:
            case = f"frame-{frame}"
            save = frame == 70 and (phase != "development" or c["alpha"] == 0)
            jobs.append(dict(id=config_id(c) + "/" + case, config=c, case=case, save=save))
    return dict(version=1, phase=phase, identity=identity(binary), references=references,
                selection_sha256=digest(root / ("frozen.json" if phase == "evaluation" else "development-selection.json"))
                if selected is not None else None, input_sha256=input_hashes, jobs=jobs)


def command_for(binary, destination, job, plan, inputs):
    c = job["config"]
    command = [str(binary), "--output", str(destination), "--methods", c["method"],
               "--refinement", "both", "--baseline-controls", "--reuse-selection", "--extended-controls",
               "--composite-alpha", str(c["alpha"]), "--composite-beta", str(c["beta"]),
               "--luma-clip", str(c["luma_clip"]), "--gradient-clip", str(c["gradient_clip"]),
               "--cost-cutoff", str(c["cutoff"]), "--uniqueness-margin", str(c["margin"]),
               "--runs", "3" if job["case"].startswith("frame-") else "1", "--warmup", "1"]
    for arm, ref in plan["references"].items():
        command += [f"--reference-{arm}-cutoff", str(ref["cutoff"]),
                    f"--reference-{arm}-margin", str(ref["margin"])]
    if not job["save"]:
        command.append("--metrics-only")
    if job["case"].startswith("synthetic-"):
        command += ["--synthetic", "--synthetic-seed", job["case"].split("-")[1]]
    else:
        frame = job["case"].split("-")[1]
        command += ["--left", str(inputs / f"frame-{frame}-left.pgm"),
                    "--right", str(inputs / f"frame-{frame}-right.pgm")]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["smoke", "regression", "development", "sensitivity", "evaluation"], required=True)
    parser.add_argument("--resume", action="store_true", help="Skip verified completed jobs; never overwrite partial jobs")
    parser.add_argument("--plan-only", action="store_true", help="Write command plan without running the matcher")
    parser.add_argument("--compiler", default="g++", help="Compiler used to build --binary, recorded for provenance")
    args = parser.parse_args()
    binary, root = args.binary.resolve(), args.output.resolve()
    inputs = args.inputs.resolve() if args.inputs else None
    plan = make_plan(args.phase, root, inputs, binary)  # Validate freeze before creating evaluation files.
    phase_root = root / args.phase
    phase_root.mkdir(parents=True, exist_ok=True)
    plan_path = phase_root / "plan.json"
    if plan_path.exists():
        if not args.resume or read_json(plan_path) != plan:
            raise ValueError("Existing phase plan; use --resume with identical source, binary, inputs and selection")
    else:
        write_json(plan_path, plan)
        for source in SOURCES:
            snapshot = phase_root / "sources" / source
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT / source).read_bytes())
        compiler = subprocess.run([args.compiler, "--version"], capture_output=True, text=True, check=True)
        write_json(phase_root / "environment.json", dict(platform=platform.platform(), compiler=compiler.stdout,
                   omp_num_threads=8, binary=str(binary)))
    commands = [command_for(binary, phase_root / job["id"], job, plan, inputs) for job in plan["jobs"]]
    write_json(phase_root / "commands.json", commands)
    if args.plan_only:
        print(f"Planned {len(commands)} jobs in {phase_root}")
        return
    env = dict(os.environ, OMP_NUM_THREADS="8")
    for index, (job, command) in enumerate(zip(plan["jobs"], commands), 1):
        destination = phase_root / job["id"]
        complete = destination / "complete.json"
        if complete.exists() and args.resume:
            saved = read_json(complete)
            if saved["job"] != job or any(digest(destination / p) != h for p, h in saved["output_sha256"].items()):
                raise ValueError(f"Completed job changed: {destination}")
            continue
        if destination.exists():
            raise ValueError(f"Refusing to overwrite partial job {destination}; archive it before retrying")
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(commands)}] {job['id']}", flush=True)
        with destination.with_name(destination.name + ".log").open("x") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        checksums = {str(p.relative_to(destination)): digest(p) for p in sorted(destination.rglob("*")) if p.is_file()}
        write_json(complete, dict(job=job, command=command, output_sha256=checksums))
    print(f"Completed {len(commands)} jobs; analyze the {args.phase} phase before continuing.")


if __name__ == "__main__":
    main()
