#!/usr/bin/env python3
"""Run the opt-in masked-census ablation, without touching the live runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["development", "evaluation"], required=True)
    parser.add_argument("--deadband", type=int, default=2)
    parser.add_argument("--cost-cutoff", type=int, default=120)
    parser.add_argument("--margin", type=int, default=3)
    args = parser.parse_args()
    binary = args.binary.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="8")
    commands = []

    def run(variant, case, flags, cutoff=120, margin=3, deadband=2, save=False):
        destination = root / variant / case
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite {destination}")
        command = [str(binary), "--output", str(destination), "--methods", "baseline,noise",
                   "--noise-deadband", str(deadband), "--min-information", "120",
                   "--cost-cutoff", str(cutoff), "--uniqueness-margin", str(margin),
                   "--runs", "3" if case.startswith("frame-") else "1", "--warmup", "1"]
        if not save:
            command.append("--metrics-only")
        command += flags
        print(f"Running {variant}/{case}", flush=True)
        log = root / f"{variant}-{case}.log"
        with log.open("w") as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        commands.append(command)
        (root / f"commands-{args.phase}.json").write_text(json.dumps(commands, indent=2) + "\n")

    if args.phase == "development":
        for deadband in [0, 1, 2, 4]:
            for seed in [1709, 991]:
                run(f"dev-tau{deadband}", f"synthetic-{seed}",
                    ["--synthetic", "--extended-controls", "--synthetic-seed", str(seed)], deadband=deadband)
        # Full bidirectional/filter/support reruns: these are executable operating
        # points, not confidence ranks on a previously frozen acceptance mask.
        for cutoff in [60, 90, 120]:
            for margin in [0, 3, 6, 12]:
                run(f"calibration-cut{cutoff}-gap{margin}", "synthetic-1709",
                    ["--synthetic", "--extended-controls", "--synthetic-seed", "1709"], cutoff, margin)
    else:
        variant = f"eval-tau{args.deadband}-cut{args.cost_cutoff}-gap{args.margin}"
        for seed in [4201, 7877, 12011, 32003]:
            run(variant, f"synthetic-{seed}",
                ["--synthetic", "--extended-controls", "--synthetic-seed", str(seed)],
                args.cost_cutoff, args.margin, args.deadband, save=seed == 4201)
        hashes = {}
        for frame in [20, 40, 70, 100, 140, 180, 210, 240, 270]:
            paths = [args.inputs / f"frame-{frame}-{side}.pgm" for side in ["left", "right"]]
            for path in paths:
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            run(variant, f"frame-{frame}", ["--left", str(paths[0]), "--right", str(paths[1])],
                args.cost_cutoff, args.margin, args.deadband, save=frame == 70)
        (root / "input-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (root / f"binary-hash-{args.phase}.txt").write_text(hashlib.sha256(binary.read_bytes()).hexdigest() + "\n")


if __name__ == "__main__":
    main()
