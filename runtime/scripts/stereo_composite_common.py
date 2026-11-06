"""Shared, standard-library-only protocol definitions for the offline trial."""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "depth-model/python"),
                             str(_source_root / "depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import csv
import hashlib
import json
import math
from pathlib import Path

DEVELOPMENT_SEEDS = [1709, 991, 56039, 65537]
EVALUATION_SEEDS = [104729, 130363, 15485863, 32452843]
REGRESSION_SEEDS = [4201, 7877, 12011, 32003]
FRAMES = [20, 40, 70, 100, 140, 180, 210, 240, 270]
ARMS = ["integer", "parabola"]
SCENES = ["textured-integer", "textured-fractional", "weak-texture-fractional",
          "step-silhouette-fractional", "blank-ambiguous", "repetitive-ambiguous",
          "repetitive-alias-ambiguous", "independent-noise-1", "independent-noise-3",
          "independent-noise-6", "gain-bias", "gain-bias-noise", "clipped-gain",
          "vertical-half-pixel", "vertical-one-pixel", "thin-silhouette", "unrelated-pair"]
AMBIGUOUS = {"blank-ambiguous", "repetitive-ambiguous", "repetitive-alias-ambiguous"}
NOISY = ["independent-noise-3", "independent-noise-6"]
ROOT = (PATHS.repo / 'runtime')
SOURCES = ["include/stereo_matcher_experiment.hpp", "include/depth_speckle_filter.hpp",
           "tools/compare_stereo_matchers.cpp", "tools/test_stereo_matcher_experiment.cpp",
           "tools/test_stereo_composite_workflow.py",
           "scripts/run_stereo_composite_experiment.py", "scripts/analyze_stereo_composite_experiment.py",
           "scripts/stereo_composite_common.py", "docs/stereo_composite_unary_plan.md", "Makefile"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(binary):
    return {"binary_sha256": digest(binary), "source_sha256": {p: digest(ROOT / p) for p in SOURCES}}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def pooled(records, field, weight):
    values = [(number(r[field]), float(r[weight])) for r in records]
    values = [(v, w) for v, w in values if v is not None and w > 0]
    total = sum(w for _, w in values)
    return sum(v * w for v, w in values) / total if total else None


def config(method="baseline", alpha=0, beta=0.5, luma_clip=16, gradient_clip=8, cutoff=120, margin=3):
    return dict(method=method, alpha=alpha, beta=beta, luma_clip=luma_clip,
                gradient_clip=gradient_clip, cutoff=cutoff, margin=margin)


def config_id(c):
    return (f"{c['method']}-a{c['alpha']:g}-b{c['beta']:g}-l{c['luma_clip']:g}"
            f"-g{c['gradient_clip']:g}-c{c['cutoff']}-m{c['margin']}")


def grid(method, alpha=0, beta=.5, luma_clip=16, gradient_clip=8):
    return [config(method, alpha, beta, luma_clip, gradient_clip, cutoff, margin)
            for cutoff in [60, 90, 120] for margin in [0, 3, 6, 12]]


def load_phase(root, phase):
    """Refuse partial/tampered runs instead of declaring gates from a subset."""
    directory = Path(root) / phase
    plan = read_json(directory / "plan.json")
    result = []
    for job in plan["jobs"]:
        folder = directory / job["id"]
        complete = read_json(folder / "complete.json")
        if complete["job"] != job:
            raise ValueError(f"Completion manifest mismatch: {folder}")
        for file, checksum in complete["output_sha256"].items():
            if digest(folder / file) != checksum:
                raise ValueError(f"Output hash mismatch: {folder / file}")
        loaded = {name: rows(folder / (name + ".csv")) for name in ["metrics", "comparison", "fixed-mask"]}
        expected = set(SCENES if job["case"].startswith("synthetic-") else ["recorded"])
        for arm in ARMS:
            for method in [job["config"]["method"], "default", "calibrated"]:
                for suffix in ["", "-speckle"]:
                    found = [r["scene"] for r in loaded["metrics"] if r["method"] == method + "-" + arm + suffix]
                    if len(found) != len(expected) or set(found) != expected:
                        raise ValueError(f"Missing/duplicate scenes for {method}/{arm}{suffix}: {folder}")
        result.append(dict(job=job, **loaded))
    return plan, result
