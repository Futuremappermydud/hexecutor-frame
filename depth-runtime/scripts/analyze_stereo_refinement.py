#!/usr/bin/env python3
"""Diagnose saved matcher stages and subpixel fits; never rerun acceptance.

Uses only Python's standard library. Refinement errors use the original SGM
supported/GT mask (or its intersection with baseline), with no new LR, range,
support or speckle decisions. They are attribution diagnostics, not candidate
accuracy/coverage measurements or timing benchmarks.
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
from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        data = path.read_bytes()
        self.hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return data

    def floats(self, path, count):
        data = self.read(path)
        if len(data) != count * 4:
            raise ValueError(f"Unexpected float data size: {path}")
        values = array("f")
        values.frombytes(data)
        if sys.byteorder != "little":
            values.byteswap()
        if any(not math.isfinite(v) for v in values):
            raise ValueError(f"Nonfinite float data: {path}")
        return values

    def pgm(self, path):
        # The exporter writes a three-line P5 header without comments.
        magic, dimensions, maximum, pixels = self.read(path).split(b"\n", 3)
        width, height = map(int, dimensions.split())
        if magic != b"P5" or maximum != b"255" or len(pixels) != width * height:
            raise ValueError(f"Unexpected exported PGM: {path}")
        return width, height, pixels


def census(image, width, height):
    output = [0] * len(image)
    for y in range(height):
        for x in range(width):
            center = image[y * width + x]
            bits = bit = 0
            for dy in range(-3, 4):
                yy = min(height - 1, max(0, y + dy))
                for dx in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    xx = min(width - 1, max(0, x + dx))
                    if image[yy * width + xx] > center:
                        bits |= 1 << bit
                    bit += 1
            output[y * width + x] = bits
    return output


def unary(left, right, index, disparity, width):
    return sum((left[index + offset] ^ right[index + offset - disparity]).bit_count()
               for offset in (0, -2, 2, -2 * width, 2 * width))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stage_losses(root, inputs):
    rows = []
    for case in sorted((root / "default").glob("frame-*")):
        folder = case / "recorded"
        archived = list(csv.DictReader(inputs.read(case / "metrics.csv").decode().splitlines()))
        width, height, _ = inputs.pgm(folder / "left.pgm")
        count = width * height
        for method in ("baseline", "edge", "sgm"):
            left = inputs.floats(folder / f"{method}-left-disparity.f32", count)
            pre = inputs.floats(folder / f"{method}-pre-support-depth.f32", count)
            depth = inputs.floats(folder / f"{method}-depth.f32", count)
            speckle = inputs.floats(folder / f"{method}-speckle-depth.f32", count)
            rw, rh, rejected = inputs.pgm(folder / f"{method}-lr-rejection.pgm")
            if (rw, rh) != (width, height):
                raise ValueError("Rejection dimensions differ")
            counts = dict(left_match=sum(v > 0 for v in left),
                          lr_lost=sum(bool(v & 1) for v in rejected),
                          range_lost=sum(bool(v & 2) for v in rejected),
                          low_disparity_lost=sum(bool(v & 4) for v in rejected),
                          pre_support=sum(v > 0 for v in pre),
                          supported=sum(v > 0 for v in depth),
                          speckled=sum(v > 0 for v in speckle))
            if counts["left_match"] != (counts["lr_lost"] + counts["range_lost"] +
                                        counts["low_disparity_lost"] + counts["pre_support"]):
                raise ValueError("Filter stage counts do not partition accepted matches")
            if any((d > 0 and p <= 0) or (s > 0 and d <= 0)
                   for p, d, s in zip(pre, depth, speckle)):
                raise ValueError("Support or speckle stage adds pixels")
            counts["support_lost"] = counts["pre_support"] - counts["supported"]
            counts["speckle_lost"] = counts["supported"] - counts["speckled"]
            saved = next(row for row in archived if row["method"] == method)
            saved_speckle = next(row for row in archived if row["method"] == method + "-speckle")
            for name, expected in (("left_match", saved["left_match_pct"]),
                                   ("supported", saved["supported_pct"]),
                                   ("speckled", saved_speckle["supported_pct"])):
                if abs(counts[name] * 100 / count - float(expected)) > 2e-6:
                    raise ValueError("Stage counts do not reproduce archived metrics")
            rows.append(dict(frame=case.name, method=method, pixels=count,
                             **{k + "_pct": v * 100 / count for k, v in counts.items()}))
    return rows


def refinement(root, inputs):
    rows = []
    for seed in (1709, 991):
        for scene in ("textured-fractional", "weak-texture-fractional",
                      "step-silhouette-fractional"):
            reference = root / "default" / f"synthetic-{seed}" / scene
            width, height, left = inputs.pgm(reference / "left.pgm")
            rw, rh, right = inputs.pgm(reference / "right.pgm")
            if (rw, rh) != (width, height):
                raise ValueError("Stereo dimensions differ")
            count = width * height
            lc, rc = census(left, width, height), census(right, width, height)
            for variant in ("default", "p4-24", "p16-96"):
                first_row = len(rows)
                folder = root / variant / f"synthetic-{seed}" / scene
                vw, vh, variant_left = inputs.pgm(folder / "left.pgm")
                vrw, vrh, variant_right = inputs.pgm(folder / "right.pgm")
                if ((vw, vh) != (width, height) or (vrw, vrh) != (width, height) or
                        variant_left != left or variant_right != right):
                    raise ValueError("Penalty variants do not share identical inputs")
                tw, th, evaluated = inputs.pgm(folder / "truth-evaluated.pgm")
                if (tw, th) != (width, height):
                    raise ValueError("GT dimensions differ")
                truth = inputs.floats(folder / "truth-disparity.f32", count)
                depth = inputs.floats(folder / "sgm-depth.f32", count)
                disparity = inputs.floats(folder / "sgm-left-disparity.f32", count)
                integer = inputs.floats(folder / "sgm-left-integer-disparity.f32", count)
                raw_saved = inputs.floats(folder / "sgm-left-raw-cost.f32", count)
                baseline_depth = inputs.floats(reference / "baseline-depth.f32", count)
                baseline_d = inputs.floats(reference / "baseline-left-disparity.f32", count)
                raw_refined = list(disparity)
                used_raw = set()
                for i in range(count):
                    if not (evaluated[i] and depth[i] > 0):
                        continue
                    if truth[i] <= 0 or disparity[i] <= 0:
                        raise ValueError("Accepted GT sample lacks positive disparity")
                    d = int(integer[i])
                    if d != integer[i]:
                        raise ValueError("Noninteger winner")
                    center = unary(lc, rc, i, d, width)
                    if center != raw_saved[i]:
                        raise ValueError("Recomputed unary differs from saved selected unary")
                    max_d = min(95, i % width - 5)
                    if not 1 < d < max_d:
                        continue
                    before = unary(lc, rc, i, d - 1, width)
                    after = unary(lc, rc, i, d + 1, width)
                    curvature = before - 2 * center + after
                    # Only refine a genuine local raw minimum. Otherwise keep
                    # the archived regularized fit; never shift the integer basin.
                    if curvature > 0 and center <= min(before, after):
                        offset = (before - after) / (2 * curvature)
                        raw_refined[i] = d + max(-0.5, min(0.5, offset))
                        used_raw.add(i)
                blended = [(a + b) * 0.5 for a, b in zip(disparity, raw_refined)]
                agreed = [b if abs(a - b) <= 0.125 else a
                          for a, b in zip(disparity, raw_refined)]
                for mask_name in ("sgm-accepted", "baseline-sgm-common"):
                    indices = [i for i in range(count) if evaluated[i] and depth[i] > 0
                               and (mask_name == "sgm-accepted" or baseline_depth[i] > 0)]
                    if not indices:
                        raise ValueError("Empty diagnostic evaluation mask")
                    fits = [("sgm-regularized", disparity), ("sgm-integer", integer),
                            ("sgm-guarded-raw-fit", raw_refined),
                            ("sgm-half-raw-blend", blended),
                            ("sgm-raw-agreement-0.125", agreed)]
                    if mask_name == "baseline-sgm-common":
                        fits.append(("baseline", baseline_d))
                    for name, values in fits:
                        errors = [abs(values[i] - truth[i]) for i in indices]
                        bad = sum(abs(24.282608 / values[i] - 24.282608 / truth[i]) >
                                  max(0.04, 0.05 * 24.282608 / truth[i]) for i in indices)
                        rows.append(dict(variant=variant, seed=seed, scene=scene,
                                         fixed_mask=mask_name, fit=name, samples=len(indices),
                                         mae_disparity_px=statistics.mean(errors),
                                         bad_depth_pct=bad * 100 / len(indices),
                                         raw_fit_eligible_pct=sum(i in used_raw for i in indices) *
                                         100 / len(indices),
                                         winner_disagrees_with_truth_round_pct=sum(
                                             integer[i] != round(truth[i]) for i in indices) *
                                             100 / len(indices)))
                # Check the attribution mask and original error against the
                # archived evaluation before interpreting alternative fits.
                metrics_path = folder.parent / "metrics.csv"
                metrics_text = inputs.read(metrics_path).decode()
                original = next(row for row in csv.DictReader(metrics_text.splitlines())
                                if row["scene"] == scene and row["method"] == "sgm")
                reproduced = next(row for row in rows[first_row:]
                                  if row["fixed_mask"] == "sgm-accepted" and
                                  row["fit"] == "sgm-regularized")
                if (reproduced["samples"] != int(original["accepted_gt_pixels"]) or
                        abs(reproduced["mae_disparity_px"] -
                            float(original["accepted_mae_disparity_px"])) > 2e-6 or
                        abs(reproduced["bad_depth_pct"] -
                            float(original["accepted_bad_depth_pct"])) > 2e-6):
                    raise ValueError("Original SGM diagnostic does not reproduce saved metrics")
            print(f"Analyzed seed {seed}, {scene}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = Inputs()
    stages = stage_losses(args.directory, inputs)
    if len(stages) != 27:
        raise ValueError("Expected nine recorded frames with three methods each")
    fits = refinement(args.directory, inputs)
    write_csv(args.output / "stage-losses.csv", stages)
    write_csv(args.output / "fixed-mask-refinement.csv", fits)
    means = {}
    for method in ("baseline", "edge", "sgm"):
        selected = [row for row in stages if row["method"] == method]
        means[method] = {key: statistics.mean(row[key] for row in selected)
                         for key in selected[0] if key.endswith("_pct")}
    inputs.read((PATHS.repo / 'depth-runtime/scripts/analyze_stereo_refinement.py'))
    inputs.read(args.directory / "manifest.json")
    (args.output / "summary.json").write_text(json.dumps(dict(
        scope="Offline saved-output attribution; fixed masks, no acceptance rerun or hardware timing",
        recorded_ground_truth=None, mean_recorded_stages=means,
        input_sha256=inputs.hashes), indent=2, allow_nan=False) + "\n")
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
