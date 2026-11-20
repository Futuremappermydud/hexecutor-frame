#!/usr/bin/env python3
"""Summarize isolated comparison outputs; coverage never implies real accuracy."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    rows = []
    fields = None
    for path in sorted(root.glob("*/frame-*/metrics.csv")) + sorted(root.glob("*/synthetic-*/metrics.csv")):
        variant, case = path.parent.parent.name, path.parent.name
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            if fields is None:
                fields = reader.fieldnames
            elif fields != reader.fieldnames:
                raise ValueError(f"Different metric schema: {path}")
            rows.extend(dict(variant=variant, case=case, **row) for row in reader)
    if not rows:
        raise ValueError("No comparison metric files found")
    with (root / "combined-metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["variant", "case"] + fields)
        writer.writeheader()
        writer.writerows(rows)

    real = {}
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant and row["kind"] == "recorded"]
        methods = {}
        for method in sorted({row["method"] for row in selected}):
            matched = [row for row in selected if row["method"] == method]
            coverage = [float(row["supported_pct"]) for row in matched]
            timing = [float(row["cpu_wall_p50_ms"]) for row in matched]
            timing = [value for value in timing if math.isfinite(value)]
            methods[method] = dict(frames=len(matched), mean_supported_pct=statistics.mean(coverage),
                                   median_frame_cpu_p50_ms=statistics.median(timing) if timing else None)
        real[variant] = methods
    with (root / "summary.json").open("w") as file:
        json.dump(dict(real=real, real_ground_truth=None,
                       note="Real coverage/agreement are proxies. CPU timing is not GPU or live timing."),
                  file, indent=2, allow_nan=False)
        file.write("\n")

    lines = ["# Stereo matcher comparison results", "", "These are offline measurements on Frame. "
             "The live runtime was co-running. Recorded images have no independent depth ground truth; "
             "coverage and agreement alone do not establish accuracy.", "",
             "## Recorded inputs", "",
             "| Variant | Method/stage | Frames | Mean image support | Median frame CPU p50 |",
             "| --- | --- | ---: | ---: | ---: |"]
    for variant, methods in real.items():
        for method, result in methods.items():
            timing = result["median_frame_cpu_p50_ms"]
            lines.append(f"| {variant} | {method} | {result['frames']} | "
                         f"{result['mean_supported_pct']:.3f}% | "
                         f"{f'{timing:.1f} ms' if timing is not None else 'unmeasured'} |")
    lines += ["", "`*-speckle` uses unchanged production connected-component cleanup. "
              "The CPU reference timing includes allocations, census, matching, LR/range and neighborhood support; "
              "it excludes speckle cleanup. GPU production timings are in `inputs/inputs.csv` and measure "
              "host process time after one warmup with five samples per frame. Neither is display latency.", "",
              "## Controlled synthetic inputs", "",
              "Bad depth means accepted error >max(0.04 m, 5% true depth). Missing pixels remain in the "
              "evaluation denominator. Values below use the supported-depth stage, before speckle removal.", "",
              "| Variant | Seed | Scene | Method | Missing GT | Accepted disparity MAE | Accepted bad depth |",
              "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        if row["kind"] != "synthetic" or row["method"].endswith("-speckle"):
            continue
        if row["scene"] not in ("weak-texture-fractional", "textured-fractional", "step-silhouette-fractional"):
            continue
        lines.append(f"| {row['variant']} | {row['case'].split('-')[-1]} | {row['scene']} | {row['method']} | "
                     f"{float(row['missing_gt_pct']):.2f}% | {float(row['accepted_mae_disparity_px']):.3f} px | "
                     f"{float(row['accepted_bad_depth_pct']):.2f}% |")
    lines += ["", "Blank and periodic ambiguity checks, complete per-pixel outputs, boundary leakage and "
              "foreground missing/error metrics are retained in the per-case metrics and `combined-metrics.csv`. "
              "Synthetic pairs do not model camera synchronization, calibration errors, real noise or optics. "
              "Review per-case README files for the exact generator and parameters.", ""]
    (root / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
