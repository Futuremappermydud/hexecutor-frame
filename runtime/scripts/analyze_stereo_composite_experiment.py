#!/usr/bin/env python3
"""Summarize complete sweeps, select development finalists, and freeze evaluation."""
import argparse
import csv
import html
import math
from pathlib import Path
import statistics

from stereo_composite_common import (AMBIGUOUS, ARMS, NOISY, SCENES,
    config, config_id, digest, load_phase, number, pooled, read_json, write_json)

FIELDS = {"missing_gt_pct": "evaluated_gt_pixels",
          "accepted_bad_depth_pct": "accepted_gt_pixels",
          "accepted_bad_of_all_gt_pct": "evaluated_gt_pixels",
          "background_boundary_leakage_of_all_pct": "evaluated_boundary_pixels",
          "accepted_mae_disparity_px": "accepted_gt_pixels",
          "foreground_bad_of_all_pct": "evaluated_foreground_pixels",
          "foreground_missing_pct": "evaluated_foreground_pixels"}


def summarize(records):
    return {scene: dict({field: pooled([r for r in records if r["scene"] == scene], field, weight)
                         for field, weight in FIELDS.items()},
                        supported_pct=pooled([r for r in records if r["scene"] == scene], "supported_pct", "pixels"))
            for scene in SCENES}


def compare_common(records, reference, method):
    return {scene: {field: pooled([r for r in records if r["scene"] == scene and
                                  r["reference"] == reference and r["method"] == method], field, "common_gt_pixels")
                    for field in ["common_reference_mae_px", "common_candidate_mae_px",
                                  "common_reference_bad_pct", "common_candidate_bad_pct"]}
            for scene in SCENES}


def gate(candidate, reference, common, require_gain=True):
    """Conservative per-scene gates. None is never interpreted as zero error."""
    reasons = []
    for scene in SCENES:
        c, r = candidate[scene], reference[scene]
        if scene in AMBIGUOUS:
            if c["supported_pct"] is None or c["supported_pct"] != 0:
                reasons.append(f"{scene}: ambiguity support")
            continue
        if scene == "unrelated-pair":
            if c["supported_pct"] is None or r["supported_pct"] is None or c["supported_pct"] > r["supported_pct"] + 1e-7:
                reasons.append(f"{scene}: false support")
            continue
        for field in ["missing_gt_pct", "accepted_bad_depth_pct", "accepted_bad_of_all_gt_pct",
                      "background_boundary_leakage_of_all_pct"]:
            cv, rv = c[field], r[field]
            if rv is None:
                # No boundary exists on planes; no accepted baseline pixels in
                # some strict operating points. Missing/error gates still apply.
                continue
            if cv is None or cv > rv + 1e-7:
                reasons.append(f"{scene}: {field}")
    if require_gain:
        for scene in NOISY:
            values = common[scene]
            a, b = values["common_candidate_mae_px"], values["common_reference_mae_px"]
            if a is None or b is None or not a < b:
                reasons.append(f"{scene}: no common-mask MAE gain")
    return reasons


def ranking(c, summary, common):
    scores = [common[s]["common_candidate_mae_px"] for s in NOISY]
    if any(v is None for v in scores):
        score = math.inf
    else:
        score = statistics.mean(scores)
    coverage = statistics.mean(100 - summary[s]["missing_gt_pct"] for s in NOISY)
    return (score, -coverage, c["cutoff"], -c["margin"], c["alpha"], c["beta"], c["luma_clip"], c["gradient_clip"])


def analyze(plan, jobs):
    grouped = {}
    for job in jobs:
        if not job["job"]["case"].startswith("synthetic-"):
            continue
        key = config_id(job["job"]["config"])
        grouped.setdefault(key, []).append(job)
    expected_seeds = {j["case"] for j in plan["jobs"] if j["case"].startswith("synthetic-")}
    reports = {}
    for key, cases in grouped.items():
        if {j["job"]["case"] for j in cases} != expected_seeds:
            raise ValueError(f"Incomplete synthetic split: {key}")
        c = cases[0]["job"]["config"]
        metrics = [r for case in cases for r in case["metrics"]]
        comparisons = [r for case in cases for r in case["comparison"]]
        arms = {}
        for arm in ARMS:
            method = c["method"] + "-" + arm
            candidate = summarize([r for r in metrics if r["method"] == method])
            default = summarize([r for r in metrics if r["method"] == "default-" + arm])
            calibrated = summarize([r for r in metrics if r["method"] == "calibrated-" + arm])
            common = compare_common(comparisons, "default-" + arm, method)
            calibrated_common = compare_common(comparisons, "calibrated-" + arm, method)
            baseline = c["method"] == "baseline"
            reasons = ["default/" + r for r in gate(candidate, default, common, not baseline)]
            # First sweep preselects against default; sensitivity/evaluation
            # additionally compares to each independently calibrated baseline.
            if plan["phase"] in ["sensitivity", "evaluation"]:
                reasons += ["calibrated/" + r for r in gate(candidate, calibrated, calibrated_common)]
            cleaned_method = method + "-speckle"
            cleaned = summarize([r for r in metrics if r["method"] == cleaned_method])
            cleaned_default = summarize([r for r in metrics if r["method"] == "default-" + arm + "-speckle"])
            cleaned_calibrated = summarize([r for r in metrics if r["method"] == "calibrated-" + arm + "-speckle"])
            cleaned_common = compare_common(comparisons, "default-" + arm + "-speckle", cleaned_method)
            reasons += ["speckle/default/" + r for r in gate(cleaned, cleaned_default, cleaned_common, not baseline)]
            if plan["phase"] in ["sensitivity", "evaluation"]:
                cc = compare_common(comparisons, "calibrated-" + arm + "-speckle", cleaned_method)
                reasons += ["speckle/calibrated/" + r for r in gate(cleaned, cleaned_calibrated, cc)]
            regressions = []
            for case in cases:
                own = summarize([r for r in case["metrics"] if r["method"] == method])
                ref = summarize([r for r in case["metrics"] if r["method"] == "default-" + arm])
                regressions += [case["job"]["case"] + "/" + r for r in gate(own, ref, common, False)]
                own_clean = summarize([r for r in case["metrics"] if r["method"] == cleaned_method])
                ref_clean = summarize([r for r in case["metrics"] if r["method"] == "default-" + arm + "-speckle"])
                regressions += [case["job"]["case"] + "/speckle/" + r for r in gate(own_clean, ref_clean, cleaned_common, False)]
            arms[arm] = dict(scenes=candidate, default=default, calibrated=calibrated, common=common,
                             after_cleanup=cleaned, after_cleanup_common=cleaned_common,
                             calibrated_common=calibrated_common, passed=not reasons, failure_reasons=reasons,
                             individual_seed_regressions=regressions)
        reports[key] = dict(config=c, arms=arms)
    return reports


def select_development(reports, plan):
    baseline = {}
    for arm in ARMS:
        eligible = [v for v in reports.values() if v["config"]["method"] == "baseline" and v["arms"][arm]["passed"]]
        if not eligible:
            raise ValueError(f"No valid baseline control for {arm}")
        winner = min(eligible, key=lambda v: ranking(v["config"], v["arms"][arm]["scenes"], v["arms"][arm]["common"]))
        baseline[arm] = winner["config"]
    finalists = []
    for method in ["composite-raw", "composite-normalized"]:
        eligible = [v for v in reports.values() if v["config"]["method"] == method and
                    v["config"]["alpha"] > 0 and v["arms"]["integer"]["passed"]]
        if eligible:
            best = min(eligible, key=lambda v: ranking(v["config"], v["arms"]["integer"]["scenes"], v["arms"]["integer"]["common"]))
            finalists.append(best["config"])
    return dict(identity=plan["identity"], baseline=baseline, finalists=finalists,
                status="sensitivity-required" if finalists else "failed-quality-gates")


def freeze(reports, plan, root):
    if (root / "evaluation").exists() or (root / "frozen.json").exists():
        raise ValueError("Refusing to replace a frozen configuration or inspect new settings after evaluation")
    finalists = []
    for method in ["composite-raw", "composite-normalized"]:
        # Selection priority remains integer-only. Fractional fitting may fail
        # independently, in which case freeze only the integer mode.
        eligible = [v for v in reports.values() if v["config"]["method"] == method and v["arms"]["integer"]["passed"]]
        if eligible:
            best = min(eligible, key=lambda v: ranking(v["config"], v["arms"]["integer"]["scenes"], v["arms"]["integer"]["common"]))
            finalists.append(dict(config=best["config"], modes=[a for a in ARMS if best["arms"][a]["passed"]]))
    evidence = [root / "development-selection.json", root / "development/plan.json", root / "sensitivity/plan.json"]
    evidence += list((root / "development").rglob("complete.json")) + list((root / "sensitivity").rglob("complete.json"))
    result = dict(version=1, identity=plan["identity"], baseline=plan["references"], finalists=finalists,
                  status="frozen-for-evaluation" if finalists else "failed-quality-gates",
                  development_evidence_sha256={str(p.relative_to(root)): digest(p) for p in evidence})
    write_json(root / "frozen.json", result)
    return result


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_curves(path, reports):
    # Standalone SVG: each point is a complete rerun, not a rank on a frozen mask.
    colors = {"baseline": "#333333", "composite-raw": "#0072b2", "composite-normalized": "#d55e00"}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="570" viewBox="0 0 1000 570">',
             '<rect width="1000" height="570" fill="white"/>',
             '<text x="30" y="25" font-family="sans-serif">Noise ±3: bad accepted depth versus retained GT coverage (complete reruns)</text>']
    for index, arm in enumerate(ARMS):
        x0, y0 = 70 + index * 490, 480
        points = []
        for key, v in reports.items():
            s = v["arms"][arm]["scenes"]["independent-noise-3"]
            if s["missing_gt_pct"] is not None and s["accepted_bad_depth_pct"] is not None:
                points.append((key, v["config"]["method"], 100 - s["missing_gt_pct"], s["accepted_bad_depth_pct"]))
        ymax = max([p[3] for p in points] + [1.])
        parts += [f'<path d="M{x0} 60 V{y0} H{x0+380}" fill="none" stroke="black"/>',
                  f'<text x="{x0}" y="50" font-family="sans-serif">{arm}; y max {ymax:.3f}%</text>']
        for key, method, coverage, error in points:
            parts.append(f'<circle cx="{x0+3.8*coverage:.3f}" cy="{y0-420*error/ymax:.3f}" r="3" '
                         f'fill="{colors[method]}"><title>{html.escape(key)}: retained {coverage:.3f}%, bad {error:.4f}%</title></circle>')
        parts.append(f'<text x="{x0}" y="505" font-family="sans-serif">Retained GT: 0% → 100%</text>')
    parts.append('<text x="30" y="545" font-family="sans-serif">Black: census baseline; blue: raw; orange: normalized. Hover for configuration.</text></svg>')
    path.write_text("\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--phase", choices=["smoke", "regression", "development", "sensitivity", "evaluation"], required=True)
    parser.add_argument("--freeze", action="store_true", help="Only after the complete sensitivity phase")
    parser.add_argument("--archive", type=Path, help="Original matcher archive for saved default baseline parity")
    args = parser.parse_args()
    if args.freeze and args.phase != "sensitivity":
        parser.error("--freeze is only valid for sensitivity")
    root = args.directory.resolve()
    plan, jobs = load_phase(root, args.phase)
    reports = analyze(plan, jobs)
    checks = {}
    parity = []
    for case in jobs:
        if case["job"]["config"]["alpha"] != 0 or case["job"]["config"]["method"] == "baseline":
            continue
        folder = root / args.phase / case["job"]["id"]
        for file in folder.rglob("default-*"):
            if file.suffix not in [".f32", ".pgm"] or "-vs-" in file.name:
                continue
            equivalent = file.with_name(file.name.replace("default-", case["job"]["config"]["method"] + "-", 1))
            if equivalent.exists():
                if digest(file) != digest(equivalent):
                    raise ValueError(f"Zero-weight full-stage parity failed: {equivalent}")
                parity.append(str(equivalent.relative_to(root)))
    checks["saved_zero_weight_parity_files"] = parity
    if args.archive:
        archive_matches = []
        for case in jobs:
            if case["job"]["case"] != "frame-70" or not case["job"]["save"]:
                continue
            folder = root / args.phase / case["job"]["id"] / "recorded"
            for old in sorted((args.archive / "default/frame-70/recorded").glob("baseline-*")):
                new = folder / old.name.replace("baseline-", "default-parabola-", 1)
                if new.exists():
                    if digest(new) != digest(old):
                        raise ValueError(f"Archived baseline stage parity failed: {new}")
                    archive_matches.append(str(new.relative_to(root)))
        if len(archive_matches) < 18:
            raise ValueError("Too few archived baseline stages checked")
        checks["archived_baseline_stage_files"] = archive_matches
    selection = None
    if args.phase == "development":
        selection = select_development(reports, plan)
        path = root / "development-selection.json"
        if path.exists() and read_json(path) != selection:
            raise ValueError("Refusing to replace a changed development selection")
        if not path.exists():
            write_json(path, selection)
    if args.freeze:
        selection = freeze(reports, plan, root)
    if args.phase == "evaluation":
        frozen = read_json(root / "frozen.json")
        checks["evaluated_frozen_modes"] = {config_id(v["config"]):
            {arm: reports[config_id(v["config"])]["arms"][arm]["passed"] for arm in v["modes"]}
            for v in frozen["finalists"]}
    output = root / args.phase
    summary = dict(phase=args.phase, configurations=reports, checks=checks, selection=selection)
    write_json(output / "summary.json", summary)
    for kind in ["metrics", "comparison", "fixed-mask"]:
        records = [dict(config_id=config_id(job["job"]["config"]), case=job["job"]["case"], **row)
                   for job in jobs for row in job[kind]]
        write_csv(output / ("combined-" + kind + ".csv"), records)
    plot_curves(output / "confidence-curves.svg", reports)
    lines = ["# Composite unary results", "", f"Phase: {args.phase}. Offline CPU replay; recorded coverage has no depth GT.", "",
             "Each row pools seeds per scene. Integer and parabola gates are independent. Fixed-mask rows are attribution only.", "",
             "| Configuration | Integer gate | Parabola gate |", "| --- | --- | --- |"]
    for key, value in reports.items():
        lines.append(f"| {key} | {'pass' if value['arms']['integer']['passed'] else 'fail'} | "
                     f"{'pass' if value['arms']['parabola']['passed'] else 'fail'} |")
    if selection:
        lines += ["", "Selection status: " + selection["status"] + "."]
    lines += ["", "Per-scene values, individual-seed regressions and rejection reasons: [summary.json](summary.json).",
              "Complete rerun confidence points: [confidence-curves.svg](confidence-curves.svg).",
              "Own/common integer-mask refinement diagnostics: [combined-fixed-mask.csv](combined-fixed-mask.csv).", ""]
    (output / "results.md").write_text("\n".join(lines))
    print(f"Analyzed {len(jobs)} verified jobs, {len(reports)} configurations; "
          f"{sum(v['arms']['integer']['passed'] for v in reports.values())} integer gates pass.")
    if selection:
        print(selection["status"])


if __name__ == "__main__":
    main()
