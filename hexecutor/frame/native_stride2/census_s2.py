#!/usr/bin/env python3
"""Census actual QDRX stride-2 Xwb commands for the native S2 experiment."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FRAME = HERE.parent
ROOT = FRAME.parent
sys.path.insert(0, str(FRAME))
from resident_image import unpack  # noqa: E402

VARIANT = "ConvLayer_s2.opt@CB*2.Xwb.Fi"


def model(path):
    graph = unpack(path.read_bytes())
    tensors = {t["id"]: t for t in graph["tensors"]}
    result = []
    for c in graph["commands"]:
        if c["variant"] != VARIANT:
            continue
        x, w, fi = (tensors[i] for i in c["inputs"][:3])
        y = tensors[c["outputs"][0]]
        result.append({
            "sequence": c["sequence"],
            "input_dims": x["shape"]["dimensions"],
            "input_max": x["shape"]["max_dimensions"],
            "input_padding": x["shape"]["padding"],
            "input_slice": x["storage"]["slice_offset"],
            "input_pages": len(x["storage"]["blocks"]),
            "output_dims": y["shape"]["dimensions"],
            "output_max": y["shape"]["max_dimensions"],
            "output_padding": y["shape"]["padding"],
            "output_slice": y["storage"]["slice_offset"],
            "output_pages": len(y["storage"]["blocks"]),
            "weights_dims": w["shape"]["dimensions"],
            "weight_class": w["class_name"],
            "fi_dims": fi["shape"]["dimensions"],
        })
    return result


def main():
    cases = {
        "primary": model(FRAME / "build/resident-primary.qdrx"),
        "alternate": model(FRAME / "build/resident-alternate.qdrx"),
    }
    report = {"variant": VARIANT, "models": {}}
    for name, rows in cases.items():
        groups = {}
        for r in rows:
            key = (r["weights_dims"][2], r["weights_dims"][3], r["input_dims"][3])
            groups[str(key)] = groups.get(str(key), 0) + 1
        report["models"][name] = {"count": len(rows), "weight_ci_co_input_ci_counts": groups,
                                  "sequences": [r["sequence"] for r in rows], "cases": rows}
    out = HERE / "s2_census.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {"count": item["count"], "groups": item["weight_ci_co_input_ci_counts"]}
                      for name, item in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
