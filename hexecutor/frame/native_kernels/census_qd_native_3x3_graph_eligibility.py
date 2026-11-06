#!/usr/bin/env python3
"""First-reason census for the current bounded native 3x3 graph bridge."""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "frame"), str(ROOT / "tools")]
from resident_image import unpack

MODELS = ("primary", "alternate")


def classify(cmd, tensors):
    if cmd.get("opcode") != 6 or "ConvLayer_s1" not in cmd.get("variant", ""):
        return None, None
    if len(cmd.get("inputs", ())) < 3 or len(cmd.get("outputs", ())) != 1:
        return "operation_arity", None
    x, w, fi = (tensors[i] for i in cmd["inputs"][:3])
    y = tensors[cmd["outputs"][0]]
    xd, xm, xp = x["shape"]["dimensions"], x["shape"]["max_dimensions"], x["shape"]["padding"]
    yd, ym, yp = y["shape"]["dimensions"], y["shape"]["max_dimensions"], y["shape"]["padding"]
    wd, wm = w["shape"]["dimensions"], w["shape"]["max_dimensions"]
    fd, fm = fi["shape"]["dimensions"], fi["shape"]["max_dimensions"]
    checks = (
        (x["class_name"] == y["class_name"] == "CB" and x["dtype"] == y["dtype"] == "uint8", "activation_class_dtype"),
        (w["class_name"] == "Xwb" and w["dtype"] == "int8", "not_Xwb_i8"),
        (fi["class_name"] == "Fi" and fi["dtype"] == "int32", "not_Fi_i32"),
        (xd[0] == yd[0] == xm[0] == ym[0] == 1, "batch_geometry"),
        (wd[:2] == wm[:2] == [3, 3], "kernel_not_3x3"),
        (len(wm) == 4 and 32 <= wm[2] <= 256 and wm[2] % 32 == 0, "CI_outside_32_256"),
        (len(wm) == 4 and 32 <= wm[3] <= 128 and wm[3] % 32 == 0, "physical_CO_outside_32_128"),
        (wd[2] == wm[2] and 0 < wd[3] <= wm[3], "weight_logical_dims"),
        (xd[3] == xm[3] == wm[2] and ym[3] == wm[3] and 0 < yd[3] <= wm[3], "activation_channels"),
        (xd[1] == 10 and yd[1] == 8 and xm[1] == 24 and ym[1] == 8, "height_profile"),
        (xp == [0, 7, 7, 0] and yp == [0, 0, 0, 0], "padding_profile"),
        (xd[1] == yd[1] + 2 and xd[2] == yd[2] + 2, "stride1_extent"),
        (xm[2] <= 272 and ym[2] <= 256 and xm[2] % 8 == ym[2] % 8 == 0 and yd[2] % 8 == 0, "width_profile"),
        (x.get("quantization", {}).get("zero_point", 0) == 0, "input_zero_point"),
        (fm == [1, wm[3] // 32, 1, 64] and fd == fm, "Fi_row_geometry"),
        (w["storage"].get("direct") is not None and fi["storage"].get("direct") is not None, "not_direct_operands"),
    )
    for ok, reason in checks:
        if not ok:
            return reason, None
    if yd[3] != wm[3] and not (
        wm[2] == wm[3] == 32 and yd[3] == 2 and xd[1:3] == [10, 258] and
        xm[2] == 272 and yd[1:3] == [8, 256] and ym[1:3] == [8, 256]
    ):
        return "unproven_physical_output_tail", None
    icb, ocb = wm[2] // 32, wm[3] // 32
    ipages = (xm[1] // 8) * (xm[2] // 8) * icb
    opages = (ym[1] // 8) * (ym[2] // 8) * ocb
    xblocks, yblocks = x["storage"].get("blocks", []), y["storage"].get("blocks", [])
    xslice, yslice = x["storage"].get("slice_offset", 0), y["storage"].get("slice_offset", 0)
    if ipages > 256 or opages > 256:
        return "adapter_table_capacity", None
    if xslice % icb:
        return "input_slice_alignment", None
    if xslice + ipages > len(xblocks):
        return "input_block_table_span", None
    if yslice + opages > len(yblocks):
        return "output_block_table_span", None
    pool_bytes = graph_pool_bytes
    def active_refs(t, blocks, start, pages):
        refs=[]
        for b in blocks[start:start+pages]:
            pool, off=b.get("pool", -1), b.get("offset", -1)
            if pool not in pool_bytes or off < 0 or off % 2048 or off + 2048 > pool_bytes[pool]:
                return None
            refs.append((pool,off))
        return refs
    xr=active_refs(x,xblocks,xslice,ipages);yr=active_refs(y,yblocks,yslice,opages)
    if xr is None or yr is None:
        return "adapter_pool_bounds", None
    if len(set(yr)) != len(yr):
        return "adapter_output_alias", None
    ws=w["storage"].get("direct",{});fs=fi["storage"].get("direct",{})
    wbytes=w["element_bytes"]
    for d in wm:wbytes*=d
    fbytes=fi["element_bytes"]
    for d in fm:fbytes*=d
    if ws.get("offset",-1)%128 or fs.get("offset",-1)%128:
        return "direct_operand_alignment", None
    for direct,nbytes in ((ws,wbytes),(fs,fbytes)):
        pool,off=direct.get("pool",-1),direct.get("offset",-1)
        if pool not in pool_bytes or off<0 or off+nbytes>pool_bytes[pool]:
            return "direct_operand_pool_bounds", None
    result={"sequence":cmd["sequence"],"CI":wm[2],"CO":wm[3],"logical_CO":yd[3],
            "input_maxW":xm[2],"output_maxW":ym[2],"input_pages":ipages,"output_pages":opages,
            "input_dims_below_max":xd[1:3] != xm[1:3],"output_spatial_crop":yd[1:3] != ym[1:3],
            "input_slice":xslice,"output_slice":yslice}
    if yslice % ocb:
        return "output_slice_alignment", result
    return "eligible_adapter_metadata", result


def main():
    global graph_pool_bytes
    report = {}
    for model in MODELS:
        path = ROOT / f"frame/build/resident-{model}.qdrx"
        if not path.exists():
            report[model] = {"missing": str(path.relative_to(ROOT))}
            continue
        graph = unpack(path.read_bytes())
        graph_pool_bytes = {int(k): int(v) for k, v in graph["pool_bytes"].items() if k.isdigit()}
        tensors = {t["id"]: t for t in graph["tensors"]}
        reasons = collections.Counter()
        reject_profiles = collections.defaultdict(collections.Counter)
        eligible = []
        y_slice_candidate=[]
        considered = 0
        total = 0
        for cmd in graph["commands"]:
            reason, row = classify(cmd, tensors)
            if reason is None:
                continue
            considered += 1
            reasons[reason] += 1
            if reason == "height_profile":
                x, w, fi = (tensors[i] for i in cmd["inputs"][:3])
                y = tensors[cmd["outputs"][0]]
                xd=x["shape"]["dimensions"];xm=x["shape"]["max_dimensions"]
                yd=y["shape"]["dimensions"];ym=y["shape"]["max_dimensions"]
                key=f"X{xd[1]}x{xd[2]}/max{xm[1]}x{xm[2]} -> Y{yd[1]}x{yd[2]}/max{ym[1]}x{ym[2]}"
                reject_profiles[reason][key]+=1
            if row:
                if reason=="output_slice_alignment":y_slice_candidate.append(row)
                else:eligible.append(row)
        total = len(graph["commands"])
        profile_counts=collections.Counter((r["CI"],r["CO"],r["logical_CO"],r["input_maxW"],r["output_maxW"]) for r in eligible)
        candidate_profile_counts=collections.Counter((r["CI"],r["CO"],r["logical_CO"],r["input_maxW"],r["output_maxW"]) for r in y_slice_candidate)
        report[model] = {"image": str(path.relative_to(ROOT)), "commands": total,
                         "s1_conv_commands_considered": considered,
                         "eligible_adapter_metadata_count": len(eligible),
                         "candidate_count_if_only_Y_slice_modulo_guard_is_removed":len(eligible)+len(y_slice_candidate),
                         "Y_slice_only_candidate_count":len(y_slice_candidate),
                         "Y_slice_only_candidate_profiles":{"/".join(map(str,k)):v for k,v in sorted(candidate_profile_counts.items())},
                         "first_rejection_reason_counts": dict(sorted(reasons.items())),
                         "eligible_profile_counts": {"/".join(map(str,k)):v for k,v in sorted(profile_counts.items())},
                         "eligible_crop_counts": {"input_dims_below_max":sum(r["input_dims_below_max"] for r in eligible),
                                                  "output_spatial_crop":sum(r["output_spatial_crop"] for r in eligible),
                                                  "logical_CO_below_physical":sum(r["logical_CO"]<r["CO"] for r in eligible)},
                         "height_profile_rejection_shapes":dict(sorted(reject_profiles["height_profile"].items())),
                         "coverage_scope":"Metadata, direct-operand alignment and QDRX block-table footprint census. Does not validate HMX lease state or execute per-command numerical proofs.",
                         "eligible_sequence_ids": [r["sequence"] for r in eligible]}
    out = Path(__file__).resolve().parent / "qd_native_3x3_graph_eligibility_census.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name:{k:v for k,v in row.items() if k!="eligible_sequence_ids"}
                      for name,row in report.items()},indent=2))


if __name__ == "__main__":
    main()
