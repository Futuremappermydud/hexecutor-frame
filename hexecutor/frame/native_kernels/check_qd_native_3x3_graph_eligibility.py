#!/usr/bin/env python3
"""Check the graph bridge's narrow metadata gates against five real QDRX ops."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "frame"), str(ROOT / "tools")]
from resident_image import unpack

IMAGE = ROOT / "frame/build/resident-primary.qdrx"
SEQUENCES = (247, 874, 3589, 3527, 5019)


def require(ok, seq, why):
    if not ok:
        raise SystemExit(f"sequence {seq}: bridge eligibility mismatch: {why}")


def check(seq, cmd, tensors):
    require(cmd["opcode"] == 6 and "ConvLayer_s1" in cmd["variant"], seq,
            f"unexpected operation {cmd['opcode']} / {cmd['variant']}")
    x, w, fi = (tensors[i] for i in cmd["inputs"][:3])
    y = tensors[cmd["outputs"][0]]
    xd, xm, xp = x["shape"]["dimensions"], x["shape"]["max_dimensions"], x["shape"]["padding"]
    yd, ym, yp = y["shape"]["dimensions"], y["shape"]["max_dimensions"], y["shape"]["padding"]
    wd, wm = w["shape"]["dimensions"], w["shape"]["max_dimensions"]
    fd, fm = fi["shape"]["dimensions"], fi["shape"]["max_dimensions"]
    require(x["class_name"] == y["class_name"] == "CB", seq, "CB input/output")
    require(x["dtype"] == y["dtype"] == "uint8", seq, "u8 activation tensors")
    require(w["class_name"] == "Xwb" and w["dtype"] == "int8", seq, "Xwb int8 weights")
    require(fi["class_name"] == "Fi" and fi["dtype"] == "int32", seq, "Fi int32")
    require(xd[0] == yd[0] == xm[0] == ym[0] == 1, seq, "batch dimensions")
    require(wd[:2] == wm[:2] == [3, 3], seq, "3x3 weights")
    ci, co = wm[2:]
    require(ci in (32, 64, 128, 256) and co in (32, 64, 128), seq, "covered CI/physical CO")
    require(wd[2] == ci and 0 < wd[3] <= co, seq, "logical/physical weight channels")
    require(xd[3] == xm[3] == ci and ym[3] == co and 0 < yd[3] <= co,
            seq, "CB/weight channel geometry")
    require(xm[1] == 24 and xd[1] == 10 and ym[1] == yd[1] == 8, seq, "proven height geometry")
    require(xp == [0, 7, 7, 0] and yp == [0, 0, 0, 0], seq, "pad geometry")
    require(xd[1] == yd[1] + 2 and xd[2] == yd[2] + 2, seq, "stride-1 3x3 extent")
    require(xm[2] <= 272 and ym[2] <= 256 and xm[2] % 8 == ym[2] % 8 == 0 and yd[2] % 8 == 0,
            seq, "bounded aligned max widths")
    zp = x.get("quantization", {}).get("zero_point", 0)
    require(zp == 0, seq, "input zero point")
    require(fm == [1, co // 32, 1, 64] and fd == fm, seq, "Fi per-CO-block [1,CO/32,1,64]")
    ws, fs = w["storage"]["direct"], fi["storage"]["direct"]
    require(ws["offset"] % 128 == fs["offset"] % 128 == 0, seq, "direct operand alignment")
    wbytes = w["element_bytes"]
    for d in wm:
        wbytes *= d
    fbytes = fi["element_bytes"]
    for d in fm:
        fbytes *= d
    require(wbytes == 9 * ci * co and fbytes == co * 8, seq, "max-dimension W/FI byte counts")
    if yd[3] != co:
        require((ci, co, yd[3], xd[1:3], xm[2], yd[1:3], ym[1:3]) ==
                (32, 32, 2, [10, 258], 272, [8, 256], [8, 256]),
                seq, "physical CO32 tail exception")
    print(f"PASS {seq}: CI={ci}, physical CO={co}, logical CO={yd[3]}, W={wbytes} B, Fi={fbytes} B")


def main():
    graph = unpack(IMAGE.read_bytes())
    tensors = {t["id"]: t for t in graph["tensors"]}
    commands = {c["sequence"]: c for c in graph["commands"]}
    for seq in SEQUENCES:
        check(seq, commands[seq], tensors)
    print(f"PASS all {len(SEQUENCES)} recorded graph geometries satisfy bridge metadata gates")


if __name__ == "__main__":
    main()
