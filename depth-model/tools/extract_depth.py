#!/usr/bin/env python3
"""Extract Model stereo ExecuTorch layers and the raw QNN context, as data only.

Requires only Python's standard library. Does not execute model code or modify
the serialized graph. pte.py is the reader recovered with the original notes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

from pte import FB, parse, delegate_bytes


def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def unwrap(data):
    if data[4:8] == b"ET12":
        return data, {"container": "ExecuTorch", "offset": 0}
    offset = data.find(b"etz0", 0, 256)
    if offset >= 0:
        expected = struct.unpack_from("<Q", data, offset + 8)[0]
        if data[offset + 16:offset + 23] != b"deflate":
            raise ValueError("unsupported compression")
        result = zlib.decompress(data[offset + 64:])
        if len(result) != expected or result[4:8] != b"ET12":
            raise ValueError("decompressed size or ExecuTorch identifier mismatch")
        return result, {"container": "etz0/deflate", "offset": offset}
    offset = data.find(b"ET12", 4, 256) - 4
    if offset < 0 or not data.startswith(b"TH1"):
        raise ValueError("unrecognized model container")
    return data[offset:], {"container": "TH1/ExecuTorch", "offset": offset}


def checked_delegate(data, info):
    d = info["plans"][0]["delegates"]
    if len(d) != 1 or d[0]["location"] != 1:
        raise ValueError("expected one segment-backed delegate")
    offset, size = info["segments"][d[0]["index"]]
    start = info["segment_base"] + offset
    if start < info["program_size"] or start + size > len(data):
        raise ValueError("delegate segment outside file")
    blob = delegate_bytes(data, info)
    if len(blob) != size:
        raise ValueError("truncated delegate")
    return blob


def io_spec(plan, direction):
    return [dict(position=n, value_index=i, **plan["values"][i])
            for n, i in enumerate(plan[direction])]


def json_default(value):
    if isinstance(value, bytes):
        try:
            s = value.decode("ascii")
            if s.isprintable():
                return {"ascii": s, "hex": value.hex()}
        except UnicodeDecodeError:
            pass
        return {"hex": value.hex()}
    raise TypeError(type(value).__name__)


def extract(source, output):
    raw = source.read_bytes()
    outer, container = unwrap(raw)
    outer_info = parse(outer)
    if outer_info["plans"][0]["delegates"][0]["id"] != "HexagonRpcBackend":
        raise ValueError("expected HexagonRpcBackend")
    inner = checked_delegate(outer, outer_info)
    inner_info = parse(inner)
    if inner_info["plans"][0]["delegates"][0]["id"] != "QnnBackend":
        raise ValueError("expected nested QnnBackend")
    wrapped = checked_delegate(inner, inner_info)
    fb = FB(wrapped)
    root = fb.root()
    name = fb.string(root, 0)
    input_ids = fb.vec_scalars(root, 1, "I", 4)
    input_ranks = fb.vec_scalars(root, 2, "I", 4)
    output_ids = fb.vec_scalars(root, 3, "I", 4)
    output_ranks = fb.vec_scalars(root, 4, "I", 4)
    begin, size = fb.vec(root, 5)
    if not name or "zelda_qvga" not in name or not begin or begin + size > len(wrapped):
        raise ValueError("unrecognized Bolt QNN wrapper")
    context = fb.vec_bytes(root, 5)
    if context[:8] != bytes.fromhex("0000000200000002"):
        raise ValueError("unexpected QNN serialization version prefix")
    op, ip = outer_info["plans"][0], inner_info["plans"][0]
    if len(input_ids) != len(ip["inputs"]) or len(output_ids) != len(ip["outputs"]):
        raise ValueError("wrapper/ExecuTorch tensor count mismatch")
    if input_ranks != [len(v["sizes"]) for v in io_spec(ip, "inputs")]:
        raise ValueError("wrapper input rank mismatch")
    if output_ranks != [len(v["sizes"]) for v in io_spec(ip, "outputs")]:
        raise ValueError("wrapper output rank mismatch")
    output.mkdir(parents=True, exist_ok=True)
    files = {"outer.pte": outer, "dsp.pte": inner,
             "qnn_delegate.fb": wrapped, "context.bin": context}
    for filename, data in files.items():
        (output / filename).write_bytes(data)
    manifest = {
        "source": str(source.resolve()), "source_digest": digest(raw),
        "container": container,
        "artifacts": {k: digest(v) for k, v in files.items()},
        "qnn_wrapper": {"graph_name": name, "context_offset": begin,
                        "input_ids": input_ids, "input_ranks": input_ranks,
                        "output_ids": output_ids, "output_ranks": output_ranks},
        "outer_inputs": io_spec(op, "inputs"),
        "outer_outputs": io_spec(op, "outputs"),
        "inner_inputs": io_spec(ip, "inputs"),
        "inner_outputs": io_spec(ip, "outputs"),
        "outer_output_positions_in_inner": [ip["outputs"].index(i) for i in op["outputs"]],
        "scalar_methods": {p["name"]: p["values"] for p in outer_info["plans"][1:]},
        "outer_program": outer_info, "inner_program": inner_info,
        "status": "Extracted unchanged; NPU execution and input semantics unverified."
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=json_default) + "\n")
    print(f"{source.name}: {len(context):,} byte QNN context -> {output / 'context.bin'}")
    print(f"SHA256 {digest(context)['sha256']}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.source, args.output)
