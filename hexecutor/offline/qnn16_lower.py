#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Lower a decoded QAIRT FP16 graph and captured run lists to a qnn16 package.

Offline development only. Prefer `--tensor-dumps`, captured by the simulator trace
hook from runtime tensor pointers. `--pool2` is an unverified fallback because
graph offsets are not proven to be file offsets. Every emitted package is marked
development-only; native execution is not implied.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from .qnn16_oracle import load, map_entries, read_trace, read_worker_checkpoints
except ImportError:
    from qnn16_oracle import load, map_entries, read_trace, read_worker_checkpoints

BLOCK = 2048
VTCM_BYTES = 8 << 20
DTYPE_BYTES = {"float32": 4, "float16": 2, "uint8": 1, "int8": 1,
               "uint16": 2, "int16": 2, "int32": 4, "int32_quantized": 4}


def align(n, a=128):
    return (n + a - 1) // a * a


def _shape(t):
    shape = t.get("shape") or {}
    return list(shape.get("dimensions") or [])


def _storage_shape(t):
    """Stored extent: packed weights/tables keep max_dimensions lanes (for
    example [256, 32] storage for a [256, 8] Xpe), so dumps and exports use it."""
    shape = t.get("shape") or {}
    return list(shape.get("max_dimensions") or shape.get("dimensions") or [])


def _storage_ranges(t):
    """Yield (pool, offset, byte_count) ranges represented by tensor storage."""
    store = t.get("storage") or {}
    direct = store.get("direct")
    n = math.prod(_storage_shape(t)) * DTYPE_BYTES.get(t.get("dtype"), t.get("element_bytes", 1))
    if direct:
        # HMX Xpe/Xwe records are addressed in 2 KiB tiles; their logical
        # tensor dimensions can describe only the valid lanes. Preserve the
        # full serialized tile extent when exporting or budgeting them.
        if str(t.get("class_name", "")).startswith("X"):
            n = align(n, BLOCK)
        yield direct["pool"], direct["offset"], n
    for ent in store.get("blocks") or []:
        if ent:
            yield ent["pool"], ent["offset"], BLOCK


def _pool2_ranges(graph):
    ranges = {}
    for t in graph["tensors"]:
        for pool, off, size in _storage_ranges(t):
            if pool == 2:
                key = (int(off), int(size))
                ranges.setdefault(key, []).append(int(t["id"]))
    return ranges


def write_tensor_config(graph, rows, entries, destination):
    """Write capture lines for pool-2 inputs and pool-3 Const inputs."""
    tensors = {t["id"]: t for t in graph["tensors"]}
    # A multithreaded node occupies several worker rows, run by a thread-split
    # kernel whose argument structs do not hold tensor handles in the standard
    # slots (QAIRT 2.50 Transpose_impl on 4 HVX threads). Capture shared
    # inputs at a single-row consumer; inputs only a split node reads stay
    # uncaptured, and lowering fails if it needs them.
    node_location, row_count = {}, {}
    for key, node in entries.items():
        if node is not None:
            seq = node["sequence"]
            row_count[seq] = row_count.get(seq, 0) + 1
            if seq not in node_location or key < node_location[seq]:
                node_location[seq] = key
    const_ids = {tid for node in graph["nodes"] if node["type_name"] == "Const" for tid in node["outputs"]}
    chosen = {}
    for node in graph["nodes"]:
        location = node_location.get(node["sequence"])
        if location is None or row_count[node["sequence"]] > 1:
            continue
        for input_index, tid in enumerate(node["inputs"]):
            t = tensors[tid]
            if not any(pool == 2 or (pool == 3 and tid in const_ids)
                       for pool, _off, _size in _storage_ranges(t)):
                continue
            if tid in chosen:
                continue
            kind = node["type_name"]
            if "weights_to_vtcm" in kind or "zero_to_vtcm" in kind:
                slot = 1 + input_index
            elif kind.startswith("ConvLayer.fp16"):
                # QAIRT 2.50 HMX conv op stores its tensor handles in the
                # extended argument area; direct pool-2 Fi inputs use slot 8.
                slot = 8
            elif kind == "layernorm_q16_va_normvals_fixedpoint.tcm@Fi.t.ni.nf*3.Fi*2" and input_index == 0:
                # The variadic normvals wrapper exposes the epsilon tensor in slot 5.
                slot = 5
            else:
                # Elementwise and format adapters store one tensor handle per
                # input in the standard argument vector.
                slot = 1 + input_index
            store = t.get("storage") or {}
            # Indirect Const tensors: QAIRT places small ones (e.g. a 12-wide soft-argmax grid) in pool 2
            # as well as pool 3; either way the hook must follow the block-pointer table.
            if not store.get("direct") and tid in const_ids and any(
                    e and e.get("pool") in (2, 3) for e in store.get("blocks") or []):
                full = _storage_shape(t)
                size = full[0] * -(-full[1] // 8) * -(-full[2] // 4) * -(-full[3] // 32) * BLOCK
                if store.get("slice_offset", 0):
                    raise ValueError(f"indirect constant {tid} has a nonzero block slice")
                slot |= 0x80000000  # capture hook dereferences the block-pointer table
            else:
                size = next(size for pool, _off, size in _storage_ranges(t)
                            if pool == 2 or (pool == 3 and tid in const_ids))
            chosen[tid] = (location[0], location[1], slot, size)
    expected = {tid for ids in _pool2_ranges(graph).values() for tid in ids}
    missing = sorted(expected - set(chosen))
    if missing:
        raise ValueError(f"pool-2 tensors have no capturable run-list input: {missing[:8]}")
    Path(destination).write_text("".join(f"{lst} {idx} {slot} {tid} {size}\n"
                                         for tid, (lst, idx, slot, size) in sorted(chosen.items())))
    return len(chosen)


def _read_tensor_dumps(path):
    data = Path(path).read_bytes()
    pos, values = 0, {}
    while pos < len(data):
        if len(data) - pos < 12:
            raise ValueError("truncated QD_TENSORS record header")
        seq, tid, size = __import__("struct").unpack_from("<III", data, pos)
        pos += 12
        if size > len(data) - pos:
            raise ValueError(f"truncated QD_TENSORS record for tensor {tid}")
        payload = data[pos:pos + size]
        pos += size
        if tid in values and values[tid] != payload:
            raise ValueError(f"simulator tensor {tid} changed between captures")
        values[tid] = payload
    return values


def lower(context_dir, trace_dir, pool2_path, output_prefix, known=None, tensor_dumps=None):
    context_dir, trace_dir = Path(context_dir), Path(trace_dir)
    graph = load(context_dir / "graph.json")
    rows = read_trace(trace_dir)
    entries = map_entries(graph, rows, known, read_worker_checkpoints(trace_dir))
    tensors = {t["id"]: t for t in graph["tensors"]}
    if bool(pool2_path) == bool(tensor_dumps):
        raise ValueError("provide exactly one of pool2_path or tensor_dumps")
    captured = _read_tensor_dumps(tensor_dumps) if tensor_dumps else None
    pool2 = Path(pool2_path).read_bytes() if pool2_path else None

    # Export immutable pool-2 regions, preserving all offsets through an explicit
    # relocation table. Block references are deduplicated; direct ranges are copied
    # byte-exactly. Alignment keeps future native readers straightforward.
    ranges = set(_pool2_ranges(graph))
    blob = bytearray()
    relocations = []
    copied = {}
    for off, size in sorted(ranges):
        key = (off, size)
        if captured is not None:
            ids = _pool2_ranges(graph)[key]
            missing = [tid for tid in ids if tid not in captured]
            if missing:
                raise ValueError(f"QD_TENSORS is missing pool-2 tensor captures {missing} for range [{off}, {off + size})")
            payloads = [captured[tid] for tid in ids]
            if any(len(payload) != size for payload in payloads):
                raise ValueError(f"QD_TENSORS byte count mismatch for pool-2 range [{off}, {off + size})")
            if any(payload != payloads[0] for payload in payloads[1:]):
                raise ValueError(f"conflicting QD_TENSORS captures for pool-2 range [{off}, {off + size})")
            payload = payloads[0]
        else:
            if off < 0 or size <= 0 or off + size > len(pool2):
                raise ValueError(f"pool-2 range [{off}, {off + size}) outside --pool2 image ({len(pool2)} bytes)")
            payload = pool2[off:off + size]
        if key not in copied:
            at = align(len(blob))
            blob.extend(bytes(at - len(blob)))
            blob.extend(payload)
            copied[key] = at
        relocations.append({"source_offset": off, "bytes": size, "export_offset": copied[key]})

    tensor_rows = []
    pool1_extent = 0
    for tid, t in sorted(tensors.items()):
        placements = []
        for pool, off, size in _storage_ranges(t):
            placements.append({"pool": int(pool), "offset": int(off), "bytes": int(size)})
            if pool == 1:
                pool1_extent = max(pool1_extent, int(off) + int(size))
        tensor_rows.append({"id": tid, "producer": t.get("producer"), "dtype": t.get("dtype"),
                            "element_bytes": t.get("element_bytes"), "class": t.get("class_name"),
                            "scalar": t.get("scalar"), "shape_values": t.get("shape_values"),
                            "shape": _shape(t), "max_shape": (t.get("shape") or {}).get("max_dimensions"),
                            "padding": (t.get("shape") or {}).get("padding"),
                            "storage": t.get("storage"), "placements": placements})

    def io_rows(ids):
        out = []
        for index, tid in enumerate(ids):
            t = tensors[tid]
            size = math.prod(_shape(t)) * DTYPE_BYTES.get(t.get("dtype"), t.get("element_bytes", 1))
            out.append({"index": index, "tensor_id": tid, "shape": _shape(t),
                        "dtype": t.get("dtype"), "bytes": size})
        return out

    run_lists = {"0": [], "1": []}
    for row in rows:
        seq, lst, idx, fn, n = row
        node = entries[(lst, idx)]
        run_lists[str(lst)].append({"index": idx, "sequence": seq, "function": fn,
                                    "node_id": None if node is None else node["sequence"],
                                    "type": None if node is None else node["type_name"],
                                    "inputs": [] if node is None else node["inputs"],
                                    "outputs": [] if node is None else node["outputs"],
                                    "checkpoints": [] if node is None else node.get("checkpoints", []),
                                    "framework_words": [] if node is None else node.get("framework_words", []),
                                    "framework_vector": [] if node is None else node.get("framework_vector", []),
                                    "node_metadata": {} if node is None else {k: v for k, v in node.items()
                                        if k not in ("inputs", "outputs", "sequence", "type_name")}})

    if pool1_extent > VTCM_BYTES:
        raise ValueError(f"pool-1 placement extent {pool1_extent} exceeds {VTCM_BYTES}-byte VTCM")

    prefix = Path(output_prefix)
    manifest_path = prefix.with_suffix(".q16.json")
    constants_path = prefix.with_suffix(".q16.bin")
    constants_path.parent.mkdir(parents=True, exist_ok=True)
    constants_path.write_bytes(blob)
    manifest = {
        "format": "qnn16-v1",
        "context_sha256": graph["context_sha256"],
        "package_status": "development-only",
        "constants_source_kind": "qnn-simulator-tensor-dump" if captured is not None else "unverified-pool2-image",
        "constants_source_verified": captured is not None,
        "constants_source_file": Path(tensor_dumps if tensor_dumps else pool2_path).name,
        "constants_source_sha256": hashlib.sha256(Path(tensor_dumps if tensor_dumps else pool2_path).read_bytes()).hexdigest(),
        "pool2_backing_sha256": hashlib.sha256(pool2).hexdigest() if pool2 is not None else None,
        "constant_capture_count": len(captured) if captured is not None else None,
        "constants_sha256": hashlib.sha256(blob).hexdigest(),
        "constants_file": constants_path.name,
        "constants_bytes": len(blob),
        "constant_relocations": relocations,
        "inputs": io_rows(graph["graph_inputs"]),
        "outputs": io_rows(graph["graph_outputs"]),
        "external_io_bytes": {"input": sum(x["bytes"] for x in io_rows(graph["graph_inputs"])),
                               "output": sum(x["bytes"] for x in io_rows(graph["graph_outputs"]))},
        "pool1_vtcm_extent": align(pool1_extent, BLOCK),
        "tensors": tensor_rows,
        "run_lists": run_lists,
        "run_list_counts": {k: len(v) for k, v in run_lists.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("context", type=Path, help="htp_graph.py output directory")
    ap.add_argument("trace", type=Path, help="captured trace directory")
    ap.add_argument("--pool2", type=Path, help="unverified pool-2 backing image (prefer --tensor-dumps)")
    ap.add_argument("--tensor-dumps", type=Path, help="QD_TENSORS captured from simulator tensor pointers")
    ap.add_argument("--write-tensor-config", type=Path, help="write QD_TENSOR_CONFIG for pool-2 graph inputs and exit")
    ap.add_argument("--known", type=Path, help="function-offset to operation-type map")
    ap.add_argument("--output", type=Path, help="output prefix")
    a = ap.parse_args()
    graph = load(a.context / "graph.json")
    rows = read_trace(a.trace)
    entries = map_entries(
        graph, rows, load(a.known) if a.known else None, read_worker_checkpoints(a.trace)
    )
    if a.write_tensor_config:
        count = write_tensor_config(graph, rows, entries, a.write_tensor_config)
        print(f"{count} constant tensor capture entries -> {a.write_tensor_config}")
        return
    if a.output is None or bool(a.pool2) == bool(a.tensor_dumps):
        ap.error("provide --output and exactly one of --pool2 or --tensor-dumps")
    m = lower(a.context, a.trace, a.pool2, a.output, load(a.known) if a.known else None, a.tensor_dumps)
    print(f"{m['format']}: {len(m['tensors'])} tensors, {m['run_list_counts']} run-list entries, "
          f"{m['constants_bytes']} constant bytes -> {a.output}.q16.json")


if __name__ == "__main__":
    main()
