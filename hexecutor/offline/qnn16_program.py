#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Lower a decoded QAIRT 2.50 FP16 context to a Q16P program and constant pool.

Inputs, all local and ignored:
  CONTEXT   htp_graph.py output directory (graph.json)
  TRACE     trace-hook run (trace.log) giving the executed run-list order
  TENSORS   QD_TENSORS dump of every pool-2 input (see qnn16_lower.py
            --write-tensor-config), the source of the constant pool

The program executes the traced entries serially in simulator order with
QNN's own VTCM placements. The format is hexecutor/native/qnn16/program.h.
Offline development only: no device access and no model execution.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

try:
    from .qnn16_oracle import load, map_entries, read_trace, read_worker_checkpoints
except ImportError:
    from qnn16_oracle import load, map_entries, read_trace, read_worker_checkpoints

BLOCK = 2048
VTCM = 8 << 20
COMPLETION = VTCM - BLOCK
NONE = 0xFFFFFFFF
MAGIC, VERSION = 0x50363151, 1
KINDS = ["COPY", "ZERO", "CONV1X1", "DW1X1", "DW3X3", "CONV3X3", "S2D", "DW3X3_S2", "IM2COL",
         "INPUT", "OUTPUT", "FORMAT", "CAST_F32", "CAST_F16", "MUL", "MINMAX", "SIGMOID",
         "RESHAPE", "CONCAT", "SLICE", "SHUFFLE", "TRANSPOSE", "AVGPOOL", "X2S", "ZAP", "PW1X1", "SWISH", "WELFORD_PARTIAL", "CONV3X3_S2", "WELFORD_MERGE", "MOVE", "ADD_SWISH", "GROUP_MOMENTS", "DECIMATE", "SLICE_F32", "INSTANCE_NORM", "RESIZE_BILINEAR", "SOFTMAX", "PW_SWISH", "REDUCE_SUM", "RESIZE_NEAREST",
         "L2NORM", "REDUCE_MEAN", "PW_TANH", "AFFINE", "INPUT_U16", "OUTPUT_U16", "SLICE_U16", "SPILL_FILL", "REARRANGE_U16", "MOMENTS_U16", "NORMVALS_U16", "NORMALIZE_U16", "PW_XSB_U16", "LINCLIP_U16", "PACK_XPB_U16", "XPB_BIAS", "XPB_FI", "PW_XPB_U16",
         "SOFTMAX_U16", "MUL_U16", "ADD_U16", "GELU_U16"]
KIND = {name: i + 1 for i, name in enumerate(KINDS)}
SKIP = ("@DmaCheckpointSet", "@DmaCheckpointWait", "@SyncOp", "@Dummy", "Const", "Shape")


def align(n, a=128):
    return (n + a - 1) // a * a


def read_dumps(path):
    data = Path(path).read_bytes()
    pos, out = 0, {}
    while pos < len(data):
        _seq, tid, size = struct.unpack_from("<III", data, pos)
        pos += 12
        payload = data[pos:pos + size]
        if len(payload) != size:
            raise ValueError("truncated QD_TENSORS record")
        if tid in out and out[tid] != payload:
            raise ValueError(f"pool-2 tensor {tid} changed between captures")
        out[tid] = payload
        pos += size
    return out


TANH_CLAMPS = struct.pack("<5H", 0x3C00, 0, 0xBC00, 0, 0)  # halves 9..13 of QAIRT's 14-half tanh parameters


def _tanh_ok(halves):
    slope = struct.unpack("<e", struct.pack("<H", halves[6]))[0]
    return (halves[9], halves[11]) == (0x3C00, 0xBC00) and 0.5 < slope < 1.5


def context_tanh_parameters(binary):
    """QAIRT 2.50's piecewise-tanh parameters, read from the context binary.

    The 14 FP16 halves are a fixed QAIRT constant (byte-identical in unrelated models). When QAIRT gives a fused
    tanh conv a direct pool-2 copy, the run-list argument holds no handle to it, so the trace hook cannot capture
    it. Every serialized copy (the +-1 clamps at halves 9 and 11) must agree on the slope, which is all lowering
    uses besides the clamps."""
    data, found, at = Path(binary).read_bytes(), {}, 0
    while (at := data.find(TANH_CLAMPS, at)) >= 0:
        if at >= 18:
            halves = struct.unpack_from("<14H", data, at - 18)
            if _tanh_ok(halves):
                found.setdefault(halves[6], halves)
        at += 1
    if len(found) != 1:
        raise ValueError(f"{binary}: expected one tanh slope among the context's tanh tables, found {len(found)}")
    return next(iter(found.values()))


class Lowering:
    def __init__(self, graph, dumps, context_binary=None):
        self.graph = graph
        self.context_binary = context_binary
        self.T = {t["id"]: t for t in graph["tensors"]}
        self.dumps = dumps
        self.tensor_index = {}
        self.tensor_rows = []
        self.words = []
        self.ops = []
        self.constants = bytearray()
        self.constant_at = {}
        self.fi_words = {}
        self.op_nodes = []
        self.current = None
        self.produced = {tid for n in graph["nodes"] if n["type_name"] not in ("Const", "Shape")
                         for tid in n["outputs"]}
        # Pool 2 contains both immutable weights and mutable DDR results.
        # Only the latter need an arena mapping; derive its extent from graph outputs.
        pool2_end = 0
        for t in graph["tensors"]:
            direct = (t.get("storage") or {}).get("direct") or {}
            shape = t.get("shape")
            if t["id"] in self.produced and direct.get("pool") == 2 and shape:
                pool2_end = max(pool2_end, direct["offset"] +
                                math.prod(shape["max_dimensions"]) * (t.get("element_bytes") or 1))
        self.pool3_base = align(pool2_end)
        # Only grouped-moment constants are materialized from direct pool 3.
        # Other pool-3 records remain in the immutable constant pool.
        pool3_ids = {n["inputs"][i] for n in graph["nodes"]
                     if n["type_name"].startswith("va_parallel_welford_reduce_grouped_moments_sf")
                     for i in (0, 1, 4)}
        pool3_end = 0
        for tid in pool3_ids:
            t = self.T[tid]
            direct = (t.get("storage") or {}).get("direct") or {}
            shape = t.get("shape")
            if direct.get("pool") == 3 and shape:
                pool3_end = max(pool3_end, direct["offset"] + math.prod(shape["max_dimensions"]) * (t.get("element_bytes") or 1))
        self.pool3_end = align(pool3_end)
        self.indirect_arena = self.pool3_base + self.pool3_end
        self.arena_bytes = 0
        self.spill_bytes = 0
        self.copy_const = {}  # COPY op index -> pool-2 tensor id

    # ----- tensors -----
    def dtype(self, t):
        if t["class_name"].startswith("X") or t["class_name"] in ("Fi", "fi", "Fs"):
            return 2  # packed weights or scale/bias tables: raw bytes
        if t.get("dtype") == "float32":
            return 1
        if t.get("dtype") == "uint16":
            return 3
        if t.get("element_bytes") == 2:
            return 0
        raise ValueError(f"tensor {t['id']} ({t['class_name']}) is neither FP16 nor float32")

    def tensor(self, tid, exact_bytes=None):
        if tid in self.tensor_index:
            return self.tensor_index[tid]
        t = self.T[tid]
        shape = t.get("shape")
        if not shape:
            raise ValueError(f"tensor {tid} has no shape")
        store = t.get("storage") or {}
        dims, full, pad = shape["dimensions"], shape["max_dimensions"], shape["padding"]
        if (t["class_name"] == "F5H" and t["dtype"] == "uint16" and store.get("direct")
                and len(dims) == 5 and dims == full and not any(pad)):
            dims = full = [dims[0] * dims[1], *dims[2:]]
            pad = [0] * 4
        if len(dims) != 4:
            raise ValueError(f"tensor {tid} is not rank 4")
        dtype = self.dtype(t)
        esize = (2, 4, t.get("element_bytes") or 1, 2)[dtype]
        direct = store.get("direct")
        if direct:
            pool = direct["pool"]
            if pool not in (1, 2, 3):
                raise ValueError(f"tensor {tid} is in unsupported pool {pool}")
            nbytes = exact_bytes or math.prod(full) * esize
            if t["class_name"].startswith("X"):
                nbytes = max(nbytes, align(math.prod(full) * esize, BLOCK))
            if pool == 1:
                if direct["offset"] + nbytes > COMPLETION:
                    raise ValueError(f"tensor {tid} overlaps the completion tile")
                row = [1, dtype, direct["offset"], nbytes, 0, 0]
            else:
                # Preserve aliases by mapping pool offsets into disjoint arena regions.
                if pool == 2 and direct["offset"] + nbytes > self.pool3_base:
                    raise ValueError(f"tensor {tid} is not a mutable pool-2 result")
                if pool == 3 and direct["offset"] + nbytes > self.pool3_end:
                    raise ValueError(f"tensor {tid} exceeds the planned pool-3 arena")
                offset = direct["offset"] + (0 if pool == 2 else self.pool3_base)
                self.arena_bytes = max(self.arena_bytes, align(offset + nbytes))
                row = [2, dtype, offset, nbytes, 0, 0]
        else:
            if dtype not in (0, 3) or t["class_name"][0] not in "Cc":
                raise ValueError(f"tensor {tid} ({t['class_name']}) is not a supported 16-bit crouton tensor")
            count = full[0] * -(-full[1] // 8) * -(-full[2] // 4) * -(-full[3] // 32)
            start = store.get("slice_offset", 0)
            blocks = store.get("blocks") or []
            # QAIRT also places small indirect Const tensors (e.g. a 12-wide soft-argmax grid) in pool 2.
            if any(ent and (ent.get("pool") == 3 or (ent.get("pool") == 2 and tid in self.dumps))
                   for ent in blocks[start:start + count]):
                if tid in self.produced or start or len(self.dumps.get(tid, b"")) != count * BLOCK:
                    raise ValueError(f"tensor {tid}: indirect constant capture is missing or unsupported")
                raw = self.dumps[tid]
                compact = bytearray()
                nbw, nbc = -(-full[2] // 4), -(-full[3] // 32)
                for b in range(dims[0]):
                    for y in range(dims[1]):
                        for x in range(dims[2]):
                            for c in range(dims[3]):
                                fb, fy, fx, fc = b + pad[0], y + pad[1], x + pad[2], c + pad[3]
                                block = ((fb * -(-full[1] // 8) + fy // 8) * nbw + fx // 4) * nbc + fc // 32
                                within = (((fy % 8) * 2 + (fx % 4) // 2) * 32 + fc % 32) * 4 + (fx % 2) * 2
                                compact.extend(raw[block * BLOCK + within:block * BLOCK + within + 2])
                self.dumps[tid] = bytes(compact)
                nbytes = len(compact)
                offset = self.indirect_arena
                self.indirect_arena = align(offset + nbytes)
                self.arena_bytes = max(self.arena_bytes, self.indirect_arena)
                row = [2, dtype, offset, nbytes, 0, 0] + list(dims) + list(dims) + [0] * 4 + [0, 0]
                index = len(self.tensor_rows)
                self.tensor_index[tid] = index
                self.tensor_rows.append(row)
                off, size = self.constant(tid)
                self.copy_const[len(self.ops)] = tid
                self.op("COPY", out=index, p=(off, size))
                return index
            first = len(self.words)
            for k in range(start, start + count):
                ent = blocks[k] if k < len(blocks) else None
                if ent and ent.get("pool") == 1:
                    if ent["offset"] % BLOCK or ent["offset"] + BLOCK > COMPLETION:
                        raise ValueError(f"tensor {tid} block outside VTCM")
                    self.words.append(ent["offset"])
                elif ent and ent.get("pool") != 1:
                    raise ValueError(f"tensor {tid}: indirect pool-{ent.get('pool')} constant requires capture")
                else:
                    self.words.append(NONE)
            row = [0, dtype, 0, 0, first, count]
        row += list(dims) + list(full) + list(pad) + [0, 0]
        self.tensor_index[tid] = len(self.tensor_rows)
        self.tensor_rows.append(row)
        return self.tensor_index[tid]

    def vtcm_block(self, offset):
        """A direct FP16 tensor covering one raw 2 KiB VTCM block."""
        if offset % BLOCK or offset + BLOCK > COMPLETION:
            raise ValueError("zap block outside VTCM")
        key = ("vtcm_block", offset)
        if key not in self.tensor_index:
            dims = [1, 8, 4, 32]
            self.tensor_index[key] = len(self.tensor_rows)
            self.tensor_rows.append([1, 0, offset, BLOCK, 0, 0] + dims + dims + [0] * 4 + [0, 0])
        return self.tensor_index[key]

    def contiguous_f32(self, tid):
        t = self.T[tid]
        s = t["shape"]
        if (self.dtype(t) != 1 or not (t.get("storage") or {}).get("direct") or
                s["dimensions"] != s["max_dimensions"] or any(s["padding"])):
            raise ValueError(f"tensor {tid} is not a contiguous float32 tensor")
        return self.tensor(tid)

    def materialize_f32(self, tid):
        idx = self.contiguous_f32(tid)
        if self.tensor_rows[idx][0] == 2 and tid not in self.produced:
            off, n = self.constant(tid)
            if n != self.tensor_rows[idx][3]:
                raise ValueError(f"constant tensor {tid} size mismatch")
            self.copy_const[len(self.ops)] = tid
            self.op("COPY", out=idx, p=(off, n))
        return idx

    def constant(self, tid):
        if tid not in self.dumps:
            raise ValueError(f"pool-2 tensor {tid} was not captured")
        if tid not in self.constant_at:
            self.constants.extend(bytes(align(len(self.constants)) - len(self.constants)))
            self.constant_at[tid] = len(self.constants)
            self.constants.extend(self.dumps[tid])
        return self.constant_at[tid], len(self.dumps[tid])

    def const_values(self, tid, dtype):
        import array
        a = array.array(dtype)
        a.frombytes(self.dumps[tid])
        return list(a)

    # ----- operations -----
    def op(self, kind, x=NONE, y=NONE, z=NONE, out=NONE, p=()):
        p = list(p) + [0] * (11 - len(p))
        self.ops.append([KIND[kind], x, y, z, out] + [v & 0xFFFFFFFF for v in p])
        self.op_nodes.append(self.current)

    def compact_stride2_weights(self):
        """Copy only the diagonal of stride-2 depthwise weights.

        A t=514 stride-2 Xwe tensor holds nine 32x32 tiles per channel group
        (tap tile order {1,4,0,7,8,6,3,5,2}) of which the kernels read only the
        diagonal: row r keeps channel 2r in bytes [8r, 8r + 4) and 2r + 1 in
        [8r + 4, 8r + 8). When every reader of a COPY is DW3X3_S2 weights, the
        pool holds one 128-byte vector per (group, tap ky*3+kx) instead, and
        those operations get p1 = 1. The constant pool is rebuilt afterwards."""
        tile_of_tap = (1, 4, 0, 7, 8, 6, 3, 5, 2)
        uses = {}
        for i, o in enumerate(self.ops):
            if o[0] == KIND["COPY"]:
                continue
            for slot in (1, 2, 3):
                if o[slot] != NONE:
                    uses.setdefault(o[slot], []).append((i, slot))
            if o[0] == KIND["CONCAT"]:
                for k in range(o[7]):
                    uses.setdefault(self.words[o[6] + k], []).append((i, 0))
        compact = {}
        for i, o in enumerate(self.ops):
            if o[0] != KIND["COPY"]:
                continue
            u = uses.get(o[4], [])
            if u and all(self.ops[j][0] == KIND["DW3X3_S2"] and slot == 2 for j, slot in u):
                compact[i] = u
        pool, at = bytearray(), {}
        for i, o in enumerate(self.ops):
            if o[0] != KIND["COPY"]:
                continue
            tid = self.copy_const[i]
            key = (tid, i in compact)
            if key not in at:
                data = self.dumps[tid]
                if i in compact:
                    if len(data) % (9 * BLOCK):
                        raise ValueError(f"stride-2 weights {tid}: {len(data)} bytes")
                    out = bytearray()
                    for g in range(len(data) // (9 * BLOCK)):
                        for k in range(9):
                            tile = data[(g * 9 + tile_of_tap[k]) * BLOCK:][:BLOCK]
                            vec = bytearray(128)
                            for r in range(16):
                                vec[8 * r:8 * r + 8] = tile[r * 128 + 8 * r:r * 128 + 8 * r + 8]
                            out += vec
                    data = bytes(out)
                pool.extend(bytes(align(len(pool)) - len(pool)))
                at[key] = (len(pool), len(data))
                pool.extend(data)
            o[5], o[6] = at[key]
        for u in compact.values():
            for j, _ in u:
                self.ops[j][6] = 1
        self.constants = pool
        return len(compact)

    def s4(self, node, i):
        return self.T[node["inputs"][i]]["shape_values"]

    def lower_node(self, node):
        name = node["type_name"]
        base = name.split("@")[0]
        ins, outs = node["inputs"], node["outputs"]
        T = self.T
        if name.startswith(SKIP):
            return
        if name == "layernorm_q16_2d_moments_reshaped.tcm@Fi.CH":
            src, dst = T[ins[0]], T[outs[0]]
            if src["dtype"] != "uint16" or src["class_name"] != "CH" or dst["dtype"] != "int32" or dst["class_name"] != "Fi":
                raise ValueError("U16 LayerNorm moments require Crouton input and dense int32 output")
            self.op("MOMENTS_U16", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        if name == "layernorm_q16_va_normvals_fixedpoint.tcm@Fi.t.ni.nf*3.Fi*2":
            if len(ins) != 9 or len(outs) != 1 or T[ins[1]].get("scalar") != 1024:
                raise ValueError("U16 LayerNorm normvals requires four 256-channel partials")
            if T[ins[0]]["dtype"] != "float32" or len(self.dumps.get(ins[0], b"")) != 4:
                raise ValueError("U16 LayerNorm normvals requires captured float32 epsilon")
            if any(T[t]["dtype"] != "int32" or T[t]["class_name"] != "Fi" for t in ins[5:] + outs):
                raise ValueError("U16 LayerNorm normvals requires dense int32 tables")
            epsilon, = struct.unpack("<I", self.dumps[ins[0]])
            scales = [T[t]["scalar_word"] for t in ins[2:5]]
            partials = [self.tensor(t) for t in ins[5:]]
            self.op("NORMVALS_U16", *partials[:3], out=self.tensor(outs[0]),
                    p=(partials[3], epsilon, *scales))
            return
        if name == "space_rearrange@CH*2":
            src, dst = T[ins[0]], T[outs[0]]
            if any(t["dtype"] != "uint16" or t["class_name"] != "CH" for t in (src, dst)):
                raise ValueError("space_rearrange requires U16 Crouton tensors")
            if not src.get("quantization") or src["quantization"] != dst.get("quantization"):
                raise ValueError("space_rearrange input and output encodings differ")
            self.op("REARRANGE_U16", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        if name == "layernorm_q16_2d_normalize_fixedpoint.tcm@CH*2.Fi.Fh.Fi":
            if [(T[t]["class_name"], T[t]["dtype"]) for t in ins + outs] != [
                ("CH", "uint16"), ("Fi", "int32"), ("Fh", "int16"), ("Fi", "int32"), ("CH", "uint16"),
            ]:
                raise ValueError("U16 LayerNorm normalize requires Crouton input/output and typed dense tables")
            src, norms, gamma, beta = [self.tensor(t) for t in ins]
            self.op("NORMALIZE_U16", src, norms, gamma, self.tensor(outs[0]), p=(beta,))
            return
        if name in ("ConvLayer_s1.opt@CH*2.Xsb.Fi.t.fi", "ConvLayer_s1.opt@CH*2.Xpb.Fi.t.fi"):
            split = "Xpb" in name
            weights = "Xpb" if split else "Xsb"
            if [(T[t]["class_name"], T[t]["dtype"]) for t in ins[:3] + outs] != [
                ("CH", "uint16"), (weights, "int8"), ("Fi", "int32"), ("CH", "uint16"),
            ] or len(ins) != 5 or T[ins[3]].get("scalar") != (197 if split else 8):
                raise ValueError(f"{weights} convolution requires the SHARP U16 pointwise layout")
            source, weight, target = [T[t].get("quantization") for t in (*ins[:2], outs[0])]
            if not source or not weight or not target or weight["zero_point"]:
                raise ValueError(f"{weights} convolution requires asymmetric activations and symmetric weights")
            scale = (65536 if split else 256) * source["scale"] * weight["scale"] / target["scale"]
            scale_bits, = struct.unpack("<I", struct.pack("<f", scale))
            params = (scale_bits, target["zero_point"])
            if not split:
                # Xsb Fi stores one exponent and mantissa per output lane.
                producer = self.graph["nodes"][T[ins[2]]["producer"]]
                if (producer["type_name"] != "ConvLayer.opt.weights_to_vtcm@Fi.fi."
                        or len(producer["inputs"]) != 1
                        or producer["inputs"][0] not in self.dumps):
                    raise ValueError("Xsb Fi lane scales require a captured constant table")
                fi = self.const_values(producer["inputs"][0], "I")
                if not fi or len(fi) % 128:
                    raise ValueError("Xsb Fi table must contain full 128-word groups")
                lanes = [(fi[group + half + lane], fi[group + half + lane + 1])
                         for group in range(0, len(fi), 128) for half in (0, 64)
                         for lane in range(0, 32, 2)]
                values = [math.ldexp(1024 + (man & 1023), ((exp >> 10) & 31) - 26)
                          for exp, man in lanes]
                if (len({man >> 10 for _, man in lanes}) != 1
                        or min(abs(value / scale - 1) for value in values) > 0.0015):
                    raise ValueError("unsupported Xsb Fi lane scale encoding")
                params += (1,)
            self.op("PW_XPB_U16" if split else "PW_XSB_U16", *[self.tensor(t) for t in ins[:3]], out=self.tensor(outs[0]),
                    p=params)
            return
        if name in ("linearclip@FH*2.fi", "linearclip_simple_rq@CH*2.fi"):
            source, target = T[ins[0]], T[outs[0]]
            if source["dtype"] != "uint16" or target["dtype"] != "uint16":
                raise ValueError("linearclip requires U16 input and output")
            fi = self.const_values(ins[1], "I")
            if len(fi) != 13 or fi[0] != 514 or fi[12] != 1 or fi[6] & 0x80000000:
                raise ValueError("unsupported U16 linearclip descriptor")
            multiplier, shift = fi[10] & 65535, 15 - (fi[10] >> 24)
            input_zero, output_zero = fi[11] & 65535, fi[11] >> 16
            if (not multiplier or not 1 <= shift <= 31
                    or input_zero != source["quantization"]["zero_point"]
                    or output_zero != target["quantization"]["zero_point"]):
                raise ValueError("U16 linearclip constants disagree with tensor encodings")
            if name == "linearclip@FH*2.fi":
                if fi[1] > 1 << 18 or fi[2] > 0xffff:
                    raise ValueError("unsupported flat U16 linearclip coefficients")
                # The flat kernel rounds once to Q3, adds its Q3 bias,
                # then rounds back to an integer output.
                params = (fi[1], 18, 0, 0, ((fi[2] + 4) << 15) + (1 << 14))
            else:
                params = (multiplier, shift, input_zero, output_zero, 1 << (shift - 1))
            self.op("LINCLIP_U16", self.tensor(ins[0]), out=self.tensor(outs[0]), p=params)
            return
        if name == "ConvLayer.opt.convert_weights_to_signed.packed.tcm@Xpb.FH.t.fs.t":
            source, output = T[ins[0]], T[outs[0]]
            source_q, output_q = source.get("quantization"), output.get("quantization")
            if (source["class_name"] != "FH" or source["dtype"] != "uint16" or output["class_name"] != "Xpb"
                    or output["dtype"] != "int8" or not source_q or not output_q
                    or source_q["zero_point"] != 32768 or output_q["zero_point"] != 0
                    or source_q["scale"] != output_q["scale"] or T[ins[1]].get("shape_values") != [1] * 4
                    or T[ins[3]].get("scalar") != 1):
                raise ValueError("Xpb packing requires symmetric weights with the SHARP pointwise layout")
            scales = self.const_values(ins[2], "i")
            if len(scales) < source["shape"]["dimensions"][-1] or any(v != 65536 for v in scales):
                raise ValueError("Xpb packing requires unity Q16 weight scales")
            self.op("PACK_XPB_U16", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        if name == "ConvLayer.opt.bias_weight_update@Fs.Xpb.ni*2.t.fs":
            weight, seed, output = (T[t] for t in (ins[0], ins[4], outs[0]))
            input_zero, mode, code = (T[t].get("scalar") for t in ins[1:4])
            if (weight["class_name"] != "Xpb" or mode != 2 or code != 197
                    or not isinstance(input_zero, int) or not 0 <= input_zero <= 65535
                    or seed["dtype"] != "int32_quantized" or output["dtype"] != "int32_quantized"
                    or seed["shape"]["dimensions"] != output["shape"]["dimensions"]
                    or seed.get("quantization") != output.get("quantization")):
                raise ValueError("unsupported Xpb bias update descriptor")
            seed_bytes = self.dumps.get(ins[4])
            if seed_bytes is None or len(seed_bytes) != 4 * math.prod(seed["shape"]["dimensions"]) or any(seed_bytes):
                raise ValueError("Xpb bias update requires a captured zero base bias")
            self.op("XPB_BIAS", self.tensor(ins[0]), out=self.tensor(outs[0]), p=(input_zero,))
            return
        if name == "ConvLayer.opt.bias_scale_shuff@Fi.fi.Fs.t":
            if T[ins[2]].get("scalar") != 2 or [T[t]["class_name"] for t in ins[:2] + outs] != ["fi", "Fs", "Fi"]:
                raise ValueError("unsupported Xpb Fi shuffle mode")
            if ins[0] not in self.fi_words:
                scales = self.const_values(ins[0], "I")
                channels = T[ins[1]]["shape"]["dimensions"][-1]
                if len(scales) != channels * 2:
                    raise ValueError("Xpb Fi scale and bias table sizes differ")
                self.fi_words[ins[0]] = len(self.words)
                self.words.extend(scales)
            self.op("XPB_FI", self.tensor(ins[1]), out=self.tensor(outs[0]), p=(self.fi_words[ins[0]],))
            return
        if name in ("Softmax_Crouton@CH*2.t", "mul_op_crt@CH*3.fi", "Add.tcm@CH*3", "gelu_16_crt@CH*2"):
            binary = name in ("mul_op_crt@CH*3.fi", "Add.tcm@CH*3")
            values = ins[:2] if binary else ins[:1]
            if any(T[t]["dtype"] != "uint16" or T[t]["class_name"] != "CH" or not T[t].get("quantization")
                   for t in values + outs):
                raise ValueError(f"{name} requires quantized U16 Crouton tensors")
            encodings = [T[t]["quantization"] for t in values + outs]
            src, dst = encodings[0], encodings[-1]
            if name == "Softmax_Crouton@CH*2.t":
                if self.dumps.get(ins[1]) != struct.pack("<f", 1.0) or dst != {"scale": 2 ** -16, "zero_point": 0}:
                    raise ValueError("U16 softmax requires unity beta and Q0.16 output")
                kind = "SOFTMAX_U16"
                params = struct.unpack("<I", struct.pack("<f", src["scale"]))
            elif name == "gelu_16_crt@CH*2":
                kind = "GELU_U16"
                scales = struct.unpack("<2I", struct.pack("<2f", src["scale"], 1 / dst["scale"]))
                params = (*scales, src["zero_point"], dst["zero_point"])
            elif name == "mul_op_crt@CH*3.fi":
                kind = "MUL_U16"
                fi = self.const_values(ins[2], "I")
                if (len(fi) != 6 or fi[0] != 0xA0302 or fi[1] or fi[3] != (
                        src["zero_point"] | (encodings[1]["zero_point"] << 16))):
                    raise ValueError("unsupported U16 multiplication descriptor")
                params = (fi[2], fi[3] & 65535, fi[3] >> 16, fi[4], fi[5])
            else:
                kind = "ADD_U16"
                ratios = [v["scale"] / dst["scale"] for v in encodings[:2]]
                shift = 14 - math.frexp(max(ratios))[1]
                if not 1 <= shift <= 30:
                    raise ValueError("unsupported U16 addition scale")
                multipliers = [math.floor(math.ldexp(r, shift + 1) + 0.5) for r in ratios]
                offset = int((math.ldexp(dst["zero_point"], shift + 1) -
                              sum(m * v["zero_point"] for m, v in zip(multipliers, encodings))) / 2)
                params = (*multipliers, shift, offset)
            self.op(kind, *[self.tensor(t) for t in values], out=self.tensor(outs[0]), p=params)
            return
        if name in ("@Spill", "@Fill") and "spill_groups" in node:
            self.spill_fill(node)
            return
        if name == "@BlockZap":
            for table in node["zap_tables"]:
                code = table["code"]
                if (code & 0xF0) == 0xC0 and 1 <= (code & 0xF) <= 8:
                    rs, cs = 8 - (code & 0xF), 0
                elif code in (0x50, 0x4C, 0x54):
                    rs, cs = 0, {0x50: 3, 0x4C: 2, 0x54: 1}[code]
                else:
                    raise ValueError(f"@BlockZap {node['sequence']}: unknown code {code:#x}")
                # QAIRT 2.50 copy tables: each source block is first copied to
                # its destination, whose region is then cleared.
                for src, dst in zip(table.get("sources", ()), table["blocks"]):
                    self.op("MOVE", self.vtcm_block(src), out=self.vtcm_block(dst))
                first = len(self.words)
                for b in table["blocks"]:
                    if b % BLOCK or b + BLOCK > COMPLETION:
                        raise ValueError("zap block outside VTCM")
                    self.words.append(b)
                self.op("ZAP", p=(first, len(table["blocks"]), rs, cs))
            return
        if base in ("ConvLayer.opt.weights_to_vtcm", "ConvLayer.opt.bias_to_vtcm"):
            off, n = self.constant(ins[0])
            self.copy_const[len(self.ops)] = ins[0]
            self.op("COPY", out=self.tensor(outs[0], exact_bytes=n), p=(off, n))
            return
        if base == "zero_to_vtcm.fp16":
            self.op("ZERO", out=self.tensor(outs[0]))
            return
        if base.startswith("ConvLayer.fp16.s1") or base.startswith("ConvLayer.fp16.s2"):
            x, w, fi, t, _cfg = ins
            qt = T[t].get("scalar")
            wd = T[w]["shape"]["dimensions"]
            tables = T[fi]["shape"]["dimensions"][2]
            out = outs[0]
            if tables == 5 and qt == 544 and wd == [1, 1, 4, 32]:
                # The observed five-table form fuses two identity depthwise
                # lanes, addition and a piecewise SiLU. Use mathematical SiLU;
                # this approximates QAIRT's piecewise function.
                producer = self.graph["nodes"][T[w]["producer"]]
                if producer["type_name"].split("@")[0] != "ConvLayer.opt.weights_to_vtcm":
                    raise ValueError("fused add-swish weights are not a constant copy")
                raw = self.dumps[producer["inputs"][0]]
                for c in range(32):
                    for j in range(2):
                        offset = ((c // 2) * 4 + j * 3) * 2
                        if struct.unpack_from("<H", raw, offset)[0] != 0x3c00:
                            raise ValueError("fused add-swish weights are not identity")
                if T[x]["shape"]["dimensions"][3] != 64 or T[out]["shape"]["dimensions"][3] != 32:
                    raise ValueError("fused add-swish channel shape")
                self.op("ADD_SWISH", self.tensor(x), out=self.tensor(out))
                return
            if tables == 5 and qt == 0 and wd[:2] == [1, 1]:
                self.op("PW_SWISH", self.tensor(x), self.tensor(w), self.tensor(fi), self.tensor(out), (wd[2],))
                return
            if tables == 6 and qt == 0 and wd[:2] == [1, 1] and not base.startswith("ConvLayer.fp16.s2"):
                # Pointwise convolution fused with QAIRT's piecewise tanh: tables 4 and 5 clamp to
                # +-1 (their high halves), and table 3's low halves hold the central slope times the
                # convolution scale. The slope alone is half 6 of the last input. Use mathematical tanh.
                producer = self.graph["nodes"][T[fi]["producer"]]
                if producer["type_name"].split("@")[0] != "ConvLayer.opt.weights_to_vtcm":
                    raise ValueError("fused tanh tables are not a constant copy")
                raw = self.dumps[producer["inputs"][0]]
                halves = struct.unpack("<14H", self.dumps[ins[4]])
                if not _tanh_ok(halves) and T[ins[4]].get("storage", {}).get("direct", {}).get("pool") == 2:
                    # Direct pool-2 copy: not reachable from the run-list arguments (see context_tanh_parameters).
                    if self.context_binary is None:
                        raise ValueError(f"conv {node['sequence']}: pool-2 tanh parameters need the context binary")
                    halves = context_tanh_parameters(self.context_binary)
                slope = struct.unpack("<e", struct.pack("<H", halves[6]))[0]
                if not _tanh_ok(halves):
                    raise ValueError(f"conv {node['sequence']}: unexpected fused tanh parameters")
                for g in range(-(-wd[3] // 32)):
                    for lane in range(min(32, wd[3] - 32 * g)):
                        hi = lambda t: struct.unpack_from("<H", raw, ((g * 6 + t) * 64 + lane) * 4 + 2)[0]
                        if (hi(4), hi(5)) != (0x3c00, 0xbc00):
                            raise ValueError(f"conv {node['sequence']}: fused tanh clamps are not +-1")
                self.op("PW_TANH", self.tensor(x), self.tensor(w), self.tensor(fi), self.tensor(out),
                        (wd[2], struct.unpack("<I", struct.pack("<f", slope))[0]))
                return
            if tables not in (1, 2):
                raise ValueError(f"conv {node['sequence']}: {tables} Fi tables")
            if base.startswith("ConvLayer.fp16.s2"):
                if qt == 514:
                    self.op("DW3X3_S2", self.tensor(x), self.tensor(w), self.tensor(fi), self.tensor(out), (tables,))
                elif qt == 0 and wd[:2] == [3, 3]:
                    self.op("CONV3X3_S2", self.tensor(x), self.tensor(w), self.tensor(fi), self.tensor(out), (tables, wd[2]))
                else:
                    raise ValueError(f"stride-2 conv {node['sequence']} t={qt}")
            elif qt in (0, 3) and wd[:2] == [1, 1] and wd[2] % 32:
                # Partial last K tile: Xpe packs ceil(ci/2) pairs per output group.
                self.op("PW1X1", self.tensor(x), self.tensor(w), self.tensor(fi), self.tensor(out),
                        (tables, wd[2]))
            elif qt in (0, 3) and wd[:2] == [1, 1]:
                self.conv1x1(node, x, w, fi, out, wd, tables)
            elif qt == 0 and wd[:2] == [3, 3]:
                self.op("CONV3X3", self.tensor(x), self.tensor(w), self.tensor(fi), self.tensor(out),
                        (tables, wd[2], wd[3]))
            elif qt == 544:
                self.op("DW1X1", self.tensor(x), self.tensor(w), self.tensor(fi), self.tensor(out), (tables,))
            elif qt == 514:
                self.op("DW3X3", self.tensor(x), self.tensor(w), self.tensor(fi), self.tensor(out), (tables,))
            else:
                raise ValueError(f"conv {node['sequence']}: unsupported t={qt} kernel {wd[:2]}")
            return
        if base == "ConvLayer_SpaceToDepth_stride2.tcm":
            self.op("S2D", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        if base == "ConvLayer.opt.inconv_im2col_cast.tcm":
            self.op("IM2COL", self.contiguous_f32(ins[0]), out=self.tensor(outs[0]))
            return
        if base in ("*InputSlicePad", "*InputSlice"):
            pad, start = self.s4(node, 0), self.s4(node, 1)
            slot = self.s4(node, 4 if base == "*InputSlicePad" else 3)[-1]
            origin = (start[1] - pad[1], start[2] - pad[2], start[0] - pad[0])
            self.check_io_slot("inputs", slot, self.s4(node, -1))
            if T[outs[0]].get("dtype") == "uint16":
                if base != "*InputSlice" or any(pad) or any(x < 0 for x in origin):
                    raise ValueError("U16 input requires an unpadded slice")
                params = self.io_quantization("inputs", slot, outs[0])
                self.op("INPUT_U16", out=self.tensor(outs[0]), p=(slot, *origin, *params))
            else:
                self.op("INPUT", out=self.contiguous_f32(outs[0]), p=(slot, *origin))
            return
        if base in ("*OutputSlice", "*OutputCast_fp16_to_fp32"):
            origin = self.s4(node, 3 if base == "*OutputCast_fp16_to_fp32" else 2)
            slot = self.s4(node, -1)[-1]
            self.check_io_slot("outputs", slot, self.s4(node, 1))
            params = (slot, origin[1], origin[2], origin[0])
            if base == "*OutputSlice" and T[ins[0]].get("dtype") == "uint16":
                self.op("OUTPUT_U16", self.tensor(ins[0]),
                        p=(*params, *self.io_quantization("outputs", slot, ins[0])))
            else:
                x = self.contiguous_f32(ins[0]) if base == "*OutputSlice" else self.tensor(ins[0])
                if base == "*OutputCast_fp16_to_fp32" and self.dtype(T[ins[0]]) != 0:
                    raise ValueError("output cast source is not FP16")
                self.op("OUTPUT", x, p=params)
            return
        if base.startswith("ForceFormat_"):
            source, output = T[ins[0]], T[outs[0]]
            if "uint16" in (source["dtype"], output["dtype"]) and (
                    source["dtype"] != output["dtype"] or not source.get("quantization")
                    or source["quantization"] != output.get("quantization")):
                raise ValueError("U16 format conversion requires matching quantization")
            self.op("FORMAT", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        if base.startswith("Cast_fp16_to_fp32"):
            self.op("CAST_F32", self.tensor(ins[0]), out=self.contiguous_f32(outs[0]))
            return
        if base.startswith("Cast_fp32_to_fp16"):
            self.op("CAST_F16", self.contiguous_f32(ins[0]), out=self.tensor(outs[0]))
            return
        if base in ("Mul.fp16", "Mul.fp16_no_bc"):
            if base == "Mul.fp16_no_bc" and self.dumps[ins[2]] != bytes(4):
                raise ValueError("unsupported Mul.fp16_no_bc options")
            self.op("MUL", self.tensor(ins[0]), self.tensor(ins[1]), out=self.tensor(outs[0]))
            return
        if base.startswith("ReluMinMax.fp16"):
            lo = self.const_values(ins[1], "f")[0]
            hi = self.const_values(ins[2], "f")[0]
            self.op("MINMAX", self.tensor(ins[0]), out=self.tensor(outs[0]),
                    p=(struct.unpack("<I", struct.pack("<f", lo))[0],
                       struct.unpack("<I", struct.pack("<f", hi))[0]))
            return
        if base.startswith("Sigmoid.fp16"):
            self.op("SIGMOID", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        if base == "reduce_sum.fp16.tcm.crouton":
            self.op("REDUCE_SUM", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        if base == "reduce_mean.fp16.tcm.crouton":
            # The axis input is implied by the shapes: every reduced dimension is 1 in the output.
            self.op("REDUCE_MEAN", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        if base == "norm.fp16.tcm":
            # L2 normalization along one axis. Inputs: axis, a layout descriptor, epsilon.
            axis = self.const_values(ins[1], "i")
            if len(axis) != 1 or not 0 <= axis[0] < 4:
                raise ValueError(f"norm {node['sequence']}: axis {axis}")
            self.op("L2NORM", self.tensor(ins[0]), out=self.tensor(outs[0]),
                    p=(axis[0], struct.unpack("<I", self.dumps[ins[3]])[0]))
            return
        if base == "Flatten":
            self.op("RESHAPE", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        if base == "Batchnorm.fp16.scalar.tcm":
            # out = x * scale + bias with one FP16 scale and one float32 bias.
            scale = struct.unpack("<e", self.dumps[ins[1]])[0]
            self.op("AFFINE", self.tensor(ins[0]), out=self.tensor(outs[0]),
                    p=(struct.unpack("<I", struct.pack("<f", scale))[0], struct.unpack("<I", self.dumps[ins[2]])[0]))
            return
        if base == "fp16_flat_output_scalar_fm.tcm":
            # float32 out = FP16 x times one FP16 scalar.
            scale = struct.unpack("<e", self.dumps[ins[1]])[0]
            self.op("AFFINE", self.tensor(ins[0]), out=self.contiguous_f32(outs[0]),
                    p=(struct.unpack("<I", struct.pack("<f", scale))[0], 0))
            return
        if base == "InstanceNorm_sf.fp16.tcm":
            self.op("INSTANCE_NORM", self.tensor(ins[0]), self.contiguous_f32(ins[1]),
                    self.tensor(ins[2]), self.tensor(outs[0]), (self.contiguous_f32(ins[3]),))
            return
        if base == "va_parallel_welford_reduce_grouped_moments_sf":
            factor = self.materialize_f32(ins[0]); ratio = self.materialize_f32(ins[1])
            gamma = self.materialize_f32(ins[4]); source = self.contiguous_f32(ins[6])
            inverse = T[ins[2]]["scalar"]
            epsilon = self.const_values(ins[3], "f")[0]
            bits = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
            self.op("GROUP_MOMENTS", source, factor, ratio, self.contiguous_f32(outs[0]),
                    (gamma, bits(inverse), bits(epsilon), T[ins[5]]["scalar"]))
            return
        if base in ("flat_from_vtcm", "flat_to_vtcm"):
            self.op("MOVE", self.contiguous_f32(ins[0]), out=self.contiguous_f32(outs[0]))
            return
        if base == "va_qf32_parallel_welford_no_moments_sf.tcm":
            first = len(self.words)
            ids = [self.contiguous_f32(i) for i in ins[2:]]
            self.words.extend(ids)
            self.op("WELFORD_MERGE", self.contiguous_f32(ins[0]), self.contiguous_f32(ins[1]),
                    out=self.contiguous_f32(outs[0]), p=(first, len(ids)))
            return
        if base == "InstanceNorm.fp16.compute_welford_partial_sums_sf.tcm":
            self.op("WELFORD_PARTIAL", self.tensor(ins[0]), self.contiguous_f32(ins[1]),
                    out=self.contiguous_f32(outs[0]))
            return
        if base == "Swish_fp16":
            self.op("SWISH", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        if base == "ResizeNear_slice.tcm_special":
            dy, dx = [self.const_values(i, "i") for i in ins[1:3]]
            if len(dy) != 8 or len(dx) != 8 or dy[2] != -2147483648 or dx[2] != -2147483648:
                raise ValueError("unsupported nearest resize descriptor")
            self.op("RESIZE_NEAREST", self.tensor(ins[0]), out=self.tensor(outs[0]), p=(dy[5], dx[5]))
            return
        if base == "Softmax":
            if struct.unpack("<e", self.dumps[ins[1]])[0] != 1.0:
                raise ValueError("unsupported softmax scale")
            self.op("SOFTMAX", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        if base == "ResizeBilinear.fp16.tcm^010":
            if self.dumps[ins[1]] != b"\x00" or self.dumps[ins[2]] != b"\x01":
                raise ValueError("unsupported bilinear resize options")
            coords = [self.const_values(tid, "f")[0] for tid in ins[3:7]]
            bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in coords]
            self.op("RESIZE_BILINEAR", self.tensor(ins[0]), out=self.tensor(outs[0]), p=bits)
            return
        if base == "Decimate.tcm":
            stride = self.s4(node, 1)
            if stride[0] != 1 or stride[3] != 1 or not stride[1] or not stride[2]:
                raise ValueError("unsupported decimation stride")
            self.op("DECIMATE", self.tensor(ins[0]), out=self.tensor(outs[0]), p=stride[1:3])
            return
        if name == "Slice_reduce_batch@FH.F5H.s5.s4":
            source, output = T[ins[0]], T[outs[0]]
            start, size = (self.s4(node, i) for i in (1, 2))
            shape = source["shape"]["dimensions"]
            if (source["dtype"] != "uint16" or output["dtype"] != "uint16"
                    or not source.get("quantization") or source["quantization"] != output.get("quantization")
                    or len(shape) != 5 or size != output["shape"]["dimensions"] or size != shape[1:]
                    or len(start) != 5 or any(start[1:]) or not 0 <= start[0] < shape[0]):
                raise ValueError("U16 batch reduction requires a complete outer slice")
            self.op("SLICE_U16", self.tensor(ins[0]), out=self.tensor(outs[0]), p=(start[0] * shape[1], 0, 0, 0))
            return
        if name == "Slice_contig.tcm@FH*2.s4*2":
            source, output = T[ins[0]], T[outs[0]]
            start, end = (self.s4(node, i) for i in (1, 2))
            if (source["dtype"] != "uint16" or output["dtype"] != "uint16"
                    or not source.get("quantization") or source["quantization"] != output.get("quantization")
                    or len(start) != 4 or any(end)):
                raise ValueError("U16 contiguous slice requires matching encodings and an unpadded crop")
            self.op("SLICE_U16", self.tensor(ins[0]), out=self.tensor(outs[0]), p=start)
            return
        if base == "SlicePad_shape_inplace":
            pad, start, end = (self.s4(node, i) for i in (1, 2, 3))
            if name == "SlicePad_shape_inplace@FH*2.s4*3.cH":
                src, dst = T[ins[0]], T[outs[0]]
                if any(pad) or any(end) or src["dtype"] != "uint16" or dst["dtype"] != "uint16":
                    raise ValueError("U16 SlicePad requires an unpadded crop")
                if not src.get("quantization") or src["quantization"] != dst.get("quantization"):
                    raise ValueError("U16 SlicePad input and output encodings differ")
                for tensor in (src, dst):
                    shape = tensor["shape"]
                    if (not tensor["storage"].get("direct") or shape["dimensions"] != shape["max_dimensions"]
                            or any(shape["padding"])):
                        raise ValueError("U16 SlicePad requires compact direct tensors")
                if any(offset < 0 or offset + size > extent for offset, size, extent in zip(
                        start, dst["shape"]["dimensions"], src["shape"]["dimensions"])):
                    raise ValueError("U16 SlicePad crop is outside the input")
                self.op("SLICE_U16", self.tensor(ins[0]), out=self.tensor(outs[0]), p=start)
            else:
                if any(pad) or any(end) or any(start[:3]):
                    raise ValueError("unsupported SlicePad offsets")
                self.op("SLICE_F32", self.contiguous_f32(ins[0]), out=self.contiguous_f32(outs[0]), p=(start[3],))
            return
        if base == "Reshape":
            self.op("RESHAPE", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        if base in ("Concat", "Concat_1h"):
            operands = [T[t] for t in ins[1:] + outs]
            if any(t["dtype"] == "uint16" for t in operands) and (
                    not operands[-1].get("quantization") or
                    any(t["dtype"] != "uint16" or t.get("quantization") != operands[-1]["quantization"]
                        for t in operands)):
                raise ValueError("U16 concatenation requires matching encodings")
            if "scalar" in T[ins[0]]:
                axis = T[ins[0]]["scalar"]
            else:
                # Some QAIRT concat axis handles use an extended argument
                # layout, so the generic capture slot is not the axis. Infer
                # it only when the shapes uniquely identify the axis.
                shapes = [T[i]["shape"]["dimensions"] for i in ins[1:]]
                output_shape = T[outs[0]]["shape"]["dimensions"]
                axes = [a for a in range(4) if all(
                    sum(s[d] for s in shapes) == output_shape[d] if d == a else
                    all(s[d] == output_shape[d] for s in shapes) for d in range(4))]
                if len(axes) != 1:
                    raise ValueError(f"concat axis ambiguous: {axes}")
                axis = axes[0]
            # Register the operands first: a tensor seen here for the first time appends its block table.
            ids = [self.tensor(i) for i in ins[1:]]
            out = self.tensor(outs[0])
            first = len(self.words)
            self.words.extend(ids)
            self.op("CONCAT", out=out, p=(axis, first, len(ids)))
            return
        if base == "AlignedSlice":
            self.op("SLICE", self.tensor(ins[0]), out=self.tensor(outs[0]), p=(3, self.s4(node, 1)[3]))
            return
        if base.startswith("ChannelShuffle"):
            self.op("SHUFFLE", self.tensor(ins[0]), out=self.tensor(outs[0]), p=(T[ins[1]]["scalar"],))
            return
        if base == "Transpose.2D" or base.startswith("Transpose_impl"):
            if name == "Transpose_impl^010@FH*2.fi.t" and T[ins[0]]["dtype"] == "uint16":
                source, output = T[ins[0]], T[outs[0]]
                dims = source["shape"]["dimensions"]
                if (len(dims) != 4 or len(set(dims)) != 4 or output["dtype"] != "uint16"
                        or not source.get("quantization") or source["quantization"] != output.get("quantization")
                        or output["shape"]["dimensions"] != [dims[0], dims[1], dims[3], dims[2]]):
                    raise ValueError("unsupported U16 transpose")
                perm = [0, 1, 3, 2]
            else:
                perm = self.const_values(ins[2 if base.startswith("Transpose_impl") else 1], "i")
            if sorted(perm) != [0, 1, 2, 3]:
                raise ValueError(f"transpose {node['sequence']}: perm {perm}")
            self.op("TRANSPOSE", self.tensor(ins[0]), out=self.tensor(outs[0]), p=perm)
            return
        if base.startswith("AvgPool_with_pad"):
            k = self.s4(node, 1)[1]
            self.op("AVGPOOL", self.tensor(ins[0]), out=self.tensor(outs[0]), p=(k, k))
            return
        if base == "x2s_opt":
            if self.const_values(ins[1], "i") != [2, 2]:
                raise ValueError("x2s_opt block is not 2x2")
            self.op("X2S", self.tensor(ins[0]), out=self.tensor(outs[0]))
            return
        raise ValueError(f"operation {node['sequence']} has no handler: {name}")

    def conv1x1(self, node, x, w, fi, out, wd, tables):
        T = self.T
        xi, oi = self.tensor(x), self.tensor(out)
        xr, orow = self.tensor_rows[xi], self.tensor_rows[oi]
        if xr[0] or orow[0]:
            raise ValueError(f"1x1 conv {node['sequence']} needs crouton tensors")
        xfull = T[x]["shape"]["max_dimensions"]
        K, N = -(-xfull[3] // 32), -(-wd[3] // 32)
        M = xr[5] // K
        if -(-wd[2] // 32) != K or M * K != xr[5] or orow[5] != M * N:
            raise ValueError(f"1x1 conv {node['sequence']}: block tables do not match M={M} K={K} N={N}")
        a = self.words[xr[4]:xr[4] + M * K]
        c = self.words[orow[4]:orow[4] + M * N]
        if NONE in a or NONE in c:
            raise ValueError(f"1x1 conv {node['sequence']}: absent block in an HMX operand")
        first = len(self.words)
        self.words.extend(a + c)
        self.op("CONV1X1", xi, self.tensor(w), self.tensor(fi), oi, (M, K, N, tables, first))

    def spill_fill(self, node):
        first = len(self.words)
        words = []
        extent = self.spill_bytes
        for group in node["spill_groups"]:
            offset = group["offset"]
            if offset < 0 or offset % BLOCK or not group["transfers"]:
                raise ValueError("invalid Q16P spill group offset or count")
            for tagged, size in group["transfers"]:
                vtcm = tagged & ~(BLOCK - 1)
                if (tagged % BLOCK != 2 or vtcm < 0 or size <= 0 or size % BLOCK
                        or vtcm + size > COMPLETION or offset + size > 0x20000000):
                    raise ValueError("unsupported Q16P spill/fill transfer")
                words.extend((vtcm, offset, size))
                offset += size
            extent = max(extent, offset)
        if not words:
            raise ValueError("empty Q16P spill/fill")
        self.words.extend(words)
        self.spill_bytes = extent
        self.op("SPILL_FILL", p=(first, len(words) // 3, int(node["type_name"] == "@Spill")))

    def io_quantization(self, kind, slot, tid):
        port = self.T[self.graph["graph_" + kind][slot]]
        tile = self.T[tid]
        quant = port.get("quantization")
        if port["dtype"] != "uint16" or not quant or tile.get("quantization") != quant:
            raise ValueError(f"U16 {kind} slot {slot} and tensor {tid} encodings differ")
        scale = struct.unpack("<I", struct.pack("<f", quant["scale"]))[0]
        zero = quant["zero_point"]
        if not 0x00800000 <= scale < 0x77800000 or not 0 <= zero <= 65535:
            raise ValueError(f"invalid U16 {kind} slot {slot} encoding")
        return scale, zero

    def check_io_slot(self, kind, slot, dims):
        ids = self.graph["graph_inputs" if kind == "inputs" else "graph_outputs"]
        if slot >= len(ids) or self.T[ids[slot]]["shape"]["dimensions"] != list(dims):
            raise ValueError(f"{kind} slot {slot} does not match {dims}")


def find_context_binary(context_dir, graph):
    """The context binary beside the htp_graph.py output directory whose SHA-256 the graph records, or None."""
    for path in sorted(Path(context_dir).resolve().parent.glob("*.bin")):
        if hashlib.sha256(path.read_bytes()).hexdigest() == graph["context_sha256"]:
            return path
    return None


def lower(context_dir, trace_dir, tensors_path, output_prefix, known=None, context_binary=None):
    graph = load(Path(context_dir) / "graph.json")
    if context_binary is not None:
        if hashlib.sha256(Path(context_binary).read_bytes()).hexdigest() != graph["context_sha256"]:
            raise ValueError(f"{context_binary} is not the context {context_dir} was decoded from")
    else:
        context_binary = find_context_binary(context_dir, graph)
    rows = read_trace(trace_dir)
    entries = map_entries(graph, rows, known, read_worker_checkpoints(trace_dir))
    L = Lowering(graph, read_dumps(tensors_path), context_binary)
    seen = set()
    for row in rows:
        node = entries[(row[1], row[2])]
        if node is not None and node["sequence"] not in seen:
            seen.add(node["sequence"])
            L.current = node["sequence"]
            L.lower_node(node)
    L.compact_stride2_weights()

    io = []
    exchange = 0
    for kind in ("graph_inputs", "graph_outputs"):
        for tid in graph[kind]:
            dims = L.T[tid]["shape"]["dimensions"]
            nbytes = math.prod(dims) * 4
            io.append(list(dims) + [exchange, nbytes, 0, 0])
            exchange = align(exchange + nbytes)

    header_bytes = 128
    tensor_offset = header_bytes
    op_offset = tensor_offset + 80 * len(L.tensor_rows)
    word_offset = op_offset + 64 * len(L.ops)
    io_offset = word_offset + 4 * len(L.words)
    total = io_offset + 32 * len(io)
    constants = bytes(L.constants)
    scratch = scratch_bytes(L) + L.arena_bytes
    version = 3 if L.spill_bytes else 2 if L.arena_bytes else VERSION
    header = struct.pack("<16I", MAGIC, version, total, len(L.tensor_rows), len(L.ops), len(L.words),
                         len(graph["graph_inputs"]), len(graph["graph_outputs"]),
                         tensor_offset, op_offset, word_offset, io_offset,
                         len(constants), scratch, exchange, VTCM)
    header += hashlib.sha256(constants).digest() + struct.pack("<8I", L.arena_bytes, L.spill_bytes, 0, 0, 0, 0, 0, 0)
    body = b"".join(struct.pack("<20I", *r) for r in L.tensor_rows)
    body += b"".join(struct.pack("<16I", *o) for o in L.ops)
    body += struct.pack(f"<{len(L.words)}I", *L.words)
    body += b"".join(struct.pack("<8I", *r) for r in io)
    image = header + body
    assert len(image) == total
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".q16p").write_bytes(image)
    prefix.with_suffix(".q16c").write_bytes(constants)
    counts = {}
    for o in L.ops:
        counts[KINDS[o[0] - 1]] = counts.get(KINDS[o[0] - 1], 0) + 1
    summary = {"format": "Q16P-3" if L.spill_bytes else "Q16P-1", "context_sha256": graph["context_sha256"],
               "image_bytes": total, "image_sha256": hashlib.sha256(image).hexdigest(),
               "constant_bytes": len(constants), "constant_sha256": hashlib.sha256(constants).hexdigest(),
               "scratch_bytes": scratch, "exchange_bytes": exchange, "spill_bytes": L.spill_bytes, "tensors": len(L.tensor_rows),
               "ops": len(L.ops), "op_counts": counts,
               "inputs": [r[:4] for r in io[:len(graph["graph_inputs"])]],
               "outputs": [r[:4] for r in io[len(graph["graph_inputs"]):]],
               "constants_source": "qnn-simulator-tensor-dump", "package_status": "development-only",
               "op_nodes": L.op_nodes}
    prefix.with_suffix(".q16.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def elements(row):
    return row[6] * row[7] * row[8] * row[9]


def scratch_bytes(L):
    """Mirror of q16_op_scratch in program.c: logical FP16/float32 staging."""
    R = L.tensor_rows

    def logical(t):
        return align(elements(R[t]) * 2)

    def f32(n):
        return align(n * 4)

    need = 0
    for o in L.ops:
        kind, x, y, out = KINDS[o[0] - 1], o[1], o[2], o[4]
        if kind in ("DW1X1", "DW3X3", "CONV3X3", "DW3X3_S2", "CONV3X3_S2", "S2D", "PW1X1", "PW_SWISH", "PW_TANH"):
            oc, xc = R[out][9], R[x][9]
            w = {"DW3X3": oc * 9, "DW3X3_S2": oc * 9, "DW1X1": oc * 2, "CONV3X3": oc * 9 * xc,
                 "PW1X1": oc * xc, "PW_SWISH": oc * xc, "PW_TANH": oc * xc,
                 "CONV3X3_S2": oc * 9 * (xc // 4)}.get(kind, 0)
            if kind == "S2D":
                oc = 0
            n = f32(elements(R[x])) + f32(elements(R[out])) + f32(w) + 3 * f32(oc)
        elif kind in ("MINMAX", "SIGMOID", "SWISH", "RESHAPE",
                      "SLICE", "SHUFFLE", "TRANSPOSE", "AVGPOOL", "X2S", "FORMAT", "ADD_SWISH"):
            n = logical(x) + logical(out)
        elif kind in ("REDUCE_SUM", "REDUCE_MEAN"):
            n = logical(x) + f32(elements(R[out]))
        elif kind == "L2NORM":
            n = logical(x) + f32(elements(R[x]))
        elif kind == "AFFINE":
            n = logical(x)
        elif kind == "INSTANCE_NORM":
            n = logical(x) + logical(o[3])
        elif kind == "WELFORD_PARTIAL":
            n = logical(x)
        elif kind == "MUL":
            n = logical(x) + logical(y) + logical(out)
        elif kind == "CONCAT":
            n = logical(out) + sum(logical(t) for t in L.words[o[6]:o[6] + o[7]])
        elif kind == "CAST_F16":
            n = logical(out)
        elif kind == "OUTPUT":
            n = f32(elements(R[x]))
        elif kind == "OUTPUT_U16":
            n = logical(x)
        elif kind == "CONV1X1":
            n = align(8 * (o[5] * o[6] + o[5] * o[7])) + 128
        else:
            n = 0
        need = max(need, n)
    return align(need, 4096)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("context", type=Path)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--tensors", type=Path, required=True, help="QD_TENSORS pool-2 capture")
    ap.add_argument("--known", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="output prefix")
    ap.add_argument("--context-binary", type=Path,
                    help="the QNN context .bin (default: the one beside CONTEXT with the recorded SHA-256)")
    a = ap.parse_args()
    s = lower(a.context, a.trace, a.tensors, a.output, load(a.known), a.context_binary)
    print(f"Q16P: {s['ops']} ops, {s['tensors']} tensors, {s['constant_bytes']} constant bytes, "
          f"scratch {s['scratch_bytes']} -> {a.output}.q16p")


if __name__ == "__main__":
    main()
