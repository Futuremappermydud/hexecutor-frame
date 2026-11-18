"""Shared HXFG v1 serialization for the offline lowerings.

The source-format front ends (TFLite, ONNX) build a list of operations over
their own tensor ids and hand it here. This module owns the constant pool,
HMX weight packing, activation allocation and the binary layout checked by
`hexecutor/native/tensor/graph.c`.
"""

import hashlib
import json
import struct
import numpy as np

NONE = 0xFFFFFFFF
# Values of the HX_* operation enum in hexecutor/native/tensor/graph.h.
(CONV, DEPTHWISE, ADD, PRELU, MAXPOOL, PAD, RESIZE, RESHAPE, CONCAT, MEAN, LOGISTIC, ACT, BINARY,
 AVGPOOL, SOFTMAX, SLICE, ZEROINSERT, TRANSPOSE, REDUCE, NORM) = range(1, 21)
# HX_MATMUL: x = A [L, K] and y = B, a runtime tensor (not a constant); p[0] transB, p[1] scale.
MATMUL = 21
RMSNORM = 22
SDPA = 27
ATTN_SOFTMAX = 28
GATHER, WHERE, ARGMAX, TOPK, TILE, CAST = range(32, 38)
# HX_ACT functions, HX_BINARY and HX_REDUCE operations (graph.h).
(FN_RELU, FN_RELU6, FN_HARDSWISH, FN_HARDSIGMOID, FN_SILU, FN_TANH, FN_LEAKY,
 FN_GELU, FN_EXP, FN_SQRT, FN_ABS, FN_NEG, FN_ERF,
 FN_LOG, FN_RECIPROCAL, FN_FLOOR, FN_CEIL, FN_SIN, FN_COS, FN_NOT, FN_SIGN, FN_ROUND) = range(22)
BIN_ADD, BIN_SUB, BIN_MUL, BIN_DIV, BIN_MIN, BIN_MAX, BIN_POW, BIN_EQUAL, BIN_GREATER, BIN_LESS = range(10)
# HX_NORM modes (graph.h): per pixel over channels, or per group of channels over all pixels.
NORM_CHANNEL, NORM_GROUP = range(2)
RED_MEAN, RED_SUM, RED_MAX, RED_MIN = range(4)


def f32_bits(value):
    """Bit pattern of a float32, as stored in an operation's p[] fields."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]
# Preflight limit on any tensor dimension, including channels (graph.c).
MAX_CHANNELS = 4096
# Fused activation codes for convolution, depthwise and add.
ACT_NONE, ACT_RELU, ACT_RELU_N1_1, ACT_RELU6 = range(4)


def aligned(n):
    return (n + 127) & ~127


class ConstantPool:
    """128-byte aligned immutable constant pool."""

    def __init__(self):
        self.data = bytearray()

    def emit(self, array, dtype="<f4"):
        at = aligned(len(self.data))
        self.data.extend(bytes(at - len(self.data)))
        self.data.extend(np.asarray(array, dtype=dtype).tobytes())
        return at


def pack_hmx_weights(w):
    """Pack [kh, kw, cin, cout_stride] float weights into the FP16 HMX tile layout.

    Returns (packed array, K tile count) for convolution p[9]/p[10].
    """
    matrix = w.reshape(-1, w.shape[-1])
    k, n = matrix.shape
    kp = (k + 31) & ~31
    matrix = np.pad(matrix, ((0, kp - k), (0, 0)))
    packed = matrix.reshape(kp // 32, 16, 2, n // 32, 32).transpose(3, 0, 1, 4, 2)
    return packed, kp // 32


def hmx_fits(tiles, out_stride):
    """Mirror of the native preflight's per-op HMX scratch budget (graph.c)."""
    return (
        tiles <= 128
        and tiles * 2048 + tiles * 32 * out_stride * 2 + out_stride * 64 + 4096 + tiles * 256
        <= 0x100000
    )


def tensor_refs(op):
    """Tensor-id fields (x, y, z, out) of an op. A conv/depthwise z is a raw dilation, not an id.
    HX_MATMUL reads two tensors: x (A) and y (B, a runtime input), so y is always a live input."""
    if op[0] == MATMUL and op[2] == NONE:
        raise ValueError("HX_MATMUL needs a runtime tensor as its second operand")
    z = NONE if op[0] in (CONV, DEPTHWISE, MAXPOOL, AVGPOOL) else op[3]
    refs = [op[1], op[2], z, op[4]]
    if op[0] == SDPA and len(op) > 5 and op[5] != NONE:
        refs.append(op[5])
    return refs


def dilation_z(op):
    """Encoded z for a conv/depthwise op: NONE for dilation 1, else (dh << 8) | dw (1..8 each)."""
    z = op[3]
    if z == NONE:
        return NONE
    dh, dw = (z >> 8) & 0xFF, z & 0xFF
    if op[0] not in (CONV, DEPTHWISE, MAXPOOL, AVGPOOL) or z >> 16 or not (1 <= dh <= 8 and 1 <= dw <= 8) or (
        dh == 1 and dw == 1
    ):
        raise ValueError(f"invalid z {z:#x} on op kind {op[0]}")
    return z


def write(ops, inputs, outputs, hwc, stride, pool, destination, manifest):
    """Allocate activations and write `<destination>.hxfg/.bin/.json`.

    ops: [kind, x, y, z, out, p0..p10] over front-end tensor ids.
    hwc(i)/stride(i): static NHWC shape (batch one) and channel stride.
    pool: ConstantPool or bytearray. manifest: dict with "source",
    "source_sha256" and "ops"; arena/constant/IO fields are added here.
    """
    pool = pool.data if isinstance(pool, ConstantPool) else pool
    active = sorted(
        set(inputs + outputs + [i for op in ops for i in tensor_refs(op) if i != NONE])
    )
    ids = {old: new for new, old in enumerate(active)}
    last = {i: -1 for i in active}
    for j, op in enumerate(ops):
        inputs_of_op = [op[1], op[2]]
        if op[0] not in (CONV, DEPTHWISE, TOPK) and op[3] != NONE:
            inputs_of_op.append(op[3])
        if op[0] == SDPA and len(op) > 5 and op[5] != NONE:
            inputs_of_op.append(op[5])
        for i in inputs_of_op:
            if i != NONE:
                last[i] = j
    for i in outputs:
        last[i] = len(ops)
    # Reuse dead tensors, but never alias an op's input and output. This is
    # independently checked by the native preflight before hardware access.
    free, allocated, offsets, peak = [], {}, {}, 0

    def allocate(i):
        nonlocal peak
        n = aligned(int(np.prod(hwc(i)[:2])) * stride(i) * 4)
        choice = next((j for j, (_, size) in enumerate(free) if size >= n), None)
        if choice is None:
            at = peak
            peak += n
        else:
            at, size = free.pop(choice)
            if size > n:
                free.append((at + n, size - n))
        allocated[i] = (at, n)
        offsets[i] = at

    for i in inputs:
        allocate(i)
    for j, op in enumerate(ops):
        for i in list(allocated):
            if last[i] < j:
                free.append(allocated.pop(i))
        free.sort()
        merged = []
        for at, n in free:
            if merged and sum(merged[-1]) == at:
                merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:
                merged.append((at, n))
        free[:] = merged
        allocate(op[4])
        if op[0] == TOPK and op[3] != NONE:
            allocate(op[3])
    if not pool:
        pool.extend(bytes(128))
    tensors = b"".join(
        struct.pack("<8I", offsets[i], *hwc(i), stride(i), 0, 0, 0) for i in active
    )
    packed_ops = b"".join(
        struct.pack(
            "<16I",
            op[0],
            ids.get(op[1], NONE),
            ids.get(op[2], NONE),
            dilation_z(op) if op[0] in (CONV, DEPTHWISE, MAXPOOL, AVGPOOL) else ids.get(op[3], NONE),
            ids.get(op[4], NONE),
            ids.get(op[5], NONE) if op[0] == SDPA and len(op) > 5 and op[5] != NONE else op[5],
            *op[6:],
        )
        for op in ops
    )
    io = struct.pack(
        "<" + "I" * (len(inputs) + len(outputs)), *(ids[i] for i in inputs + outputs)
    )
    to = 256
    oo, io_at = to + len(tensors), to + len(tensors) + len(packed_ops)
    header = (
        struct.pack(
            "<12I",
            0x47465848,
            1,
            io_at + len(io),
            len(active),
            len(ops),
            len(inputs),
            len(outputs),
            peak,
            len(pool),
            to,
            oo,
            io_at,
        )
        + hashlib.sha256(pool).digest()
        + bytes(176)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".hxfg").write_bytes(header + tensors + packed_ops + io)
    destination.with_suffix(".bin").write_bytes(pool)
    full = {
        "source": manifest["source"],
        "source_sha256": manifest["source_sha256"],
        "format": "HXFG-v1",
        "arena_bytes": peak,
        "constant_bytes": len(pool),
        "ops": manifest["ops"],
        "inputs": [{"id": ids[i], "shape": manifest["shapes"][i]} for i in inputs],
        "outputs": [{"id": ids[i], "shape": manifest["shapes"][i]} for i in outputs],
    }
    full.update(manifest.get("extra", {}))
    destination.with_suffix(".json").write_text(json.dumps(full, indent=2) + "\n")
    return full
