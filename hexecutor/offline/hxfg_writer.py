# SPDX-License-Identifier: LGPL-3.0-or-later
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
# HX_ROPE (x token rows, y optional position tensor): p[0] head_dim, p[1] heads, p[2] static position,
# p[3] mode (0 split halves, 1 adjacent pairs), p[4] theta float bits (0: 10000), p[5] rotated width
# per head (0: all), p[6] 1 adds the row index to the position.
ROPE = 23
# HX_SHORTCONV: x token [1,1,C], y persistent ring [k-1,1,C], z optional gate id, p[0] taps k,
# p[1] weight offset [k, stride], p[2] bias offset or NONE, p[3] activation (0 none, 1 relu, 3 silu).
SHORTCONV = 24
# HX_LINEAR_Q: x [L,1,K], out [L,1,N]; p[0] format (0 Q4_0, 1 Q8_0), p[1] weights offset,
# p[2] bias offset or NONE, p[3] K, p[4] N, p[5] activation. Rows are independent.
LINEAR_Q = 25
# HX_KV_WRITE: x [Lin,1,C], out persistent cache [Lmax,1,C], y optional position tensor, p[0] position.
KV_WRITE = 26
SDPA = 27
ATTN_SOFTMAX = 28
GATHER_ELEMENTS = 29
GATHER_ND = 30
SCATTER_ND = 31
GATHER, WHERE, ARGMAX, TOPK, TILE, CAST, ARGMIN = range(32, 39)
# HX_ACT functions, HX_BINARY and HX_REDUCE operations (graph.h).
(FN_RELU, FN_RELU6, FN_HARDSWISH, FN_HARDSIGMOID, FN_SILU, FN_TANH, FN_LEAKY,
 FN_GELU, FN_EXP, FN_SQRT, FN_ABS, FN_NEG, FN_ERF,
 FN_LOG, FN_RECIPROCAL, FN_FLOOR, FN_CEIL, FN_SIN, FN_COS, FN_NOT, FN_SIGN, FN_ROUND,
 FN_ELU, FN_SELU, FN_CELU, FN_SOFTPLUS, FN_MISH, FN_SOFTSIGN, FN_LOGSIGMOID,
 FN_SINH, FN_COSH) = range(31)
(BIN_ADD, BIN_SUB, BIN_MUL, BIN_DIV, BIN_MIN, BIN_MAX, BIN_POW,
 BIN_EQUAL, BIN_GREATER, BIN_LESS, BIN_GE, BIN_LE, BIN_AND, BIN_OR) = range(14)
BIN_GREATER_EQUAL = BIN_GE
BIN_LESS_EQUAL = BIN_LE

# HX_DTYPE enum
DTYPE_FLOAT32 = 0
DTYPE_INT32 = 1
DTYPE_BOOL = 2
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
    """128-byte aligned immutable constant pool.

    With dedupe, emitting bytes identical to an earlier emission returns the earlier offset, so a
    graph that repeats a subnetwork (one module called on several inputs) stores its weights once
    and the DSP plan keeps one resident copy. Only for callers that take every offset from emit()
    and never assume two emissions are adjacent.
    """

    def __init__(self, dedupe=False):
        self.data = bytearray()
        self.seen = {} if dedupe else None

    def emit(self, array, dtype=None):
        if dtype is None:
            arr = np.asarray(array)
            dtype = "<i4" if arr.dtype.kind in ("i", "u") and arr.dtype.itemsize == 4 else "<f4"
        raw = np.asarray(array, dtype=dtype).tobytes()
        if self.seen is not None and raw in self.seen:
            return self.seen[raw]
        at = aligned(len(self.data))
        self.data.extend(bytes(at - len(self.data)))
        self.data.extend(raw)
        if self.seen is not None:
            self.seen[raw] = at
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


def quantize_q4_0(weights):
    """Pack a float matrix [N, K] (K a multiple of 32) as HX_LINEAR_Q Q4_0 rows.

    Each 32-weight block is 16 bytes (low nibble holds element 2j, high nibble 2j+1, both offset by 8)
    followed by a float16 scale. As in GGML the scale is the block's largest-magnitude value divided by
    -8, so that value lands exactly on code 0 and every other value is within half a step.
    Returns (bytes, dequantized float32 matrix).
    """
    w = np.asarray(weights, dtype=np.float32)
    n, k = w.shape
    if k % 32:
        raise ValueError("K must be a multiple of 32")
    blocks = w.reshape(n, k // 32, 32)
    peak = np.take_along_axis(blocks, np.abs(blocks).argmax(axis=2)[..., None], axis=2)[..., 0]
    scale = (peak / -8.0).astype(np.float16)
    sf = scale.astype(np.float32)
    inv = np.where(sf != 0, 1.0 / np.where(sf != 0, sf, 1.0), 0.0)
    q = np.clip(np.floor(blocks * inv[..., None] + 8.5), 0, 15).astype(np.int32)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    out = np.concatenate([packed, scale.view(np.uint8).reshape(n, k // 32, 2)], axis=2)
    deq = ((q - 8).astype(np.float32) * sf[..., None]).reshape(n, k)
    return out.tobytes(), deq


def quantize_q8_0(weights):
    """Pack a float matrix [N, K] as HX_LINEAR_Q Q8_0 rows (32 int8 plus a float16 scale per block)."""
    w = np.asarray(weights, dtype=np.float32)
    n, k = w.shape
    if k % 32:
        raise ValueError("K must be a multiple of 32")
    blocks = w.reshape(n, k // 32, 32)
    amax = np.abs(blocks).max(axis=2)
    scale = (amax / 127.0).astype(np.float16)
    sf = scale.astype(np.float32)
    inv = np.where(sf > 0, 1.0 / np.where(sf > 0, sf, 1.0), 0.0)
    q = np.clip(np.rint(blocks * inv[..., None]), -127, 127).astype(np.int8)
    out = np.concatenate([q.view(np.uint8), scale.view(np.uint8).reshape(n, k // 32, 2)], axis=2)
    deq = (q.astype(np.float32) * sf[..., None]).reshape(n, k)
    return out.tobytes(), deq


def rope_theta(cos, sin, rot):
    """The base of the rotary table cos/sin(position * base ** (-2j / rot)), as the float32 the engine
    uses. Raises ValueError unless the caches are exactly that table (to 5e-4 over their first 4096
    positions), so a scaled or otherwise modified rotary embedding is never lowered as a plain one."""
    cos, sin = np.asarray(cos, np.float64), np.asarray(sin, np.float64)
    half = rot // 2
    if cos.ndim != 2 or cos.shape != sin.shape or cos.shape[1] != half:
        raise ValueError("cos and sin caches must be constant [max_position, rotary_dim / 2] tables")
    if half == 1:
        theta = 10000.0  # one frequency of exactly 1: the base never matters
    else:
        angle = float(np.arctan2(sin[1, half - 1], cos[1, half - 1])) if cos.shape[0] > 1 else 0.0
        if not 0.0 < angle < 1.0:
            raise ValueError("cannot recover a rotary base from the caches "
                             "(position 1 must give a small angle on the slowest frequency)")
        theta = angle ** (-rot / (rot - 2.0))
    theta = float(np.float32(theta))
    rows = min(cos.shape[0], 4096)
    ang = np.arange(rows)[:, None] * theta ** (-2.0 * np.arange(half) / rot)[None, :]
    if np.abs(np.cos(ang) - cos[:rows]).max() > 5e-4 or np.abs(np.sin(ang) - sin[:rows]).max() > 5e-4:
        raise ValueError("cos/sin caches are not the plain base ** (-2j / rot) rotary table")
    return theta


def write(ops, inputs, outputs, hwc, stride, pool, destination, manifest, dtypes=None, persistent=None):
    """Allocate activations and write `<destination>.hxfg/.bin/.json`.

    ops: [kind, x, y, z, out, p0..p10] over front-end tensor ids.
    hwc(i)/stride(i): static NHWC shape (batch one) and channel stride.
    pool: ConstantPool or bytearray. manifest: dict with "source",
    "source_sha256" and "ops"; arena/constant/IO fields are added here.
    dtypes: dict or callable mapping tensor id -> dtype enum (0: float32, 1: int32, 2: bool).
    persistent: dict or iterable of tensor ids that live in the persistent state region (conv rings,
    KV caches). Their offsets are relative to that region, assigned once and never reused, and the
    region size is reported as `state_bytes`. The host must supply it (graph state) and keep it
    between runs; the graph never initialises it.
    """
    persistent = dict(persistent) if isinstance(persistent, dict) else {i: 0x1000 + n for n, i in enumerate(persistent or ())}
    dtype_fn = dtypes if callable(dtypes) else (dtypes.get if isinstance(dtypes, dict) else (lambda i: 0))
    get_dtype = lambda i: dtype_fn(i) or 0
    pool = pool.data if isinstance(pool, ConstantPool) else pool
    active = sorted(
        set(inputs + outputs + [i for op in ops for i in tensor_refs(op) if i != NONE])
    )
    ids = {old: new for new, old in enumerate(active)}
    persistent = {i: v for i, v in persistent.items() if i in ids}
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

    state_offsets, state_peak = {}, 0
    for i in sorted(persistent):
        state_offsets[i] = state_peak
        state_peak += aligned(int(np.prod(hwc(i)[:2])) * stride(i) * 4)
    offsets.update(state_offsets)

    def allocate(i):
        nonlocal peak
        if i in persistent:
            offsets[i] = state_offsets[i]
            return
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
        if i in persistent:
            raise ValueError("a graph input cannot be persistent state")
        allocate(i)
    for j, op in enumerate(ops):
        for i in list(allocated):
            if last[i] < j and i not in persistent:
                free.append(allocated.pop(i))
        free.sort()
        merged = []
        for at, n in free:
            if merged and sum(merged[-1]) == at:
                merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:
                merged.append((at, n))
        free[:] = merged
        if op[4] not in allocated and op[4] not in offsets:
            allocate(op[4])
        if op[0] == TOPK and op[3] != NONE:
            allocate(op[3])
    if not pool:
        pool.extend(bytes(128))
    tensors = b"".join(
        struct.pack("<8I", offsets[i], *hwc(i), stride(i), int(i in persistent),
                    persistent.get(i, 0), get_dtype(i))
        for i in active
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
        "inputs": [{"id": ids[i], "shape": manifest["shapes"][i], "dtype": get_dtype(i)} for i in inputs],
        "outputs": [{"id": ids[i], "shape": manifest["shapes"][i], "dtype": get_dtype(i)} for i in outputs],
        "tensor_dtypes": {str(ids[i]): get_dtype(i) for i in active},
        "state_bytes": state_peak,
        "persistent": {str(ids[i]): {"state_id": persistent[i], "offset": state_offsets[i]}
                       for i in sorted(persistent) if i in ids},
    }
    full.update(manifest.get("extra", {}))
    destination.with_suffix(".json").write_text(json.dumps(full, indent=2) + "\n")
    return full
