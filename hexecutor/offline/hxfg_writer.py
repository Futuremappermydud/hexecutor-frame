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
 AVGPOOL) = range(1, 15)
# HX_ACT functions and HX_BINARY operations (graph.h).
(FN_RELU, FN_RELU6, FN_HARDSWISH, FN_HARDSIGMOID, FN_SILU, FN_TANH, FN_LEAKY,
 FN_GELU) = range(8)
BIN_ADD, BIN_SUB, BIN_MUL = range(3)


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


def write(ops, inputs, outputs, hwc, stride, pool, destination, manifest):
    """Allocate activations and write `<destination>.hxfg/.bin/.json`.

    ops: [kind, x, y, z, out, p0..p10] over front-end tensor ids.
    hwc(i)/stride(i): static NHWC shape (batch one) and channel stride.
    pool: ConstantPool or bytearray. manifest: dict with "source",
    "source_sha256" and "ops"; arena/constant/IO fields are added here.
    """
    pool = pool.data if isinstance(pool, ConstantPool) else pool
    active = sorted(
        set(inputs + outputs + [i for op in ops for i in op[1:5] if i != NONE])
    )
    ids = {old: new for new, old in enumerate(active)}
    last = {i: -1 for i in active}
    for j, op in enumerate(ops):
        for i in op[1:4]:
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
    if not pool:
        pool.extend(bytes(128))
    tensors = b"".join(
        struct.pack("<8I", offsets[i], *hwc(i), stride(i), 0, 0, 0) for i in active
    )
    packed_ops = b"".join(
        struct.pack("<16I", op[0], *(ids.get(i, NONE) for i in op[1:5]), *op[5:])
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
