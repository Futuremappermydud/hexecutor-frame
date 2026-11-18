"""HXET v1 delegate blob: the payload stored in a .pte for the hexecutor backend.

All integers are little-endian with no padding:

    b"HXET", u32 version (1), u32 backend (1 = tensor_f32 HXFG),
    u32 n_inputs, u32 n_outputs
    for each input, then each output:
        u32 layout  (0 = as-is; 1 = PyTorch NCHW, runtime transposes to/from NHWC)
        u32 rank, u32 dims[rank]   (PyTorch-side dims, float32 implied)
    u64 image_size, u64 constants_size
    image bytes (.hxfg), constants bytes (.bin constant pool)

Pure Python with no dependencies so the runtime contract is easy to audit.
The C++ reader must match this file exactly.
"""

from dataclasses import dataclass
import struct

MAGIC = b"HXET"
VERSION = 1
BACKEND_TENSOR_F32 = 1
LAYOUT_AS_IS = 0
LAYOUT_NCHW = 1


@dataclass(frozen=True)
class TensorSpec:
    layout: int
    dims: tuple

    @classmethod
    def for_shape(cls, dims):
        dims = tuple(int(d) for d in dims)
        return cls(LAYOUT_NCHW if len(dims) == 4 else LAYOUT_AS_IS, dims)


@dataclass(frozen=True)
class Blob:
    version: int
    backend: int
    inputs: tuple
    outputs: tuple
    image: bytes
    constants: bytes


def pack(image, constants, inputs, outputs, backend=BACKEND_TENSOR_F32):
    """Serialize an HXET v1 blob. inputs/outputs are sequences of TensorSpec."""
    out = bytearray(MAGIC)
    out += struct.pack("<4I", VERSION, backend, len(inputs), len(outputs))
    for spec in list(inputs) + list(outputs):
        out += struct.pack(f"<2I{len(spec.dims)}I", spec.layout, len(spec.dims), *spec.dims)
    out += struct.pack("<2Q", len(image), len(constants))
    out += bytes(image)
    out += bytes(constants)
    return bytes(out)


def unpack(data):
    """Parse an HXET v1 blob, validating magic, version and all lengths."""
    data = bytes(data)
    if data[:4] != MAGIC:
        raise ValueError("not an HXET blob")
    pos = 4

    def take(fmt):
        nonlocal pos
        size = struct.calcsize(fmt)
        if pos + size > len(data):
            raise ValueError("truncated HXET blob")
        values = struct.unpack_from(fmt, data, pos)
        pos += size
        return values

    version, backend, n_in, n_out = take("<4I")
    if version != VERSION:
        raise ValueError(f"unsupported HXET version {version}")
    specs = []
    for _ in range(n_in + n_out):
        layout, rank = take("<2I")
        if layout not in (LAYOUT_AS_IS, LAYOUT_NCHW):
            raise ValueError(f"unknown layout {layout}")
        specs.append(TensorSpec(layout, take(f"<{rank}I") if rank else ()))
    image_size, constants_size = take("<2Q")
    if pos + image_size + constants_size != len(data):
        raise ValueError("HXET blob length mismatch")
    image = data[pos : pos + image_size]
    constants = data[pos + image_size :]
    return Blob(
        version=version,
        backend=backend,
        inputs=tuple(specs[:n_in]),
        outputs=tuple(specs[n_in:]),
        image=image,
        constants=constants,
    )
