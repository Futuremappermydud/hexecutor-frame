"""ExecuTorch BackendDetails for the hexecutor tensor_f32 (HXFG) backend.

`preprocess` runs at export time only. It lowers the delegated edge-dialect
subgraph to an HXFG image and constant pool (via hxfg_writer, in a temporary
directory) and wraps them in the HXET v1 blob described in blob.py. The
delegate's external I/O is float32 PyTorch NCHW for rank-4 tensors; the runtime
transposes those to and from HXFG NHWC (layout = 1 in the blob).
"""

import tempfile
from pathlib import Path

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult

from . import blob
from .lowering import lower_program


class HexecutorBackend(BackendDetails):
    @staticmethod
    def preprocess(edge_program, compile_specs):
        with tempfile.TemporaryDirectory(prefix="hexecutor-hxfg-") as root:
            summary, image, constants = lower_program(edge_program, Path(root) / "graph")
        payload = blob.pack(
            image,
            constants,
            [blob.TensorSpec.for_shape(s) for s in summary["inputs"]],
            [blob.TensorSpec.for_shape(s) for s in summary["outputs"]],
            backend=blob.BACKEND_TENSOR_F32,
        )
        return PreprocessResult(processed_bytes=payload)
