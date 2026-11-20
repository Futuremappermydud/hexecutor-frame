"""The stereo model's deployment contract, independent of executor internals."""
import numpy as np
from hexecutor import ProgramSignature, TensorSpec

DEPTH_SIGNATURE = ProgramSignature(
    inputs=(TensorSpec('x', (320, 256), 'uint8'), TensorSpec('y', (320, 256), 'uint8'),
            TensorSpec('hidden', (1, 2, 320, 256), 'float32'), TensorSpec('t', (1, 2, 4, 4), 'float32')),
    outputs=(TensorSpec('uncertainty', (1, 2, 320, 256), 'uint8'),
             TensorSpec('disparity', (1, 2, 320, 256), 'float32')))


def depth_buffers(inputs):
    """Adapt legacy flat/indexed depth inputs at the model boundary."""
    return [np.ascontiguousarray(inputs[i], dtype=spec.dtype).reshape(spec.shape)
            for i, spec in enumerate(DEPTH_SIGNATURE.inputs)]
