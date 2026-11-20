# SPDX-License-Identifier: LGPL-3.0-or-later
"""Offline program/buffer development API; the live host API is C++."""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import math

import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError('tensor name must be nonempty')
        object.__setattr__(self, 'shape', tuple(self.shape))
        if any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError('tensor dimensions must be positive integers')
        dtype = np.dtype(self.dtype)
        if dtype.hasobject or dtype.kind not in 'biuf':
            raise ValueError('only numeric tensor buffers are supported')
        object.__setattr__(self, 'dtype', dtype.str)

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return self.size * np.dtype(self.dtype).itemsize

    def buffer(self, value):
        array = np.asarray(value)
        if array.shape != self.shape or array.dtype != np.dtype(self.dtype):
            raise ValueError(f'{self.name}: expected {self.shape} {np.dtype(self.dtype)}, got {array.shape} {array.dtype}')
        return np.ascontiguousarray(array)


@dataclass(frozen=True)
class ProgramSignature:
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]

    def __post_init__(self):
        object.__setattr__(self, 'inputs', tuple(self.inputs))
        object.__setattr__(self, 'outputs', tuple(self.outputs))
        for group in (self.inputs, self.outputs):
            if not group or len({s.name for s in group}) != len(group):
                raise ValueError('input/output groups must be nonempty with unique names')

    def buffers(self, values):
        if isinstance(values, Mapping):
            expected = {s.name for s in self.inputs}
            if set(values) != expected:
                raise ValueError(f'input names must be {sorted(expected)}')
            values = [values[s.name] for s in self.inputs]
        else:
            values = list(values)
        if len(values) != len(self.inputs):
            raise ValueError('input buffer count does not match program signature')
        return [spec.buffer(value) for spec, value in zip(self.inputs, values)]

    def require_qdrx_v1(self):
        # Existing native wire layout. This is a backend capability, not a model contract.
        if len(self.inputs) != 4 or len(self.outputs) != 2:
            raise ValueError('qdrx-v1 backend ABI requires 4 inputs and 2 outputs')


@dataclass(frozen=True)
class Program:
    image: Path
    constants: Path
    signature: ProgramSignature

    def __post_init__(self):
        object.__setattr__(self, 'image', Path(self.image))
        object.__setattr__(self, 'constants', Path(self.constants))


@dataclass(frozen=True)
class ExecutionOptions:
    backend: str = 'cpu'
    engine: Path | None = None
    bootstrap: Path | None = None
    loader: Path | None = None
    prepared_program: Path | None = None

    def __post_init__(self):
        if self.backend not in ('cpu', 'dsp'):
            raise ValueError('backend must be cpu or dsp')
        if self.backend == 'dsp' and any(p is None for p in (self.engine, self.bootstrap, self.loader)):
            raise ValueError('DSP execution requires explicit engine, bootstrap, and loader paths')


def open_session(program: Program, options: ExecutionOptions | None = None):
    """Create an execution session; no weights, model names, or implicit package lookup."""
    options = options or ExecutionOptions()
    if options.backend == 'cpu':
        from .offline.native_graph import NativeGraph
        return NativeGraph(program.image, program.constants, signature=program.signature)
    from .offline.resident_dsp import ResidentDSP
    return ResidentDSP(program.image, program.constants, signature=program.signature,
                       engine=options.engine, bootstrap=options.bootstrap, loader=options.loader,
                       prepared_program=options.prepared_program)
