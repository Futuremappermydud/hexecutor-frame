"""Depth-specific convenience adapters for the generic execution backends."""
from hexecutor.offline.native_graph import NativeGraph as _NativeGraph, Stats, library
from hexecutor.offline.resident_dsp import ResidentDSP as _ResidentDSP
from hexecutor.paths import workspace
from .contract import DEPTH_SIGNATURE, depth_buffers

INPUTS = tuple((s.name, s.dtype, s.size) for s in DEPTH_SIGNATURE.inputs)
OUTPUT_SHAPES = tuple((s.shape, s.dtype) for s in DEPTH_SIGNATURE.outputs)


class NativeGraph(_NativeGraph):
    def __init__(self, image, constants, recovered=None):
        super().__init__(image, constants, signature=DEPTH_SIGNATURE)

    def run(self, inputs, first=0, last=None):
        return super().run(depth_buffers(inputs), first, last)


class ResidentDSP(_ResidentDSP):
    def __init__(self, image, constants, engine=None, direct_hmx=False, bootstrap=None,
                 prepared_program=None, loader=None):
        native = workspace().native
        super().__init__(image, constants, signature=DEPTH_SIGNATURE,
                         engine=engine or native/'build/qd_graph_dsp.elf',
                         bootstrap=bootstrap or native/'build/qd_graph_boot.elf',
                         loader=loader or native/'build/qd_loader.elf',
                         direct_hmx=direct_hmx, prepared_program=prepared_program)

    def run(self, inputs, first=0, last=None, direct_hmx=None):
        return super().run(depth_buffers(inputs), first, last, direct_hmx)
