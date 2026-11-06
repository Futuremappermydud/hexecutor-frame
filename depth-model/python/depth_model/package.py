"""Depth package identity, native artifacts, and session integration."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from hexecutor.paths import workspace
from .contract import DEPTH_SIGNATURE

@dataclass(frozen=True)
class DepthPackage:
    root: Path
    artifacts: Path | None = None
    programs: Path | None = None

    def __post_init__(self):
        object.__setattr__(self, 'root', Path(self.root).expanduser().resolve())
        paths=workspace()
        legacy=self.root/'frame'
        object.__setattr__(self, 'artifacts', Path(self.artifacts or os.environ.get('FRAME_DEPTH_ARTIFACTS', legacy/'build' if legacy.is_dir() else paths.native/'build')).expanduser().resolve())
        object.__setattr__(self, 'programs', Path(self.programs or os.environ.get('FRAME_DEPTH_PROGRAMS', legacy/'native_program/examples' if legacy.is_dir() else paths.model_deployment/'programs')).expanduser().resolve())

    def resolve(self, identity):
        # Stable pinned package identities support separately installed v1 packages.
        if identity.startswith('frame/build/'):
            return self.artifacts/identity.removeprefix('frame/build/')
        if identity.startswith('frame/native_program/examples/'):
            return self.programs/identity.removeprefix('frame/native_program/examples/')
        raise ValueError(f'unknown depth artifact identity: {identity}')

    def constants(self, model):
        if model not in ('primary','alternate'):raise ValueError('unknown depth model')
        return self.root/'recovered'/model/'constants/constant_pool_5.bin'

# Pinned artifacts for the prepared backend, all validated on Frame 2026-11-01:
# full-model smoke/random outputs equal the emulator sha256 for both models.
PREPARED_ENGINE='frame/build/qd_graph_dsp-scalar-dynamic-tcm-fallback-l2-blocked-glue-fast-spill-native-1x1-native-3x3-direct-range-prepared-reference-vtcm-lease.elf'
PREPARED_BOOTSTRAP='frame/build/qd_graph_boot-c0-preserve-vtcm-guarded-entry.elf'
PREPARED_SHA256={
    PREPARED_ENGINE:'c432a7140ac633d9e20366ac8d801de1b42acdd2bf21dc2fdde65d415e9ce13b',
    PREPARED_BOOTSTRAP:'628e362de7563a2c3b0b5cf2a02cae49493b80931e47e52d652fedae449fa147',
    'frame/build/qd_loader.elf':'828ca0638750a6d8b69b3aaa536cb6c23508ce821b2333e270b455c321414572',
    'frame/native_program/examples/primary-v2.nprg':'c6ae3022dbbdab04f474751bd987e9bfc0547337d5e9b95becb2e100eed86013',
    'frame/native_program/examples/alternate-v2.nprg':'21953104d4960e19dff0f61116dd787e2eb533ef1e4c8eeeedfba197787f69a4',
    'frame/build/resident-primary.qdrx':'dc8fcddcf4876d11f7a717ec4f111e1d3d3a40d4f3163758b590234e0b50b92b',
    'frame/build/resident-alternate.qdrx':'54b2949bbb25655ee459c903432f67cacd26e66519854dcb9745a91777eb7298',
}
# QNN run-list engine, validated on Frame 2026-11-02 (claim_20261102i): primary
# smoke+random exact vs the hw oracle, 22.7 ms steady. Run list is primary-only.
QNN_ENGINE='frame/build/ab/claim_20261102i/eng-claim.elf'
QNN_BOOTSTRAP='frame/build/ab/claim_20261102i/boot-claim.elf'
PREPARED_SHA256[QNN_ENGINE]='678611d0d66eeeb28b01d76f403494cfe206babd2cd3b7fe4192417608629068'
PREPARED_SHA256[QNN_BOOTSTRAP]='cc30661b9496c489fcf6aad3fd79ad6009ab320f76e69edc3bc0aa319e0f798e'


@contextmanager
def prepared_session(root,model,engine=PREPARED_ENGINE,bootstrap=PREPARED_BOOTSTRAP):
    """Yields an infer(...) callable on a prepared executor engine."""
    from hexecutor.offline.resident_dsp import ResidentDSP
    package=DepthPackage(root)
    program=f'frame/native_program/examples/{model}-v2.nprg'
    image=f'frame/build/resident-{model}.qdrx'
    # Every pinned artifact is checked before the DSP lock is taken.
    for name in (engine,bootstrap,'frame/build/qd_loader.elf',program,image):
        if hashlib.sha256(package.resolve(name).read_bytes()).hexdigest()!=PREPARED_SHA256[name]:
            raise ValueError(f'prepared artifact {name} does not match its validated sha256')
    with ResidentDSP(package.resolve(image),package.constants(model),signature=DEPTH_SIGNATURE,loader=package.resolve('frame/build/qd_loader.elf'),
                     engine=package.resolve(engine),bootstrap=package.resolve(bootstrap),
                     prepared_program=package.resolve(program)) as dsp:
        if not dsp.preserve_vtcm:raise ValueError('preserving bootstrap required')
        def infer(images,hidden,transforms):
            outputs=dsp.run([images[0],images[1],hidden,transforms])
            trace=dsp.last.get('native_direct_trace',[])
            if not dsp.last.get('native_prepared_program') or len(trace)!=16 or trace[8]!=dsp.commands:
                raise RuntimeError(f'prepared run did not execute the full model: {trace}')
            return outputs[1],outputs[0]
        yield infer
    print(f'[ModelDepth] Prepared session closed; VTCM lease {dsp.vtcm_lease}',flush=True)
