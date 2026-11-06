"""Explicit source/build/model locations shared by development tools.

Runtime backends receive artifact paths directly. Model paths here are only
for development tools, and can be overridden without filesystem aliases.
"""
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Workspace:
    repo: Path
    native: Path
    offline: Path
    benchmarks: Path
    model_data: Path
    model_tools: Path
    model_deployment: Path
    model_reference: Path


def workspace():
    repo = Path(os.environ.get('FRAME_SOURCE_ROOT', Path(__file__).resolve().parent.parent)).resolve()
    data = Path(os.environ.get('FRAME_DEPTH_DATA', repo / 'depth-model')).expanduser().resolve()
    return Workspace(repo, repo/'hexecutor/native', repo/'hexecutor/offline',
                     repo/'hexecutor/benchmarks', data, repo/'depth-model/tools',
                     repo/'depth-model/deployment', repo/'depth-model/python/depth_model/reference')
