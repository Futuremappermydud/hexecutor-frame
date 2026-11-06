"""Model-independent execution contracts and native Hexagon backends."""
from .api import ExecutionOptions, Program, ProgramSignature, TensorSpec, open_session

__all__ = ['ExecutionOptions', 'Program', 'ProgramSignature', 'TensorSpec', 'open_session']
