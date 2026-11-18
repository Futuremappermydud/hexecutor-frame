"""Partitioner that tags the edge-dialect nodes the HXFG tensor_f32 backend can run.

The support rules live in lowering.py (`select_nodes`), so the partitioner and
the backend always agree. Everything else stays on the ExecuTorch CPU path.
"""

from executorch.exir.backend.partitioner import (
    DelegationSpec,
    PartitionResult,
    Partitioner,
)
from executorch.exir.backend.utils import tag_constant_data
import torch

from .backend import HexecutorBackend
from .lowering import kind_name, select_nodes

TAG = "hexecutor_tensor_f32"


class HexecutorPartitioner(Partitioner):
    def __init__(self):
        super().__init__()
        self.delegation_spec = DelegationSpec(HexecutorBackend.__name__, [])

    def ops_to_not_decompose(self, ep):
        # PReLU and linear would otherwise be rewritten by the core ATen decomposition
        # (where/mul, permute/addmm), which HXFG does not express. ExecuTorch requires every
        # kept op to be delegated, so keep exactly the nodes that partition() will select,
        # decided on this pre-decomposition graph.
        selected = set(select_nodes(ep))

        def keep(node):
            return node in selected and kind_name(node) in (
                "prelu", "linear", "act", "bmm", "sdpa", "rmsnorm", "norm"
            )

        # Non-core activations and custom norms would otherwise be rewritten into clamp/mul or var/mean sequences.
        kept = [torch.ops.aten.prelu.default, torch.ops.aten.linear.default,
                torch.ops.aten.hardswish.default, torch.ops.aten.hardsigmoid.default,
                torch.ops.aten.silu.default,
                torch.ops.aten.bmm.default,
                torch.ops.aten.scaled_dot_product_attention.default]
        for op in (
            getattr(torch.ops.aten, "rms_norm", None),
            getattr(torch.ops.aten, "instance_norm", None),
            getattr(torch.ops.aten, "native_instance_norm", None),
        ):
            if op is not None and hasattr(op, "default"):
                kept.append(op.default)
        return (kept, keep)

    def partition(self, exported_program) -> PartitionResult:
        nodes = select_nodes(exported_program)
        partition_tags = {}
        for node in nodes:
            node.meta["delegation_tag"] = TAG
        if nodes:
            partition_tags[TAG] = self.delegation_spec
            # Weights and biases used only by tagged nodes travel inside the delegate.
            tag_constant_data(exported_program)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
