# SPDX-License-Identifier: LGPL-3.0-or-later
"""Export a PyTorch module to an ExecuTorch .pte with hexecutor-delegated partitions.

Usage (from the repository root, with the pinned venv):

    hexecutor/.work/executorch-venv/bin/python -m hexecutor.offline.executorch.export \
        --factory my_package.models:build --input-shape 1,3,32,32 --out out/model.pte

`--factory` names a zero-argument callable returning an nn.Module, or a
(module, example_inputs) pair. Outputs must stay under hexecutor/.work/ or
another ignored artifact directory; no weights are committed.
"""

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower

from .partitioner import HexecutorPartitioner
from .decompositions import custom_table, uses_custom_ops
from .volume import has_high_rank, has_nhwc_linear, has_oversized_conv, rewrite_volumes


def _ensure_flatc():
    """ExecuTorch serializes the .pte by calling `flatc` from PATH.

    The pinned wheel installs it in the venv's bin directory; when the venv
    interpreter is run without activating the venv, that directory is missing.
    """
    if shutil.which("flatc"):
        return
    venv_bin = Path(sys.prefix) / "bin"
    if (venv_bin / "flatc").exists():
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"


_ensure_flatc()


def export_program(model, example_inputs, partitioner=None):
    """torch.export -> edge (with hexecutor partitioning) -> ExecuTorch program manager."""
    model = model.eval()
    if not isinstance(example_inputs, tuple):
        example_inputs = (example_inputs,)
    with torch.no_grad():
        exported = torch.export.export(model, example_inputs)
    if (has_high_rank(exported) or has_nhwc_linear(exported) or has_oversized_conv(exported)
            or uses_custom_ops(exported)):
        # Rank-5 cost volumes are re-traced as rank-4 depth-row layouts (volume.py); vector norms and
        # im2col are decomposed into operations HXFG has (decompositions.py).
        table = torch.export.default_decompositions()
        for op in HexecutorPartitioner._kept_ops():
            table.pop(op, None)
        exported = rewrite_volumes(exported, example_inputs, custom_table(table), force=uses_custom_ops(exported))
    # Fold identity repeat([1]) nodes before edge lowering so the partitioner
    # sees through them (InstanceNorm2d decomposes to batch_norm_legit with
    # repeat on weight/bias).
    g = exported.graph
    modified = False
    for n in list(g.nodes):
        if n.op == "call_function" and "repeat" in str(n.target):
            sizes = n.args[1] if len(n.args) > 1 else None
            if isinstance(sizes, (list, tuple)) and all(s == 1 for s in sizes):
                n.replace_all_uses_with(n.args[0])
                g.erase_node(n)
                modified = True
    if modified:
        g.lint()
        exported._graph_module = torch.fx.GraphModule(exported.graph_module, g)
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[partitioner or HexecutorPartitioner()],
        compile_config=EdgeCompileConfig(),
    )
    return edge.to_executorch(config=ExecutorchBackendConfig())


def export(model, example_inputs, out_pte):
    """Write the .pte for `model`; returns the path."""
    program = export_program(model, example_inputs)
    out_pte = Path(out_pte)
    out_pte.parent.mkdir(parents=True, exist_ok=True)
    out_pte.write_bytes(program.buffer)
    return out_pte


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--factory", required=True, help="module:callable")
    parser.add_argument("--input-shape", required=True, help="comma-separated, e.g. 1,3,32,32")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    module_name, _, attr = args.factory.partition(":")
    built = getattr(importlib.import_module(module_name), attr)()
    if isinstance(built, tuple):
        model, example = built
    else:
        model = built
        example = torch.randn(*[int(v) for v in args.input_shape.split(",")])
    path = export(model, example, args.out)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
