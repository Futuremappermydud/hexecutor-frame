#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inspect QAIRT's compiler schematic as data, without executing its Python.

QAIRT 2.50 emits Python-like dictionaries plus expansion loops. Only literal
assignments and bounded scalar arithmetic are read here. Compiler binaries,
weights, schematic dumps and decoded graph data belong in ignored directories.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
import operator
from pathlib import Path

ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.truediv,
}


class LiteralArithmetic(ast.NodeTransformer):
    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if (
            type(node.op) in ARITHMETIC
            and isinstance(node.left, ast.Constant)
            and isinstance(node.right, ast.Constant)
            and type(node.left.value) in (int, float)
            and type(node.right.value) in (int, float)
        ):
            value = ARITHMETIC[type(node.op)](node.left.value, node.right.value)
            if abs(value) > 2**64:
                raise ValueError("schematic arithmetic exceeds supported range")
            return ast.copy_location(ast.Constant(value), node)
        return node


def read_schematic(path):
    if path.stat().st_size > 32 << 20:
        raise ValueError("schematic exceeds inspection budget")
    module = ast.parse(path.read_text())
    result = {}
    for node in module.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        try:
            result[node.targets[0].id] = ast.literal_eval(
                LiteralArithmetic().visit(node.value)
            )
        except (ValueError, TypeError, ZeroDivisionError):
            continue
    for key in (
        "graph_before_optimization",
        "graph_after_optimization",
        "concrete_ops_info",
        "concrete_ops_info_key",
    ):
        if key not in result:
            raise ValueError("unsupported compiler schematic: missing " + key)
    return result


def inspect(path):
    data = read_schematic(path)
    before, after = data["graph_before_optimization"], data["graph_after_optimization"]
    classes = data["concrete_ops_info_key"]
    concrete = {key: classes[value] for key, value in data["concrete_ops_info"].items()}

    def counts(items):
        return dict(Counter(items).most_common())

    report = {
        "schematic": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "graph_name": data.get("qnn_graph_name"),
        "before_nodes": len(before),
        "after_nodes": len(after),
        "before_operations": counts(n["operation"] for n in before.values()),
        "after_operations": counts(n["operation"] for n in after.values()),
        "after_tensor_types": counts(
            t["dtype"] for n in after.values() for t in n["outputs"]
        ),
        "concrete_kernel_classes": counts(concrete.values()),
        "runlists": {},
        "timing": "Compiler topology only. No device latency is established by this schematic.",
    }
    for name in ("runlist1", "runlist2", "runlist_vec", "runlist_mtx", "runlist_elt"):
        ids = data.get(name, [])
        report["runlists"][name] = {
            "entries": len(ids),
            "operations": counts(after[i]["operation"] for i in ids if i in after),
        }
    blocks = data.get("tcm_block_info", {})
    if blocks:
        report["tcm_address_extent_bytes"] = max(
            at + size for at, size in blocks.values()
        )
        report["tcm_note"] = (
            "Highest block end offset, including reused/aliased blocks; not peak liveness."
        )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("schematic", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = inspect(args.schematic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "graph_name",
                    "before_nodes",
                    "after_nodes",
                    "after_operations",
                    "after_tensor_types",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
