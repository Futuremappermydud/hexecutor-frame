# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from pathlib import Path
import tempfile
import unittest

from hexecutor.offline.qnn16_lower import lower


class Qnn16LowerTest(unittest.TestCase):
    def test_variable_io_metadata_run_arguments_and_constant_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            context, trace = root / "context", root / "trace"
            context.mkdir(); trace.mkdir()
            graph = {
                "context_sha256": "a" * 64,
                "graph_inputs": [0, 1, 2], "graph_outputs": [3, 4, 5, 6],
                "nodes": [{"sequence": 0, "type_name": "Relu.fp16", "flags": 0,
                           "inputs": [0, 1], "outputs": [3], "checkpoints": [7],
                           "framework_words": [10], "framework_vector": [11, 12]}],
                "tensors": [
                    {"id": i, "producer": "input" if i < 3 else 0, "dtype": "float16",
                     "element_bytes": 2, "class_name": "Ce",
                     "shape": {"dimensions": [1, 1, 2, 2], "max_dimensions": [1, 1, 2, 2], "padding": [0, 0, 0, 0]},
                     "storage": {"direct": {"pool": 1, "offset": i * 2048}}}
                    for i in range(7)
                ] + [{"id": 7, "producer": 9, "dtype": "float16", "element_bytes": 2,
                      "class_name": "Xwe", "scalar": 2.5, "shape_values": [1, 2, 3, 4],
                      "shape": {"dimensions": [1, 1, 1, 4], "max_dimensions": [1, 1, 1, 4], "padding": [0, 0, 0, 0]},
                      "storage": {"direct": {"pool": 2, "offset": 32}}}],
            }
            (context / "graph.json").write_text(json.dumps(graph))
            (trace / "trace.log").write_text("0 0 0 0x1234 1 1\n")
            pool = bytes(range(256)) * 16
            (root / "pool2.bin").write_bytes(pool)
            out = root / "model"
            manifest = lower(context, trace, root / "pool2.bin", out)
            self.assertEqual(len(manifest["inputs"]), 3)
            self.assertEqual(len(manifest["outputs"]), 4)
            self.assertEqual(manifest["external_io_bytes"], {"input": 24, "output": 32})
            self.assertEqual(manifest["pool1_vtcm_extent"], 7 * 2048)
            self.assertEqual(manifest["constant_relocations"][0]["source_offset"], 32)
            self.assertEqual((root / "model.q16.bin").read_bytes(), pool[32:32 + 2048])
            self.assertEqual(manifest["package_status"], "development-only")
            self.assertFalse(manifest["constants_source_verified"])
            parameter = next(t for t in manifest["tensors"] if t["id"] == 7)
            self.assertEqual(parameter["scalar"], 2.5)
            self.assertEqual(parameter["shape_values"], [1, 2, 3, 4])
            self.assertEqual(parameter["storage"]["direct"], {"pool": 2, "offset": 32})
            op = manifest["run_lists"]["0"][0]
            self.assertEqual(op["inputs"], [0, 1])
            self.assertEqual(op["checkpoints"], [7])
            self.assertEqual(op["framework_vector"], [11, 12])


    def test_exports_captured_runtime_tensor_and_marks_source_verified(self):
        import struct
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "context").mkdir(); (root / "trace").mkdir()
            graph = {"context_sha256": "c" * 64, "graph_inputs": [], "graph_outputs": [],
                     "nodes": [], "tensors": [{"id": 7, "producer": 9, "dtype": "float16",
                     "element_bytes": 2, "class_name": "Xwe",
                     "shape": {"dimensions": [1, 1, 1, 4], "max_dimensions": [1, 1, 1, 4], "padding": [0, 0, 0, 0]},
                     "storage": {"direct": {"pool": 2, "offset": 32}}}]}
            (root / "context/graph.json").write_text(json.dumps(graph))
            (root / "trace/trace.log").write_text("")
            payload = bytes((i * 17) % 256 for i in range(2048))
            dumps = root / "tensors.bin"
            dumps.write_bytes(struct.pack("<III", 0, 7, len(payload)) + payload)
            manifest = lower(root / "context", root / "trace", None, root / "captured", tensor_dumps=dumps)
            self.assertTrue(manifest["constants_source_verified"])
            self.assertEqual(manifest["constants_source_kind"], "qnn-simulator-tensor-dump")
            self.assertEqual((root / "captured.q16.bin").read_bytes(), payload)
            self.assertEqual(manifest["constant_capture_count"], 1)

            duplicate = root / "conflicting.bin"
            duplicate.write_bytes(struct.pack("<III", 0, 7, len(payload)) + payload +
                                  struct.pack("<III", 1, 7, len(payload)) + bytes(len(payload)))
            with self.assertRaisesRegex(ValueError, "changed between captures"):
                lower(root / "context", root / "trace", None, root / "bad", tensor_dumps=duplicate)

            graph["tensors"].append({**graph["tensors"][0], "id": 8})
            (root / "context/graph.json").write_text(json.dumps(graph))
            with self.assertRaisesRegex(ValueError, "missing pool-2 tensor captures"):
                lower(root / "context", root / "trace", None, root / "incomplete", tensor_dumps=dumps)

    def test_rejects_pool2_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "context").mkdir(); (root / "trace").mkdir()
            graph = {"context_sha256": "b" * 64, "graph_inputs": [], "graph_outputs": [],
                     "nodes": [], "tensors": [{"id": 0, "producer": "const", "dtype": "float16",
                     "element_bytes": 2, "shape": {"dimensions": [1], "max_dimensions": [1], "padding": [0]},
                     "storage": {"direct": {"pool": 2, "offset": 7}}}]}
            (root / "context/graph.json").write_text(json.dumps(graph))
            (root / "trace/trace.log").write_text("")
            (root / "pool2.bin").write_bytes(b"1234")
            with self.assertRaisesRegex(ValueError, "outside --pool2"):
                lower(root / "context", root / "trace", root / "pool2.bin", root / "out")


if __name__ == "__main__":
    unittest.main()
