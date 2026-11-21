# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from pathlib import Path
import tempfile
import unittest

from hexecutor.offline.qnn16_lower import lower
from hexecutor.offline.qnn16_program import KIND, Lowering
from hexecutor.offline.qnn16_oracle import map_entries, read_trace, read_worker_checkpoints


class Qnn16LowerTest(unittest.TestCase):
    def test_worker_checkpoint_pairs_separate_adjacent_split_operations(self):
        graph = {"nodes": [
            {"sequence": 10, "type_name": "Transpose", "flags": 4, "checkpoints": [2, 5]},
            {"sequence": 11, "type_name": "Transpose", "flags": 4, "checkpoints": [3, 6]},
        ]}
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp)
            (trace / "trace.log").write_text(
                "# worker-checkpoints 0 0x1008 0x1014\n"
                "# worker-checkpoints 1 0x1008 0x1014\n"
                "# worker-checkpoints 2 0x100c 0x1018\n"
                "# worker-checkpoints 3 0x100c 0x1018\n"
                "0 1 1 0x20 1 0\n"
                "1 1 0 0x20 2 0\n"
                "2 1 3 0x20 1 0\n"
                "3 1 2 0x20 2 0\n"
            )
            rows = read_trace(trace)
            checkpoints = read_worker_checkpoints(trace)
            with self.assertRaisesRegex(ValueError, "ambiguous or unhandled split"):
                map_entries(graph, rows)
            entries = map_entries(graph, rows, worker_checkpoints=checkpoints)
            self.assertEqual([entries[(1, i)]["sequence"] for i in range(4)], [10, 10, 11, 11])
            checkpoints[2] = (0x100C, 0x101C)
            with self.assertRaisesRegex(ValueError, "worker checkpoints do not match"):
                map_entries(graph, rows, worker_checkpoints=checkpoints)

    def test_layernorm_moments_keep_unsigned_input_and_raw_output(self):
        def shape(dims):
            return {"dimensions": dims, "max_dimensions": dims, "padding": [0] * 4}

        padded_table = [None] + [{"pool": 1, "offset": i * 2048} for i in range(72)]
        src = {"id": 1, "class_name": "CH", "dtype": "uint16", "element_bytes": 2,
               "shape": shape([1, 8, 36, 256]),
               "storage": {"blocks": padded_table, "slice_offset": 1}}
        dst = {"id": 2, "class_name": "Fi", "dtype": "int32", "element_bytes": 4,
               "shape": shape([1, 1, 3, 288]), "storage": {"direct": {"pool": 1, "offset": 72 * 2048}}}
        node = {"sequence": 56, "type_name": "layernorm_q16_2d_moments_reshaped.tcm@Fi.CH",
                "inputs": [1], "outputs": [2]}
        lowerer = Lowering({"nodes": [node], "tensors": [src, dst]}, {})
        lowerer.lower_node(node)
        self.assertEqual(lowerer.ops[0][0], KIND["MOMENTS_U16"])
        self.assertEqual(lowerer.words, [i * 2048 for i in range(72)])
        self.assertEqual(lowerer.tensor_rows[0][1], 3)
        self.assertEqual(lowerer.tensor_rows[1][1:4], [2, 72 * 2048, 3456])
        dst["dtype"] = "float32"
        with self.assertRaisesRegex(ValueError, "dense int32 output"):
            lowerer.lower_node(node)

    def test_uint16_space_rearrange_preserves_crouton_storage(self):
        def tensor(tid, dims, full, padding, blocks, offset):
            return {"id": tid, "class_name": "CH", "dtype": "uint16", "element_bytes": 2,
                    "shape": {"dimensions": dims, "max_dimensions": full, "padding": padding},
                    "storage": {"blocks": [{"pool": 1, "offset": offset + i * 2048}
                                           for i in range(blocks)]},
                    "quantization": {"scale": 0.25, "zero_point": 32768}}

        narrow = tensor(1, [1, 1, 577, 32], [1, 8, 580, 32], [0, 7, 0, 0], 145, 0)
        wide = tensor(2, [1, 8, 76, 32], [1, 8, 76, 32], [0] * 4, 19, 145 * 2048)
        node = {"sequence": 23, "type_name": "space_rearrange@CH*2",
                "inputs": [1], "outputs": [2]}
        lowerer = Lowering({"nodes": [node], "tensors": [narrow, wide]}, {})
        lowerer.lower_node(node)
        self.assertEqual(lowerer.ops[0][0], KIND["REARRANGE_U16"])
        self.assertEqual(lowerer.words, [i * 2048 for i in range(164)])
        self.assertEqual(lowerer.tensor_rows[0][14:18], [0, 7, 0, 0])
        wide["quantization"]["zero_point"] = 0
        with self.assertRaisesRegex(ValueError, "encodings differ"):
            lowerer.lower_node(node)

    def test_grouped_spill_transfers_preserve_order_and_ddr_offsets(self):
        graph = {"nodes": [], "tensors": []}
        lowerer = Lowering(graph, {})
        node = {"sequence": 14, "type_name": "@Spill", "inputs": [], "outputs": [],
                "spill_groups": [{"offset": 4096, "transfers": [[2, 2048], [4098, 2048]]},
                                 {"offset": 0, "transfers": [[8194, 4096]]}]}
        lowerer.lower_node(node)
        self.assertEqual(lowerer.words, [0, 4096, 2048, 4096, 6144, 2048, 8192, 0, 4096])
        self.assertEqual(lowerer.spill_bytes, 8192)
        self.assertEqual(lowerer.ops[0][0], KIND["SPILL_FILL"])
        self.assertEqual(lowerer.ops[0][5:8], [0, 3, 1])
        node["type_name"] = "@Fill"
        lowerer.lower_node(node)
        self.assertEqual(lowerer.ops[1][5:8], [9, 3, 0])
        node["spill_groups"][1]["transfers"][0][0] = 8193
        with self.assertRaisesRegex(ValueError, "unsupported Q16P spill/fill transfer"):
            lowerer.lower_node(node)
        self.assertEqual(len(lowerer.words), 18)

    def test_uint16_slicepad_lowers_an_in_bounds_crop_only(self):
        quant = {"scale": 0.25, "zero_point": 32768}

        def tensor(tid, dims, offset):
            return {"id": tid, "class_name": "FH", "dtype": "uint16", "element_bytes": 2,
                    "shape": {"dimensions": dims, "max_dimensions": dims, "padding": [0] * 4},
                    "storage": {"direct": {"pool": 1, "offset": offset}},
                    "quantization": quant.copy()}

        source = tensor(1, [1, 2, 5, 8], 0)
        output = tensor(5, [1, 1, 2, 4], 2048)
        zero = {"id": 2, "shape_values": [0, 0, 0, 0]}
        origin = {"id": 3, "shape_values": [0, 1, 2, 3]}
        node = {"sequence": 0, "type_name": "SlicePad_shape_inplace@FH*2.s4*3.cH",
                "inputs": [1, 2, 3, 2, 4], "outputs": [5]}
        graph = {"nodes": [node], "tensors": [source, zero, origin, {"id": 4}, output]}
        lowerer = Lowering(graph, {})
        lowerer.lower_node(node)
        self.assertEqual(lowerer.ops[0][0], KIND["SLICE_U16"])
        self.assertEqual(lowerer.ops[0][5:9], [0, 1, 2, 3])
        origin["shape_values"][-1] = 5
        with self.assertRaisesRegex(ValueError, "crop is outside the input"):
            lowerer.lower_node(node)
        origin["shape_values"][-1] = 3
        output["quantization"]["zero_point"] = 0
        with self.assertRaisesRegex(ValueError, "encodings differ"):
            lowerer.lower_node(node)

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
