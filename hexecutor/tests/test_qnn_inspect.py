import tempfile
import unittest
from pathlib import Path
from hexecutor.offline.inspect_qnn import inspect, read_schematic

SCHEMATIC = """
graph_before_optimization = {1: {'operation': 'q::Input', 'inputs': [],
    'outputs': [{'dtype': 'Float32', 'dims': [1, 2, 2, 3]}]}}
graph_after_optimization = {2: {'operation': 'q::ConvLayer.fp16.s1.tcm',
    'inputs': [], 'outputs': [{'dtype': 'Float16', 'dims': [1, 2, 2, 32]}],
    'storage': 2 * 2048}}
concrete_ops_info = {2: 0}
concrete_ops_info_key = ['synthetic_fp16_kernel']
runlist2 = [2]
runlist_vec = []
tcm_block_info = {0: (4096, 2 * 2048)}
qnn_graph_name = 'synthetic'
"""


class QnnInspectionTests(unittest.TestCase):
    def test_compiler_arithmetic_and_runlist_summary(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "schematic"
            path.write_text(SCHEMATIC)
            self.assertEqual(
                read_schematic(path)["graph_after_optimization"][2]["storage"], 4096
            )
            result = inspect(path)
            self.assertEqual(result["after_tensor_types"], {"Float16": 1})
            self.assertEqual(result["runlists"]["runlist2"]["entries"], 1)
            self.assertEqual(result["tcm_address_extent_bytes"], 8192)

    def test_schematic_code_is_never_executed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "schematic"
            marker = Path(root) / "executed"
            path.write_text(
                SCHEMATIC + f"\n__import__('pathlib').Path({str(marker)!r}).touch()\n"
            )
            self.assertEqual(inspect(path)["graph_name"], "synthetic")
            self.assertFalse(marker.exists())

    def test_unknown_required_expression_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "schematic"
            path.write_text(
                SCHEMATIC.replace(
                    "'storage': 2 * 2048", "'storage': unknown_function()"
                )
            )
            with self.assertRaises(ValueError):
                read_schematic(path)


if __name__ == "__main__":
    unittest.main()
