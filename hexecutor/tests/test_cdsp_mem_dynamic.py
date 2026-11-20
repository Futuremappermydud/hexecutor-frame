# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test dynamic CDSP memory manager and reservation range."""

import unittest
from unittest.mock import patch
import hexecutor.offline.cdsp_mem_manager as cmm


class TestDynamicCdspMem(unittest.TestCase):
    def test_constants_and_limits(self):
        self.assertEqual(cmm.CDSP_MEM_MIN_SIZE_MB, 128)
        self.assertEqual(cmm.CDSP_MEM_MAX_SIZE_MB, 1536)  # 1.5 GiB
        self.assertEqual(cmm.CDSP_MEM_CHUNK_MB, 2)

    def test_validation_bounds(self):
        # Below 128 MiB
        with self.assertRaises(ValueError):
            cmm.set_cdsp_mem_mb(64)
        # Above 1.5 GiB (1536 MiB)
        with self.assertRaises(ValueError):
            cmm.set_cdsp_mem_mb(2048)
        # Odd size
        with self.assertRaises(ValueError):
            cmm.set_cdsp_mem_mb(255)

    def test_reservation_steps(self):
        # Test valid steps up to 1.5 GiB
        for size in (128, 256, 512, 768, 1024, 1280, 1536):
            with patch("hexecutor.offline.cdsp_mem_manager._dev_exists", return_value=False), \
                 patch("hexecutor.offline.cdsp_mem_manager._sysfs_exists", return_value=False):
                res = cmm.set_cdsp_mem_mb(size)
                self.assertEqual(res, size)

    def test_lease_lifecycle(self):
        state = {"size": 128}

        def mock_get():
            return state["size"]

        def mock_set(val):
            state["size"] = val
            return val

        with patch("hexecutor.offline.cdsp_mem_manager.get_cdsp_mem_mb", side_effect=mock_get), \
             patch("hexecutor.offline.cdsp_mem_manager.set_cdsp_mem_mb", side_effect=mock_set):
            self.assertEqual(cmm.get_cdsp_mem_mb(), 128)

            # Grow for 512 MiB workload
            with cmm.cdsp_mem_lease(512) as size:
                self.assertEqual(size, 512)
                self.assertEqual(cmm.get_cdsp_mem_mb(), 512)

                # Nested grow for 1.5 GiB workload
                with cmm.cdsp_mem_lease(1536) as nested_size:
                    self.assertEqual(nested_size, 1536)
                    self.assertEqual(cmm.get_cdsp_mem_mb(), 1536)

                # Shrunk back to 512
                self.assertEqual(cmm.get_cdsp_mem_mb(), 512)

            # Shrunk back to 128 baseline
            self.assertEqual(cmm.get_cdsp_mem_mb(), 128)


if __name__ == "__main__":
    unittest.main()
