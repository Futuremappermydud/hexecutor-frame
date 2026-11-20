import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import tsdf_planes as planes


def rectangle(center, u, v, hu, hv, step=0.025):
    center = np.asarray(center, float)
    u, v = np.asarray(u, float), np.asarray(v, float)
    nu, nv = max(1, int(np.ceil(2 * hu / step))), max(1, int(np.ceil(2 * hv / step)))
    a = np.linspace(-hu, hu, nu + 1)
    b = np.linspace(-hv, hv, nv + 1)
    x, y = np.meshgrid(a, b)
    corners = center + x[..., None] * u + y[..., None] * v
    triangles = []
    for j in range(nv):
        for i in range(nu):
            p00, p10 = corners[j, i], corners[j, i + 1]
            p01, p11 = corners[j + 1, i], corners[j + 1, i + 1]
            triangles.extend(((p00, p10, p11), (p00, p11, p01)))
    return np.asarray(triangles)


def at_offset(found, normal, offset, tolerance=0.005):
    return [p for p in found if planes.angle(p['normal'], normal) < 1
            and abs(abs(p['offset']) - offset) < tolerance]


class TsdfPlaneTests(unittest.TestCase):
    def test_parallel_planes_stay_separate(self):
        triangles = np.concatenate((rectangle((0, 1, -2), (1, 0, 0), (0, 1, 0), 1, 1),
                                    rectangle((0, 1, -1.9), (1, 0, 0), (0, 1, 0), 0.4, 0.4)))
        found = planes.detect_snapshot(triangles)
        self.assertEqual(len(at_offset(found, (0, 0, 1), 2.0)), 1)
        self.assertEqual(len(at_offset(found, (0, 0, 1), 1.9)), 1)

    def test_disconnected_coplanar_patches_do_not_merge(self):
        triangles = np.concatenate((rectangle((-0.75, 1, -2), (1, 0, 0), (0, 1, 0), 0.4, 0.8),
                                    rectangle((0.75, 1, -2), (1, 0, 0), (0, 1, 0), 0.4, 0.8)))
        found = at_offset(planes.detect_snapshot(triangles), (0, 0, 1), 2.0)
        self.assertEqual(len(found), 2)
        for patch in found:
            x = np.asarray(sorted(patch['voxels']))[:, 0]
            self.assertTrue(np.all(x < 0) or np.all(x > 0))

    def test_doorway_has_no_wall_footprint(self):
        triangles = np.concatenate((rectangle((-0.8, 1.25, -2), (1, 0, 0), (0, 1, 0), 0.4, 1.25),
                                    rectangle((0.8, 1.25, -2), (1, 0, 0), (0, 1, 0), 0.4, 1.25),
                                    rectangle((0, 2.25, -2), (1, 0, 0), (0, 1, 0), 0.4, 0.25)))
        found = at_offset(planes.detect_snapshot(triangles), (0, 0, 1), 2.0)
        self.assertTrue(found)
        for patch in found:
            for x, y, _ in patch['voxels']:
                centre = (np.asarray((x, y)) + 0.5) * planes.VOXEL_M
                self.assertFalse(abs(centre[0]) < 0.35 and 0.1 < centre[1] < 1.9)

    def test_curved_surface_does_not_wrap_into_one_plane(self):
        angles = np.linspace(0, 2 * np.pi, 73)
        ring = np.column_stack((0.3 * np.cos(angles), 0.3 * np.sin(angles)))
        triangles = []
        for i in range(72):
            a, b = ring[i], ring[i + 1]
            triangles.extend((((a[0], 0, a[1]), (b[0], 0, b[1]), (b[0], 1, b[1])),
                              ((a[0], 0, a[1]), (b[0], 1, b[1]), (a[0], 1, a[1]))))
        found = planes.detect_snapshot(np.asarray(triangles))
        widest = 0.0
        for patch in found:
            voxels = np.asarray(sorted(patch['voxels']))
            centres = (voxels + 0.5) * planes.VOXEL_M
            arc = np.sort(np.degrees(np.arctan2(centres[:, 0], centres[:, 2])) % 360)
            gaps = np.diff(np.concatenate((arc, arc[:1] + 360)))
            widest = max(widest, float(360 - gaps.max()))
        self.assertLessEqual(widest, 60)

    def test_small_protrusion_does_not_move_wall(self):
        triangles = np.concatenate((rectangle((0, 1.5, -2), (1, 0, 0), (0, 1, 0), 1.5, 1),
                                    rectangle((0, 1.5, -1.98), (1, 0, 0), (0, 1, 0), 0.15, 0.15)))
        found = at_offset(planes.detect_snapshot(triangles), (0, 0, 1), 2.0, 0.03)
        wall = max(found, key=lambda p: len(p['voxels']))
        self.assertLess(abs(abs(wall['offset']) - 2.0), 0.005)
        self.assertEqual(sum(len(p['voxels']) > 40 for p in found), 1)


if __name__ == '__main__':
    unittest.main()
