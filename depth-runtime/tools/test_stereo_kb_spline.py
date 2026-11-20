"""Synthetic vendor-oracle vectors; no device calibration or imagery embedded.

Oracle library SHA256 44ca80763343d6588798bb52418611304fd50e2fd7de6ddfce5cbd31989e1984.
Reproducer: local temporal-selfcal-v53/spline-oracle/synthetic_check.py.
"""
import copy
import unittest
import numpy as np
from stereo_kb_spline import KBSpline

INTRINSICS = {'cameraModel': 'kb_spline', 'gridSize': [5, 5], 'fx': 320.0, 'fy': 325.0, 'cx': 528.0, 'cy': 512.0, 'k1': 0.03, 'k2': -0.007, 'k3': -0.004, 'k4': 0.0, 'nodeSpacing': [228.0, 222.0], 'sU': [-0.75, -0.6875, -0.625, -0.5625, -0.5, -0.4375, -0.375, 0.0, -0.25, -0.1875, -0.125, 0.0, 0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75], 'sV': [-1.375, -0.5, 0.375, 1.25, -0.75, 0.125, 1.0, 0.0, -0.125, 0.75, -1.25, 0.0, 0.0, 1.375, -0.625, 0.25, 1.125, -0.875, 0.0, 0.875, -1.125, -0.25, 0.625, -1.375, -0.5]}
POINTS = [[0.2947745963976931, -1.2800898824551257, 1.0], [1.1968751685409604, 1.9437984821551086, 1.0], [1.4872773318889372, 1.592358650003916, 1.0], [0.9526959463790055, 2.6780967539777727, 1.0], [-1.818015994093258, 2.160775589561839, 1.0], [2.8277989126379284, -1.1175883665423676, 1.0], [1.187387223455618, -0.6981708231079453, 1.0], [2.0635412441461063, -1.6173665867345945, 1.0], [0.0, 0.0, 1.0], [20.0, 0.0, 1.0], [-20.0, 0.0, 1.0], [0.0, 20.0, 1.0], [0.0, -20.0, 1.0], [1.0, 1.0, 0.01]]
EXPECTED = [[595.3044441369963, 215.23295375479728], [726.157298755631, 838.8073353620654], [782.1427545842924, 787.9844830383971], [662.5485066950472, 895.7084226024509], [270.2847244795975, 822.9832393260142], [906.8079245438744, 360.4616770684752], [793.1729248132386, 354.46921158221073], [837.2492674923838, 266.87096864722633], [528.0, 512.0], [1006.715990826401, 512.6450380086899], [49.26977337905474, 512.3631067574024], [528.1135163754225, 996.9303272670653], [527.8828597664833, 25.232845216378877], [872.9715835844958, 861.3977757187971]]

BILINEAR_EXPECTED = [[595.305782221367, 215.25743934431623], [726.1617434833597, 838.8075321856654], [782.1419270679355, 787.9562113936136], [662.5435127489288, 895.6786541944135], [270.27586028994034, 822.9747285546349], [906.8080122748063, 360.46942050632845], [793.1734871824956, 354.4815299159433], [837.2465002410852, 266.88249358464384], [528.0, 512.0], [1006.7232051909922, 512.6445156310083], [49.26459508286426, 512.3662137210952], [528.1112979147462, 996.9355983051368], [527.8848626644832, 25.252874178488668], [872.9721173666449, 861.4210328720045]]

class SplineTest(unittest.TestCase):
    def test_vendor_projection_interior_axis_and_clamped_edges(self):
        got = np.array(KBSpline(INTRINSICS).project(np.array(POINTS).T)).T
        np.testing.assert_allclose(got, EXPECTED, atol=5e-6, rtol=0)

    def test_zero_spline_matches_independent_equidistant_projection(self):
        k = copy.deepcopy(INTRINSICS)
        k.update(k1=0.,k2=0.,k3=0.,k4=0.,sU=[0.]*25,sV=[0.]*25)
        u,v = KBSpline(k).project(np.array([[1.,0.,0.],[0.,1.,0.],[1.,1.,1.]]))
        np.testing.assert_allclose(u,[528+320*np.pi/4,528,528],atol=1e-12)
        np.testing.assert_allclose(v,[512,512+325*np.pi/4,512],atol=1e-12)

    def test_vendor_bilinear_and_inverse_roundtrip(self):
        camera=KBSpline(INTRINSICS,interpolation='bilinear')
        np.testing.assert_allclose(np.array(camera.project(np.array(POINTS).T)).T,
                                   BILINEAR_EXPECTED,atol=5e-6,rtol=0)
        points=np.array(POINTS[:9])
        pixels=np.array(camera.project(points.T)).T
        expected=points/np.linalg.norm(points,axis=1)[:,None]
        np.testing.assert_allclose(camera.unproject(pixels),expected,atol=1e-12,rtol=0)
        with self.assertRaises(ValueError):KBSpline(INTRINSICS).unproject(pixels)
        with self.assertRaises(ValueError):camera.unproject([[10000.,10000.]])

    def test_invalid_schema_and_fixed_nodes_reject(self):
        for field,value in [('gridSize',[6,5]),('nodeSpacing',[0,222]),('fx',float('nan')),('sV',[0.]*24)]:
            k=copy.deepcopy(INTRINSICS);k[field]=value
            with self.assertRaises(ValueError):KBSpline(k)
        k=copy.deepcopy(INTRINSICS);k['sU'][7]=1.
        with self.assertRaises(ValueError):KBSpline(k)
        with self.assertRaises(ValueError):KBSpline(INTRINSICS).project([0,0,0])

if __name__ == '__main__':unittest.main()
