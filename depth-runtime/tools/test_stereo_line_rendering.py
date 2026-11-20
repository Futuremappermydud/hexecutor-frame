import unittest
from audit_stereo_line_rendering import control

class RenderedLineTests(unittest.TestCase):
    def test_full_image_detector_and_rotation_pipeline(self):
        for yaw in (0.,.006):
            with self.subTest(yaw=yaw):
                result=control([.002,yaw,-.001],seed=1813,independent_eye_consensus=True)
                self.assertTrue(result['fit']['accepted'],result)
                self.assertLess(result['max_horizontal_truth_probe_error_px'],.3)

if __name__=='__main__':unittest.main()
