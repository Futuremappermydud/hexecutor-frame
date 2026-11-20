import unittest
import cv2
import numpy as np
from audit_stereo_eye_skew import RotationTimeline,projection_effect

class EyeSkewTests(unittest.TestCase):
    def test_constant_angular_motion_and_independent_projection(self):
        velocity=np.array([.1,.7,-.2])
        rows=[]
        for t in (0,1000000,2000000):
            r=cv2.Rodrigues(velocity*(t/1e9))[0]
            rows.append(dict(mono_ns=str(t),valid='1',result='200',**{f'm{i}{j}':str(r[i,j]) for i in range(3) for j in range(3)}))
        for row in rows:
            for axis in range(3):row[f'm{axis}3']=str([.4,-.2,.1][axis]*int(row['mono_ns'])/1e9)
        timeline=RotationTimeline(rows)
        a,_=timeline.at(250000);b,_=timeline.at(1750000)
        _,position,_=timeline.at(750000,include_translation=True)
        np.testing.assert_allclose(position,np.array([.4,-.2,.1])*.00075,atol=1e-12)
        translation=projection_effect(np.eye(3),translation_drift_m=[.001,0,0])
        self.assertAlmostEqual(translation['maximum_absolute_xy_px'][0],380.*.001/.7,places=10)
        self.assertAlmostEqual(translation['maximum_absolute_xy_px'][1],0.,places=10)
        np.testing.assert_allclose(a.T@b,cv2.Rodrigues(velocity*.0015)[0],atol=1e-12)
        k=np.array([[380.,12.,308.],[0.,390.,310.],[0.,0.,1.]])
        effect=projection_effect(a.T@b,k)
        points=np.array([[x,y,1.] for x in (-.5,0.,.5) for y in (-.5,0.,.5)])
        # Direct homogeneous projection includes skew, independent of helper.
        q=(b.T@a@points.T).T@k.T;original=points@k.T
        delta=q[:,:2]/q[:,2,None]-original[:,:2]/original[:,2,None]
        np.testing.assert_allclose(effect['maximum_absolute_xy_px'],np.max(abs(delta),axis=0),atol=1e-10)
        with self.assertRaises(ValueError):timeline.at(-1)
        with self.assertRaises(ValueError):timeline.at(500000,max_bracket_ns=100)
        rows[1]['valid']='0'
        with self.assertRaises(ValueError):timeline.at(500000)

if __name__=='__main__':unittest.main()
