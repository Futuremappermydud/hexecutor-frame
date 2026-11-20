import unittest
import cv2
import numpy as np
from stereo_affine_matching import refine_matches

class AffineMatchingTests(unittest.TestCase):
    def test_known_affine_motion_and_photometric_change(self):
        rng=np.random.default_rng(2311)
        image=cv2.GaussianBlur(rng.integers(30,220,(256,256),dtype=np.uint8),(3,3),.7)
        a=np.array([[1.06,.04],[-.025,.97]]);t=np.array([.8,-.5])+128-a@np.array([128.,128.])
        target=cv2.warpAffine(image,np.c_[a,t],(256,256),flags=cv2.INTER_LINEAR).astype(float)*.7+20
        points=np.array([(x,y) for x in (65.,95.,125.,155.,185.) for y in (65.,95.,125.,155.,185.)])
        truth=points@a.T+t;seeds=np.c_[points,truth+[.4,-.3]];mask=np.ones_like(image,bool)
        refined,kept=refine_matches([image,target],[mask,mask],seeds,True)
        self.assertGreaterEqual(len(kept),20)
        self.assertLess(np.max(np.linalg.norm(refined[:,2:]-truth[kept],axis=1)),.15)
        self.assertEqual(len(refine_matches([image,target],[mask,np.zeros_like(mask)],seeds)),0)
        self.assertEqual(refine_matches([image,target],[mask,mask],[]).shape,(0,4))

    def test_linear_edge_has_no_observable_affine_patch_position(self):
        ramp=np.tile(np.arange(256,dtype=float),(256,1));mask=np.ones_like(ramp,bool)
        result=refine_matches([ramp,ramp*.7+20],[mask,mask],[[100.,100.,100.8,100.4]])
        self.assertEqual(len(result),0)

if __name__=='__main__':unittest.main()
