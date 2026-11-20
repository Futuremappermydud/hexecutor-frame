import unittest
import cv2
import numpy as np
from stereo_line_rotation import fit_line_rotation, extract_lines

class LineRotationTests(unittest.TestCase):
    def test_independent_translated_three_dimensional_lines(self):
        rng=np.random.default_rng(1712);truth=np.array([.003,-.006,.002])
        relative=cv2.Rodrigues(truth)[0]
        directions=np.array([[1.,.2,.3],[.1,1.,-.2],[.3,.1,1.]])
        directions/=np.linalg.norm(directions,axis=0)[None]
        axes=cv2.Rodrigues(np.array([.3,-.2,.15]))[0] @ directions
        rotations=np.array([cv2.Rodrigues(np.array([.01*i,-.005*i,.003*i]))[0] for i in range(8)])
        normals=[];eyes=[];frames=[]
        for f,r in enumerate(rotations):
            center=np.array([.02*f,.01*f,0.])
            for eye in (0,1):
                for axis in axes.T:
                    for _ in range(12):
                        a=rng.uniform(-1,1,3)+[0,0,3];b=a+.5*axis
                        points=(np.array([a,b])-center)@r
                        if eye:points=(points-[.105,0,0])@relative
                        # Independent pinhole endpoint projection, then line plane.
                        uv=points[:,:2]/points[:,2,None]
                        n=np.cross(np.r_[uv[0],1.],np.r_[uv[1],1.]);n/=np.linalg.norm(n)
                        normals.append(n);eyes.append(eye);frames.append(f)
                for _ in range(5):
                    n=rng.normal(size=3);normals.append(n/np.linalg.norm(n));eyes.append(eye);frames.append(f)
        result=fit_line_rotation(normals,eyes,frames,rotations,np.ones(len(normals)))
        self.assertTrue(result['accepted'],result)
        np.testing.assert_allclose(result['rotation_vector_rad'],truth,atol=.0005)
        self.assertEqual(result['metric_acceptance'],'unevaluated')
        independent=fit_line_rotation(normals,eyes,frames,rotations,np.ones(len(normals)),independent_eye_consensus=True)
        self.assertTrue(independent['accepted'],independent)
        np.testing.assert_allclose(independent['rotation_vector_rad'],truth,atol=.0005)
        self.assertGreaterEqual(len(independent['direction_matches']),2)
        # Replace the right scene's direction families with incompatible ones.
        # Nearby individual line planes must not masquerade as shared axes.
        incompatible=np.array(normals,copy=True);eye_array=np.array(eyes);frame_array=np.array(frames)
        q=cv2.Rodrigues(np.array([.2,.13,.17]))[0]
        for i in np.flatnonzero(eye_array==1):
            r=rotations[frame_array[i]]
            incompatible[i]=r.T @ q @ r @ incompatible[i]
        rejected=fit_line_rotation(incompatible,eyes,frames,rotations,np.ones(len(normals)),independent_eye_consensus=True)
        self.assertFalse(rejected['accepted'],rejected)
        self.assertEqual(rejected['reason'],'insufficient_mutual_direction_matches')



    def test_independent_physical_lens_and_rotation_recovery(self):
        from test_stereo_temporal_lens import fixture
        from test_stereo_lens_calibration import K, independent_projection
        _,_,_,cameras,truth=fixture()
        rng=np.random.default_rng(1814)
        rotations=np.array([cv2.Rodrigues(np.array([.01*i,-.005*i,.003*i]))[0] for i in range(8)])
        normals=[];eyes=[];frames=[];endpoints=[]
        axes=np.array([[1.,.1,.2],[.2,1.,-.1],[.1,.2,1.]])
        for f,r in enumerate(rotations):
            for eye in (0,1):
                for axis in axes:
                    for _ in range(16):
                        a=rng.uniform(-1,1,3)+[0,0,3.];points=(np.array([a,a+.6*axis])-[.02*f,.01*f,0])@r
                        if eye:points=(points-[.105,0.,0.])@cv2.Rodrigues(truth[:3])[0]
                        uv=independent_projection(points,cameras[eye].rotation,truth[4+eye],np.exp(truth[3]) if eye else 1.)
                        ray=np.c_[uv,np.ones(2)]@np.linalg.inv(K).T
                        n=np.cross(ray[0],ray[1]);n/=np.linalg.norm(n)
                        normals.append(n);eyes.append(eye);frames.append(f);endpoints.append(uv.ravel())
        result=fit_line_rotation(normals,eyes,frames,rotations,np.ones(len(normals)),endpoints,cameras)
        self.assertTrue(result['accepted'],result)
        np.testing.assert_allclose(result['parameters_vector'],truth,atol=.001)

    def test_single_repeated_line_per_frame_cannot_supply_direction_support(self):
        # Even many temporal views of one line do not meet per-image support.
        rng=np.random.default_rng(1713)
        normals=rng.normal(size=(120,3));normals[:,2]=0
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        result=fit_line_rotation(normals,np.zeros(120),np.arange(120),
                                 np.tile(np.eye(3),(120,1,1)),np.ones(120))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'],'insufficient_shared_directions')

    def test_missing_line_support_and_mask(self):
        result=fit_line_rotation(np.empty((0,3)),np.empty(0),np.empty(0),np.eye(3)[None],np.empty(0))
        self.assertFalse(result['accepted'])
        image=np.zeros((128,128),np.uint8);cv2.line(image,(15,30),(110,90),255,2)
        lines,n,w=extract_lines(image,np.zeros_like(image),np.eye(3))
        self.assertEqual(len(lines),0)
        with self.assertRaises(ValueError):fit_line_rotation([[0,0,2]],[0],[0],np.eye(3)[None],[1])

if __name__=='__main__':unittest.main()
