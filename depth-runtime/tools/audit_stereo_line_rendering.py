#!/usr/bin/env python3
"""Independent rendered-image control for natural-line stereo rotation.

Projects actual 3D segments with OpenCV, rasterizes at 4x resolution, then uses
only grayscale images/masks and recorded rotations for extraction and fitting.
Truth geometry never supplies labels, directions, or line normals to the fit.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from stereo_line_rotation import extract_lines, fit_line_rotation

K=np.array([[380.,0.,308.],[0.,380.,308.],[0.,0.,1.]])


def control(rotation_vector, seed=1811, gain=.7, blur=.5, independent_eye_consensus=False):
    rng=np.random.default_rng(seed);rotation_vector=np.asarray(rotation_vector,float)
    # Two nonorthogonal families, varied positions and depths, fixed across eyes/time.
    axes=np.array([[1.,.12,.10],[-.18,1.,.15]])
    axes/=np.linalg.norm(axes,axis=1)[:,None]
    segments=[]
    for direction in axes:
        for _ in range(24):
            center=np.r_[rng.uniform(-.9,.9,2),rng.uniform(1.6,3.0)]
            segments.append(np.array([center-.55*direction,center+.55*direction]))
    rotations=np.array([cv2.Rodrigues(np.array([.004*i,-.006*i,.002*i]))[0] for i in range(8)])
    normals=[];eyes=[];frames=[];weights=[];counts=[]
    for frame,r in enumerate(rotations):
        center=np.array([.018*frame,.009*frame,0.]);counts.append([])
        for eye in (0,1):
            image=np.full((2464,2464),28,np.uint8)
            for segment in segments:
                points=(segment-center)@r
                if eye:points=(points-[.105,0.,0.])@cv2.Rodrigues(rotation_vector)[0]
                # projectPoints applies R world->camera independently of estimator code.
                uv,_=cv2.projectPoints(points,np.zeros(3),np.zeros(3),K,None)
                a,b=np.rint(uv.reshape(2,2)*4).astype(int)
                cv2.line(image,tuple(a),tuple(b),220,5,cv2.LINE_AA)
            image=cv2.resize(image,(616,616),interpolation=cv2.INTER_AREA)
            if blur:image=cv2.GaussianBlur(image,(5,5),blur)
            if eye:image=np.clip(image.astype(float)*gain+17.,0,255).astype(np.uint8)
            _,n,w=extract_lines(image,np.full_like(image,255),K)
            normals.extend(n);weights.extend(w);eyes.extend([eye]*len(n));frames.extend([frame]*len(n));counts[-1].append(len(n))
    fit=fit_line_rotation(normals,eyes,frames,rotations,weights,independent_eye_consensus=independent_eye_consensus)
    result={'truth_rotation_vector_rad':rotation_vector.tolist(),'line_counts_by_frame_eye':counts,'fit':fit,
            'scope':'Known rendered pinhole segment scene; no real-capture accuracy claim.'}
    if fit['accepted']:
        rays=np.array([[x,y,1.] for x in (-.5,0.,.5) for y in (-.5,0.,.5)])
        def projected(vector):
            p=rays@cv2.Rodrigues(np.asarray(vector))[0]@K.T
            return p[:,:2]/p[:,2,None]
        result['max_horizontal_truth_probe_error_px']=float(np.max(np.abs(projected(fit['rotation_vector_rad'])[:,0]-projected(rotation_vector)[:,0])))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');p.add_argument('--independent-eye-consensus',action='store_true');a=p.parse_args()
    results=[]
    for seed in (1811,1812,1813):
        for yaw in (-.006,0.,.006):
            result=control([.002,yaw,-.001],seed=seed,independent_eye_consensus=a.independent_eye_consensus);result['seed']=seed;results.append(result)
            print(seed,yaw,result['fit']['accepted'],result.get('max_horizontal_truth_probe_error_px'),flush=True)
    Path(a.output).write_text(json.dumps({'controls':results,'horizontal_truth_limit_px':.3},indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
