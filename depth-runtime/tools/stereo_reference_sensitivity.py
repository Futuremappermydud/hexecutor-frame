"""Offline sensitivity of conditional left-only planar reference reconstruction.

Illustrative corner perturbations are not measured sensor uncertainty, truth,
confidence intervals or acceptance thresholds. Right observations never affect
pose, perturbation draws, predictions, failure counts or sensitivity statistics.
"""
import cv2
import numpy as np
from stereo_lens_calibration import project_right


def left_pose_prediction(pixels,left,right,parameters,baseline,tag_size):
    pixels=np.asarray(pixels,float);p=np.asarray(parameters,float)
    if pixels.shape!=(4,2) or not np.isfinite(pixels).all() or not np.isfinite(tag_size) or tag_size<=0:
        raise ValueError('four finite corners and positive tag size required')
    half=tag_size/2;obj=np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]],float)
    k=left.intrinsic
    rays=left.rays(pixels,p[4]);h=rays@k.T;corrected=h[:,:2]/h[:,2,None]
    # Undo skew for OpenCV PnP, retaining exact intrinsic geometry in reporting.
    pnp=corrected.copy();pnp[:,0]-=k[0,1]*(pnp[:,1]-k[1,2])/k[1,1]
    pk=k.copy();pk[0,1]=0
    ok,rv,tv=cv2.solvePnP(obj,pnp,pk,None,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:raise ValueError('left pose fit failed')
    points=obj@cv2.Rodrigues(rv)[0].T+tv.ravel()
    if np.any(points[:,2]<=0):raise ValueError('left pose behind camera')
    predicted_left=left.project(points,p[4]);predicted_right=project_right(points,right,p,baseline)
    return predicted_right,{'left_reprojection_rms_xy_px':np.sqrt(np.mean((predicted_left-pixels)**2,axis=0)).tolist(),
        'center_depth_m':float(np.mean(points[:,2])),'span_xy_px':np.ptp(pixels,axis=0).tolist()}


def reference_sensitivity(observations,left,right,parameters,baseline,tag_size,sigmas=(0.,.05,.1,.25),draws=256,seed=2111):
    if isinstance(draws,bool) or not isinstance(draws,int) or not 2<=draws<=2048 or any(not np.isfinite(s) or s<0 for s in sigmas):
        raise ValueError('bounded draws and finite nonnegative perturbations required')
    rng=np.random.default_rng(seed);entries=[]
    for index,observation in enumerate(observations):
        pixels=np.asarray(observation['left'],float)
        nominal,geometry=left_pose_prediction(pixels,left,right,parameters,baseline,tag_size)
        entry={'observation_index':index,'ordinal':observation.get('ordinal'),'geometry':geometry,'perturbations':[]}
        for sigma in sigmas:
            predictions=[];failures=0
            for _ in range(draws):
                try:prediction,_=left_pose_prediction(pixels+rng.normal(0,sigma,(4,2)),left,right,parameters,baseline,tag_size)
                except (ValueError,cv2.error):failures+=1;continue
                predictions.append(prediction)
            values=np.array(predictions)
            entry['perturbations'].append({'assumed_independent_corner_sigma_px':sigma,'successful_draws':len(values),'failed_draws':failures,
                'rms_prediction_change_xy_px':np.sqrt(np.mean((values-nominal)**2,axis=(0,1))).tolist() if len(values) else None,
                'mean_prediction_change_xy_px':np.mean(values-nominal,axis=(0,1)).tolist() if len(values) else None})
        entries.append(entry)
    return {'scope':__doc__,'draws_per_sigma_per_observation':draws,'seed':seed,'observations':entries,'metric_acceptance':'unevaluated'}
