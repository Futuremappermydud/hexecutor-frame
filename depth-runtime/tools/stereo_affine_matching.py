"""Offline bidirectional affine patch refinement with local contrast normalization.

Six image-warp parameters are estimated; no epipolar, depth, pose or tag inputs
are used. Source points stay fixed. Default matchers are not modified.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

_TRIAL_OFFSETS=np.array([(x,y) for y in range(-14,15) for x in range(-14,15)],float)
_OFFSETS=np.array([(x,y) for y in range(-7,8) for x in range(-7,8)],float)


def _sample(image,xy):
    return map_coordinates(image,[xy[:,1],xy[:,0]],order=1,mode='nearest',prefilter=False)


def _supported(mask,xy):
    lo=np.floor(xy).astype(int);hi=lo+1
    if np.any(lo<0) or np.any(hi[:,0]>=mask.shape[1]) or np.any(hi[:,1]>=mask.shape[0]):return False
    return bool(np.all(mask[lo[:,1],lo[:,0]]) and np.all(mask[lo[:,1],hi[:,0]]) and np.all(mask[hi[:,1],lo[:,0]]) and np.all(mask[hi[:,1],hi[:,0]]))


def _fit(source,target,source_mask,target_mask,left,seed,gradients):
    if not _supported(target_mask,seed+_TRIAL_OFFSETS):return None
    xy=left+_OFFSETS
    if not _supported(source_mask,xy):return None
    template=_sample(source,xy);sd=template.std()
    if sd<4:return None
    template=(template-template.mean())/sd
    cached=None;values=None;jacobian=None
    def evaluate(p):
        nonlocal cached,values,jacobian
        if cached is not None and np.array_equal(cached,p):return values,jacobian
        a=np.eye(2)+p[2:].reshape(2,2);q=seed+p[:2]+_OFFSETS@a.T
        patch=_sample(target,q);std=max(float(patch.std()),1e-6);z=(patch-patch.mean())/std
        gx=_sample(gradients[0],q);gy=_sample(gradients[1],q)
        j=np.column_stack((gx,gy,gx*_OFFSETS[:,0],gx*_OFFSETS[:,1],gy*_OFFSETS[:,0],gy*_OFFSETS[:,1]))
        jacobian=(j-j.mean(axis=0)-z[:,None]*np.mean(z[:,None]*j,axis=0))/std
        values=z-template;cached=p.copy();return values,jacobian
    bounds=np.array([2.,2.,.15,.15,.15,.15])
    opt=least_squares(lambda p:evaluate(p)[0],np.zeros(6),jac=lambda p:evaluate(p)[1],bounds=(-bounds,bounds),
                      loss='soft_l1',f_scale=.2,max_nfev=40,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    p=opt.x;a=np.eye(2)+p[2:].reshape(2,2);q=seed+p[:2]+_OFFSETS@a.T
    if not opt.success or np.any(np.abs(p)>=bounds*.999) or not _supported(target_mask,q):return None
    information=np.linalg.svd(opt.jac,compute_uv=False)
    if information[0]<=1e-8 or information[-1]/information[0]<1e-4:return None
    singular=np.linalg.svd(a,compute_uv=False)
    if singular.min()<.75 or singular.max()>1.25 or np.linalg.det(a)<.7:return None
    patch=_sample(target,q);std=patch.std()
    if std<4:return None
    correlation=float(np.mean(template*(patch-patch.mean())/std))
    if correlation<.8:return None
    return seed+p[:2]


def refine_matches(images,masks,matches,return_indices=False):
    if len(images)!=2 or len(masks)!=2:raise ValueError('two images and masks required')
    images=[np.asarray(i,float) for i in images];masks=[np.asarray(m,bool) for m in masks]
    if any(i.ndim!=2 or not np.isfinite(i).all() or m.shape!=i.shape for i,m in zip(images,masks)):raise ValueError('finite grayscale images and same-sized masks required')
    points=np.asarray(matches,float)
    if not points.size:points=np.empty((0,4))
    if points.ndim!=2 or points.shape[1]!=4 or not np.isfinite(points).all():raise ValueError('finite N-by-4 matches required')
    gradients=[]
    for im in images:
        gy,gx=np.gradient(im);gradients.append((gx,gy))
    output=[];indices=[]
    for i,p in enumerate(points):
        right=_fit(images[0],images[1],masks[0],masks[1],p[:2],p[2:],gradients[1])
        if right is None or np.linalg.norm(right-p[2:])>2:continue
        reverse=_fit(images[1],images[0],masks[1],masks[0],right,p[:2],gradients[0])
        if reverse is None or np.linalg.norm(reverse-p[:2])>.5:continue
        output.append(np.r_[p[:2],right]);indices.append(i)
    result=np.asarray(output,float).reshape(-1,4)
    return (result,np.asarray(indices,int)) if return_indices else result
