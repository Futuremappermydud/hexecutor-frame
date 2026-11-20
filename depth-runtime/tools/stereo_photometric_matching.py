"""Offline exploratory ROI-normalized OpenCV LK, frozen before evaluation.

Each 41x41 ROI is extracted on an integer pixel grid, independently normalized
to mean 128/std 32, and passed to original OpenCV LK. This is exploratory and
preserves original geometry; independent tests are required.
"""
import cv2
import numpy as np


def _segment(mask, start, end):
    n=max(1,int(np.ceil(2*np.linalg.norm(end-start))))
    q=np.rint(np.linspace(start,end,n+1)).astype(int)
    ok=(q[:,0]>=0)&(q[:,0]<mask.shape[1])&(q[:,1]>=0)&(q[:,1]<mask.shape[0])
    return bool(np.all(ok) and np.all(mask[q[:,1],q[:,0]]))


def _roi(image, point):
    cx,cy=np.rint(point).astype(int); r=20
    if cx-r<0 or cy-r<0 or cx+r>=image.shape[1] or cy+r>=image.shape[0]: return None
    a=np.asarray(image[cy-r:cy+r+1,cx-r:cx+r+1],dtype=float)
    sd=float(a.std())
    if sd<4: return None
    return np.clip(np.rint((a-a.mean())*(32./sd)+128.),0,255).astype(np.uint8),np.array([cx-r,cy-r],float)


def refine_matches(images,masks,matches,return_indices=False):
    if len(images)!=2 or len(masks)!=2:raise ValueError('expected two images and masks')
    images=[np.asarray(x, dtype=np.float32) for x in images];masks=[np.asarray(x) for x in masks]
    if any(x.ndim!=2 or not np.isfinite(x).all() for x in images):raise ValueError('images must be finite grayscale arrays')
    if any(x.ndim!=2 or x.shape!=im.shape for x,im in zip(masks,images)):raise ValueError('invalid masks')
    points=np.asarray(matches,dtype=float)
    if not points.size:points=np.empty((0,4))
    if points.ndim!=2 or points.shape[1]!=4 or not np.isfinite(points).all():raise ValueError('invalid matches')
    kw=dict(winSize=(15,15),maxLevel=0,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,minEigThreshold=1e-4)
    out=[];indices=[]
    for i,p in enumerate(points):
        left,seed=p[:2],p[2:]
        if not _segment(masks[0],left,left) or not _segment(masks[1],seed,seed):continue
        lr,rr=_roi(images[0],left),_roi(images[1],seed)
        if lr is None or rr is None:continue
        a,oa=lr;b,ob=rr
        l=np.asarray(left-oa,dtype=np.float32).reshape(1,1,2); r=np.asarray(seed-ob,dtype=np.float32).reshape(1,1,2)
        dest,status,_=cv2.calcOpticalFlowPyrLK(a,b,l,r.copy(),**kw)
        if dest is None or not status[0,0] or not np.isfinite(dest).all():continue
        back,status2,_=cv2.calcOpticalFlowPyrLK(b,a,dest,l.copy(),**kw)
        if back is None or not status2[0,0] or not np.isfinite(back).all():continue
        right=dest.ravel().astype(float)+ob; reverse=back.ravel().astype(float)+oa
        if np.linalg.norm(right-seed)>2 or np.linalg.norm(reverse-left)>.5:continue
        if not _segment(masks[0],left,reverse) or not _segment(masks[1],seed,right):continue
        # Apply the original texture and ZNCC gate to original intensities.
        pa=cv2.getRectSubPix(images[0],(15,15),tuple(map(float,left))).astype(float)
        pb=cv2.getRectSubPix(images[1],(15,15),tuple(map(float,right))).astype(float)
        pa-=pa.mean();pb-=pb.mean();norm=np.linalg.norm(pa)*np.linalg.norm(pb)
        if pa.std()<4 or pb.std()<4 or norm==0 or np.sum(pa*pb)/norm<.8:continue
        out.append([*left,*right]);indices.append(i)
    result=np.asarray(out,dtype=float).reshape(-1,4)
    return (result,np.asarray(indices,dtype=int)) if return_indices else result
