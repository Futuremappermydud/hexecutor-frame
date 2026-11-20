"""Global nominal stereo bearing initialization with fixed supplied rotations.

Solves all point positions and camera centers jointly. Each independent group's
first center fixes its gauge; the supplied stereo baseline is the only scale.
This is an initialization, not a calibration estimate or acceptance criterion.
"""
import hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr


def initialize_rotation_graph(left_pixels,right_pixels,frame_indices,track_indices,rotations,track_count,gauges,intrinsic_left,intrinsic_right,baseline):
    fi=np.asarray(frame_indices,int);ti=np.asarray(track_indices,int);rot=np.asarray(rotations,float)
    nf=len(rot);free=[i for i in range(nf) if i not in set(gauges)]
    slots={f:i for i,f in enumerate(free)}
    left=np.c_[left_pixels,np.ones(len(fi))]@np.linalg.inv(intrinsic_left).T
    right=np.c_[right_pixels,np.ones(len(fi))]@np.linalg.inv(intrinsic_right).T
    row_indices=[];columns=[];values=[];rhs=[]
    for i,(f,t) in enumerate(zip(fi,ti)):
        for eye,ray in enumerate((left[i],right[i])):
            d=rot[f]@ray;d/=np.linalg.norm(d);p=np.eye(3)-np.outer(d,d)
            base=6*i+3*eye
            for a in range(3):
                for b in range(3):
                    row_indices.append(base+a);columns.append(3*t+b);values.append(p[a,b])
                    if f in slots:
                        row_indices.append(base+a);columns.append(3*track_count+3*slots[f]+b);values.append(-p[a,b])
            rhs.extend(p@(rot[f]@np.array([baseline,0.,0.])) if eye else np.zeros(3))
    a=coo_matrix((values,(row_indices,columns)),shape=(6*len(fi),3*track_count+3*len(free))).tocsr()
    solution=lsmr(a,np.asarray(rhs),atol=1e-10,btol=1e-10,maxiter=4000)
    if solution[1] not in (0,1,2,4,5) or not np.isfinite(solution[0]).all():return None
    world=solution[0][:3*track_count].reshape(-1,3);centers=np.zeros((nf,3))
    centers[free]=solution[0][3*track_count:].reshape(-1,3)
    return world,centers,{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lsmr_stop':int(solution[1]),'lsmr_iterations':int(solution[2]),'linear_residual_norm':float(solution[3])}
