"""Offline orthogonal nuisance-block coordinates for bundle conditioning.

Only unbounded disjoint pose/point blocks rotate into their initial Jacobian's
right-singular basis. Existing Jacobian column scaling then acts along those
local singular directions. No variable is removed and no prior is introduced.
"""
import hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import eye


def block_rotation(jacobian, blocks, bounds):
    n=jacobian.shape[1]
    lower=np.broadcast_to(np.asarray(bounds[0],float),(n,))
    upper=np.broadcast_to(np.asarray(bounds[1],float),(n,))
    transform=eye(n,format='lil');used=set();rotated=0;skipped=0
    for start,width in blocks:
        if (isinstance(start,bool) or isinstance(width,bool) or not isinstance(start,int) or
                not isinstance(width,int) or width not in (3,6) or start<0 or start+width>n):
            raise ValueError('in-range pose/point blocks of width three or six required')
        columns=set(range(start,start+width))
        if used&columns:raise ValueError('nuisance blocks must be disjoint')
        used|=columns
        if np.isfinite(lower[start:start+width]).any() or np.isfinite(upper[start:start+width]).any():
            raise ValueError('bounded parameters must retain original coordinates')
        selected=jacobian[:,start:start+width].tocsr()
        active=np.unique(selected.nonzero()[0])
        block=selected[active].toarray()
        if not np.isfinite(block).all():raise ValueError('finite nuisance Jacobian required')
        _,singular,vt=np.linalg.svd(block,full_matrices=False)
        # Keep numerically rank-deficient blocks in their original coordinates;
        # do not expose a numerical null direction to inverse column scaling.
        if len(singular)!=width or singular[-1]<=max(singular[0]*np.sqrt(np.finfo(float).eps),np.finfo(float).tiny):
            skipped+=1;continue
        transform[start:start+width,start:start+width]=vt.T
        rotated+=1
    return transform.tocsc(),{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rotated_blocks':rotated,'rank_deficient_blocks_unchanged':skipped,
        'map':'orthogonal_block_rotation','variables_removed':0,'rotation_basis':'initial_unweighted_jacobian'}
