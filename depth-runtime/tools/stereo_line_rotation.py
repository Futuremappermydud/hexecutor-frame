"""Experimental offline stereo rotation from shared natural line directions.

Each line plane normal is measured in its eye's nominal virtual camera. Supplied
camera-to-world rotations provide temporal alignment only. Directions are inferred
from left lines; no scene depth, translations, tags or Manhattan axes are inputs.
Default mode fixes nominal focal/lens parameters. Explicit physical mode fits
right physical focal scale and both KB4 k1 changes from endpoint rays. Neither
mode certifies accuracy.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares


def extract_lines(image, mask, intrinsic, minimum_length=40.):
    image,mask=np.asarray(image),np.asarray(mask)
    k=np.asarray(intrinsic,float)
    if image.ndim!=2 or image.dtype!=np.uint8 or mask.shape!=image.shape or k.shape!=(3,3):
        raise ValueError('uint8 grayscale image, same-size mask and intrinsic required')
    raw=cv2.createLineSegmentDetector().detect(image)[0]
    lines=[]
    for line in raw.reshape(-1,4) if raw is not None else []:
        a,b=line[:2],line[2:];length=np.linalg.norm(b-a)
        if length<minimum_length:continue
        q=np.rint(np.linspace(a,b,int(np.ceil(2*length))+1)).astype(int)
        if np.any(q<0) or np.any(q[:,0]>=mask.shape[1]) or np.any(q[:,1]>=mask.shape[0]):continue
        if not np.all(mask[q[:,1],q[:,0]]):continue
        lines.append(line)
    lines=np.asarray(lines,float).reshape(-1,4)
    inv=np.linalg.inv(k)
    a=np.c_[lines[:,:2],np.ones(len(lines))]@inv.T
    b=np.c_[lines[:,2:],np.ones(len(lines))]@inv.T
    normals=np.cross(a,b);norm=np.linalg.norm(normals,axis=1)
    valid=norm>1e-10
    return lines[valid],normals[valid]/norm[valid,None],np.minimum(np.linalg.norm(lines[valid,2:]-lines[valid,:2],axis=1),150.)


def supported_frames(frame_ids, minimum_lines=3):
    _, counts = np.unique(frame_ids, return_counts=True)
    return int(np.sum(counts >= minimum_lines))


def discover_directions(normals, frames, weights, threshold=.003, minimum_lines=20, minimum_frames=5):
    n=np.asarray(normals,float);frames=np.asarray(frames);weights=np.asarray(weights,float)
    remaining=np.ones(len(n),bool);directions=[];reports=[];rng=np.random.default_rng(1701)
    for _ in range(6):
        indices=np.flatnonzero(remaining)
        if len(indices)<minimum_lines:break
        pairs=rng.choice(indices,(3000,2),replace=True)
        candidates=np.cross(n[pairs[:,0]],n[pairs[:,1]])
        lengths=np.linalg.norm(candidates,axis=1);candidates=candidates[lengths>.02]/lengths[lengths>.02,None]
        if not len(candidates):break
        # Bounded blocks keep the experiment's memory use independent of line count.
        best=None;best_score=-1
        for start in range(0,len(candidates),100):
            c=candidates[start:start+100];inside=np.abs(c@n[indices].T)<threshold
            scores=inside@weights[indices];scores[inside.sum(axis=1)<minimum_lines]=-1
            j=int(np.argmax(scores))
            if scores[j]>best_score:best_score=float(scores[j]);best=c[j]
        if best is None or best_score<0:break
        keep=remaining & (np.abs(n@best)<threshold)
        _,sing,v=np.linalg.svd(n[keep]*np.sqrt(weights[keep,None]),full_matrices=False)
        direction=v[-1];keep=remaining & (np.abs(n@direction)<threshold)
        frame_count=supported_frames(frames[keep]);ratio=float(sing[1]/sing[0])
        remaining[keep]=False
        if frame_count<minimum_frames or ratio<.02:continue
        if any(abs(direction@d)>np.cos(np.radians(20)) for d in directions):continue
        directions.append(direction);reports.append({'lines':int(keep.sum()),'frames_with_at_least_three_lines':frame_count,'normal_span_ratio':ratio})
    return np.asarray(directions,float).reshape(-1,3),reports


def fit_line_rotation(normals, eyes, frames, rotations, weights, line_endpoints=None, camera_ray_models=None, independent_eye_consensus=False):
    n=np.asarray(normals,float);eye=np.asarray(eyes);frame=np.asarray(frames);rot=np.asarray(rotations,float);weights=np.asarray(weights,float)
    if (n.ndim!=2 or n.shape[1]!=3 or any(a.shape!=(len(n),) for a in (eye,frame,weights)) or
        rot.ndim!=3 or rot.shape[1:]!=(3,3) or not np.isfinite(n).all() or not np.isfinite(rot).all() or
        not np.isfinite(weights).all() or np.any(weights<=0) or np.any(~np.isin(eye,[0,1])) or
        np.any(frame!=np.floor(frame)) or np.any(frame<0) or np.any(frame>=len(rot)) or
        not np.allclose(np.linalg.norm(n,axis=1),1.,atol=1e-6) or
        not np.allclose(rot.transpose(0,2,1)@rot,np.eye(3),atol=1e-6) or not np.allclose(np.linalg.det(rot),1.,atol=1e-6)):
        raise ValueError('unit normals, eye/frame IDs, proper rotations and positive weights required')
    physical = camera_ray_models is not None
    if physical != (line_endpoints is not None):
        raise ValueError('physical mode requires both line endpoints and camera ray models')
    if physical:
        from stereo_lens_calibration import CameraRayModel
        endpoints=np.asarray(line_endpoints,float)
        if endpoints.shape!=(len(n),4) or not np.isfinite(endpoints).all() or len(camera_ray_models)!=2 or any(not isinstance(c,CameraRayModel) for c in camera_ray_models):
            raise ValueError('finite line endpoints and two physical CameraRayModel instances required')
    ncal=6 if physical else 3
    def measured_normals(p):
        if not physical:return n.copy()
        out=np.empty_like(n)
        for e,c in enumerate(camera_ray_models):
            mask=eye==e;pixels=endpoints[mask]
            a=c.rays(pixels[:,:2],p[4+e],np.exp(p[3]) if e else 1.)
            b=c.rays(pixels[:,2:],p[4+e],np.exp(p[3]) if e else 1.)
            value=np.cross(a,b);length=np.linalg.norm(value,axis=1)
            if np.any(length<1e-10):raise ValueError('degenerate physical line endpoints')
            out[mask]=value/length[:,None]
        return out
    n=measured_normals(np.zeros(ncal))
    frame=frame.astype(int);world=np.einsum('nij,nj->ni',rot[frame],n)
    directions,support=discover_directions(world[eye==0],frame[eye==0],weights[eye==0])
    result={'accepted':False,'metric_acceptance':'unevaluated','directions':directions.tolist(),'left_direction_support':support,
            'model':'physical6' if physical else 'rotation3',
            'scope':'Natural-line direction rotation with supplied temporal alignment; training only, no metric certification.'}
    if len(directions)<2:result['reason']='insufficient_shared_directions';return result
    cost=np.abs(world@directions.T);labels=np.argmin(cost,axis=1)
    chosen=cost[np.arange(len(n)),labels]<.01
    if not isinstance(independent_eye_consensus,bool):
        raise ValueError('independent_eye_consensus must be boolean')
    result['independent_eye_consensus']=independent_eye_consensus
    if independent_eye_consensus:
        right_directions,right_support=discover_directions(world[eye==1],frame[eye==1],weights[eye==1])
        result['right_direction_support']=right_support
        result['right_directions']=right_directions.tolist()
        if len(right_directions)<2:
            result['reason']='insufficient_independent_right_directions';return result
        angular=np.arccos(np.clip(np.abs(directions@right_directions.T),0.,1.))
        matches=[]
        for l in range(len(directions)):
            r=int(np.argmin(angular[l]))
            if np.argmin(angular[:,r])==l and angular[l,r]<np.radians(3.):matches.append((l,r))
        result['direction_matches']=[{'left':l,'right':r,'nominal_angle_degrees':float(np.degrees(angular[l,r]))} for l,r in matches]
        if len(matches)<2:
            result['reason']='insufficient_mutual_direction_matches';return result
        right_cost=np.abs(world[eye==1]@right_directions.T)
        right_labels=np.argmin(right_cost,axis=1)
        mapped=np.full(len(right_labels),-1,int)
        for l,r in matches:mapped[right_labels==r]=l
        labels[eye==1]=np.maximum(mapped,0)
        chosen[eye==1]=(mapped>=0) & (right_cost[np.arange(len(mapped)),right_labels]<.003)
        chosen[eye==0]=cost[eye==0,labels[eye==0]]<.003
        chosen &= np.isin(labels,[l for l,r in matches])
    active=[]
    for d in range(len(directions)):
        if all(np.sum(chosen & (labels==d) & (eye==e))>=20 and supported_frames(frame[chosen & (labels==d) & (eye==e)])>=5 for e in (0,1)):active.append(d)
    chosen &= np.isin(labels,active)
    if len(active)<2:result['reason']='insufficient_two_eye_support';return result
    # Two tangent coordinates per direction avoid an unobservable normalization DOF.
    tangent=[]
    for d in directions:
        a=np.eye(3)[np.argmin(np.abs(d))];u=np.cross(d,a);u/=np.linalg.norm(u)
        tangent.append(np.column_stack((u,np.cross(d,u))))
    tangent=np.asarray(tangent)
    def axes(p):
        d=directions+np.einsum('nij,nj->ni',tangent,p[ncal:].reshape(-1,2))
        return d/np.linalg.norm(d,axis=1)[:,None]
    def errors(p):
        corrected=measured_normals(p);corrected[eye==1]=corrected[eye==1]@cv2.Rodrigues(p[:3])[0].T
        normal=np.einsum('nij,nj->ni',rot[frame],corrected)
        return np.sum(normal*axes(p)[labels],axis=1)
    # Inactive directions have no nuisance columns in optimization.
    used=np.array(active);initial=np.zeros(ncal+2*len(used))
    def expand(p):
        full=np.zeros(ncal+2*len(directions));full[:ncal]=p[:ncal]
        full[ncal:].reshape(-1,2)[used]=p[ncal:].reshape(-1,2);return full
    scale=np.sqrt(weights/np.mean(weights[chosen]))
    bounds=np.full(len(initial),.05)
    if physical:bounds[3:6]=[.03,.02,.02]
    opt=least_squares(lambda p:errors(expand(p))[chosen]*scale[chosen],initial,bounds=(-bounds,bounds),loss='soft_l1',f_scale=.002,max_nfev=100)
    p=expand(opt.x);res=errors(p);inlier=chosen & (np.abs(res)<.003)
    support_final=[]
    for d in active:
        support_final.append({'direction_index':int(d),'eyes':[{'lines':int(np.sum(inlier & (labels==d) & (eye==e))),
            'frames_with_at_least_three_lines':supported_frames(frame[inlier & (labels==d) & (eye==e)])} for e in (0,1)]})
    reasons=[]
    if not opt.success:reasons.append('solver_not_converged')
    if np.any(np.abs(opt.x)>=bounds*.999):reasons.append('parameter_bound_reached')
    if any(s['lines']<20 or s['frames_with_at_least_three_lines']<5 for d in support_final for s in d['eyes']):reasons.append('insufficient_final_support')
    singular=np.linalg.svd(opt.jac,compute_uv=False)
    if singular[-1]/singular[0]<1e-4:reasons.append('rank_or_conditioning_failure')
    result.update(accepted=not reasons,reasons=reasons,rotation_vector_rad=p[:3].tolist(),final_direction_support=support_final,
        chosen_lines=int(chosen.sum()),inlier_lines=int(inlier.sum()),rms_plane_sine=float(np.sqrt(np.mean(res[inlier]**2))) if inlier.any() else None,
        singular_values=singular.tolist(),fitted_directions=axes(p).tolist())
    if physical:result['parameters_vector']=p[:6].tolist()
    return result
