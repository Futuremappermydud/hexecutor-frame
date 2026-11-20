"""Offline validation nuisance fit with frozen physical stereo calibration.

Only supplied support tracks enter pose/point fitting. Evaluation tracks must
be passed separately to left-only triangulation. Calibration is never optimized.
The fixed baseline sets scale; recorded rotations remain a conditional input.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from stereo_global_initialization import initialize_rotation_graph
from stereo_lens_calibration import calibrated_left_camera


def fit_fixed_geometry(rows, rotations, cameras, baseline, parameters):
    rows=np.asarray(rows,float);rot=np.asarray(rotations,float);p=np.asarray(parameters,float).copy()
    if rows.ndim!=2 or rows.shape[1]!=6 or not np.isfinite(rows).all():raise ValueError('finite N-by-6 support observations required')
    if p.shape not in ((6,),(11,)) or not np.isfinite(p).all():raise ValueError('frozen six- or eleven-element physical calibration required')
    if len(p)==11 and np.any(p[6:10]!=0):raise ValueError('fixed geometry requires nominal baseline direction and no principal shift')
    if not np.isfinite(baseline) or baseline<=0:raise ValueError('positive baseline required')
    if np.any(rows[:,:2]!=np.floor(rows[:,:2])) or len(np.unique(rows[:,:2],axis=0))!=len(rows):raise ValueError('unique integer frame/track IDs required')
    frames,fi=np.unique(rows[:,0],return_inverse=True);tracks,ti=np.unique(rows[:,1],return_inverse=True);nf=len(frames);nt=len(tracks)
    if rot.shape!=(nf,3,3) or not np.isfinite(rot).all() or not np.allclose(rot.transpose(0,2,1)@rot,np.eye(3),atol=1e-6) or not np.allclose(np.linalg.det(rot),1,atol=1e-6) or (nf and not np.allclose(rot[0],np.eye(3),atol=1e-6)):raise ValueError('proper rotations in sorted frame order and first-camera gauge required')
    result={'accepted':False,'metric_acceptance':'unevaluated','parameters_vector':p.tolist(),'poses':[],'points':[], 'diagnostics':{'scope':'Validation support-only nuisance fit with frozen calibration; not calibration acceptance or metric truth.','reasons':[]}}
    def reject(reason):result['diagnostics']['reasons']=[reason];return result
    if nf<3 or nt<30 or len(rows)<90 or np.min(np.bincount(fi))<12 or np.min(np.bincount(ti))<3:return reject('insufficient_support')
    seen_f={0};seen_t=set()
    while True:
        next_t=seen_t|set(ti[np.isin(fi,list(seen_f))].tolist())
        next_f=seen_f|set(fi[np.isin(ti,list(next_t))].tolist())
        if next_f==seen_f and next_t==seen_t:break
        seen_f,seen_t=next_f,next_t
    if len(seen_f)!=nf or len(seen_t)!=nt:return reject('disconnected_support')
    left,right=cameras;left=calibrated_left_camera(left,p);relative=cv2.Rodrigues(p[:3])[0]
    lr=left.rays(rows[:,2:4],p[4]);rr=right.rays(rows[:,4:6],p[5],np.exp(p[3]))@relative.T
    def pixels(rays,k):
        q=rays@k.T;return q[:,:2]/q[:,2,None]
    initial=initialize_rotation_graph(pixels(lr,left.intrinsic),pixels(rr,right.intrinsic),fi,ti,rot,nt,[0],left.intrinsic,right.intrinsic,baseline)
    if initial is None:return reject('initialization_failed')
    world,centers,diagnostic=initial;result['diagnostics']['initialization']=diagnostic
    def unpack(x):return np.vstack((np.zeros(3),x[:3*(nf-1)].reshape(-1,3))),x[3*(nf-1):].reshape(-1,3)
    def residual(x,domain=False):
        c,w=unpack(x);local=np.einsum('nji,nj->ni',rot[fi],w[ti]-c[fi]);lp,lv=left.project_for_optimization(local,p[4]);rp,rv=right.project_for_optimization((local-[baseline,0,0])@relative,p[5],np.exp(p[3]));e=np.c_[lp-rows[:,2:4],rp-rows[:,4:6]]
        return (e,lv&rv) if domain else e.ravel()
    start=np.r_[centers[1:].ravel(),world.ravel()];sparsity=lil_matrix((4*len(rows),len(start)),dtype=np.int8)
    for i,(f,t) in enumerate(zip(fi,ti)):
        if f:sparsity[4*i:4*i+4,3*(f-1):3*f]=1
        k=3*(nf-1)+3*t;sparsity[4*i:4*i+4,k:k+3]=1
    opt=least_squares(residual,start,jac_sparsity=sparsity.tocsr(),loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=100,ftol=1e-7,xtol=1e-7,gtol=1e-7,tr_options={'atol':1e-10,'btol':1e-10,'maxiter':1000})
    e,valid=residual(opt.x,True);c,w=unpack(opt.x);inliers=np.all(np.linalg.norm(e.reshape(-1,2,2),axis=2)<=1.,axis=1)
    result['diagnostics'].update(converged=bool(opt.success),evaluations=opt.nfev,projection_domain_fraction=float(valid.mean()),inlier_fraction=float(inliers.mean()),support_observations=len(rows),support_tracks=nt)
    if not opt.success:return reject('solver_not_converged')
    if valid.mean()<.95:return reject('projection_domain')
    if inliers.mean()<.6:return reject('insufficient_inliers')
    if np.max(np.linalg.norm(c,axis=1))<.01:return reject('insufficient_translation')
    result['accepted']=True
    result['poses']=[{'frame_id':int(f),'rotation_vector_rad':cv2.Rodrigues(r)[0].ravel().tolist(),'translation_m':v.tolist()} for f,r,v in zip(frames,rot,c)]
    result['points']=[{'track_id':int(t),'position_m':v.tolist()} for t,v in zip(tracks,w)]
    return result
