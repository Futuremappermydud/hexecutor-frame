"""Offline descriptive residual structure for physical natural-track fits.

No residual group selects calibration, support, model parameters or publication.
Missing poses/points are counted explicitly; rejected fits remain rejected.
"""
import cv2
import numpy as np
from stereo_lens_calibration import calibrated_left_camera, baseline_direction, shifted_principal_camera


def residual_structure(rows, fit, cameras, baseline, cell_width_px=128):
    rows = np.asarray(rows, float)
    if rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all():
        raise ValueError('finite N-by-6 observations required')
    if not np.isfinite(cell_width_px) or cell_width_px <= 0 or not np.isfinite(baseline) or baseline <= 0:
        raise ValueError('positive cell width and supplied baseline required')
    if np.any(rows[:, :2] != np.floor(rows[:, :2])) or len(np.unique(rows[:, :2], axis=0)) != len(rows):
        raise ValueError('unique integer frame/track observations required')
    poses = {p['frame_id']: (cv2.Rodrigues(np.array(p['rotation_vector_rad']))[0], np.array(p['translation_m'])) for p in fit.get('poses', [])}
    points = {p['track_id']: np.array(p['position_m']) for p in fit.get('points', [])}
    selected = np.array([r[0] in poses and r[1] in points for r in rows], bool)
    report = {'training_accepted': bool(fit.get('accepted')), 'input_observations': len(rows),
              'omitted_missing_pose_or_point': int(np.sum(~selected)), 'cell_width_px': cell_width_px,
              'track_age_definition': 'Frame-ID difference from earliest observation in supplied input; not necessarily physical track birth after support pruning.',
              'scope': 'Descriptive training residuals, correlated within tracks; not truth, covariance or metric acceptance.'}
    obs = rows[selected]
    if not len(obs):
        report.update(available=False, reason='no_reconstructable_observations'); return report
    p = np.array(fit['parameters_vector']); d = baseline_direction(p)
    local = np.array([(points[t]-poses[f][1]) @ poses[f][0] for f,t in obs[:,:2]])
    lp,lv = calibrated_left_camera(cameras[0],p).project_for_optimization(local,p[4])
    right = shifted_principal_camera(cameras[1], p[8:10]) if len(p) in (10,11) else cameras[1]
    rp,rv = right.project_for_optimization((local-baseline*d) @ cv2.Rodrigues(p[:3])[0],p[5],np.exp(p[3]))
    errors = np.column_stack((lp-obs[:,2:4],rp-obs[:,4:6]))
    valid = lv & rv
    report['invalid_projection_observations'] = int(np.sum(~valid))
    earliest = {t: np.min(rows[rows[:,1] == t,0]) for t in np.unique(rows[:,1])}
    age = np.array([f-earliest[t] for f,t in obs[:,:2]])
    def stats(mask):
        e=errors[mask & valid]
        return {'observations':len(e), 'tracks':len(np.unique(obs[mask & valid,1])),
                'mean_lx_ly_rx_ry_px':np.mean(e,axis=0).tolist() if len(e) else None,
                'median_lx_ly_rx_ry_px':np.median(e,axis=0).tolist() if len(e) else None,
                'rms_lx_ly_rx_ry_px':np.sqrt(np.mean(e*e,axis=0)).tolist() if len(e) else None}
    report.update(available=True,overall=stats(np.ones(len(obs),bool)))
    report['by_frame']=[dict(frame_id=int(f),**stats(obs[:,0]==f)) for f in np.unique(obs[:,0])]
    report['by_track_age']=[dict(age_frames=int(a),**stats(age==a)) for a in np.unique(age)]
    report['by_track']=[dict(track_id=int(t),**stats(obs[:,1]==t)) for t in np.unique(obs[:,1])]
    for eye,offset in (('left',2),('right',4)):
        cells=np.floor(obs[:,offset:offset+2]/cell_width_px).astype(int)
        report['by_'+eye+'_image_cell']=[dict(cell_xy=c.tolist(),**stats(np.all(cells==c,axis=1))) for c in np.unique(cells,axis=0)]
    return report
