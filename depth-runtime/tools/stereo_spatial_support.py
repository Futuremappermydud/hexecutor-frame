"""Whole-track spatial split for offline calibration sensitivity checks.

The first supplied left observation fixes track membership in alternating image
cells. Neither right pixels, fitted residuals nor metric references assign folds.
Later observations remain with their track even if they cross a cell boundary.
"""
import numpy as np


def spatial_track_folds(rows,cell_width_px=128.):
    rows=np.asarray(rows,float)
    if rows.ndim!=2 or rows.shape[1]!=6 or not np.isfinite(rows).all():raise ValueError('finite N-by-6 observations required')
    if not np.isfinite(cell_width_px) or cell_width_px<=0:raise ValueError('positive finite cell width required')
    if np.any(rows[:,:2]!=np.floor(rows[:,:2])) or len(np.unique(rows[:,:2],axis=0))!=len(rows):raise ValueError('unique integer frame/track IDs required')
    labels=np.empty(len(rows),int);assignments=[]
    for track in np.unique(rows[:,1]):
        indices=np.flatnonzero(rows[:,1]==track);i=indices[np.argmin(rows[indices,0])]
        cell=np.floor(rows[i,2:4]/cell_width_px).astype(int)
        fold=int(np.sum(cell)%2);labels[indices]=fold
        assignments.append({'track_id':int(track),'first_frame_id':int(rows[i,0]),'first_left_cell_xy':cell.tolist(),'fold':fold})
    return labels,{'cell_width_px':cell_width_px,'assignments':assignments,'scope':__doc__}
