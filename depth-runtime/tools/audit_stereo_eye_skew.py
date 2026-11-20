#!/usr/bin/env python3
"""Offline rotation-only image effect of recorded left/right raw timestamp skew.

This is conditional on raw skew representing exposure skew, the explicit global
exposure offset, recorded pose accuracy and nominal camera/head axes. Unknown
translation scale, rolling readout timing and per-eye exposure delays are not inferred.
An additional diagnostic includes logged translations at their recorded scale and
the nominal right-camera lever arm; that scale is an assumption, not metric truth.
No natural pixels or tags select an offset or correction.
"""
import argparse
import bisect
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from stereo_pose_constraints import rectified_in_head, basis
from evaluate_stereo_temporal_calibration import K, sha


class RotationTimeline:
    def __init__(self, rows):
        self.rows=rows;self.times=[int(r['mono_ns']) for r in rows]
        if len(rows)<2 or any(a>=b for a,b in zip(self.times,self.times[1:])):
            raise ValueError('strictly increasing multi-sample pose timeline required')

    def at(self,target,max_bracket_ns=5000000,include_translation=False):
        j=bisect.bisect_left(self.times,target)
        indices=[j] if j<len(self.times) and self.times[j]==target else [j-1,j]
        if min(indices)<0 or max(indices)>=len(self.times):raise ValueError('target outside pose timeline')
        if self.times[indices[-1]]-self.times[indices[0]]>max_bracket_ns:raise ValueError('pose bracket too wide')
        rotations=[];translations=[]
        for i in indices:
            row=self.rows[i]
            if row['valid']!='1' or row['result']!='200':raise ValueError('invalid or degraded bracket pose')
            r=np.array([[float(row[f'm{a}{b}']) for b in range(3)] for a in range(3)])
            if not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(r),1.,atol=1e-4):raise ValueError('nonrigid bracket pose')
            u,_,v=np.linalg.svd(r);rotations.append(u@v)
            if include_translation:
                t=np.array([float(row[f'm{a}3']) for a in range(3)])
                if not np.isfinite(t).all():raise ValueError('nonfinite bracket position')
                translations.append(t)
        if len(indices)==1:return (rotations[0],translations[0],0) if include_translation else (rotations[0],0)
        t=(target-self.times[indices[0]])/(self.times[indices[1]]-self.times[indices[0]])
        delta=cv2.Rodrigues(rotations[0].T@rotations[1])[0].ravel()
        r=rotations[0]@cv2.Rodrigues(t*delta)[0];gap=self.times[indices[1]]-self.times[indices[0]]
        return (r,(1-t)*translations[0]+t*translations[1],gap) if include_translation else (r,gap)


def projection_effect(relative_rotation,intrinsic=K,translation_drift_m=None):
    rays=np.array([[x,y,1.] for x in (-.5,0.,.5) for y in (-.5,0.,.5)])
    if translation_drift_m is not None:
        rays=np.concatenate([rays*z for z in (.7,1.5,2.5)])
    else:translation_drift_m=np.zeros(3)
    def project(p):
        q=p@intrinsic.T
        if np.any(q[:,2]<=0):raise ValueError('probe behind camera')
        return q[:,:2]/q[:,2,None]
    difference=project((rays-np.asarray(translation_drift_m))@relative_rotation)-project(rays)
    return {'maximum_absolute_xy_px':np.max(np.abs(difference),axis=0).tolist(),
            'rms_xy_px':np.sqrt(np.mean(difference*difference,axis=0)).tolist()}


def run(windows,poses,config,exposure_offset_ms):
    with Path(poses).open(newline='') as stream:timeline=RotationTimeline(list(csv.DictReader(stream)))
    device=json.loads(Path(config).read_text());entries=[]
    for manifest_path in windows:
        manifest=json.loads(Path(manifest_path).read_text());calibration=json.loads(Path(manifest['calibration']).read_text())
        extrinsic=rectified_in_head(calibration,device)
        right_camera=next(c for c in calibration['cameras'] if c['sourceCamera']=='slam_right')
        if right_camera['extrinsics'].get('__units')!='mm':raise ValueError('explicit millimetre camera position required')
        cad=device['cv']['cad_from_cal'];head=device['head']
        right_origin=basis(head).T@(basis(cad)@(np.array(right_camera['extrinsics']['position'],float)/1000.)+np.array(cad['position'])-np.array(head['position']))
        pairs={r['left_file']:r for r in (json.loads(l) for l in Path(manifest['pairs']).read_text().splitlines() if l.strip())}
        frames=[]
        for frame in manifest['frames']:
            pair=pairs[Path(frame['left']).name]
            if pair['right_file']!=Path(frame['right']).name:raise ValueError('stereo file mismatch')
            skew=int(pair['right_raw_ns'])-int(pair['left_raw_ns'])
            left_time=int(pair['left_monotonic_ns'])-round(exposure_offset_ms*1e6)
            entry={'ordinal':frame['ordinal'],'source_pair_id':frame['source_pair_id'],'right_minus_left_raw_ns':skew}
            try:
                left,lp,lg=timeline.at(left_time,include_translation=True);right,rp,rg=timeline.at(left_time+skew,include_translation=True)
                relative=extrinsic.T@left.T@right@extrinsic
                head_drift=(rp-lp)@(left@extrinsic)
                lever_drift=((right-left)@right_origin)@(left@extrinsic)
                total_drift=head_drift+lever_drift
                entry.update(available=True,maximum_pose_bracket_ns=max(lg,rg),
                    rotation_vector_rad=cv2.Rodrigues(relative)[0].ravel().tolist(),projection=projection_effect(relative),
                    logged_head_translation_drift_m=head_drift.tolist(),right_lever_translation_drift_m=lever_drift.tolist(),
                    rotation_and_logged_translation=projection_effect(relative,translation_drift_m=total_drift))
            except ValueError as error:entry.update(available=False,reason=str(error))
            frames.append(entry)
        effects=[e['projection']['maximum_absolute_xy_px'] for e in frames if e['available']]
        entries.append({'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),'pairs_sha256':sha(manifest['pairs']),
            'frames':frames,'available_frames':len(effects),'maximum_rotation_and_logged_translation_probe_shift_xy_px':np.max([e['rotation_and_logged_translation']['maximum_absolute_xy_px'] for e in frames if e['available']],axis=0).tolist() if effects else None,'maximum_absolute_probe_shift_xy_px':np.max(effects,axis=0).tolist() if effects else None})
    return {'source_sha256':sha(__file__),'poses_sha256':sha(poses),'device_config_sha256':sha(config),
        'exposure_offset_ms':exposure_offset_ms,'pose_bracket_limit_ns':5000000,'windows':entries,
        'scope':__doc__,'metric_acceptance':'unevaluated'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('poses','config','output'):p.add_argument(n)
    p.add_argument('--window',action='append',required=True);p.add_argument('--exposure-offset-ms',type=float,required=True)
    a=p.parse_args();r=run(a.window,a.poses,a.config,a.exposure_offset_ms)
    Path(a.output).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
