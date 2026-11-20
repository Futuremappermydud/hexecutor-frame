#!/usr/bin/env python3
"""Offline shared calibration and leave-one-window-out tests on long trajectories.

Training support, robust cleanup, physical models and original minima are held
fixed. No tags participate in window selection or optimization. Camera motion
remains metric-scale-free apart from the explicitly supplied stereo baseline.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from evaluate_stereo_temporal_lens import prepare_window
from evaluate_stereo_long_trajectory import BUDGET as SINGLE_BUDGET
from evaluate_stereo_grouped_lens import stereo_frame_identities, probe_stability
from evaluate_stereo_temporal_calibration import K, sha, select_external_tags
from evaluate_stereo_lens_calibration import tag_score
from stereo_temporal_calibration import fit_temporal_calibration
from stereo_support_refit import refit_physical_support

BUDGET = {'max_frames':150, 'max_tracks':2000, 'max_observations':45000}


def retained_problem(rows, rotations, report):
    if not report.get('attempted') or not report.get('refit_training_accepted'):
        return rows, rotations
    selected=rotations[report['retained_original_frame_indices']]
    return rows[report['retained_input_observation_indices']], selected[0].T @ selected


def run(windows, poses, config, offset, external_tag_manifest=None,right_principal_point=False,principal_point_bound_px=2.,anchor_refinement=False,global_rotation_initialization=False,grouped_support_cleanup=False,interleave_after=False,interpolate_poses=False,free_rotations=False,left_focal_scale=False,rotation_initialization_only=False,refine_head_rotation=False):
    if not isinstance(refine_head_rotation,bool) or (refine_head_rotation and (free_rotations or rotation_initialization_only)):
        raise ValueError('head-axis refinement requires constrained recorded rotations')
    if not isinstance(left_focal_scale,bool) or (left_focal_scale and right_principal_point):
        raise ValueError('choose left focal or right principal calibration, not both')
    if not isinstance(rotation_initialization_only,bool) or (rotation_initialization_only and not (free_rotations and global_rotation_initialization)):
        raise ValueError('rotation seeds require free rotations and global rotation initialization')
    if not isinstance(free_rotations, bool) or (free_rotations and global_rotation_initialization and not rotation_initialization_only):
        raise ValueError('free rotations require PnP initialization')
    if not 3 <= len(windows) <= 5:
        raise ValueError('three to five explicit independent long windows required')
    single_budget=dict(SINGLE_BUDGET,max_physical_principal_shift_px=principal_point_bound_px)
    if interleave_after:single_budget.update(max_frames=60,max_observations=18000)
    shared_budget=dict(BUDGET,max_physical_principal_shift_px=principal_point_bound_px)
    if interleave_after:shared_budget.update(max_frames=300,max_observations=90000)
    models=('physical6','physical_focal7') if left_focal_scale else ('physical6','physical_center8') if right_principal_point else ('physical4','physical6')
    entries=[];problems={};seen=set();cameras=None;baseline=None
    for index,(manifest,frozen) in enumerate(windows):
        data,rows,held,rot,current,_,_=prepare_window(manifest,frozen,poses,config,offset,
            prune_support=True,maximum_frames=60 if interleave_after else 30,training_track_limit=400,anchor_refinement=anchor_refinement,interpolate_poses=interpolate_poses)
        if interleave_after:
            meta=json.loads(Path(manifest).read_text())
            records={r['pair']:r for r in (json.loads(line) for line in Path(meta['pairs']).read_text().splitlines() if line.strip())}
            roles=[records[f['source_pair_id']]['role'] for f in meta['frames']]
            if roles.count('reference')!=30 or roles.count('after') not in (29,30) or set(roles)!={'reference','after'}:
                raise ValueError('exactly 30 reference and 29/30 after captures required, no lit images')
        elif len(data['ordinals']) != 30:raise ValueError('exactly 30 references per window required')
        identities=stereo_frame_identities(json.loads(Path(manifest).read_text()),data['observed_provenance'])
        if seen & identities:raise ValueError('windows reuse raw stereo frames')
        seen |= identities
        if cameras is None:cameras=current;baseline=data['baseline_m']
        if not np.isclose(baseline,data['baseline_m'],atol=1e-9,rtol=0):raise ValueError('baseline differs')
        for a,b in zip(cameras,current):
            if any(not np.allclose(getattr(a,n),getattr(b,n),atol=1e-9,rtol=0) for n in ('intrinsic','rotation','lens')):
                raise ValueError('nominal physical cameras differ')
        fits={};cleanups={}
        for model in models:
            fit=fit_temporal_calibration(rows,K,K,baseline,config=single_budget,model=model,camera_rotations=rot if not free_rotations or rotation_initialization_only else None,rotation_initialization_only=rotation_initialization_only,camera_ray_models=cameras,global_rotation_initialization=global_rotation_initialization,refine_camera_rotation=refine_head_rotation)
            fit,cleanup=refit_physical_support(rows,rot,cameras,baseline,fit,single_budget,use_recorded_rotations=not free_rotations,rotation_initialization_only=rotation_initialization_only)
            problems[index,model]=retained_problem(rows,rot,cleanup)
            fits[model]=fit;cleanups[model]=cleanup
        entries.append({'manifest':str(manifest),'frozen':str(frozen),'prepared':data,'fits':fits,'cleanup':cleanups})
        print('prepared',index,flush=True)
    eligible=[i for i,e in enumerate(entries) if all(f['accepted'] for f in e['fits'].values())]
    result={'source_sha256':sha(__file__),'estimator_sha256':sha(Path(__file__).with_name('stereo_temporal_calibration.py')),
        'windows':entries,'eligible_window_indices':eligible,'budget':shared_budget,'models':{},'metric_acceptance':'unevaluated',
        'selection':'Intersection of the two declared physical model standalone training acceptances after explicit support cleanup; no external tag inputs.',
        'head_rotation_refinement_enabled':refine_head_rotation,'rotation_initialization_only':rotation_initialization_only,'camera_rotation_mode':'shared_axis' if refine_head_rotation else 'free_seeded' if rotation_initialization_only else 'free' if free_rotations else 'recorded','interleaved_after':interleave_after,'unique_raw_stereo_hash_pairs':len(seen)}
    if len(eligible)<3:result['reason']='insufficient_training_accepted_windows';return result
    tags=[];tag_size=.12
    if external_tag_manifest:
        excluded=set()
        for i in eligible:excluded.update(f['source_pair_id'] for f in json.loads(Path(windows[i][0]).read_text())['frames'])
        tags,provenance=select_external_tags(external_tag_manifest,excluded)
        external=json.loads(Path(external_tag_manifest).read_text());tag_size=external['tag_size_m']
        if sha(external['calibration']) != entries[0]['prepared']['observed_provenance']['calibration_sha256']:raise ValueError('tag calibration differs')
        result['external_tag_provenance']=provenance
    for model in models:
        def fit_indices(indices):
            rows=[];rotations=[];groups=[]
            for i in indices:
                obs,rot=problems[i,model];obs=obs.copy()
                if np.max(obs[:,1])>=1000000:raise ValueError('track ID range exceeded')
                obs[:,0]+=i*100;obs[:,1]+=i*1000000
                rows.append(obs);rotations.extend(rot);groups.extend([i]*len(rot))
            fit=fit_temporal_calibration(np.concatenate(rows),K,K,baseline,config=shared_budget,model=model,
                camera_rotations=np.asarray(rotations) if not free_rotations or rotation_initialization_only else None,rotation_initialization_only=rotation_initialization_only,camera_ray_models=cameras,frame_groups=groups,global_rotation_initialization=global_rotation_initialization,refine_camera_rotation=refine_head_rotation)
            if grouped_support_cleanup:
                from stereo_support_refit import refit_grouped_physical_support
                fit,report=refit_grouped_physical_support(np.concatenate(rows),np.asarray(rotations),groups,cameras,baseline,fit,shared_budget,global_initialization=global_rotation_initialization,use_recorded_rotations=not free_rotations,rotation_initialization_only=rotation_initialization_only)
                fit=dict(fit,grouped_support_cleanup=report)
            return fit
        fit=fit_indices(eligible);print('shared',model,fit['accepted'],flush=True)
        omitted=[]
        for i in eligible:
            omitted.append({'omitted_index':i,'fit':fit_indices([j for j in eligible if i!=j])})
            print('omitted',model,i,flush=True)
        entry={'fit':fit,'leave_one_window_out':omitted,
            'training_consistency':probe_stability([fit,*[e['fit'] for e in omitted]],*cameras,baseline)}
        if fit['accepted']:entry['external_tags']=tag_score(tags,*cameras,fit['parameters_vector'],baseline,tag_size)
        result['models'][model]=entry
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('poses','config','output'):p.add_argument(name)
    p.add_argument('--window',nargs=2,action='append',required=True)
    p.add_argument('--exposure-offset-ms',type=float,required=True)
    p.add_argument('--left-focal-scale',action='store_true');p.add_argument('--external-tag-manifest');p.add_argument('--right-principal-point',action='store_true');p.add_argument('--principal-point-bound-px',type=float,default=2.)
    p.add_argument('--anchor-refinement',action='store_true');p.add_argument('--global-rotation-initialization',action='store_true');p.add_argument('--grouped-support-cleanup',action='store_true');p.add_argument('--interleave-after',action='store_true')
    p.add_argument('--refine-head-rotation',action='store_true')
    p.add_argument('--free-rotations',action='store_true');p.add_argument('--rotation-initialization-only',action='store_true')
    a=p.parse_args();result=run(a.window,a.poses,a.config,a.exposure_offset_ms,a.external_tag_manifest,a.right_principal_point,a.principal_point_bound_px,a.anchor_refinement,a.global_rotation_initialization,a.grouped_support_cleanup,a.interleave_after,left_focal_scale=a.left_focal_scale,free_rotations=a.free_rotations,rotation_initialization_only=a.rotation_initialization_only,refine_head_rotation=a.refine_head_rotation)
    destination=Path(a.output);temporary=destination.with_name('.'+destination.name+'.partial-'+str(os.getpid()))
    try:
        with temporary.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():temporary.unlink()
if __name__=='__main__':main()
