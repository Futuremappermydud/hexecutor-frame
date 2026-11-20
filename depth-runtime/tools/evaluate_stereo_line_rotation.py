#!/usr/bin/env python3
"""Offline line-direction rotation evaluation on explicit frozen saved images."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from evaluate_stereo_temporal_calibration import extract, K, sha, select_external_tags
from evaluate_stereo_lens_calibration import camera_models, tag_score
from stereo_pose_constraints import recorded_rotations
from stereo_line_rotation import extract_lines, fit_line_rotation


def run(manifest_path,frozen_path,poses,config,offset,external_tag_manifest=None,physical_lens=False,independent_eye_consensus=False):
    cv2.setNumThreads(1)
    manifest=json.loads(Path(manifest_path).read_text())
    frames,observed=extract(manifest_path,json.loads(Path(frozen_path).read_text()))
    if any(f['split']!='train' for f in manifest['frames']):raise ValueError('training images only')
    ordinals=sorted(frames)
    if not 5<=len(ordinals)<=30:raise ValueError('five to thirty explicit frames required')
    rotations,provenance=recorded_rotations(manifest_path,poses,config,offset)
    rotations=np.array([rotations[i] for i in ordinals]);rotations=rotations[0].T@rotations
    normals=[];eyes=[];frame_ids=[];weights=[];counts=[];endpoints=[]
    for index,ordinal in enumerate(ordinals):
        count=[]
        for eye in (0,1):
            lines,n,w=extract_lines(frames[ordinal]['images'][eye],frames[ordinal]['masks'][eye],K)
            endpoints.extend(lines);normals.extend(n);eyes.extend([eye]*len(n));frame_ids.extend([index]*len(n));weights.extend(w);count.append(len(n))
        counts.append(count)
    left,right,baseline=camera_models(manifest,observed)
    fit=fit_line_rotation(np.asarray(normals).reshape(-1,3),eyes,frame_ids,rotations,weights,
        np.asarray(endpoints).reshape(-1,4) if physical_lens else None, (left,right) if physical_lens else None, independent_eye_consensus)
    result={'source_sha256':sha(__file__),'estimator_sha256':sha(Path(__file__).with_name('stereo_line_rotation.py')),
        'observed_provenance':observed,'pose_provenance':provenance,'line_counts_by_frame_eye':counts,'fit':fit,
        'metric_acceptance':'unevaluated','selection':'All mask-safe LSD segments >=40 px; fixed support settings, no tags.'}
    left,right,baseline=camera_models(manifest,observed)
    if external_tag_manifest:
        tags,tag_provenance=select_external_tags(external_tag_manifest,{f['source_pair_id'] for f in manifest['frames']})
        external=json.loads(Path(external_tag_manifest).read_text())
        if sha(external['calibration'])!=observed['calibration_sha256']:raise ValueError('external lens snapshot differs')
        result['external_tag_provenance']=tag_provenance
        result['uncorrected_external_tags']=tag_score(tags,left,right,np.zeros(6),baseline,external['tag_size_m'])
        if fit['accepted']:
            result['external_tags']=tag_score(tags,left,right,fit.get('parameters_vector',np.r_[fit['rotation_vector_rad'],0.,0.,0.]),baseline,external['tag_size_m'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','frozen','poses','config','output'):p.add_argument(name)
    p.add_argument('--exposure-offset-ms',type=float,required=True);p.add_argument('--external-tag-manifest');p.add_argument('--physical-lens',action='store_true');p.add_argument('--independent-eye-consensus',action='store_true')
    a=p.parse_args();r=run(a.manifest,a.frozen,a.poses,a.config,a.exposure_offset_ms,a.external_tag_manifest,a.physical_lens,a.independent_eye_consensus)
    Path(a.output).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
