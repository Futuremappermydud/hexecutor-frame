#!/usr/bin/env python3
"""Post-screen attribution on saved seed-1709 silhouette stages; no new matching."""
import argparse
from array import array
from pathlib import Path
from stereo_composite_common import digest,rows,write_json


def floats(path):
    values=array('f');values.frombytes(path.read_bytes());return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    root=p.parse_args().root
    result={};hashes={}
    for guard in range(4):
        for exposure in [0,1]:
            key=f'g{guard}-e{exposure}';result[key]={}
            metrics=rows(root/key/'synthetic-1709/metrics.csv')
            for scene,width in [('step-silhouette-fractional',64),('thin-silhouette',8)]:
                folder=root/key/'synthetic-1709'/scene
                evaluated_path=folder/'truth-evaluated.pgm'
                evaluated=evaluated_path.read_bytes().split(b'\n',3)[3]
                result[key][scene]={}
                for arm in ['integer','parabola']:
                    name='guarded-adaptive-'+arm
                    paths={field:folder/f'{name}-{field}.f32' for field in
                           ['depth','left-anchors','left-information','left-support-bits','left-anchor-stops']}
                    data={field:floats(path) for field,path in paths.items()}
                    for path in list(paths.values())+[evaluated_path]:hashes[str(path.relative_to(root))]=digest(path)
                    boundary=[];leaked=[]
                    for i,z in enumerate(data['depth']):
                        x,y=i%256,i//256
                        foreground=112<=x<112+width and 36<=y<124
                        near_vertical=28<=y<132 and (abs(x-112)<=8 or abs(x-(112+width))<=8)
                        near_horizontal=104<=x<120+width and (abs(y-36)<=8 or abs(y-124)<=8)
                        if evaluated[i] and not foreground and (near_vertical or near_horizontal):
                            boundary.append(i)
                            if z>0 and 24.282608/z>13.375:leaked.append(i)
                    measured=100*len(leaked)/len(boundary)
                    original=next(r for r in metrics if r['scene']==scene and r['method']==name)
                    if abs(measured-float(original['background_boundary_leakage_of_all_pct']))>1e-7:
                        raise ValueError('Diagnostic leakage count differs from saved metric')
                    anchors=data['left-anchors']
                    result[key][scene][arm]=dict(evaluated_boundary_pixels=len(boundary),leaked_pixels=len(leaked),
                        leakage_pct=measured,leaks_without_left_anchor=sum(anchors[i]==0 for i in leaked),
                        leaks_with_incorrect_left_anchor=sum(anchors[i]>0 and abs(anchors[i]-8.25)>1 for i in leaked),
                        leaks_with_anchor_arm_stop=sum(data['left-anchor-stops'][i]>0 for i in leaked),
                        leaks_with_censored_descriptor_bits=sum(data['left-support-bits'][i]<48*data['left-information'][i] for i in leaked),
                        boundary_pixels_with_left_anchor=sum(anchors[i]>0 for i in boundary))
    write_json(root/'boundary-diagnostics.json',dict(scope='post-hoc seed-1709 attribution only, not selection or held-out evidence',
               source_sha256=digest(Path(__file__)),input_sha256=hashes,configurations=result))
    print('Saved boundary attribution; leakage counts reproduce saved metrics')


if __name__=='__main__':main()
