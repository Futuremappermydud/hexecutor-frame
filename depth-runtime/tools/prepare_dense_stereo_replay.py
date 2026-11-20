#!/usr/bin/env python3
"""Prepare a read-only dense replay and missing-file transfer plan.

No network access is performed. Run `verify` after a future transfer to record
local hashes and detect size errors; unknown remote hashes cannot be validated.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import shlex
import sys

ANCHORS = (0, 10, 30, 35, 40)
WINDOW_SIZE = 12
RAW_SIZE = 1056 * 1024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_name(name):
    if not isinstance(name, str) or not name or '\x00' in name or '\n' in name or '\r' in name:
        raise ValueError('invalid source filename')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
        raise ValueError('unsafe source filename')
    p = PurePosixPath(name)
    if p.is_absolute() or len(p.parts) != 1 or p.name != name or name in ('.', '..'):
        raise ValueError('source filenames must be single relative path components')
    return name


def index_pairs(rows):
    by_id, names = {}, set()
    for row in rows:
        if row.get('role') not in ('reference', 'lit', 'after'):
            raise ValueError('pair role missing or unsupported')
        pair = row.get('pair')
        if isinstance(pair, bool) or not isinstance(pair, int) or pair in by_id:
            raise ValueError('duplicate or invalid pair id')
        for side in ('left', 'right'):
            name = safe_name(row.get(side + '_file'))
            if name in names:
                raise ValueError('duplicate image filename')
            names.add(name)
            t = row.get(side + '_raw_ns')
            if isinstance(t, bool) or not isinstance(t, int):
                raise ValueError('missing or invalid raw timestamp')
        by_id[pair] = row
    return by_id


def select_windows(manifest, rows, window_size=WINDOW_SIZE, include_after=False):
    """Find frozen anchor IDs in ordered reference records, stop before holdout."""
    if isinstance(window_size, bool) or not isinstance(window_size, int) or not 3 <= window_size <= 60:
        raise ValueError('window size must be an integer between 3 and 60')
    if not isinstance(include_after,bool) or (include_after and window_size>30):
        raise ValueError('interleaved off-phase windows require at most 30 references')
    index_pairs(rows)
    references = [r for r in rows if r['role'] == 'reference']
    if any(a['left_raw_ns'] >= b['left_raw_ns'] for a, b in zip(references, references[1:])):
        raise ValueError('reference timestamps must increase strictly')
    by_left = {r['left_file']: i for i, r in enumerate(references)}
    frames = manifest['frames']
    if len({f['ordinal'] for f in frames}) != len(frames):
        raise ValueError('duplicate v2 ordinals')
    frames = {f['ordinal']: f for f in frames}
    for ordinal in (*ANCHORS, 15, 45):
        if ordinal not in frames:
            raise ValueError('missing required frozen ordinal')
        expected = 'heldout' if ordinal in (15, 45) else 'train'
        if frames[ordinal]['split'] != expected:
            raise ValueError('frozen split does not match expected training quarters')
    def row_index(ordinal):
        f = frames[ordinal]
        left, right = Path(f['left']).name, Path(f['right']).name
        if left not in by_left:
            raise ValueError('frozen frame does not match reference pair')
        i = by_left[left]
        if references[i]['right_file'] != right:
            raise ValueError('frozen left/right files do not match pair')
        return i
    cutoffs = {0: row_index(15), 30: row_index(45)}
    if not (row_index(0) < cutoffs[0] <= row_index(30) < cutoffs[30]):
        raise ValueError('frozen training boundaries are inconsistent')
    windows = []
    for ordinal in ANCHORS:
        start = row_index(ordinal)
        q = 0 if ordinal < 15 else 30
        selection = references[start:start + window_size]
        if len(selection) != window_size:
            raise ValueError(f'anchor {ordinal}: insufficient reference pairs')
        if start < row_index(q) or start + window_size > cutoffs[q]:
            raise ValueError(f'anchor {ordinal}: selection crosses held-out boundary')
        missing_after=[]
        if include_after:
            after={}
            for row in rows:
                if row['role']=='after':
                    if row.get('cycle') in after:raise ValueError('duplicate after cycle')
                    after[row.get('cycle')]=row
            combined=[]
            for ref in selection:
                cycle=ref.get('cycle')
                if isinstance(cycle,bool) or not isinstance(cycle,int):raise ValueError('explicit reference cycle required')
                combined.append(ref)
                if cycle not in after:missing_after.append(ref['pair']);continue
                following=after[cycle]
                if any(following[side+'_raw_ns']<=ref[side+'_raw_ns'] for side in ('left','right')):
                    raise ValueError('after timestamp does not follow reference')
                combined.append(following)
            if any(a[side+'_raw_ns']>=b[side+'_raw_ns'] for side in ('left','right') for a,b in zip(combined,combined[1:])):
                raise ValueError('interleaved timestamps must increase in both eyes')
            boundary=references[cutoffs[q]]
            if any(row[side+'_raw_ns']>=boundary[side+'_raw_ns'] for row in combined for side in ('left','right')):
                raise ValueError('after frame crosses held-out boundary')
            selection=combined
        windows.append({'anchor_v2_ordinal': ordinal, 'training_quarter_start_ordinal': q,
                        'heldout_boundary_ordinal': 15 if q == 0 else 45,
                        'heldout_boundary_left_raw_ns': references[cutoffs[q]]['left_raw_ns'],
                        'pairs': selection, **({'missing_after_reference_pair_ids':missing_after} if include_after else {})})
    return windows


def write_replay_script(plan_path, windows, original_path):
    """Create a reviewable offline runner without materializing data manifests."""
    plan_path = Path(plan_path).resolve()
    out = plan_path.parent / 'materialized'
    out.mkdir(parents=True, exist_ok=True)
    commands, results = [], []
    long_mode = any(len(w.get('frames', w.get('pairs', []))) > 12 for w in windows)
    for window in windows:
        anchor = window['anchor_v2_ordinal']
        manifest = out / f'window-{anchor}.json'
        baseline = out / f'window-{anchor}-baseline.json'
        result = out / f'window-{anchor}-temporal.json'
        commands.append(['python3', 'depth-runtime/tools/prepare_dense_stereo_replay.py', 'check-replay', str(manifest)])
        commands.append(['python3', 'depth-runtime/tools/evaluate_stereo_self_calibration.py', '--matching', 'lk', str(manifest), str(baseline)])
        if not long_mode:
            commands.append(['python3', 'depth-runtime/tools/evaluate_stereo_temporal_calibration.py', str(manifest), str(baseline), str(result), '--start', '0', '--window-length', '12', '--external-tag-manifest', original_path])
        results.append(str(result))
    if not long_mode:
        commands.append(['python3', 'depth-runtime/tools/prepare_dense_stereo_replay.py', 'aggregate', str(plan_path), '--output', str(out / 'aggregate-stability.json'), *sum((['--result', r] for r in results), [])])
    runner = out / 'run-dense-replay.sh'
    runner.write_text('#!/bin/sh\nset -eu\n' + '\n'.join(' '.join(map(shlex.quote, c)) for c in commands) + '\n')
    if long_mode:
        with runner.open('a') as stream:
            stream.write('# Long sequences are baseline-matched only here. Run evaluate_stereo_long_trajectory.py with explicit recorded pose/config paths for geometry.\n')
    runner.chmod(0o755)
    return runner


def build(manifest_path, output_dir, remote_root, session_path=None, v2_result_path=None,
          window_size=WINDOW_SIZE, local_source_dirs=(), include_after=False):
    manifest_path, out = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if not isinstance(remote_root, str) or ':' not in remote_root or any(c in remote_root for c in '\n\r\x00'):
        raise ValueError('remote root must be explicit HOST:relative-path or HOST:/absolute-path')
    host, remote_path = remote_root.split(':', 1)
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*', host) or
            not re.fullmatch(r'[A-Za-z0-9_./-]+', remote_path) or '..' in PurePosixPath(remote_path).parts):
        raise ValueError('invalid remote root')
    remote_root = remote_root.rstrip('/') + '/'
    data = json.loads(manifest_path.read_text())
    v2 = json.loads(Path(v2_result_path).read_text()) if v2_result_path else None
    if v2 is not None and sha(manifest_path) != v2['manifest_sha256']:
        raise ValueError('frozen v2 manifest hash mismatch')
    pairs_path = Path(data['pairs']).resolve()
    rows = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    windows = select_windows(data, rows, window_size, include_after)
    if v2 is not None:
        for field, key in (('pairs', 'pairs_sha256'), ('rectifier_source', 'rectifier_sha256'),
                           ('calibration', 'calibration_sha256'), ('observations', 'observations_sha256')):
            if sha(data[field]) != v2[key]:
                raise ValueError('frozen v2 source mismatch: ' + field)
    out.mkdir(parents=True, exist_ok=True)
    target_dir = out / 'raw'
    existing = {}
    original_by_name = {}
    references_by_left = {r['left_file']: r for r in rows if r['role'] == 'reference'}
    for f in data['frames']:
        left, right = Path(f['left']).name, Path(f['right']).name
        if left not in references_by_left or references_by_left[left]['right_file'] != right:
            raise ValueError('v2 frame does not identify one reference pair')
        for side in ('left', 'right'):
            p = Path(f[side]).resolve()
            name = safe_name(p.name)
            original_by_name[name] = f['ordinal']
            if p.exists():
                if p.stat().st_size != RAW_SIZE:
                    raise ValueError('existing raw file has unexpected size: ' + str(p))
                existing[name] = p
    local_source_dirs = [Path(directory).resolve() for directory in local_source_dirs]
    expected_by_name = {}
    if v2 is not None:
        for path, digest in v2['input_sha256'].items():
            name = Path(path).name
            if name in expected_by_name and expected_by_name[name] != digest:
                raise ValueError('ambiguous frozen input basename')
            expected_by_name[name] = digest
    missing = set()
    all_files = {}
    for window in windows:
        result_frames = []
        for dense_ordinal, row in enumerate(window.pop('pairs')):
            frame = {'dense_ordinal': dense_ordinal, 'source_pair_id': row['pair'],
                     'original_v2_ordinal': original_by_name.get(row['left_file']),
                     'role': row['role'], 'capture_metadata': dict(row), 'left_raw_ns': row['left_raw_ns'],
                     'right_raw_ns': row['right_raw_ns'],
                     'absolute_skew_us': abs(row['left_raw_ns']-row['right_raw_ns'])/1000,
                     'left_seq': row.get('left_seq'), 'right_seq': row.get('right_seq')}
            for side in ('left', 'right'):
                name = safe_name(row[side+'_file'])
                source = existing.get(name)
                if source is None:
                    source = next((directory/name for directory in local_source_dirs if (directory/name).is_file()), None)
                    if source is not None and source.stat().st_size != RAW_SIZE:
                        raise ValueError('local raw file has unexpected size: '+str(source))
                path = source if source else target_dir / name
                if source is None:
                    missing.add(name)
                if name not in all_files:
                    if source is not None and name in expected_by_name and sha(source) != expected_by_name[name]:
                        raise ValueError('existing selected frame differs from frozen v2 hash: ' + str(source))
                    all_files[name] = {'local_path': str(path), 'remote_path': remote_root + name,
                                       'expected_size_bytes': RAW_SIZE,
                                       'present_at_plan_time': source is not None,
                                       'known_local_sha256': sha(source) if source else None,
                                       'remote_sha256': None}
                frame[side] = str(path)
            result_frames.append(frame)
        window['frames'] = result_frames
    src = {'frozen_manifest': {'path': str(manifest_path), 'sha256': sha(manifest_path)},
           'pair_metadata': {'path': str(pairs_path), 'sha256': sha(pairs_path)}}
    for key in ('rectifier_source', 'calibration', 'observations'):
        p = Path(data[key]).resolve()
        src[key] = {'path': str(p), 'sha256': sha(p)}
    if v2_result_path is not None:
        v2_result_path = Path(v2_result_path).resolve()
        src['v2_result'] = {'path': str(v2_result_path), 'sha256': sha(v2_result_path)}
    if session_path is not None:
        session_path = Path(session_path).resolve()
        src['session_metadata'] = {'path': str(session_path), 'sha256': sha(session_path)}
    files_from = out / 'missing-files.txt'
    files_from.write_text(''.join(name + '\n' for name in sorted(missing)))
    plan = {'version': 1, 'generator_sha256': sha(__file__), 'sources': src,
            'window_size': window_size, 'include_after':include_after, 'local_source_dirs': list(map(str, local_source_dirs)),
            'remote_root': remote_root, 'transfer_destination': str(target_dir),
            'files_from': str(files_from), 'files_from_sha256': sha(files_from),
            'raw_frame_expected_size_bytes': RAW_SIZE, 'windows': windows,
            'files': all_files, 'missing_file_count': len(missing),
            'missing_expected_bytes': len(missing) * RAW_SIZE,
            'heldout_tag_evaluation': {'external_v2_manifest': str(manifest_path),
                'external_observations': str(Path(data['observations']).resolve()),
                'mapping': 'Observation ordinal always refers to the original v2 manifest, never to a dense ordinal.',
                'heldout_v2_ordinals': [f['ordinal'] for f in data['frames'] if f['split'] == 'heldout']},
            'caveats': ['Remote path and file availability are not checked while device offline.',
                        'Missing files have no verified remote hash; byte length alone is not end-to-end integrity.',
                        'reference role is capture metadata and does not independently prove no illumination.',
                        'No sensor, hardware or network action is performed by this planner.']}
    output = out / 'plan.json'
    output.write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    write_replay_script(output, windows, str(manifest_path))
    # Scripts are explicit opt-in artifacts; generating them never invokes rsync.
    args = ' '.join(map(shlex.quote, ['rsync', '--ignore-existing', '--files-from='+str(files_from), remote_root, str(target_dir)+'/']))
    dryargs = ' '.join(map(shlex.quote, ['rsync', '--dry-run', '--itemize-changes', '--ignore-existing', '--files-from='+str(files_from), remote_root, str(target_dir)+'/']))
    for name, cmd in (('transfer-dry-run.sh', dryargs), ('transfer.sh', args)):
        path = out / name
        path.write_text('#!/bin/sh\nset -eu\nmkdir -p ' + shlex.quote(str(target_dir)) + '\n' + cmd + '\n')
        path.chmod(0o755)
    return output


def verify(plan_path):
    p = Path(plan_path).resolve()
    plan = json.loads(p.read_text())
    source_failures = []
    if sha(__file__) != plan['generator_sha256']:
        source_failures.append('generator')
    for name, record in plan['sources'].items():
        try:
            if sha(record['path']) != record['sha256']:
                source_failures.append(name)
        except OSError:
            source_failures.append(name)
    if sha(plan['files_from']) != plan['files_from_sha256']:
        source_failures.append('files_from')
    results, missing, wrong_size, changed = {}, 0, 0, 0
    for name, info in plan['files'].items():
        safe_name(name)
        path = Path(info['local_path'])
        item = {'local_path': str(path), 'remote_sha256': None}
        if not path.is_file():
            missing += 1; item['status'] = 'missing'
        elif path.stat().st_size != info['expected_size_bytes']:
            wrong_size += 1; item['status'] = 'wrong_size'; item['size_bytes'] = path.stat().st_size
        else:
            actual = sha(path); item['sha256'] = actual; item['size_bytes'] = path.stat().st_size
            if info['known_local_sha256'] and actual != info['known_local_sha256']:
                changed += 1; item['status'] = 'changed_from_known_local_hash'
            else:
                item['status'] = 'matched_known_hash' if info['known_local_sha256'] else 'size_ok_remote_hash_unknown'
        results[name] = item
    return {'plan_sha256': sha(p), 'source_failures': source_failures,
            'files': results, 'missing_count': missing, 'wrong_size_count': wrong_size,
            'changed_known_hash_count': changed, 'files_present_and_sources_unchanged': not (source_failures or missing or wrong_size or changed),
            'remote_content_verified': False,
            'note': 'Files without a known hash are not verified against remote content.'}


def materialize(plan_path):
    """After all files exist, make isolated v2-compatible manifests per window."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    check = verify(plan_path)
    if not check['files_present_and_sources_unchanged']:
        raise ValueError('cannot materialize: files missing/invalid or sources changed')
    original_path = plan['sources']['frozen_manifest']['path']
    original = json.loads(Path(original_path).read_text())
    original_obs = json.loads(Path(original['observations']).read_text())
    out = plan_path.parent / 'materialized'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'integrity.json').write_text(json.dumps(check, indent=2) + '\n')
    tool_dir = Path(__file__).resolve().parent
    replay_sources = {key: {'path': str(tool_dir / name), 'sha256': sha(tool_dir / name)} for key, name in (
        ('source_sha256', 'evaluate_stereo_temporal_calibration.py'),
        ('temporal_sha256', 'stereo_temporal_calibration.py'),
        ('evaluator_sha256', 'evaluate_stereo_self_calibration.py'),
        ('estimator_sha256', 'stereo_self_calibration.py'),
        ('validation_sha256', 'stereo_selfcal_validation.py'))}
    for window in plan['windows']:
        anchor = window['anchor_v2_ordinal']
        observations = []
        frames = []
        for f in window['frames']:
            ordinal = f['original_v2_ordinal']
            if ordinal is not None:
                for o in original_obs:
                    if o['ordinal'] == ordinal:
                        mapped = dict(o)
                        mapped['original_v2_ordinal'] = ordinal
                        mapped['source_pair_id'] = f['source_pair_id']
                        mapped['ordinal'] = f['dense_ordinal']
                        observations.append(mapped)
            left_name, right_name = Path(f['left']).name, Path(f['right']).name
            frames.append({'ordinal': f['dense_ordinal'], 'split': 'train',
                           'left': f['left'], 'right': f['right'],
                           'source_pair_id': f['source_pair_id'], 'original_v2_ordinal': ordinal,
                           'left_observed_sha256': check['files'][left_name]['sha256'],
                           'right_observed_sha256': check['files'][right_name]['sha256'],
                           'left_identity_status': check['files'][left_name]['status'],
                           'right_identity_status': check['files'][right_name]['status']})
        obs_path = out / f'window-{anchor}-observations.json'
        obs_path.write_text(json.dumps(observations, indent=2) + '\n')
        dense = {k: original[k] for k in ('rectifier_source', 'calibration', 'pairs', 'tag_size_m')}
        dense.update(observations=str(obs_path), observations_sha256=sha(obs_path), frames=frames,
                     selection=f"{len(window['frames'])} original reference/after off-phase pairs; one predeclared training window",
                     source_plan=str(plan_path), source_plan_sha256=sha(plan_path),
                     original_v2_manifest=original_path, original_v2_manifest_sha256=sha(original_path),
                     anchor_v2_ordinal=anchor, replay_sources=replay_sources,
                     identity_note='Observed local SHA-256 recorded; newly transferred files have no independently verified remote hash.')
        manifest = out / f'window-{anchor}.json'
        manifest.write_text(json.dumps(dense, indent=2) + '\n')
    write_replay_script(plan_path, plan['windows'], original_path)
    return out


def aggregate(plan_path, result_paths):
    from stereo_selfcal_validation import assess_update_stability
    import numpy as np
    plan = json.loads(Path(plan_path).read_text())
    out = Path(plan_path).resolve().parent / 'materialized'
    expected = {w['anchor_v2_ordinal'] for w in plan['windows']}
    by_hash = {sha(out / f'window-{anchor}.json'): anchor for anchor in expected}
    found = {}
    for path in result_paths:
        d = json.loads(Path(path).read_text())
        anchor = by_hash.get(d.get('manifest_sha256'))
        if anchor is None or anchor in found:
            raise ValueError('unexpected or duplicate dense result manifest')
        if len(d['windows']) != 1 or d['window_length'] != 12 or d['windows'][0]['start'] != 0:
            raise ValueError('result is not the required 12-frame window')
        dense = json.loads((out / f'window-{anchor}.json').read_text())
        expected_window = next(w for w in plan['windows'] if w['anchor_v2_ordinal'] == anchor)
        if (dense.get('source_plan_sha256') != sha(plan_path) or
                [f['source_pair_id'] for f in dense['frames']] != [f['source_pair_id'] for f in expected_window['frames']]):
            raise ValueError('dense manifest no longer agrees with source plan')
        for key, record in dense['replay_sources'].items():
            if d.get(key) != record['sha256'] or sha(record['path']) != record['sha256']:
                raise ValueError('replay source mismatch: ' + key)
        baseline_path = out / f'window-{anchor}-baseline.json'
        if sha(baseline_path) != d['v2_result_sha256']:
            raise ValueError('dense baseline hash differs from result')
        w = d['windows'][0]
        ext = w.get('external_tag_provenance')
        if 'fit' in w.get('temporal_relative4', {}) and not ext:
            raise ValueError('external heldout tag provenance missing for fitted window')
        if ext and (ext['manifest_sha256'] != plan['sources']['frozen_manifest']['sha256'] or
                    ext['observations_sha256'] != plan['sources']['observations']['sha256'] or
                    ext['pairs_sha256'] != plan['sources']['pair_metadata']['sha256']):
            raise ValueError('external tag source differs from frozen manifest')
        found[anchor] = d
    omitted = sorted(expected - set(found))
    failures = []
    fits = []
    for anchor, d in sorted(found.items()):
        w = d['windows'][0]
        fit = w.get('temporal_relative4', {}).get('fit', {})
        if fit.get('accepted'):
            fits.append(fit)
        else:
            failures.append(anchor)
    for key in ('source_sha256', 'temporal_sha256', 'estimator_sha256', 'evaluator_sha256', 'validation_sha256'):
        if len({d[key] for d in found.values()}) > 1:
            raise ValueError('results use mixed source hashes: ' + key)
    baseline_values = {d['baseline_m'] for d in found.values()}
    if len(baseline_values) > 1:
        raise ValueError('baseline differs across results')
    if len(fits) >= 2:
        K = np.array([[380.,0,308],[0,380.,308],[0,0,1]])
        guard = assess_update_stability(fits[0], fits[1:], K, K, next(iter(baseline_values)))
    else:
        guard = {'stability_passed': False, 'reasons': ['insufficient_independent_accepted_window_fits']}
    guard.update(source_plan_sha256=sha(plan_path), result_sha256={str(Path(p).resolve()): sha(p) for p in result_paths}, expected_anchors=sorted(expected), missing_result_anchors=omitted,
                 failed_or_rejected_anchors=failures, accepted_window_count=len(fits),
                 session_stability_passed=bool(guard['stability_passed'] and not omitted and not failures),
                 metric_acceptance='unevaluated')
    return guard


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    create = sub.add_parser('plan'); create.add_argument('manifest'); create.add_argument('output_dir'); create.add_argument('--remote-root', required=True); create.add_argument('--session'); create.add_argument('--v2-result', required=True); create.add_argument('--window-size', type=int, default=WINDOW_SIZE); create.add_argument('--local-source-dir', action='append', default=[]); create.add_argument('--include-after',action='store_true')
    check = sub.add_parser('verify'); check.add_argument('plan'); check.add_argument('--output', required=True)
    make = sub.add_parser('materialize'); make.add_argument('plan')
    replay_check = sub.add_parser('check-replay'); replay_check.add_argument('manifest')
    combine = sub.add_parser('aggregate'); combine.add_argument('plan'); combine.add_argument('--result', action='append', default=[]); combine.add_argument('--output', required=True)
    args = ap.parse_args()
    if args.command == 'plan':
        print(build(args.manifest, args.output_dir, args.remote_root, args.session, args.v2_result, args.window_size, args.local_source_dir, args.include_after))
    elif args.command == 'verify':
        result = verify(args.plan)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        print('missing', result['missing_count'], 'wrong size', result['wrong_size_count'], 'changed', result['changed_known_hash_count'], 'source failures', result['source_failures'])
        if not result['files_present_and_sources_unchanged']:
            sys.exit(1)
    elif args.command == 'materialize':
        print(materialize(args.plan))
    elif args.command == 'check-replay':
        manifest = json.loads(Path(args.manifest).read_text())
        if sha(manifest['source_plan']) != manifest['source_plan_sha256']:
            raise ValueError('source plan changed since materialization')
        if sha(manifest['original_v2_manifest']) != manifest['original_v2_manifest_sha256']:
            raise ValueError('original v2 manifest changed since materialization')
        if sha(manifest['observations']) != manifest['observations_sha256']:
            raise ValueError('materialized observations changed')
        for frame in manifest['frames']:
            for side in ('left', 'right'):
                if sha(frame[side]) != frame[side + '_observed_sha256']:
                    raise ValueError('materialized frame changed: ' + frame[side])
        for key, record in manifest['replay_sources'].items():
            if sha(record['path']) != record['sha256']:
                raise ValueError('replay source changed: ' + key)
        print('local frame, source and observation hashes match materialization; remote hashes may be unknown')
    else:
        result = aggregate(args.plan, args.result)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        print('session stability passed:', result['session_stability_passed'])


if __name__ == '__main__':
    main()
