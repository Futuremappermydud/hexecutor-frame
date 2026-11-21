#!/usr/bin/env python3
"""Finalize selected Frame scans and copy the recordings to this Mac, with hashes."""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REMOTE_REPO = '/home/steamos/frame-spatial-runtime/depth-runtime'
SAFE = re.compile(r'^[A-Za-z0-9._-]+$')
SAFE_FRAMES = re.compile(r'^(rgb|ir)/[A-Za-z0-9._*?\[\]-]+$')
RAW_FRAMES = ('rgb/frame-*-raw-*.nv12', 'rgb/frame-*-raw-*.u8', 'ir/pair-*-left.u8', 'ir/pair-*-right.u8')


def is_raw_frame(relative):
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in RAW_FRAMES)


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def remote(host, command, capture_output=False):
    return run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, command],
               text=True, capture_output=capture_output)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default=os.environ.get('FRAME_HOST', 'frame'))
    ap.add_argument('--remote-dir', default=os.environ.get('FRAME_REMOTE_RUNTIME', REMOTE_REPO))
    ap.add_argument('--remote-root', default=os.environ.get('FRAME_DATASET_ROOT'),
                    help='dataset path on Frame, default REMOTE_DIR/experiments/room-dataset')
    ap.add_argument('--remote-python', default=os.environ.get('FRAME_DATASET_PYTHON'),
                    help='Python executable on Frame with the dataset dependencies')
    ap.add_argument('--session', help='one session directory name; default latest complete session')
    ap.add_argument('--all', action='store_true', help='pull all sessions')
    ap.add_argument('--raw-only', action='store_true', help='copy capture files without DSP finalization')
    ap.add_argument('--ir-yaw-deg', type=float, help='measured IR yaw for this session')
    ap.add_argument('--frames', action='append', default=[], metavar='GLOB',
                    help="raw camera frames to copy, relative to the session, e.g. 'rgb/frame-100-raw-*.nv12' or "
                         "'ir/pair-10-*.u8'; repeatable. Raw frames are otherwise left on the Frame")
    ap.add_argument('--all-frames', action='store_true', help='also copy every raw camera frame')
    ap.add_argument('--local-root', type=Path, default=(PATHS.repo / 'depth-runtime/experiments/room-dataset'))
    args = ap.parse_args()
    if args.session and args.all:
        ap.error('choose --session or --all')
    if any(not SAFE_FRAMES.fullmatch(pattern) for pattern in args.frames):
        ap.error('--frames takes a glob under rgb/ or ir/')
    remote_root = args.remote_root or f'{args.remote_dir}/experiments/room-dataset'
    python = shlex.quote(args.remote_python) if args.remote_python else '"$python"'
    prefix = f'cd {shlex.quote(args.remote_dir)} || exit; '
    if not args.remote_python:
        prefix += ('python=python3; '
                   'for candidate in "$HOME/frame-dsp/depth-model/.venv/bin/python" '
                   '"$HOME/frame-dsp/quest-depth/.venv/bin/python"; do '
                   'if test -x "$candidate"; then python="$candidate"; break; fi; done; ')
    cli = f'{python} scripts/room_dataset.py'
    root_arg = shlex.quote(remote_root)
    listing = remote(args.host, f'{prefix}{cli} index {root_arg} >/dev/null && cat {root_arg}/dataset.json',
                     capture_output=True)
    sessions = json.loads(listing.stdout)['sessions']
    allowed = ('captured', 'finalized', 'incomplete') if args.raw_only else ('captured', 'finalized')
    sessions = [s for s in sessions if s['status'] in allowed]
    if args.session:
        sessions = [s for s in sessions if s['path'] == args.session]
    elif not args.all:
        sessions = sessions[-1:]
    if not sessions:
        raise RuntimeError('no matching completed session on the Frame')
    args.local_root.mkdir(parents=True, exist_ok=True)
    failed = []
    for session in sessions:
        name = session['path']
        if not SAFE.fullmatch(name):
            raise ValueError(f'unsafe session name: {name}')
        if not args.raw_only and (session['status'] != 'finalized' or args.ir_yaw_deg is not None):
            command = f'{prefix}{cli} finalize {root_arg}/{name}'
            if args.ir_yaw_deg is not None:
                command += f' --skip-inference --ir-yaw-deg {args.ir_yaw_deg}' if session['status'] == 'finalized' else f' --ir-yaw-deg {args.ir_yaw_deg}'
            try:
                remote(args.host, command)
            except subprocess.CalledProcessError:
                failed.append(f'{name}: finalization failed; raw files will still be copied')
        destination = args.local_root / name
        destination.mkdir(parents=True, exist_ok=True)
        source = f'{args.host}:{shlex.quote(f"{remote_root}/{name}/")}'
        target = str(destination) + '/'
        # Raw camera frames are large and stay on the Frame unless requested.
        run(['rsync', '-az', '--partial', '--exclude=*.part'] + [f'--exclude={p}' for p in RAW_FRAMES] + [source, target])
        if args.all_frames:
            run(['rsync', '-a', '--partial', '--exclude=*.part', source, target])
        elif args.frames:
            run(['rsync', '-a', '--partial', '--include=*/'] + [f'--include={p}' for p in args.frames] +
                ['--exclude=*', '--prune-empty-dirs', source, target])
        manifest_path = destination / 'manifest.json'
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            for relative, expected in manifest['sha256'].items():
                file = destination / relative
                if is_raw_frame(relative) and not file.is_file():
                    continue  # not requested; a requested file that is present is still checked
                if not file.is_file() or digest(file) != expected:
                    failed.append(f'{name}: checksum mismatch for {relative}')
        elif not args.raw_only:
            failed.append(f'{name}: no manifest from finalization')
        print(f'Copied {name} to {destination}')
    run([sys.executable, str((PATHS.repo / 'depth-runtime/scripts/room_dataset.py')), 'index', str(args.local_root)])
    if failed:
        raise RuntimeError('; '.join(failed))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'pull_room_dataset: {error}', file=sys.stderr)
        sys.exit(1)
