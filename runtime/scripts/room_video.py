#!/usr/bin/env python3
"""Read camera frames from older sessions packed as FFV1 video. New scans keep raw frames."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_frame(session, kind, eye, identifier):
    session = Path(session)
    folder = session / kind
    index = json.loads((folder / 'video-index.json').read_text())
    frames = index['streams'][eye]['frames']
    position = next((n for n, frame in enumerate(frames) if frame['id'] == identifier), None)
    if position is None:
        raise KeyError(f'{identifier} not found in {kind}/{eye} video')
    # FFV1 frames are intra-coded; select uses the stored frame order, while the
    # original exposure timestamp remains in video-index.json and frame metadata.
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i',
               str(folder / index['streams'][eye]['file']), '-vf', f'select=eq(n\\,{position})',
               '-fps_mode', 'passthrough',
               '-frames:v', '1', '-pix_fmt', index['original_pixel_format'], '-f', 'rawvideo', 'pipe:1']
    data = subprocess.run(command, check=True, capture_output=True).stdout
    if len(data) != index['frame_bytes'] or digest(data) != frames[position]['sha256']:
        raise ValueError(f'video extraction mismatch: {kind}/{eye}/{identifier}')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('kind', choices=('rgb', 'ir'))
    parser.add_argument('eye', choices=('left', 'right'))
    parser.add_argument('frame', help='frame-N or pair-N identifier')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_bytes(read_frame(args.session, args.kind, args.eye, args.frame))
    print(args.output)


if __name__ == '__main__':
    main()
