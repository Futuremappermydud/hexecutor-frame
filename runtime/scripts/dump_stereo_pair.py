#!/usr/bin/env python3
"""
scripts/dump_stereo_pair.py — Capture synchronized stereo pair from Frame and save as PNG/PPM.
"""

import os
import sys
import subprocess

def main():
    print("Capturing live stereo pair from Steam Frame...")
    cmd = '''
python3 -c "
import os, subprocess

pid = subprocess.check_output(['pgrep', 'vrcompositor']).decode().strip()
fd_dir = f'/proc/{pid}/fd'
cam_fds = []
for fd in sorted(os.listdir(fd_dir), key=lambda x: int(x)):
    try:
        link = os.readlink(f'{fd_dir}/{fd}')
        if 'dmabuf' in link and os.stat(f'{fd_dir}/{fd}').st_size == 9564160:
            cam_fds.append(int(fd))
    except Exception:
        pass

if len(cam_fds) < 5:
    print('Failed to find camera FDs')
    exit(1)

right_fd = cam_fds[0]
left_fd = cam_fds[4]

import ctypes
from ctypes import CDLL, c_int, c_size_t, c_void_p, c_uint32
libc = CDLL('libc.so.6')

pfd = libc.syscall(434, int(pid), 0)
l_rfd = libc.syscall(438, pfd, right_fd, 0)
l_lfd = libc.syscall(438, pfd, left_fd, 0)

import mmap
with open(f'/proc/self/fd/{l_rfd}', 'rb') as f:
    mm_r = mmap.mmap(f.fileno(), 9564160, mmap.MAP_SHARED, mmap.PROT_READ)
    with open('/tmp/right_view.raw', 'wb') as out:
        out.write(mm_r[:2464*2464])

with open(f'/proc/self/fd/{l_lfd}', 'rb') as f:
    mm_l = mmap.mmap(f.fileno(), 9564160, mmap.MAP_SHARED, mmap.PROT_READ)
    with open('/tmp/left_view.raw', 'wb') as out:
        out.write(mm_l[:2464*2464])

print('Saved /tmp/right_view.raw and /tmp/left_view.raw on Frame')
"
'''
    subprocess.check_call(["ssh", "frame", cmd])
    os.makedirs("captures", exist_ok=True)
    subprocess.check_call(["scp", "frame:/tmp/left_view.raw", "captures/left_view.raw"])
    subprocess.check_call(["scp", "frame:/tmp/right_view.raw", "captures/right_view.raw"])

    # Convert to PPM
    for name in ["left_view", "right_view"]:
        with open(f"captures/{name}.raw", "rb") as f_in:
            data = f_in.read(2464 * 2464)
        with open(f"captures/{name}.ppm", "wb") as f_out:
            f_out.write(f"P5\n2464 2464\n255\n".encode() + data)
        print(f"Created captures/{name}.ppm (2464x2464 grayscale)")

if __name__ == "__main__":
    main()
