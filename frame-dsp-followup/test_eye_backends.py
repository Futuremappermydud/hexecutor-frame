#!/usr/bin/env python3
"""Temporarily replace the user's eye tracker, always restarting stock CDSP."""
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path('/opt/steamvr')
BIN = ROOT / 'tools/eyetracking/bin/linuxarm64/eyetracking'
WEIGHTS = ROOT / 'tools/eyetracking/resources/et_dsp_20250610_03136.weights'
OUT = Path.home() / 'frame-dsp/followup'

def pids():
    r = subprocess.run(['pgrep', '-x', 'eyetracking'], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split()]

try:
    for pid in pids():
        os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not pids():
            break
        time.sleep(.1)
    if pids():
        raise RuntimeError('Original eye tracker did not exit; refusing a second camera owner')
    for backend in ['GPU', 'CPU', 'CPUEV2']:
        with open(OUT / ('eye-' + backend + '.log'), 'w') as log:
            proc = subprocess.Popen([str(BIN), '-b', backend, '-w', str(WEIGHTS)],
                                    cwd=BIN.parent, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                rc = proc.wait(timeout=8)
                print(backend, 'exited', rc, flush=True)
            except subprocess.TimeoutExpired:
                print(backend, 'still running after 8 s', flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        print((OUT / ('eye-' + backend + '.log')).read_text()[-6000:], flush=True)
finally:
    if not pids():
        with open(OUT / 'eye-restored.log', 'a') as log:
            proc = subprocess.Popen(['taskset', '-c', '0,1', str(BIN), '-b', 'CDSP',
                                     '-w', str(WEIGHTS)], cwd=BIN.parent,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        time.sleep(2)
        print('Restored stock CDSP eye tracker:', proc.pid, 'running:', proc.poll() is None,
              flush=True)
