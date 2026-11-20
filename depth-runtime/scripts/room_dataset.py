#!/usr/bin/env python3
"""Record a synchronized room scan and finalize its replayable depth/mesh dataset.

Capture runs on Frame, without DSP inference. Finalize runs one offline Model
session, preserves both raw model outputs, converts depth with the runtime C++
geometry, detects tags, and rebuilds a TSDF mesh from exposure-aligned poses.
"""

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
import csv
import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = (PATHS.repo / 'runtime')
THERMAL = Path('/sys/class/thermal')
MODEL_PY = Path.home() / 'frame-dsp/depth-model/.venv/bin/python'
MODEL_PACKAGE = Path.home() / 'frame-dsp/depth-model/wave1-20261101'
TAG_TOOL = Path.home() / 'frame-spatial-components/build/frame_apriltag'
BELICE_PY = Path.home() / 'Documents/belice/.venv/bin/python'
BELICE_FLASH = Path.home() / 'Documents/belice/belice_flash.py'
DEFAULT_PULSE_LEAD_MS = 8.0  # unmeasured; scripts/ir_pulse_sweep.py replaces it
VRCMD = Path('/opt/steamvr/bin/linuxarm64/vrcmd')
FLOOD_SETTING = 'driver_cv.infraredEmittersMode'
FLOODS_DISABLED, FLOODS_AUTO = 0, 2
RECT_PIXELS = 616 * 616
MODEL_PIXELS = 256 * 320


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def now_clocks():
    return {'monotonic_ns': time.clock_gettime_ns(time.CLOCK_MONOTONIC),
            'raw_ns': time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
            'utc': datetime.now(timezone.utc).isoformat()}


def thermal_worker(path, stop):
    with path.open('w') as out:
        while not stop.is_set():
            stamp = now_clocks()
            for zone in sorted(THERMAL.glob('thermal_zone*')):
                try:
                    sensor = (zone / 'type').read_text().strip()
                    milli_c = int((zone / 'temp').read_text().strip())
                except (OSError, ValueError):
                    continue
                out.write(json.dumps({**stamp, 'zone': zone.name, 'sensor': sensor,
                                      'milli_c': milli_c}) + '\n')
            out.flush()
            stop.wait(0.5)


def profile_worker(path, stop, processes):
    """Sample process CPU, resident memory, and I/O without ptracing recorders."""
    with path.open('w') as out:
        while not stop.is_set():
            sample = {'clock': now_clocks(), 'loadavg': os.getloadavg(), 'processes': {}}
            for name, process in list(processes.items()):
                pid = process.pid
                try:
                    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
                    status = Path(f'/proc/{pid}/status').read_text().splitlines()
                    io = Path(f'/proc/{pid}/io').read_text().splitlines()
                    fields = dict(line.split(':', 1) for line in status + io if ':' in line)
                    sample['processes'][name] = {
                        'pid': pid, 'state': stat[0], 'user_ticks': int(stat[11]),
                        'system_ticks': int(stat[12]), 'rss_kb': fields.get('VmRSS', '').strip(),
                        'read_bytes': int(fields.get('read_bytes', '0')),
                        'write_bytes': int(fields.get('write_bytes', '0'))}
                except (OSError, IndexError, ValueError):
                    sample['processes'][name] = {'pid': pid, 'exited': True}
            out.write(json.dumps(sample) + '\n')
            out.flush()
            stop.wait(0.5)


def event(path, event_name, **fields):
    with path.open('a') as out:
        out.write(json.dumps({'clock': now_clocks(), 'event': event_name, **fields}) + '\n')


def set_floods(mode):
    """Apply the headset flood mode and confirm SteamVR holds the new value."""
    env = {**os.environ, 'LD_LIBRARY_PATH': str(VRCMD.parent)}
    subprocess.run([str(VRCMD), '--set-settings-int', FLOOD_SETTING, str(mode)], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    current = subprocess.run([str(VRCMD), '--settings-int', FLOOD_SETTING], env=env, check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=15).stdout
    if f'{FLOOD_SETTING}={mode}' not in current.split():
        raise RuntimeError(f'{FLOOD_SETTING} did not change to {mode}: {current.strip()}')


def select_lit_pairs(observations, cycles, min_shams=20):
    """Keep only pairs that a logged pulse lit and whose dots beat this session's sham pulses."""
    roles = {(entry['cycle'], entry['role']): entry for entry in observations if 'role' in entry}
    def contrast(cycle):
        lit, reference = roles.get((cycle, 'lit')), roles.get((cycle, 'reference'))
        if not lit or not reference or not reference['dot_score'] or not reference['right_dot_score']:
            return None
        return min(lit['dot_score'] / reference['dot_score'], lit['right_dot_score'] / reference['right_dot_score'])
    shams = [value for value in (contrast(c['cycle']) for c in cycles if c['status'] == 'sham') if value is not None]
    fired = [c for c in cycles if c['status'] == 'fired']
    report = {'fired': len(fired), 'sham_controls': len(shams), 'accepted': 0, 'mistimed': 0, 'below_threshold': 0}
    if len(shams) < min_shams:
        report['status'] = 'insufficient_sham_controls'
        return [], report
    threshold = float(np.percentile(shams, 95))
    report.update(threshold=threshold, sham_median=float(np.median(shams)), status='complete')
    selected, ratios = [], []
    for cycle in fired:
        value = contrast(cycle['cycle'])
        if value is None:
            continue
        ratios.append(value)
        lit = roles[(cycle['cycle'], 'lit')]
        target = lit['left_raw_ns'] - int(cycle['lead_ms'] * 1e6)
        if not cycle['on_done_raw_ns'] <= target <= cycle['off_command_raw_ns']:
            report['mistimed'] += 1
        elif value <= threshold:
            report['below_threshold'] += 1
        else:
            selected.append({**lit, 'lead_ms': cycle['lead_ms'], 'contrast': value,
                             'references': [roles[(cycle['cycle'], role)] for role in ('reference', 'after')
                                            if (cycle['cycle'], role) in roles]})
    report['accepted'] = len(selected)
    report['fired_median'] = float(np.median(ratios)) if ratios else None
    return selected, report


def capture(args):
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'output directory is not empty: {out}')
    if args.frames < 0 or (args.frames == 0 and not args.live_viewer) or args.spacing_ms < (0 if args.live_viewer else 100):
        raise ValueError('frames must be nonnegative (zero only for live capture); spacing must be nonnegative for live capture')
    if args.ir_spacing_ms < 20 or args.ir_max_pairs < 0:
        raise ValueError('IR spacing must be at least 20 ms and max pairs nonnegative')
    if args.min_free_gb < 0:
        raise ValueError('minimum free space must be nonnegative')
    if args.dots and not args.ir:
        raise ValueError('--dots needs IR recording')
    if args.live_viewer and args.model_backend == 'prepared-qnn':
        lease = Path('/sys/module/cdsp_mem/parameters/size_mb')
        if not lease.is_file() or lease.read_text().strip() != '128':
            raise RuntimeError('QNN needs a loaded 128 MB cdsp_mem lease; run scripts/setup_frame.sh --qnn from the Mac')
        fault = out.parent / '.npu-fault-boot-id'
        if fault.is_file() and fault.read_text().strip() == Path('/proc/sys/kernel/random/boot_id').read_text().strip():
            raise RuntimeError('the previous cDSP session failed to clean up; reboot the Frame before another QNN run')
    tools = {'rgb': REPO / ('bin/frame_openxr_mesh_viewer' if args.live_viewer else 'bin/capture_model_inputs'),
             'pose': (PATHS.repo / 'depth-runtime/bin/record_head_poses')}
    if not args.no_markers:
        tools['marks'] = (PATHS.repo / 'depth-runtime/bin/mark_surfaces')
    if args.ir:
        tools['ir'] = (PATHS.repo / 'depth-runtime/bin/record_ir_pairs')
    missing = [str(p) for p in tools.values() if not p.is_file()]
    if not args.keep_floods and not VRCMD.is_file():
        missing.append(str(VRCMD))
    if missing:
        raise FileNotFoundError('build Frame capture tools first: ' + ', '.join(missing))
    pulse_lead_ms, pulse_lead_source = args.pulse_lead_ms, 'argument'
    calibration = out.parent / 'ir-pulse-calibration.json'
    if pulse_lead_ms is None:
        pulse_lead_ms = json.loads(calibration.read_text())['lead_ms'] if calibration.is_file() else DEFAULT_PULSE_LEAD_MS
        pulse_lead_source = str(calibration.name) if calibration.is_file() else 'uncalibrated default'
    for existing in out.parent.glob('*/session.json'):
        prior = json.loads(existing.read_text())
        if prior.get('scene') == args.scene and prior.get('split') != args.split:
            raise ValueError(f'scene {args.scene} already belongs to {prior.get("split")} in {existing.parent}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'rgb').mkdir(); (out / 'ir').mkdir()
    duration_ms = 0 if args.frames == 0 else 12000 + args.frames * (args.spacing_ms + 250)
    session = {'version': 1, 'status': 'capturing', 'scene': args.scene, 'split': args.split,
               'notes': args.notes, 'start': now_clocks(), 'frame_target': args.frames or None,
               'rgb_spacing_ms': args.spacing_ms, 'ir_requested': args.ir,
               'ir_spacing_ms': args.ir_spacing_ms,
               'dots_requested': args.dots,
               'pulse': {'every_long_frames': args.pulse_every_frames, 'pulse_ms': args.pulse_ms,
                         'lead_ms': pulse_lead_ms, 'lead_source': pulse_lead_source,
                         'sham_percent': args.sham_percent} if args.dots else None,
               'floods_disabled_requested': not args.keep_floods,
               'rgb_color_format': None if args.no_color else 'raw stereo NV12, 2464 x 2464; rectified luma is separate',
               'clock_contract': 'RGB and IR pose matching uses CLOCK_MONOTONIC; IR raw clock also retained',
               'controller_markers_requested': not args.no_markers,
               'live_viewer': args.live_viewer,
               'live_mesh_backend': f'Model {args.model_backend} TSDF' if args.live_viewer else None,
               'controller_tip_local_m': args.controller_tip,
               'label_contract': 'marks retain full controller device pose; tip position uses the declared local offset'}
    session['capture_source_sha256'] = {
        str(path.relative_to(REPO)): sha256_file(path)
        for path in ((PATHS.repo / 'depth-runtime/scripts/room_dataset.py'), (PATHS.repo / 'depth-runtime/tools/capture_model_inputs.cpp'),
                     (PATHS.repo / 'depth-runtime/tools/record_ir_pairs.cpp'), (PATHS.repo / 'depth-runtime/tools/mark_surfaces.cpp'))}
    session['capture_binary_sha256'] = {name: sha256_file(path) for name, path in tools.items()}
    stop_file = out / 'stop.request'
    session['stop_file'] = str(stop_file.relative_to(out))
    write_json(out / 'session.json', session)
    stop = threading.Event()
    terminate = threading.Event()
    stopped_by_user = False
    stop_source = None
    stop_requested_at = None
    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, lambda *_: terminate.set())
    signal.signal(signal.SIGINT, lambda *_: terminate.set())
    workers = []
    failures = []
    warnings = []
    events = out / 'events.jsonl'
    event(events, 'capture_start', frame_target=args.frames)
    processes = {}
    with (out / 'capture.log').open('w') as capture_log:
        try:
            if not args.keep_floods:
                set_floods(FLOODS_DISABLED)
                event(events, 'floods_disabled')
            sidecars = [('pose', [str(tools['pose']), str(out / 'head_pose.csv'), str(duration_ms)], 'pose.log')]
            if not args.no_markers:
                sidecars.append(('marks', [str(tools['marks']), str(out / 'marks.csv'), str(duration_ms)] +
                                 [str(v) for v in args.controller_tip] + [str(stop_file)], 'marks.log'))
            pulse_flags = ['--pulse-every', str(args.pulse_every_frames), '--pulse-ms', str(args.pulse_ms),
                           '--lead-ms', str(pulse_lead_ms), '--sham-percent', str(args.sham_percent),
                           '--pulses-out', str(out / 'pulses.jsonl')] if args.dots else []
            for name, command, logname in sidecars + ([('ir', [str(tools['ir']), str(out / 'ir'), str(duration_ms),
                         str(args.ir_spacing_ms), str(args.ir_max_pairs)] + pulse_flags, 'ir.log')] if args.ir else []):
                log = (out / logname).open('w')
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                workers.append((name, proc, log))
                processes[name] = proc
                event(events, 'recorder_start', name=name, pid=proc.pid)
            thermal = threading.Thread(target=thermal_worker, args=(out / 'thermal.jsonl', stop), daemon=True)
            thermal.start()
            time.sleep(0.5)
            env = os.environ.copy()
            env['FRAME_CAPTURE_TIMING'] = 'v4l2'
            env['FRAME_DEPTH_SPHERICAL'] = '1'
            env['FRAME_SAVE_COLOR'] = '0' if args.no_color else '1'
            if args.save_raw:
                env['FRAME_SAVE_RAW'] = '1'
            if args.live_viewer:
                env['FRAME_DEPTH_RECORD_V2'] = str(out / 'depth')
                env['FRAME_DEPTH_RECORD_FRAMES'] = str(args.frames)
                env['FRAME_ROOM_SPACING_MS'] = str(args.spacing_ms)
                env['FRAME_DEPTH_RECORD_CLOSE_EACH_REPLY'] = '0'
                env['FRAME_ROOM_RGB_DIR'] = str(out / 'rgb')
                env['FRAME_ROOM_STOP_FILE'] = str(stop_file)
                env['FRAME_CAPTURE_UV'] = '1'
                command = [str(tools['rgb']), '--depth-model-qnn', '--tsdf']
            else:
                command = [str(tools['rgb']), str(out / 'rgb'), str(args.frames), str(args.spacing_ms)]
            rgb = subprocess.Popen(command, env=env, cwd=REPO, stdout=capture_log,
                                   stderr=subprocess.STDOUT)
            processes['rgb'] = rgb
            event(events, 'recorder_start', name='rgb', pid=rgb.pid)
            profiler = threading.Thread(target=profile_worker,
                                        args=(out / 'profile.jsonl', stop, processes), daemon=True)
            profiler.start()
            deadline = None if duration_ms == 0 else time.monotonic() + duration_ms / 1000 + 5
            next_space_check = time.monotonic()
            compositor_pid = subprocess.check_output(['pgrep', '-x', 'vrcompositor'], text=True).splitlines()[0]
            while rgb.poll() is None:
                if time.monotonic() >= next_space_check:
                    next_space_check = time.monotonic() + 1
                    free = shutil.disk_usage(out).free
                    if free < args.min_free_gb * 1024**3 and not stop_file.exists():
                        stop_file.write_text('storage_low\n')
                        event(events, 'storage_low', free_bytes=free)
                        warnings.append(f'recording stopped with {free / 1024**3:.1f} GiB free')
                if not Path('/proc', compositor_pid).exists():
                    failures.append('vrcompositor exited or restarted during capture')
                    event(events, 'compositor_lost', pid=int(compositor_pid))
                    if args.live_viewer:
                        stop_file.write_text('compositor_lost\n')
                    else:
                        rgb.terminate()
                    break
                if terminate.is_set() or stop_file.exists():
                    if not stopped_by_user:
                        stopped_by_user = True
                        stop_reason = stop_file.read_text() if stop_file.exists() else ''
                        stop_source = ('controller' if stop_reason.startswith('controller_stop') else
                                       'storage' if stop_reason.startswith('storage_low') else 'menu')
                        event(events, 'stop_requested', source=stop_source)
                        stop_requested_at = time.monotonic()
                        if args.live_viewer and not stop_file.exists():
                            stop_file.write_text('menu_stop\n')
                        if not args.live_viewer:
                            rgb.terminate()
                            break
                    if args.live_viewer and time.monotonic() - stop_requested_at > 15:
                        rgb.terminate()
                        failures.append('live viewer did not honor stop request within 15 s')
                        break
                if deadline is not None and time.monotonic() > deadline:
                    rgb.terminate()
                    failures.append('RGB capture exceeded deadline')
                    break
                time.sleep(0.1)
            try:
                rgb_code = rgb.wait(timeout=30 if args.live_viewer else 3)
            except subprocess.TimeoutExpired:
                rgb.kill(); rgb_code = rgb.wait()
            if rgb_code and not stopped_by_user and not failures:
                failures.append(f'RGB capture exited {rgb_code}')
            event(events, 'recorder_exit', name='rgb', returncode=rgb_code)
            if stopped_by_user and not list((out / 'rgb').glob('frame-*.json')):
                failures.append('stopped before the first complete RGB frame')
            if args.live_viewer and rgb_code == 0 and not stopped_by_user and (args.frames == 0 or len(list((out / 'rgb').glob('frame-*.json'))) < args.frames):
                failures.append('live viewer ended before the frame target')
        except BaseException as error:
            failures.append(f'capture interrupted: {error}')
        finally:
            stop.set()
            if 'rgb' in locals() and rgb.poll() is None:
                rgb.terminate()
                try: rgb.wait(timeout=30 if args.live_viewer else 3)
                except subprocess.TimeoutExpired: rgb.kill(); rgb.wait()
            if 'thermal' in locals():
                thermal.join(timeout=2)
            if 'profiler' in locals():
                profiler.join(timeout=2)
            for name, proc, log in workers:
                if proc.poll() is None:
                    proc.terminate()
                try:
                    code = proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill(); code = proc.wait()
                log.close()
                event(events, 'recorder_exit', name=name, returncode=code)
                if code:
                    (failures if name == 'pose' else warnings).append(f'{name} recorder exited {code}')
            if args.dots:
                # The chip holds its last level, so also cover a recorder that was killed mid-pulse.
                try:
                    subprocess.run([str(BELICE_PY), str(BELICE_FLASH), 'emitter', '--seconds', '0'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                except (OSError, subprocess.SubprocessError) as error:
                    warnings.append(f'could not confirm the emitter is off: {error}')
                pulse_path = out / 'pulses.jsonl'
                cycles = [json.loads(line) for line in pulse_path.read_text().splitlines()] if pulse_path.exists() else []
                session['dot_pulses_succeeded'] = sum(cycle['status'] == 'fired' for cycle in cycles)
                if not session['dot_pulses_succeeded']:
                    warnings.append('no successful dot pulse; no lit IR pairs')
                    event(events, 'warning', message='no successful dot pulse; no lit IR pairs')
            if not args.keep_floods:
                try:
                    set_floods(FLOODS_AUTO)
                    event(events, 'floods_restored')
                except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                    warnings.append(f'floods may still be disabled; set {FLOOD_SETTING} to {FLOODS_AUTO}: {error}')
                    event(events, 'floods_restore_failed', error=str(error))
            session['end'] = now_clocks()
            session['stopped_from_headset'] = stop_source == 'controller'
            session['stopped_from_menu'] = stop_source == 'menu'
            session['stopped_for_storage'] = stop_source == 'storage'
            session['status'] = 'captured' if not failures else 'incomplete'
            session['frames_recorded'] = len(list((out / 'rgb').glob('frame-*.json')))
            if args.live_viewer:
                session['frames_meshed'] = len(list((out / 'depth').glob('frame-*.json')))
            session['failures'] = failures
            session['warnings'] = warnings
            capture_text = (out / 'capture.log').read_text()
            if any(message in capture_text for message in (
                    'Persistent DSP workers did not exit', 'DSP state retained',
                    'explicit recovery is required')):
                session['npu_cleanup_failed'] = True
                (out.parent / '.npu-fault-boot-id').write_text(
                    Path('/proc/sys/kernel/random/boot_id').read_text())
                warnings.append('cDSP cleanup incomplete; reboot before another NPU scan')
            event(events, 'capture_end', status=session['status'], failures=failures)
            write_json(out / 'session.json', session)
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
    if failures:
        raise RuntimeError('; '.join(failures))
    print(f'Captured {out}')


def frame_paths(rgb):
    paths = sorted(rgb.glob('frame-*.json'), key=lambda p: int(p.stem.split('-')[1]))
    if not paths or [int(p.stem.split('-')[1]) for p in paths] != list(range(1, len(paths) + 1)):
        raise ValueError('RGB frames are missing or not consecutively numbered')
    return paths


def nearest_pose(csv_path, target_ns, max_error_ns=25_000_000):
    # Called after pose rows are loaded and sorted by mono_ns.
    import bisect
    times, rows = csv_path
    i = bisect.bisect_left(times, target_ns)
    choices = [rows[j] for j in (i - 1, i) if 0 <= j < len(rows) and rows[j]['valid'] == '1']
    if not choices:
        return None, None
    row = min(choices, key=lambda r: abs(int(r['mono_ns']) - target_ns))
    delta = abs(int(row['mono_ns']) - target_ns)
    return (row, delta) if delta <= max_error_ns else (None, delta)


def load_poses(path):
    with path.open(newline='') as f:
        rows = [row for row in csv.DictReader(f)]
    rows.sort(key=lambda row: int(row['mono_ns']))
    return [int(row['mono_ns']) for row in rows], rows


def load_ir_image(out, entry, eye):
    from room_video import read_frame
    path = out / 'ir' / entry[f'{eye}_file']
    data = (np.fromfile(path, np.uint8) if path.exists() else
            np.frombuffer(read_frame(out, 'ir', eye, f'pair-{entry["pair"]}'), np.uint8))
    return data.reshape(1024, 1056)


def ir_stereo_input(out, entry, mode):
    """Lit left and right images minus their dark long-exposure neighbours, per the reference mode."""
    references = {'none': [], 'before': entry['references'][:1], 'average': entry['references']}[mode]
    images = []
    for eye in ('left', 'right'):
        image = load_ir_image(out, entry, eye).astype(np.float32)
        if references:
            image -= np.mean([load_ir_image(out, reference, eye) for reference in references], axis=0)
        images.append(image)
    return images[0], images[1], len(references)


def rgb_prior(out, name):
    from ir_dot_lib import H, W
    disp_path = out / 'model' / f'{name}-spherical-disparity.f32'
    unc_path = out / 'model' / f'{name}-spherical-uncertainty.u8'
    if not disp_path.exists():
        disp_path = out / 'depth' / f'{name}-disparity.f32'
        unc_path = out / 'depth' / f'{name}-uncertainty.u8'
    return (np.fromfile(disp_path, '<f4').reshape(H, W), np.fromfile(unc_path, np.uint8).reshape(H, W))


def refine_ir(args, out, frames, poses, extrinsic):
    """Refine every pair a logged pulse lit, retaining raw input and provenance."""
    if args.ir_yaw_deg is None:
        return {'status': 'deferred_no_session_yaw', 'pairs': 0}
    pulses = out / 'pulses.jsonl'
    pairs = out / 'ir/pairs.jsonl'
    if not pulses.exists() or not pairs.exists():
        return {'status': 'no_pulsed_ir_recording', 'pairs': 0}
    os.environ['IR_YAW_DEG'] = str(args.ir_yaw_deg)
    os.environ['IR_CALIB'] = str(args.ir_calib)
    from ir_dot_lib import Rectifier, load_rgb_chain
    from ir_dot_refine_bridge import refine
    rect = Rectifier()
    chain = load_rgb_chain(args.rgb_calib or out / 'rgb/calibration-history-at-init.json')
    observations = [json.loads(line) for line in pairs.read_text().splitlines()]
    cycles = [json.loads(line) for line in pulses.read_text().splitlines()]
    selected, selection = select_lit_pairs(observations, cycles)
    rgb_times = [(int(json.loads(path.read_text())['left_end_of_readout_ns']), path.stem)
                 for path in frames]
    results = []
    for entry in selected:
        offset_ms = entry['lead_ms'] if args.ir_exposure_offset_ms is None else args.ir_exposure_offset_ms
        when = entry['left_monotonic_ns'] - int(offset_ms * 1e6)
        rgb_time, name = min(rgb_times, key=lambda item: abs(item[0] - when))
        if abs(rgb_time - when) > 250_000_000:
            results.append({'pair': entry['pair'], 'status': 'no_close_rgb_prior'}); continue
        pose_row, pose_delta = nearest_pose(poses, when)
        if pose_row is None:
            results.append({'pair': entry['pair'], 'status': 'no_ir_pose', 'pose_delta_ns': pose_delta}); continue
        disp, unc = rgb_prior(out, name)
        left, right, reference_count = ir_stereo_input(out, entry, args.ir_reference)
        kept, depth, plane = refine(rect, chain, disp, unc, left, right, rng=np.random.default_rng(7))
        prefix = f'pair-{entry["pair"]}'
        depth.astype('<f4').tofile(out / 'ir' / f'{prefix}-depth.f32')
        if plane:
            fit, grid = plane
            grid.tofile(out / 'ir' / f'{prefix}-footprint.u8')
            plane_data = {'n': fit.n.tolist(), 'c': fit.c, 'centroid': fit.cen.tolist(),
                          'u': fit.u.tolist(), 'v': fit.v.tolist(), 'inliers': fit.count,
                          'rms_m': fit.rms, 'radius_m': fit.radius}
        else:
            plane_data = None
        rotation, translation = pose_from_row(pose_row, extrinsic)
        result = {'pair': entry['pair'], 'status': 'refined', 'kept': kept, 'references_subtracted': reference_count,
                  'valid_depth_pixels': int(np.count_nonzero(depth)), 'prior_rgb_frame': name,
                  'prior_age_ns': abs(rgb_time - when), 'pose_delta_ns': pose_delta,
                  'camera_to_world_rotation': rotation, 'camera_to_world_translation_m': translation,
                  'exposure_ns': when, 'plane': plane_data,
                  'depth_file': f'{prefix}-depth.f32'}
        write_json(out / 'ir' / f'{prefix}-depth.json', result)
        results.append(result)
    write_json(out / 'ir/refinement.json', {'yaw_deg': args.ir_yaw_deg,
                                           'exposure_offset_ms': args.ir_exposure_offset_ms,
                                           'reference_mode': args.ir_reference,
                                           'selection': selection, 'results': results})
    return {'status': selection['status'], 'selection': selection,
            'pairs': sum(r['status'] == 'refined' for r in results)}


def mesh_rgb_ir(out, replay_executable):
    rgb = [(json.loads(p.read_text())['pose_time_ns'], json.loads(p.read_text()), p.stem)
           for p in (out / 'depth').glob('frame-*.json')]
    ir = [(json.loads(p.read_text())['exposure_ns'], json.loads(p.read_text()), p.stem.removesuffix('-depth'))
          for p in (out / 'ir').glob('pair-*-depth.json')]
    if not ir:
        return None
    combined = sorted([(t, 'rgb', meta, name) for t, meta, name in rgb] +
                      [(t, 'ir', meta, name) for t, meta, name in ir])
    stream = out / 'mesh/rgb-ir-frames.bin'
    with stream.open('wb') as file:
        file.write(struct.pack('<I', len(combined)))
        for _, kind, meta, name in combined:
            file.write(struct.pack('<12f', *(meta['camera_to_world_rotation'] +
                                             meta['camera_to_world_translation_m'])))
            depth = out / ('depth' if kind == 'rgb' else 'ir') / (name + '-depth.f32')
            file.write(depth.read_bytes())
            if kind == 'rgb':
                file.write((out / 'depth' / (name + '-rect-left.u8')).read_bytes())
                file.write((out / 'depth' / (name + '-confidence.f32')).read_bytes())
            else:
                file.write(bytes([128]) * RECT_PIXELS)
                file.write(np.ones(RECT_PIXELS, np.float32).tobytes())
    output = out / 'mesh/rgb-ir-mesh.ply'
    subprocess.run([str(replay_executable), str(stream), str(output)], check=True)
    return str(output.relative_to(out))


def pose_from_row(row, extrinsic):
    h = np.array([[float(row[f'm{r}{c}']) for c in range(4)] for r in range(3)])
    e = np.asarray(extrinsic['rotation_row_major'], np.float64).reshape(3, 3)
    t = np.asarray(extrinsic['translation_m'], np.float64)
    rotation = h[:, :3] @ e
    translation = h[:, :3] @ t + h[:, 3]
    return rotation.astype(np.float32).ravel().tolist(), translation.astype(np.float32).tolist()


def write_depth_metadata(frames, depth_dir, poses, extrinsic, exposure_offset_ms):
    """Write each frame's exposure-aligned pose JSON beside its converted depth; return skips."""
    skipped = []
    for path in frames:
        meta = json.loads(path.read_text())
        name = path.stem
        if not meta.get('timed') or not meta.get('left_end_of_readout_ns'):
            skipped.append({'frame': name, 'reason': 'no V4L2 exposure timestamp'})
            continue
        exposure_ns = int(meta['left_end_of_readout_ns']) - int(exposure_offset_ms * 1e6)
        row, delta = nearest_pose(poses, exposure_ns)
        if row is None:
            skipped.append({'frame': name, 'reason': 'missing capture-time pose', 'nearest_delta_ns': delta})
            continue
        rotation, translation = pose_from_row(row, extrinsic)
        outmeta = {'version': 2, 'sequence': int(name.split('-')[1]), 'files_prefix': name,
                   'camera_to_world_rotation': rotation, 'camera_to_world_translation_m': translation,
                   'capture_begin_ns': meta['capture_begin_ns'], 'capture_end_ns': meta['capture_end_ns'],
                   'pose_time_ns': int(row['mono_ns']), 'capture_time_pose': True, 'pose_valid': True,
                   'pose_timing': 'v4l2_end_of_readout_minus_offset_nearest_openvr',
                   'pose_delta_ns': delta, 'exposure_offset_ms': exposure_offset_ms,
                   'camera_sequence': meta.get('camera_sequence', 0), 'spherical_step_rad': meta['spherical_step_rad']}
        write_json(depth_dir / (name + '.json'), outmeta)
    return skipped


def finalize(args):
    out = args.output.resolve()
    stage_start = time.monotonic()
    stages = {}
    def stage(name):
        nonlocal stage_start
        elapsed = round(time.monotonic() - stage_start, 3)
        stages[name] = elapsed
        event(out / 'events.jsonl', 'finalize_stage', name=name, seconds=elapsed)
        stage_start = time.monotonic()
    session = json.loads((out / 'session.json').read_text())
    frames = frame_paths(out / 'rgb')
    if not (out / 'head_pose.csv').is_file():
        raise FileNotFoundError('head_pose.csv is required to mesh the session')
    live = session.get('live_viewer', False)
    model = out / 'model'
    if not live:
        model.mkdir(exist_ok=True)
    if not live and not args.skip_inference:
        if not args.python.is_file() or not args.package.is_dir():
            raise FileNotFoundError('Model Python/package missing; finalize on the Frame or use --skip-inference')
        command = ([str(args.python), str((PATHS.repo / 'depth-runtime/scripts/model_offline_batch.py')),
                    '--package', str(args.package), '--out', str(model)] +
                   [str(out / 'rgb' / (p.stem + '-spherical')) for p in frames])
        with (out / 'inference.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        stage('model_inference')
    if not live:
        subprocess.run([str(args.converter), str(out / 'rgb'), str(model), str(out / 'depth'), str(len(frames))], check=True)
        stage('depth_conversion')
    else:
        stage('live_depth_recording')
    depth_dir = out / 'depth'
    if live and not (depth_dir / 'capture.json').is_file():
        raise FileNotFoundError('live viewer did not save depth/capture.json')
    extrinsic = json.loads((out / 'rgb/rectified-camera-in-head.json').read_text())
    poses = load_poses(out / 'head_pose.csv')
    batch_report = model / 'batch.json'
    offline_backend = (json.loads(batch_report.read_text())['backend'] if batch_report.is_file()
                       else 'precomputed' if args.skip_inference else 'prepared-qnn')
    capture = {'version': 2, 'model': 'primary', 'backend': offline_backend, 'projection': 'spherical',
               'temporal': False, 'model_shape': [320, 256], 'rectified_shape': [616, 616],
               'right_disparity': True, 'capture_clock': 'CLOCK_MONOTONIC',
               'exposure_timestamp': True, 'config_sha256': ''}
    if not live:
        write_json(depth_dir / 'capture.json', capture)
    skipped = [] if live else write_depth_metadata(frames, depth_dir, poses, extrinsic, args.exposure_offset_ms)
    if skipped:
        write_json(out / 'skipped_frames.json', skipped)
    if not list(depth_dir.glob('frame-*.json')):
        raise RuntimeError('no exposure-aligned frames to mesh')
    stage('pose_alignment')
    tags = out / 'tags'; tags.mkdir(exist_ok=True)
    tag_status = 'not_requested' if args.skip_tags else 'complete'
    if not args.skip_tags:
        if not args.tag_tool.is_file():
            tag_status = 'tool_missing'
        else:
            for path in frames:
                name = path.stem
                for eye in ('left', 'right'):
                    image = out / 'rgb' / (name + f'-rect-{eye}.u8')
                    result = subprocess.run([str(args.tag_tool), '--raw', str(image), '--image-size',
                                             '616', '--tag-size-m', str(args.tag_size_m)],
                                            capture_output=True, text=True)
                    if result.returncode:
                        tag_status = 'partial_failure'
                    (tags / f'{name}-{eye}.txt').write_text(result.stdout + result.stderr)
                    try:
                        detections = json.loads(result.stdout)
                        write_json(tags / f'{name}-{eye}.json',
                                   {'source_frame': name, 'eye': eye, 'tag_size_m': args.tag_size_m,
                                    'detections': detections})
                    except json.JSONDecodeError:
                        pass
    stage('apriltags')
    from room_reconstruction_replay import replay
    mesh_dir = out / 'mesh'
    replay(depth_dir, mesh_dir, args.replay)
    stage('tsdf_replay')
    ir_status = refine_ir(args, out, frames, poses, extrinsic)
    rgb_ir_mesh = mesh_rgb_ir(out, args.replay) if ir_status['pairs'] else None
    stage('ir_refinement')
    session.update({'status': 'finalized', 'finalized_utc': datetime.now(timezone.utc).isoformat(),
                    'frames_recorded': len(frames), 'frames_meshed': len(list(depth_dir.glob('frame-*.json'))),
                    'pose_skips': skipped, 'tag_status': tag_status,
                    'ir_depth': ir_status, 'rgb_ir_mesh': rgb_ir_mesh,
                    'mesh': str((mesh_dir / 'mesh.ply').relative_to(out)),
                    'model_package': str(args.package) if live or not args.skip_inference else 'precomputed'})
    session['finalize_stage_seconds'] = stages
    session['finalize_binary_sha256'] = {'converter': sha256_file(args.converter) if not live else None,
                                          'replay': sha256_file(args.replay)}
    session['finalize_source_sha256'] = {
        str(path.relative_to(REPO)): sha256_file(path)
        for path in ((PATHS.repo / 'depth-runtime/scripts/model_offline_batch.py'), (PATHS.repo / 'depth-runtime/scripts/room_reconstruction_replay.py'),
                     (PATHS.repo / 'depth-runtime/src/tsdf_volume.cpp'), (PATHS.repo / 'depth-runtime/tools/model_batch_to_v2.cpp'))}
    write_json(out / 'session.json', session)
    validate(out)
    print(f'Finalized {out}; mesh: {mesh_dir / "mesh.ply"}')


def validate(out):
    out = Path(out)
    session = json.loads((out / 'session.json').read_text())
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json')
    sizes = {p.relative_to(out).as_posix(): p.stat().st_size for p in files}
    defects = []
    warnings = []
    for path in (out / 'rgb').glob('frame-*-rect-*.u8'):
        if path.stat().st_size != RECT_PIXELS:
            defects.append(f'bad RGB image size: {path.name}')
    for path in (out / 'rgb').glob('frame-*-raw-*.nv12'):
        if path.stat().st_size != 2464 * 2464 * 3 // 2:
            defects.append(f'bad stereo color NV12 size: {path.name}')
    rgb_video_path = out / 'rgb/video-index.json'
    rgb_video = json.loads(rgb_video_path.read_text()) if rgb_video_path.exists() else None
    if rgb_video:
        for eye in ('left', 'right'):
            if not (out / 'rgb' / rgb_video['streams'][eye]['file']).is_file():
                defects.append(f'missing RGB {eye} video')
            expected_ids = [frame.stem for frame in frame_paths(out / 'rgb')]
            if [frame['id'] for frame in rgb_video['streams'][eye]['frames']] != expected_ids:
                defects.append(f'RGB {eye} video index does not match frame metadata')
    if session.get('rgb_color_format'):
        for frame in frame_paths(out / 'rgb'):
            for eye in ('left', 'right'):
                indexed = rgb_video and any(row['id'] == frame.stem for row in rgb_video['streams'][eye]['frames'])
                if not (out / 'rgb' / f'{frame.stem}-raw-{eye}.nv12').is_file() and not indexed:
                    defects.append(f'missing stereo color: {frame.stem} {eye}')
    if (out / 'depth/capture.json').exists():
        try:
            from room_reconstruction_replay import load_capture
            load_capture(out / 'depth')
        except (OSError, ValueError) as error:
            defects.append(f'invalid replayable depth: {error}')
    ir_count = 0
    ir_video_path = out / 'ir/video-index.json'
    ir_video = json.loads(ir_video_path.read_text()) if ir_video_path.exists() else None
    if ir_video:
        for eye in ('left', 'right'):
            if not (out / 'ir' / ir_video['streams'][eye]['file']).is_file():
                defects.append(f'missing IR {eye} video')
    for line in (out / 'ir/pairs.jsonl').read_text().splitlines() if (out / 'ir/pairs.jsonl').exists() else []:
        entry = json.loads(line); ir_count += 1
        for eye, key in (('left', 'left_file'), ('right', 'right_file')):
            image = out / 'ir' / entry[key]
            indexed = ir_video and any(row['id'] == f'pair-{entry["pair"]}' and row['source'] == entry[key]
                                       for row in ir_video['streams'][eye]['frames'])
            if not indexed and (not image.is_file() or image.stat().st_size != 1056 * 1024):
                defects.append(f'bad IR image: {image.name}')
    if session.get('ir_requested') and session.get('status') != 'capturing' and ir_count == 0:
        warnings.append('IR was requested but no paired IR images were saved')
    pulses = [json.loads(line) for line in (out / 'pulses.jsonl').read_text().splitlines()] if (out / 'pulses.jsonl').exists() else []
    if session.get('dots_requested') and not any(event.get('status') == 'fired' for event in pulses):
        warnings.append('dots were requested but no pulse succeeded; no lit IR pairs')
    with (out / 'marks.csv').open(newline='') if (out / 'marks.csv').exists() else open(os.devnull) as file:
        mark_count = max(0, sum(1 for _ in file) - 1)
    temperatures = sum(1 for _ in (out / 'thermal.jsonl').open()) if (out / 'thermal.jsonl').exists() else 0
    tag_outputs = len(list((out / 'tags').glob('frame-*-*.txt'))) if (out / 'tags').is_dir() else 0
    frame_metas = [json.loads(path.read_text()) for path in (out / 'depth').glob('frame-*.json')]
    pose_deltas = [meta['pose_delta_ns'] for meta in frame_metas if 'pose_delta_ns' in meta]
    hashes = {p.relative_to(out).as_posix(): sha256_file(p) for p in files}
    manifest = {'version': 1, 'files': sizes, 'sha256': hashes,
                'rgb_frames': len(frame_paths(out / 'rgb')), 'ir_pairs': ir_count,
                'ir_depth_pairs': len(list((out / 'ir').glob('pair-*-depth.json'))),
                'controller_mark_samples': mark_count, 'thermal_samples': temperatures,
                'successful_dot_pulses': sum(event.get('status') == 'fired' for event in pulses),
                'tag_eye_outputs': tag_outputs,
                'meshed_frames': len(frame_metas),
                'worst_pose_delta_ms': max(pose_deltas) * 1e-6 if pose_deltas else None,
                'defects': defects, 'warnings': warnings, 'total_bytes': sum(sizes.values())}
    write_json(out / 'manifest.json', manifest)
    if defects:
        raise RuntimeError('; '.join(defects))
    print(f'Validated {manifest["rgb_frames"]} RGB frames, {ir_count} IR pairs, {manifest["total_bytes"]} bytes')


def index_dataset(root):
    root = Path(root).resolve()
    sessions = []
    scene_splits = {}
    for path in sorted(root.glob('*/session.json')):
        session = json.loads(path.read_text())
        scene, split = session.get('scene'), session.get('split')
        if not scene or split not in ('development', 'tuning', 'held_out'):
            raise ValueError(f'invalid scene/split in {path}')
        if scene in scene_splits and scene_splits[scene] != split:
            raise ValueError(f'scene {scene} appears in both {scene_splits[scene]} and {split}')
        scene_splits[scene] = split
        manifest = path.parent / 'manifest.json'
        sessions.append({'path': path.parent.relative_to(root).as_posix(), 'scene': scene,
                         'split': split, 'status': session.get('status'),
                         'rgb_frames': session.get('frames_recorded'),
                         'meshed_frames': session.get('frames_meshed'),
                         'ir_depth': session.get('ir_depth'),
                         'tag_status': session.get('tag_status'),
                         'manifest_sha256': sha256_file(manifest) if manifest.is_file() else None})
    if not sessions:
        raise ValueError(f'no sessions under {root}')
    result = {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'scene_splits': scene_splits, 'sessions': sessions}
    write_json(root / 'dataset.json', result)
    print(f'Indexed {len(sessions)} sessions across {len(scene_splits)} scenes: {root / "dataset.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cap = sub.add_parser('capture', help='run synchronized hardware recorders on Frame')
    cap.add_argument('output', type=Path)
    cap.add_argument('--scene', required=True, help='stable room/scene ID; used for scene-disjoint splits')
    cap.add_argument('--split', choices=('development', 'tuning', 'held_out'), required=True)
    cap.add_argument('--notes', default='')
    cap.add_argument('--live-viewer', action='store_true', help='single-camera live Model TSDF viewer and recorder')
    cap.add_argument('--model-backend', choices=('prepared-qnn',), default='prepared-qnn',
                     help='live viewer depth executor; the QNN run list requires the checked 128 MB cdsp_mem lease')
    cap.add_argument('--controller-tip', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                     default=[0.0, 0.0, 0.0], help='controller-local tip offset in metres')
    cap.add_argument('--no-markers', action='store_true', help='record cameras and head pose without controller overlay')
    cap.add_argument('--frames', type=int, default=20, help='zero records until stopped (live viewer only)')
    cap.add_argument('--spacing-ms', type=int, default=350)
    cap.add_argument('--ir', action='store_true', help='record raw paired IR frames')
    cap.add_argument('--ir-spacing-ms', type=int, default=80)
    cap.add_argument('--ir-max-pairs', type=int, default=500)
    cap.add_argument('--min-free-gb', type=float, default=8, help='request a clean stop when free storage drops below this level')
    cap.add_argument('--dots', action='store_true', help='pulse BELICE while IR is recording')
    cap.add_argument('--pulse-every-frames', type=int, default=6, help='long exposures per pulse; 6 is 5 Hz')
    cap.add_argument('--pulse-ms', type=float, default=10)
    cap.add_argument('--pulse-lead-ms', type=float, help='pulse centre before the lit timestamp; default from scripts/ir_pulse_sweep.py')
    cap.add_argument('--sham-percent', type=int, default=10, help='pulses left dark as a false-positive control')
    cap.add_argument('--keep-floods', action='store_true', help='leave the headset IR floods in their current mode')
    cap.add_argument('--save-raw', action='store_true', help='also save full-resolution detiled RGB luma')
    cap.add_argument('--no-color', action='store_true',
                     help='skip the full-resolution NV12 colour (about 18 MB of the 19 MB per frame)')
    fin = sub.add_parser('finalize', help='run inference, pose alignment, tags and TSDF replay')
    fin.add_argument('output', type=Path)
    fin.add_argument('--python', type=Path, default=MODEL_PY)
    fin.add_argument('--package', type=Path, default=MODEL_PACKAGE)
    fin.add_argument('--converter', type=Path, default=(PATHS.repo / 'depth-runtime/bin/model_batch_to_v2'))
    fin.add_argument('--replay', type=Path, default=(PATHS.repo / 'depth-runtime/bin/replay_room_tsdf'))
    fin.add_argument('--tag-tool', type=Path, default=TAG_TOOL)
    fin.add_argument('--tag-size-m', type=float, default=0.12)
    fin.add_argument('--exposure-offset-ms', type=float, default=8.0)
    fin.add_argument('--skip-inference', action='store_true')
    fin.add_argument('--skip-tags', action='store_true')
    fin.add_argument('--ir-yaw-deg', type=float, help='per-session IR stereo yaw; enables IR-dot depth outputs')
    fin.add_argument('--ir-calib', type=Path, default=Path('/persist/xrservice.json'))
    fin.add_argument('--rgb-calib', type=Path, help='override captured RGB calibration history for IR projection')
    fin.add_argument('--ir-reference', choices=('average', 'before', 'none'), default='average',
                     help='dark long-exposure frames to subtract from each lit pair')
    fin.add_argument('--ir-exposure-offset-ms', type=float, default=None,
                     help='override the OG01A1B exposure offset; default is each pulse lead')
    check = sub.add_parser('validate', help='validate files and refresh SHA256 manifest')
    check.add_argument('output', type=Path)
    index = sub.add_parser('index', help='build a scene-disjoint dataset index from session directories')
    index.add_argument('root', type=Path)
    args = parser.parse_args()
    if args.command == 'capture': capture(args)
    elif args.command == 'finalize': finalize(args)
    elif args.command == 'validate': validate(args.output)
    else: index_dataset(args.root)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'room_dataset: {error}', file=sys.stderr)
        sys.exit(1)
