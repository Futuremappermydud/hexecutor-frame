#!/usr/bin/env python3
"""Guided Frame capture and identical-input temporal-filter replay."""
import argparse
import datetime
import json
import pathlib
import re
import shlex
import shutil
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE = '/home/steamos/frame-spatial-runtime'
PROMPTS = {
    'warmup': 'Get ready. Look at the virtual panels.',
    'still': 'Hold your head still for eight seconds.',
    'head_turn': 'Slowly turn your head left and right. Keep your hands still.',
    'hand_motion': 'Hold your head still. Move your hand across a virtual panel, then away.',
    'recovery': 'Keep your head and hands still.',
    'finished': 'Recording finished.',
}


def ssh(host, args, **kwargs):
    return subprocess.run(['ssh', host, shlex.join(args)], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='frame')
    parser.add_argument('--matcher', choices=['baseline', 'guarded'], default='guarded')
    parser.add_argument('--frame-budget', type=int, default=512,
                        help='Maximum lossless frames; increase above 512 for runs faster than 17 Hz')
    parser.add_argument('--temporal', choices=['legacy', 'confidence'], default='legacy',
                        help='Filter shown live; replay compares both regardless')
    parser.add_argument('--confidence', nargs=6, type=float,
                        metavar=('STRONG', 'MEASURED', 'INFERRED', 'INFERRED_MS', 'MEASURED_MS', 'PENDING_MS'),
                        help='Confidence confirmation counts and expiry milliseconds, for live use and replay')
    parser.add_argument('--smoke', action='store_true', help='Three-second engineering check; no human quality evaluation')
    parser.add_argument('--download-recording', action='store_true', help='Also download the potentially 2.2 GB binary recording')
    args = parser.parse_args()
    if not 1 <= args.frame_budget <= 65536:
        parser.error('Frame budget must be in [1,65536]')
    if args.matcher == 'guarded' and (args.temporal != 'legacy' or args.confidence):
        parser.error('Guarded live mode uses legacy retention (2500 ms)')
    if args.confidence:
        counts, ages = args.confidence[:3], args.confidence[3:]
        if any(not 1 <= x <= 255 or x != int(x) for x in counts):
            parser.error('Confidence observation counts must be integers in [1,255]')
        if any(not 0 < x <= 1000 for x in ages):
            parser.error('Confidence expiry must be milliseconds in (0,1000]')
    confidence = [format(x, 'g') for x in args.confidence] if args.confidence else None
    print('Wear the headset in an ordinary room, with the virtual panels visible.\n'
          'The 30-second recording has a 3-second warmup, 8 seconds still,\n'
          '8 seconds of slow head turns, 8 seconds of hand motion, and 3 seconds recovery.\n'
          'Turn about 20 degrees each way. Move a hand closer than the panel, then remove it.\n'
          'Use ordinary surfaces; keep lighting unchanged between runs.')
    if args.smoke:
        print('SMOKE CHECK: phase names are placeholders; results are not a human motion benchmark.')
    else:
        input('Press Enter when ready to put the headset on: ')
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    local = ROOT / 'artifacts' / 'occlusion_benchmark' / stamp
    local.mkdir(parents=True)
    remote = REMOTE + '/captures/occlusion/' + stamp
    config = {'project': REMOTE, 'directory': remote, 'smoke': args.smoke,
              'matcher': args.matcher, 'temporal': args.temporal, 'confidence': confidence, 'frame_budget': args.frame_budget}
    launch = '''import hashlib,json,os,pathlib,shutil,signal,subprocess,sys,time
c=json.load(sys.stdin); p=c['project']; d=c['directory']; pathlib.Path(d).mkdir(parents=True)
required=88+c['frame_budget']*(180+616*616*11)+16
free=shutil.disk_usage(d).free
if free < required+256*1024*1024:
    raise RuntimeError('Insufficient disk space for declared lossless frame budget: need '+str(required)+' bytes plus 256 MiB reserve, available '+str(free))
# Stop only the occlusion demo, never the compositor or other applications.
old=subprocess.run(['pgrep','-f','^./bin/frame_openxr_point_cloud --occlusion-demo$'],capture_output=True,text=True)
pids=[int(x) for x in old.stdout.split()]; alive=[]
for pid in pids:
    try: os.kill(pid,signal.SIGTERM)
    except ProcessLookupError: pass
for _ in range(60):
    alive=[]
    for pid in pids:
        try: os.kill(pid,0); alive.append(pid)
        except ProcessLookupError: pass
    if not alive: break
    time.sleep(.1)
if alive: raise RuntimeError('Previous demo did not stop; refusing a second capture worker')
env=os.environ.copy(); env.update(OMP_NUM_THREADS='8',FRAME_OCCLUSION_TEMPORAL=c['temporal'],FRAME_OCCLUSION_BENCHMARK=d+'/recording.bin')
env['FRAME_OCCLUSION_STEREO']=c['matcher']
env['FRAME_OCCLUSION_PROFILE']='1'
env['FRAME_OPENXR_RENDER_PROFILE']='1'
env['FRAME_GUARDED_PROFILE']='1'
env.pop('FRAME_OCCLUSION_RECORD',None)
env['FRAME_OCCLUSION_FRAME_BUDGET']=str(c['frame_budget'])
if c['confidence']: env['FRAME_OCCLUSION_CONFIDENCE']=','.join(c['confidence'])
else: env.pop('FRAME_OCCLUSION_CONFIDENCE',None)
env['FRAME_OCCLUSION_COMPLETION']='1'
manifest=dict(c,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),omp_threads=8,completion=True,
              capture_uv=env.get('FRAME_CAPTURE_UV')=='1')
manifest['retention_policy']={'mode':c['temporal'],'legacy_max_age_ms':2500 if c['matcher']=='guarded' else 200}
manifest['guarded_settings']={key:env.get('FRAME_GUARDED_'+key,default) for key,default in
    [('ARMS','0'),('CACHE','0'),('BATCH_ROWS','8'),('YIELD_US','3000'),('LOCAL_X','8'),('LOCAL_Y','8')]}
manifest['map_generation']='runtime capture_generation and reset counters in runtime.log; each process starts empty'
manifest['calibration']='sha256 include/calibration.hpp; optical transform and intrinsics in recording header'
manifest['recording_budget_bytes']=required
manifest['free_disk_bytes_before_launch']=free
manifest['sha256']={}
for relative in ['bin/frame_openxr_point_cloud','bin/libframe_openxr_occlusion.so','shaders/stereo_match.spv','shaders/stereo_census.spv','shaders/stereo_filter.spv','shaders/stereo_support.spv','include/calibration.hpp','include/occlusion_surface.hpp','include/occlusion_completion.hpp','src/cloud_worker.cpp','src/openxr_occlusion_layer.cpp','src/occlusion_renderer.cpp','src/vulkan_stereo_pipeline.cpp','shaders/stereo_match.comp','shaders/stereo_filter.comp','shaders/stereo_support.comp']:
    manifest['sha256'][relative]=hashlib.sha256((pathlib.Path(p)/relative).read_bytes()).hexdigest()
for relative in ['src/vulkan_guarded_stereo_pipeline.cpp','include/vulkan_guarded_stereo_pipeline.hpp'] + [str(f.relative_to(p)) for f in (pathlib.Path(p)/'shaders').glob('stereo_guarded*')]:
    manifest['sha256'][relative]=hashlib.sha256((pathlib.Path(p)/relative).read_bytes()).hexdigest()
(pathlib.Path(d)/'manifest.json').write_text(json.dumps(manifest,indent=2))
if c['smoke']: env['FRAME_OCCLUSION_BENCHMARK_SMOKE']='1'
else: env.pop('FRAME_OCCLUSION_BENCHMARK_SMOKE',None)
with open(d+'/runtime.log','w') as log:
    proc=subprocess.Popen(['./scripts/run_with_occlusion.sh','./bin/frame_openxr_point_cloud','--occlusion-demo'],cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(proc.pid)
'''
    pid = ssh(args.host, ['python3', '-c', launch], input=json.dumps(config), text=True, capture_output=True).stdout.strip()
    print('Demo PID:', pid)
    seen = set()
    voice = None
    done = False
    deadline = time.monotonic() + (45 if args.smoke else 90)
    while time.monotonic() < deadline:
        result = ssh(args.host, ['cat', remote + '/runtime.log'], text=True, capture_output=True)
        log = result.stdout
        (local / 'runtime.log').write_text(log)
        for phase in re.findall(r'\[Benchmark\] phase=(\w+)', log):
            if phase not in seen:
                seen.add(phase)
                print(PROMPTS.get(phase, phase), flush=True)
                if not args.smoke and shutil.which('say'):
                    if voice and voice.poll() is None:
                        voice.terminate()
                    voice = subprocess.Popen(['say', PROMPTS.get(phase, phase)])
        if '[Benchmark] done ' in log:
            done = True
            break
        time.sleep(.5)
    if not done:
        raise RuntimeError('Recording did not finish; inspect ' + str(local / 'runtime.log'))
    replay = [REMOTE + '/bin/analyze_occlusion_benchmark', remote + '/recording.bin', remote + '/frames.csv']
    if confidence:
        replay.extend(['--confidence', *confidence])
    report = subprocess.run(
        ['ssh', args.host, shlex.join(replay)],
        text=True, capture_output=True)
    (local / 'summary.txt').write_text(('SMOKE CHECK ONLY\n' if args.smoke else '') + report.stdout + report.stderr)
    subprocess.run(['scp', args.host + ':' + remote + '/frames.csv', str(local / 'frames.csv')], check=True)
    subprocess.run(['scp', args.host + ':' + remote + '/manifest.json', str(local / 'manifest.json')], check=True)
    if args.download_recording:
        subprocess.run(['scp', args.host + ':' + remote + '/recording.bin', str(local / 'recording.bin')], check=True)
    print(report.stdout)
    print('Reports:', local)
    print('Full recording remains on Frame:', remote + '/recording.bin')
    print('The demo remains running with temporal filter:', args.temporal)
    if report.returncode:
        raise RuntimeError('Replay reported incomplete/lossy or invalid data; inspect summary.txt')


if __name__ == '__main__':
    main()
