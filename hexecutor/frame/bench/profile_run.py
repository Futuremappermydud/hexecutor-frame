"""Profile run_frame: per-call pack / DSP / readback split and cProfile of ARM side."""
import cProfile, pstats, json, sys, time, os
from pathlib import Path
pkg = Path(sys.argv[1]); out = Path(sys.argv[2]); backend = sys.argv[3] if len(sys.argv) > 3 else 'hvx'
sys.path[:0] = [str(pkg / 'frame'), str(pkg / 'tools')]
import hvx_protocol, frame_hvx
records = []
cur = {}
_pack, _result = hvx_protocol.pack, hvx_protocol.result
def pack(arena, a, b, seq):
    t = time.monotonic(); r = _pack(arena, a, b, seq); cur['pack'] = time.monotonic() - t
    cur['shape'] = r[:3]; cur['t_dispatch'] = time.monotonic(); return r
def result(arena, spec, seq):
    t = time.monotonic(); cur['dsp'] = t - cur['t_dispatch']
    r = _result(arena, spec, seq); cur['readback'] = time.monotonic() - t; return r
import conv_protocol
_pc, _cr = conv_protocol.pack_conv, conv_protocol.conv_result
def pack_conv(arena, x, w, seq, fence=None):
    t = time.monotonic(); r = _pc(arena, x, w, seq, fence); cur['pack'] = cur.get('pack', 0) + time.monotonic() - t
    cur['shape'] = [list(x.shape), list(w.shape)]; cur['t_dispatch'] = time.monotonic(); return r
def conv_result(arena, L, seq):
    t = time.monotonic(); cur['dsp'] = cur.get('dsp', 0) + t - cur['t_dispatch']
    r = _cr(arena, L, seq); cur['readback'] = cur.get('readback', 0) + time.monotonic() - t; return r
import local_transport
local_transport.pack, local_transport.result = pack, result
local_transport.pack_conv, local_transport.conv_result = pack_conv, conv_result
if backend == 'hvx':
    import frame_transport
    frame_transport.pack, frame_transport.result = pack, result
    frame_transport.pack_conv, frame_transport.conv_result = pack_conv, conv_result
_acc = frame_hvx.HvxConvolution.accumulate
def accumulate(self, x, w, zp):
    t = time.monotonic(); r = _acc(self, x, w, zp)
    cur.pop('t_dispatch', None)
    cur['total'] = time.monotonic() - t; cur['xdtype'] = str(x.dtype); records.append(dict(cur)); cur.clear(); return r
frame_hvx.HvxConvolution.accumulate = accumulate
sys.argv = ['run_frame.py', '--smoke', '--output', str(out), '--backend', backend]
import run_frame
prof = cProfile.Profile()
t0 = time.monotonic()
try:
    prof.runcall(run_frame.main)
finally:
    print('wall', time.monotonic() - t0)
    (out / 'calls.json').write_text(json.dumps(records))
    st = pstats.Stats(prof); st.sort_stats('tottime'); st.print_stats(40)
    for k in ('pack', 'dsp', 'readback', 'total'):
        print(k, sum(r.get(k, 0) for r in records))
