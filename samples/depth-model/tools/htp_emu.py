#!/usr/bin/env python3
"""Memory-level NumPy emulator for the extracted Model stereo HTP graph.

The compiled graph is a sequence of tiled kernels that read and write tensors
through block tables in VTCM (pool 1), DDR scratch (pools 3, 4), the spill
area, and the constant pool (5). This emulator keeps those pools as byte
arrays and runs each executed node in the simulator's order with a NumPy
kernel. Shared blocks, halos and @DummyOp views then need no special
handling: they alias the same bytes, as on the device.

Kernels are calibrated against traces of QAIRT 2.50's x86 HTP simulator
(tools/trace_qnn_sim.py, tools/check_kernels.py). See KERNELS for coverage.
Needs NumPy.
"""
import json
from pathlib import Path

import numpy as np

import sim_trace

BLOCK = 2048
VTCM_BYTES = 8 << 20
SPILL_BYTES = 1_769_472
NP = {'uint8': np.uint8, 'uint16': np.uint16, 'int16': np.int16, 'int8': np.int8,
      'int32': np.int32, 'float32': np.float32, 'int32_quantized': np.int32}


# --------------------------------------------------------------------------
# Tensor addressing
# --------------------------------------------------------------------------

def is_crouton(t):
    return bool((t.get('storage') or {}).get('blocks')) and t['class_name'][0] in 'Cc'


def element_offsets(t, full=False, lanes=False):
    """(pools, byte_offsets) arrays shaped like the tensor's logical dimensions.

    full=True covers the whole padded extent (max_dimensions, from 0);
    lanes=True extends a crouton's depth to its 32-lane chunk boundary.
    """
    shape, store = t['shape'], t['storage']
    dims, maxd, pad = list(shape['dimensions']), shape['max_dimensions'], shape['padding']
    if full:
        dims, pad = maxd, [0] * len(maxd)
    elif lanes and is_crouton(t):
        dims[3] = min(-(-(dims[3] + pad[3]) // 32) * 32, maxd[3]) - pad[3]
    esize = np.dtype(NP[t['dtype']]).itemsize
    grids = np.meshgrid(*[np.arange(d) + p for d, p in zip(dims, pad)], indexing='ij')
    if store.get('direct'):
        full = list(maxd)
        lin = np.zeros(dims, np.int64)
        for g, m in zip(grids, full):
            lin = lin * m + g
        pool = np.full(dims, store['direct']['pool'], np.int64)
        return pool, store['direct']['offset'] + lin * esize
    b, h, w, d = grids
    if esize == 1:
        cw = 8
        within = (h % 8) * 256 + (w % 8) * 32 + d % 32
    elif esize == 2:
        cw = 4
        within = (h % 8) * 128 + ((w % 4) // 2) * 64 + (d % 32) * 2 + w % 2
    else:
        cw = 2
        within = (h % 8) * 64 + (w % 2) * 32 + d % 32
    nbh, nbw, nbd = -(-maxd[1] // 8), -(-maxd[2] // cw), -(-maxd[3] // 32)
    bidx = ((b * nbh + h // 8) * nbw + w // cw) * nbd + d // 32 + store.get('slice_offset', 0)
    table = store['blocks']
    tp = np.array([(e or {}).get('pool', 0) if isinstance(e, dict) and 'pool' in e else 0
                   for e in table] + [0], np.int64)
    to = np.array([(e or {}).get('offset', 0) if isinstance(e, dict) else 0 for e in table] + [0],
                  np.int64)
    bidx = np.minimum(bidx, len(table))
    return tp[bidx], to[bidx] + within * esize


def tensor_bytes(t):
    """Storage bytes of a direct (flat/weight) tensor."""
    return int(np.prod(t['shape']['max_dimensions'])) * np.dtype(NP[t['dtype']]).itemsize


class Memory:
    def __init__(self, graph, const_pool, ddr_bytes=None):
        self.tensors = {t['id']: t for t in graph['tensors']}
        ddr_bytes = ddr_bytes or {}
        need = {}
        for t in graph['tensors']:
            d = (t.get('storage') or {}).get('direct')
            if d and d['pool'] in (3, 4) and t.get('shape'):
                need[d['pool']] = max(need.get(d['pool'], 0), d['offset'] + tensor_bytes(t))
        self.pools = {1: np.zeros(VTCM_BYTES, np.uint8),
                      5: np.frombuffer(const_pool, np.uint8).copy(),
                      'spill': np.zeros(SPILL_BYTES, np.uint8)}
        for p, n in need.items():
            self.pools[p] = np.zeros(max(n, ddr_bytes.get(p, 0)), np.uint8)
        self._addr = {}

    def addr(self, tid, full=False, lanes=False):
        key = (tid, full, lanes)
        if key not in self._addr:
            self._addr[key] = element_offsets(self.tensors[tid], full, lanes)
        return self._addr[key]

    def read(self, tid, full=False):
        """Logical values; full=True returns the padded extent including halos."""
        t = self.tensors[tid]
        dt = np.dtype(NP[t['dtype']])
        pools, offs = self.addr(tid, full)
        out = np.zeros(offs.shape, dt)
        for p in np.unique(pools):
            if p == 0:
                continue
            sel = pools == p
            mem = self.pools[int(p)]
            o = offs[sel]
            raw = np.stack([mem[o + i] for i in range(dt.itemsize)], -1)
            out[sel] = raw.view(dt).reshape(-1)
        return out

    def write(self, tid, value):
        """Write logical values. For croutons, depth lanes up to the 32-lane chunk
        boundary are written too: zero-filled unless `value` already covers them
        (as HVX kernels do)."""
        t = self.tensors[tid]
        dt = np.dtype(NP[t['dtype']])
        pools, offs = self.addr(tid, lanes=True)
        value = np.asarray(value).astype(dt)
        if value.ndim == 4 and value.shape[3] < offs.shape[3]:
            value = np.concatenate([value, np.zeros(value.shape[:3] + (offs.shape[3] - value.shape[3],), dt)], 3)
        value = np.broadcast_to(value, offs.shape)
        raw = np.ascontiguousarray(value).view(np.uint8).reshape(offs.shape + (dt.itemsize,))
        for p in np.unique(pools):
            if p == 0:
                continue
            sel = pools == p
            mem = self.pools[int(p)]
            o = offs[sel]
            for i in range(dt.itemsize):
                mem[o + i] = raw[sel][:, i]

    def write_region(self, tid, value, origin=(0, 0, 0, 0)):
        """Write `value` at padded coordinates `origin` (may extend past the
        logical shape into padding, as whole-chunk HVX stores do)."""
        t = self.tensors[tid]
        dt = np.dtype(NP[t['dtype']])
        pools, offs = self.addr(tid, full=True)
        sl = tuple(slice(o, o + n) for o, n in zip(origin, value.shape))
        pools, offs = pools[sl], offs[sl]
        raw = np.ascontiguousarray(np.asarray(value).astype(dt)).view(np.uint8).reshape(offs.shape + (dt.itemsize,))
        for p in np.unique(pools):
            if p == 0:
                continue
            sel = pools == p
            for i in range(dt.itemsize):
                self.pools[int(p)][offs[sel] + i] = raw[sel][:, i]

    def raw_copy(self, src, dst):
        """Byte copy of a direct tensor's storage (DMA of weights/params)."""
        s, d = self.tensors[src], self.tensors[dst]
        sd, dd = s['storage']['direct'], d['storage']['direct']
        n = min(tensor_bytes(s), tensor_bytes(d))
        self.pools[dd['pool']][dd['offset']:dd['offset'] + n] = \
            self.pools[sd['pool']][sd['offset']:sd['offset'] + n]


# --------------------------------------------------------------------------
# Arithmetic helpers
# --------------------------------------------------------------------------

def q(t):
    qq = t.get('quantization') or {}
    return qq.get('scale', 1.0), qq.get('zero_point', 0)


def fp16_scale(word):
    """Fi multiplier word -> (integer mantissa m, total right shift S): M = m / 2^S."""
    h = int(word) & 0xffff
    e, mant = (h >> 10) & 31, h & 1023
    return 1024 + mant, 34 - e


def conv_requant(acc, mult_words, sixteen):
    """HTP convolution requantization with per-channel Fi multiplier words."""
    out = np.empty(acc.shape, np.int64)
    qmax = 65535 if sixteen else 255
    for c in range(acc.shape[-1]):
        m, S = fp16_scale(mult_words[c])
        p = 8 if sixteen else S - 14
        s = S - p
        a = acc[..., c] >> p
        out[..., c] = (a * m + (1 << (s - 1))) >> s
    return np.clip(out, 0, qmax)


# --------------------------------------------------------------------------
# Kernels: fn(emu, node) using emu.mem.read/write
# --------------------------------------------------------------------------

KERNELS = {}


def kernel(*names):
    def reg(fn):
        for n in names:
            KERNELS[n] = fn
        return fn
    return reg


class Emulator:
    def __init__(self, graph, const_pool, inputs, conv_backend=None):
        self.graph = graph
        self.mem = Memory(graph, const_pool)
        self.t = self.mem.tensors
        self.inputs = inputs          # graph input index -> numpy array (native layout)
        self.conv_backend = conv_backend
        calls = Path(graph.get('_recovered', '.')) / 'kernel_calls.json'
        call_data = json.loads(calls.read_text()) if calls.exists() else {}
        if call_data.get('context_sha256', graph['context_sha256']) != graph['context_sha256']:
            raise ValueError('kernel arguments belong to a different context')
        self.calls = {int(k): v for k, v in call_data.get('calls', {}).items()}
        self.outputs = {}             # graph output index -> numpy array
        self.executed = set()

    def sv(self, tid):
        t = self.t[tid]
        if 'shape_values' in t:
            return t['shape_values']
        return t.get('scalar')

    def run_node(self, node):
        fn = KERNELS.get(node['type_name'])
        if fn is None:
            raise NotImplementedError(node['type_name'])
        fn(self, node)


@kernel('@SyncOp', '@DmaCheckpointSet', '@DmaCheckpointWait')
def k_nop(emu, node):
    pass


@kernel('*InputSlicePad@FB.s4*3.t.s4*3.')
def k_input_slice_pad(emu, node):
    if node['sequence'] in emu.executed:
        return  # second listing of the same node is a no-op
    i = node['inputs']
    pad_before, start, out_shape = emu.sv(i[0]), emu.sv(i[1]), emu.sv(i[2])
    fill = emu.sv(i[3]) or 0
    img = emu.inputs[node['graph_input_index']].reshape(emu.sv(i[6]))  # NHWC
    out = np.full(out_shape, fill, img.dtype)
    lo = [s - p for s, p in zip(start, pad_before)]  # input coords of out[0]
    src = [slice(max(l, 0), min(l + n, d)) for l, n, d in zip(lo, out_shape, img.shape)]
    dst = [slice(s.start - l, s.stop - l) for s, l in zip(src, lo)]
    out[tuple(dst)] = img[tuple(src)]
    emu.mem.write(node['outputs'][0], out)


@kernel('*InputSlice@Ff.s4*6.')
def k_input_slice(emu, node):
    out = emu.t[node['outputs'][0]]
    img = emu.inputs[node['graph_input_index']].reshape(out['shape']['dimensions'])
    emu.mem.write(node['outputs'][0], img)


@kernel('ConvLayer.opt.weights_to_vtcm@Xpb.xpb.', 'ConvLayer.opt.weights_to_vtcm@Fi.fi.',
        'ConvLayer.opt.weights_to_vtcm@Xwb.xwb.', 'ConvLayer.opt.weights_to_vtcm@FH.fH.')
def k_weights_to_vtcm(emu, node):
    emu.mem.raw_copy(node['inputs'][0], node['outputs'][0])


@kernel('ConvLayer.opt.inconv_im2col.tcm@CB.FB.s4.t', 'ConvLayer.opt.inconv_im2col.tcm@CH.FH.s4.t')
def k_im2col(emu, node):
    x = emu.mem.read(node['inputs'][0])[0]
    kh, kw, cin, _ = emu.sv(node['inputs'][1])
    _, sh, sw, _ = emu.sv(node['inputs'][2])
    out_t = emu.t[node['outputs'][0]]
    oh, ow = out_t['shape']['dimensions'][1:3]
    cols = []
    for y in range(kh):
        for xx in range(kw):
            cols.append(x[y:y + sh * oh:sh, xx:xx + sw * ow:sw, :cin])
    out = np.concatenate(cols, -1)
    emu.mem.write(node['outputs'][0], out[None, :, :, :out_t['shape']['dimensions'][3]])


# --------------------------------------------------------------------------
# HMX convolutions
#
# Weight layouts, from the SDK's SWeights/pkSWeights element_addr (see
# export_constants.weight_offset):
#   Xpb [O/32,H,W,I/4,O%32,I%4]            (packed; 1x1 / im2col layers)
#   Xwb [O/32,I/32,H,W,(I%32)/4,O%32,I%4]  (3x3 layers)
# Calibrated against the simulator (tools/check_kernels.py):
#   * stride-1 3x3 Xwb: the stored kernel's x axis is reversed,
#     true[ky, kx] = stored[ky, kw-1-kx];
#   * stride-2 Xwb: the nine HxW slots hold pixel taps in phase order S2_TAPS,
#     and the activation arrives space-to-depth'd (see k_space_to_depth).
# --------------------------------------------------------------------------

S2_TAPS = [(0, 2), (0, 0), (2, 2), (2, 0), (0, 1), (2, 1), (1, 2), (1, 0), (1, 1)]


def read_weights(mem, tid):
    t = mem.tensors[tid]
    H, W, I, O = t['shape']['max_dimensions']
    d = t['storage']['direct']
    raw = mem.pools[d['pool']][d['offset']:].view(np.int8)
    h, w, i, o = np.meshgrid(*map(np.arange, (H, W, I, O)), indexing='ij')
    if t['class_name'] in ('Xpb', 'xpb'):
        off = (((o // 32 * H + h) * W + w) * (I // 4) + i // 4) * 128 + (o % 32) * 4 + i % 4
    else:
        off = ((((o // 32 * (I // 32) + i // 32) * H + h) * W + w) * 1024
               + ((i // 4) % 8) * 128 + (o % 32) * 4 + i % 4)
    return raw[off]


def read_fi(mem, tid):
    """Fi rows of 64 int32: [multiplier word x32, bias x32] per 32 output channels."""
    t = mem.tensors[tid]
    d = t['storage']['direct']
    n = int(np.prod(t['shape']['max_dimensions']))
    rows = mem.pools[d['pool']][d['offset']:d['offset'] + 4 * n].view(np.int32).reshape(-1, 64)
    return rows[:, :32].reshape(-1), rows[:, 32:].reshape(-1)


def conv_accumulate(x, w, zp_in, stride=1):
    # Every product and partial sum is integral and below 2**53. Float64 BLAS
    # therefore gives exact integer accumulators, much faster than NumPy's
    # scalar int64 matmul. Requantization below still uses integer arithmetic.
    x = x.astype(np.float64) - zp_in
    kh, kw, ci, co = w.shape
    oh, ow = (x.shape[0] - kh) // stride + 1, (x.shape[1] - kw) // stride + 1
    acc = np.zeros((oh, ow, co), np.float64)
    assert np.max(np.abs(x), initial=0) * np.max(np.abs(w.astype(np.float64)), initial=0) * kh * kw * ci < 2 ** 53
    w = w.astype(np.float64)
    for ky in range(kh):
        for kx in range(kw):
            acc += x[ky:ky + stride * oh:stride, kx:kx + stride * ow:stride, :ci] @ w[ky, kx]
    return acc.astype(np.int64)


def s2_kernel(w):
    """True 3x3 stride-2 HWIO weights -> 2x2 kernel over the space-to-depth input.

    Input channel block j = cb*4 + 2*dy + dx holds pixel (2r+dy, 2c-dx) of input
    channels [32cb, 32cb+32); kernel tap (a, b) reads s2d[h+a, w+b].
    """
    kh, kw, ci, co = w.shape
    K = np.zeros((2, 2, 4 * ci, co), np.int64)
    for a in range(2):
        for b in range(2):
            for dy in range(2):
                for dx in range(2):
                    ky, kx = 2 * a + dy, 2 * b - dx
                    if ky < kh and 0 <= kx < kw:
                        for cb in range(ci // 32):
                            j = cb * 4 + 2 * dy + dx
                            K[a, b, 32 * j:32 * j + 32] = w[ky, kx, 32 * cb:32 * cb + 32]
    return K


def conv_input(mem, tid, depth):
    """Activation tile, including channel lanes beyond the logical depth if needed."""
    t = mem.tensors[tid]
    x = mem.read(tid)[0]
    if depth > x.shape[2]:
        p, dims = t['shape']['padding'], t['shape']['dimensions']
        x = mem.read(tid, full=True)[0][p[1]:p[1] + dims[1], p[2]:p[2] + dims[2], :depth]
    return x


@kernel('ConvLayer_s1.opt@CB*2.Xwb.Fi.t', 'ConvLayer_s1.opt@CB*2.Xpb.Fi.t',
        'ConvLayer_s1.opt@CH*2.Xwb.Fi.t', 'ConvLayer_s1.opt@CH*2.Xpb.Fi.t',
        'ConvLayer_s2.opt@CB*2.Xwb.Fi')
def k_hmx_conv(emu, node):
    xi, wi, fi = node['inputs'][:3]
    out = emu.t[node['outputs'][0]]
    w = read_weights(emu.mem, wi)
    s2 = node['type_name'].startswith('ConvLayer_s2')
    if s2:
        flat = w.reshape(9, *w.shape[2:])
        true = np.zeros_like(w)
        for k, (r, c) in enumerate(S2_TAPS):
            true[r, c] = flat[k]
        w = s2_kernel(true)
    elif emu.t[wi]['class_name'] == 'Xwb':
        w = w[:, ::-1]
    mult, bias = read_fi(emu.mem, fi)
    sixteen = emu.t[xi]['class_name'] in ('CH', 'cH')
    # HMX writes whole output chunks (8 rows x 8|4 columns x 32 lanes): extend the
    # logical output as far as the input tile's padded extent allows.
    oh, ow, co = out['shape']['dimensions'][1:]
    omax = out['shape']['max_dimensions']
    co = min(-(-co // 32) * 32, w.shape[3], len(mult))
    xs = emu.t[xi]['shape']
    xf = emu.mem.read(xi, full=True)[0]
    x = xf[xs['padding'][1]:, xs['padding'][2]:, :w.shape[2]]
    cw = 4 if sixteen else 8
    oh = min(-(-oh // 8) * 8, omax[1], x.shape[0] - w.shape[0] + 1)
    ow = min(-(-ow // cw) * cw, omax[2], x.shape[1] - w.shape[1] + 1)
    accumulate = emu.conv_backend.accumulate if emu.conv_backend is not None else conv_accumulate
    acc = accumulate(x[:oh + w.shape[0] - 1, :ow + w.shape[1] - 1], w[..., :co], q(emu.t[xi])[1])
    acc += bias[:co].astype(np.int64) * (256 if sixteen else 1)
    emu.mem.write_region(out['id'], conv_requant(acc, mult, sixteen)[None], (0,) + tuple(out['shape']['padding'][1:3]) + (0,))


@kernel('ConvLayer_SpaceToDepth_stride2.tcm@CB*2.s4')
def k_space_to_depth(emu, node):
    """out[h, w, 32*(cb*4 + 2*dy + dx) + c] = in[2h+dy, 2w-dx, 32*cb + c]."""
    i, o = node['inputs'][0], node['outputs'][0]
    ti, to = emu.t[i], emu.t[o]
    xf = emu.mem.read(i, full=True)[0]
    ph, pw = ti['shape']['padding'][1:3]
    oh, oc = to['shape']['dimensions'][1], to['shape']['dimensions'][3]
    # HVX writes whole chunks: extend the width as far as the input extent allows.
    ow = min(to['shape']['max_dimensions'][2], (xf.shape[1] - pw) // 2)
    out = np.zeros((oh, ow, oc), xf.dtype)
    ci = xf.shape[2]
    for dy in range(2):
        for dx in range(2):
            src = xf[ph + 2 * np.arange(oh) + dy][:, pw + 2 * np.arange(ow) - dx]
            for cb in range(-(-ci // 32)):
                j = cb * 4 + 2 * dy + dx
                if 32 * j < oc:
                    out[:, :, 32 * j:32 * j + 32] = src[:, :, 32 * cb:32 * cb + 32]
    emu.mem.write_region(o, out[None])


# --------------------------------------------------------------------------
# Elementwise kernels. Each mirrors one exported HVX kernel of the 2.50 x86
# simulator; the integer arguments are the per-node compiled constants in
# kernel_calls.json. Formulas were fitted on direct calls of the real kernels
# (sim/kcall.cpp): exhaustively for 8-bit and unary 16-bit kernels, on ~1M
# samples per argument set for binary 16-bit ones, and the multiply by
# replaying its identified HVX intrinsic sequence.
# --------------------------------------------------------------------------

def _s32(v):
    return ((np.asarray(v, np.int64) + 2 ** 31) % 2 ** 32) - 2 ** 31


def requantize_u8(x, P, r8, r9):
    assert r8 == 0 and r9 == 0, 'requantize_crouton_u8 offsets not calibrated'
    m, s = P & 0xffff, 15 - (P >> 24)
    return np.clip((x.astype(np.int64) * m + (1 << (s - 1))) >> s, 0, 255)


def requantize_u16(x, P, zp, _flag):
    m, s = P & 0xffff, 15 - (P >> 24)
    zin, zout = zp & 0xffff, zp >> 16
    return np.clip((((x.astype(np.int64) - zin) * m + (1 << (s - 1))) >> s) + zout, 0, 65535)


def add_u8(a, b, ma, mb, sh, off, _k):
    assert off == 0, 'elementwise_add_crouton_u8 offset not calibrated'
    S = sh + 8
    return np.clip((a.astype(np.int64) * ma + b.astype(np.int64) * mb + (1 << (S - 1))) >> S, 0, 255)


def add_u16(a, b, ma, mb, sh, off, _k):
    assert off == 0, 'elementwise_add_crouton_u16 offset not calibrated'
    return np.clip((a.astype(np.int64) * ma + b.astype(np.int64) * mb + (1 << sh)) >> (sh + 1), 0, 65535)


def addsub_u16(a, b, ma, mb, sh, zp_out):
    """a*ma - b*mb, rounded half up after a +1 bias; zero point from the output.

    The constant was pinned exactly (a single admissible value) for all three
    argument sets in this graph; its derivation from the kernel's own offset
    argument is not established.
    """
    S = sh + 1
    C = (zp_out << S) + (1 << sh) + 1
    return np.clip((a.astype(np.int64) * ma - b.astype(np.int64) * mb + C) >> S, 0, 65535)


def add1111_u16(x, scalar, ma, mb, sh, off):
    S = sh + 1
    return np.clip((x.astype(np.int64) * ma + scalar * mb + 2 * _s32(off) + (1 << sh)) >> S, 0, 65535)


def mul_u16(a, b, P1, zb, za, K, P2):
    """elementwise_mul_crouton_u16 as its HVX sequence: uh*uh -> uw products,
    minus zero-point cross terms, + K, Q31-scaled by P1 (vmpye + vmpyo:<<1 acc),
    + P2 (sat), then vasr:rnd:sat by 3 to uh."""
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    D = _s32(a * b + K - (b * (zb & 0xffff) + a * (za & 0xffff)))
    lo, hi = P1 & 0xffff, _s32(P1) >> 16
    e = (D * lo) >> 16
    qv = np.clip((2 * e + 2 * D * hi) >> 16, -2 ** 31, 2 ** 31 - 1)
    s = np.clip(qv + _s32(P2), -2 ** 31, 2 ** 31 - 1)
    return np.clip((s + 4) >> 3, 0, 65535)


def call_args(emu, node, func):
    calls = [c for c in emu.calls.get(node['sequence'], []) if c[0] == func]
    if len(calls) != 1:
        raise NotImplementedError(f'{node["type_name"]}: expected one {func} call')
    return calls[0][1], calls[0][2]


def read_scalar(emu, tid):
    v = emu.mem.read(tid)
    return int(v.reshape(-1)[0])


@kernel('linearclip_simple_rq@CB*2.fi')
def k_requant_u8(emu, node):
    regs, _ = call_args(emu, node, 'requantize_crouton_u8')
    x = emu.mem.read(node['inputs'][0])
    emu.mem.write(node['outputs'][0], requantize_u8(x, *regs[3:6]))


@kernel('linearclip_simple_rq@CH*2.fi')
def k_requant_u16(emu, node):
    regs, _ = call_args(emu, node, 'requantize_crouton_u16')
    x = emu.mem.read(node['inputs'][0])
    emu.mem.write(node['outputs'][0], requantize_u16(x, *regs[3:6]))


@kernel('Add.tcm@CB*3')
def k_add_u8(emu, node):
    regs, stack = call_args(emu, node, 'elementwise_add_crouton_u8')
    a, b = (emu.mem.read(i) for i in node['inputs'][:2])
    emu.mem.write(node['outputs'][0], add_u8(a, b, regs[4], regs[5], *stack[:3]))


@kernel('Add.tcm@CH*3')
def k_add_u16(emu, node):
    regs, stack = call_args(emu, node, 'elementwise_add_crouton_u16')
    a, b = (emu.mem.read(i) for i in node['inputs'][:2])
    emu.mem.write(node['outputs'][0], add_u16(a, b, regs[4], regs[5], *stack[:3]))


@kernel('addsub_op@CH*3.fi')
def k_addsub_u16(emu, node):
    regs, stack = call_args(emu, node, 'elementwise_addsub_crouton_u16')
    a, b = (emu.mem.read(i) for i in node['inputs'][:2])
    zp = q(emu.t[node['outputs'][0]])[1]
    emu.mem.write(node['outputs'][0], addsub_u16(a, b, regs[4], regs[5], stack[0], zp))


@kernel('addsub_op@CH*2.fH.fi')
def k_add1111_u16(emu, node):
    regs, stack = call_args(emu, node, 'elementwise_add1111_crouton_u16')
    x = emu.mem.read(node['inputs'][0])
    s = read_scalar(emu, node['inputs'][1])
    emu.mem.write(node['outputs'][0], add1111_u16(x, s, regs[4], regs[5], stack[0], stack[1]))


@kernel('mul_op_crt@CH*3.fi')
def k_mul_u16(emu, node):
    regs, stack = call_args(emu, node, 'elementwise_mul_crouton_u16')
    a, b = (emu.mem.read(i) for i in node['inputs'][:2])
    emu.mem.write(node['outputs'][0], mul_u16(a, b, regs[4], regs[5], *stack[:3]))


@kernel('mul_op@CH*2.fH.fi')
def k_mul1111_u16(emu, node):
    regs, stack = call_args(emu, node, 'elementwise_mul1111_crouton_u16')
    x = emu.mem.read(node['inputs'][0])
    s = read_scalar(emu, node['inputs'][1])
    emu.mem.write(node['outputs'][0], mul_u16(x, np.int64(s), regs[4], regs[5], *stack[:3]))


# --------------------------------------------------------------------------
# Data movement and graph I/O
# --------------------------------------------------------------------------

@kernel('flat_to_vtcm@Ff.ff.', 'flat_from_vtcm@fH.FH.',
        'ForceFormat_Crouton_f2c@CH.FH', 'ForceFormat_Crouton_f2c@CB.FB',
        'ForceFormat_Flat_c2f@FB.CB', 'ForceFormat_Flat_c2f@FH.CH',
        'ForceFormat_Crouton@CB*2', 'Reshape.hw@CB*2', 'Reshape@FB.CB')
def k_copy_values(emu, node):
    o = node['outputs'][0]
    emu.mem.write(o, emu.mem.read(node['inputs'][0]).reshape(emu.t[o]['shape']['dimensions']))


@kernel('ConvLayer.opt.activations_to_vtcm_dma@CB.cB.',
        'ConvLayer.opt.activations_to_vtcm_dma@CH.cH.',
        'ConvLayer.opt.activations_from_vtcm@cH.CH.')
def k_copy_activation(emu, node):
    i, o = node['inputs'][0], node['outputs'][0]
    emu.mem.write_region(o, emu.mem.read(i, full=True))


@kernel('Transpose.2D@CB*2.t', 'Transpose.2D@CH*2.t',
        'Transpose_impl@FB*2.fi.t', 'Transpose_impl@FH*2.fi.t')
def k_transpose(emu, node):
    perm = emu.mem.read(node['inputs'][-1]).reshape(-1)
    emu.mem.write(node['outputs'][0], emu.mem.read(node['inputs'][0]).transpose(perm))


@kernel('Concat@FB.t.FB*2')
def k_concat(emu, node):
    axis = emu.sv(node['inputs'][0])
    emu.mem.write(node['outputs'][0], np.concatenate([emu.mem.read(i) for i in node['inputs'][1:]], axis))


@kernel('Gather@FB*2.fi*3')
def k_gather(emu, node):
    i, axis, indices, _ = node['inputs']
    a = int(emu.mem.read(axis).reshape(-1)[0])
    emu.mem.write(node['outputs'][0], np.take(emu.mem.read(i), emu.mem.read(indices).reshape(-1), axis=a))


@kernel('Slice_contig.tcm@FH*2.s4*2')
def k_slice(emu, node):
    i, start, size = node['inputs']
    sl = tuple(slice(s, s + n) for s, n in zip(emu.sv(start), emu.sv(size)))
    emu.mem.write(node['outputs'][0], emu.mem.read(i)[sl])


@kernel('SlicePad_shape_inplace@FB*2.s4*3.cB', 'SlicePad_shape_inplace@CB*2.s4*3.CB')
def k_slice_pad(emu, node):
    i, pad, start, size, fill = node['inputs']
    x = emu.mem.read(i)
    lo = [s - p for s, p in zip(emu.sv(start), emu.sv(pad))]
    out = np.full(emu.sv(size), read_scalar(emu, fill), x.dtype)
    src = [slice(max(l, 0), min(l + n, d)) for l, n, d in zip(lo, out.shape, x.shape)]
    dst = [slice(s.start - l, s.stop - l) for s, l in zip(src, lo)]
    out[tuple(dst)] = x[tuple(src)]
    emu.mem.write(node['outputs'][0], out)


@kernel('Decimate.tcm@CB*2.s4')
def k_decimate(emu, node):
    x = emu.mem.read(node['inputs'][0])
    stride = emu.sv(node['inputs'][1])
    emu.mem.write(node['outputs'][0], x[tuple(slice(None, None, s) for s in stride)])


@kernel('ResizeNear_slice.tcm_special@CB*2.fi*2')
def k_resize(emu, node):
    x = emu.mem.read(node['inputs'][0])
    hp, wp = (emu.mem.read(i).reshape(-1) for i in node['inputs'][1:])
    shape = emu.t[node['outputs'][0]]['shape']['dimensions']
    # Q32 source step (word 2), fractional source origin (word 4), and
    # integer source origin (word 5). The models differ in half-pixel origin.
    h = ((np.arange(shape[1]) * (int(hp[2]) & 0xffffffff)
          + (int(hp[4]) & 0xffffffff)) >> 32) + int(hp[5])
    w = ((np.arange(shape[2]) * (int(wp[2]) & 0xffffffff)
          + (int(wp[4]) & 0xffffffff)) >> 32) + int(wp[5])
    emu.mem.write(node['outputs'][0], x[:, h][:, :, w])


@kernel('d2s.shortD.flatout@FH.CH.fi')
def k_depth_to_space(emu, node):
    x = emu.mem.read(node['inputs'][0])
    rh, rw = emu.mem.read(node['inputs'][1]).reshape(-1)
    b, h, w, c = x.shape
    out = x.reshape(b, h, w, rh, rw, c // (rh * rw)).transpose(0, 1, 3, 2, 4, 5)
    emu.mem.write(node['outputs'][0], out.reshape(b, h * rh, w * rw, -1))


@kernel('@Spill', '@Fill')
def k_spill_fill(emu, node):
    v = node['framework_vector']
    offset = v[3]
    for p, n in zip(v[4::2], v[5::2]):
        pool, addr = p & 0x7ff, p & ~0x7ff
        src, dst = emu.mem.pools[pool][addr:addr + n], emu.mem.pools['spill'][offset:offset + n]
        if node['type_name'] == '@Spill':
            dst[:] = src
        else:
            src[:] = dst
        offset += n


@kernel('*OutputSlice@CB.s4*3.', '*OutputSlice@Ff.s4*3.')
def k_output_slice(emu, node):
    i, shape, start, _ = node['inputs']
    x = emu.mem.read(i)
    idx = node['graph_output_index']
    if idx not in emu.outputs:
        emu.outputs[idx] = np.zeros(emu.sv(shape), x.dtype)
    emu.outputs[idx][tuple(slice(s, s + n) for s, n in zip(emu.sv(start), x.shape))] = x


@kernel('Quantize@FH.Ff')
def k_quantize(emu, node):
    o = node['outputs'][0]
    scale, zp = q(emu.t[o])
    x = emu.mem.read(node['inputs'][0]).astype(np.float64)
    # Simulator's conversion retains the exact product of the two float32
    # operands until rounding; a float32 intermediate changes half-way cases.
    emu.mem.write(o, np.clip(np.floor(x * np.float32(1 / scale) + 0.5) + zp, 0, 65535))


@kernel('Dequantize@Ff.CH')
def k_dequantize(emu, node):
    i = node['inputs'][0]
    scale, zp = q(emu.t[i])
    exact = (emu.mem.read(i).astype(np.float64) - zp) * float(np.float32(scale))
    rounded = exact.astype(np.float32)
    away = np.nextafter(rounded, np.where(exact < 0, np.float32(-np.inf), np.float32(np.inf)))
    # HVX's qfloat-to-sfloat conversion rounds exact halfway products away
    # from zero. NumPy's default ties-to-even loses these alternate-model bits.
    ties = exact == (rounded.astype(np.float64) + away.astype(np.float64)) * 0.5
    emu.mem.write(node['outputs'][0], np.where(ties, away, rounded))


@kernel('ElementWiseSign@FH*2', 'ElementWiseSign@CH*2')
def k_sign(emu, node):
    i, o = node['inputs'][0], node['outputs'][0]
    si, zi = q(emu.t[i]); so, zo = q(emu.t[o])
    x = np.sign(emu.mem.read(i).astype(np.int64) - zi)
    emu.mem.write(o, np.clip(np.floor(x / so + zo + 0.5), 0, 65535))


@kernel('Mean@FH*2.fi.t')
def k_mean(emu, node):
    x = emu.mem.read(node['inputs'][0]).astype(np.int64)
    # The 32-element reduction uses the simulator's two-stage rounding. Direct
    # calls pin its aggregate rounding bias to 33 (ordinary mean uses 16).
    assert x.size == 32 and q(emu.t[node['inputs'][0]]) == q(emu.t[node['outputs'][0]])
    emu.mem.write(node['outputs'][0], np.clip((x.sum() + 33) >> 5, 0, 65535))


@kernel('ConvLayer.opt.conv_scale_from_weights@ff.FB')
def k_weight_scales(emu, node):
    i, o = node['inputs'][0], node['outputs'][0]
    x = emu.mem.read(i).astype(np.int32) - q(emu.t[i])[1]
    a = np.maximum(-127 * x.min(axis=(0, 1, 2)), 128 * x.max(axis=(0, 1, 2)))
    scale = np.maximum(a.astype(np.float32) / np.float32(16256), np.float32(1))
    # The simulator rounds the float mantissa to 11 fractional bits.
    bits = (scale.view(np.uint32) + np.uint32(0x800)) & np.uint32(0xfffff000)
    emu.mem.write(o, bits.view(np.float32).reshape(emu.t[o]['shape']['dimensions']))


@kernel('invscale_to_qi32@fs.ff')
def k_invscale(emu, node):
    x = emu.mem.read(node['inputs'][0])
    emu.mem.write(node['outputs'][0], np.trunc(np.float32(65536) / x + np.float32(0.5)))


@kernel('ConvLayer.opt.combine_scales@ff*3.t')
def k_combine_scales(emu, node):
    a, b, c = node['inputs']
    emu.mem.write(node['outputs'][0], emu.mem.read(a) * emu.mem.read(b) * np.float32(emu.sv(c)))


@kernel('ConvLayer.opt.convert_weights_to_signed.packed@xpb.FB.t.fs.t')
def k_convert_weights(emu, node):
    i, _, scale, _ = node['inputs']
    o = node['outputs'][0]
    x = emu.mem.read(i).astype(np.int64) - q(emu.t[i])[1]
    s = emu.mem.read(scale).astype(np.int64)
    w = np.clip((x * s + 32768) >> 16, -128, 127).astype(np.int8)
    H, W, I, O = w.shape
    raw = w.reshape(H, W, I // 4, 4, O // 32, 32).transpose(4, 0, 1, 2, 5, 3).copy().view(np.uint8).reshape(-1)
    d = emu.t[o]['storage']['direct']
    emu.mem.pools[d['pool']][d['offset']:d['offset'] + len(raw)] = raw


@kernel('ConvLayer.opt.scale_convert@fi.ff.t')
def k_scale_convert(emu, node):
    x = emu.mem.read(node['inputs'][0])
    h = (x.view(np.uint32).astype(np.int64) + 0x1000) >> 13
    word = (h & 0x3ff) + (((h & 0x7c00) + 0x6400) & 0x7c00) + 0x400000
    emu.mem.write(node['outputs'][0], word)


@kernel('ConvLayer.opt.convert_bias.simple@Fs.s4.Ff*2.t.ni')
def k_dynamic_bias(emu, node):
    # This graph's dynamic bmm has no bias and zero input/output zero points.
    assert all(emu.sv(i) == 0 for i in node['inputs'][3:])
    emu.mem.write(node['outputs'][0], 0)


@kernel('ConvLayer.opt.bias_scale_shuff@Fi.fi.Fs.t')
def k_bias_shuffle(emu, node):
    a, b, mode = node['inputs']
    assert emu.sv(mode) == 0
    mult, bias = emu.mem.read(a).reshape(-1, 32), emu.mem.read(b).reshape(-1, 32)
    out = np.stack((mult, bias), axis=1).reshape(emu.t[node['outputs'][0]]['shape']['dimensions'])
    emu.mem.write(node['outputs'][0], out)


@kernel('linearclip@FB*2.fi', 'linearclip@CH.CB.fi', 'linearclip@CH*2.fi')
def k_linearclip(emu, node):
    i, params = node['inputs']; o = node['outputs'][0]
    p = emu.mem.read(params).reshape(-1)
    x = emu.mem.read(i).astype(np.int64)
    if emu.t[i]['dtype'] == 'uint8' and emu.t[o]['dtype'] == 'uint16':
        # Widening path uses a signed Q15 multiplier and exponent in word 1.
        m = int(p[1]) & 0xffff
        sh = 7 - (int(p[1]) >> 16)
        y = (x * m + (1 << (sh - 1))) >> sh
    else:
        P, zp = int(p[10]) & 0xffffffff, int(p[11]) & 0xffffffff
        m, sh = P & 0xffff, 15 - (P >> 24)
        m = -m if p[6] < 0 else m
        if emu.t[i]['dtype'] == 'uint8':
            # Two rounded shifts (Q15 multiply, then narrowing) cause an
            # additional half-unit at the intermediate precision.
            assert sh == 14 and zp == 0 and m >= 0
            y = (((x * m + 128) >> 8) + 32) >> 6
        else:
            y = (((x - (zp & 0xffff)) * m + (1 << (sh - 1))) >> sh) + (zp >> 16)
    emu.mem.write(o, np.clip(y, 0, np.iinfo(NP[emu.t[o]['dtype']]).max))


@kernel('Abs@FH*2', 'Abs@CH*2')
def k_abs(emu, node):
    i, o = node['inputs'][0], node['outputs'][0]
    si, zi = q(emu.t[i]); so, zo = q(emu.t[o])
    ratio = np.float32(si) * np.float32(1 / so)
    mult = int(np.rint(np.float32(ratio * 262144)))
    offset = int(np.rint(np.float32((np.float32(ratio * 32768) + zo) * 8)))
    x = emu.mem.read(i).astype(np.int64)
    _, byteoff = emu.mem.addr(i)
    rounded_lane = (byteoff // 2) % 2 == 0
    centered = np.abs(x - zi) - 32768
    y = ((centered * mult + rounded_lane * 16384) >> 15) + offset
    emu.mem.write(o, np.clip((y + 4) >> 3, 0, 65535))


@kernel('mul_op@CH*2.cH.fi', 'mul_op_flat@FH*2.fH.fi')
def k_mul_inline(emu, node):
    i, scalar, params = node['inputs']
    p = emu.mem.read(params).reshape(-1).astype(np.int64)
    x = emu.mem.read(i)
    z = int(p[3]) & 0xffffffff
    emu.mem.write(node['outputs'][0], mul_u16(x, np.int64(read_scalar(emu, scalar)),
                                           int(p[2]), z & 65535, z >> 16, int(p[4]), int(p[5])))


@kernel('addsub_op_flat@FH*3.fi')
def k_addsub_flat(emu, node):
    calls = emu.calls[node['sequence']]
    fn, regs, stack = calls[0]
    a, b = (emu.mem.read(i).astype(np.int64) for i in node['inputs'][:2])
    sign = -1 if fn == 'elementwise_sub_flat_u16' else 1
    sh = stack[0]
    y = a * regs[4] + sign * b * regs[5] + 2 * _s32(stack[1]) + (1 << sh)
    emu.mem.write(node['outputs'][0], np.clip(y >> (sh + 1), 0, 65535))


@kernel('addsub_op@CH.cH.CH.fi')
def k_addsub_inline(emu, node):
    a, b, params = node['inputs']
    p = emu.mem.read(params).reshape(-1)
    o = node['outputs'][0]
    emu.mem.write(o, addsub_u16(emu.mem.read(a), emu.mem.read(b), int(p[1]) >> 16,
                               int(p[2]), int(p[4]) >> 16, q(emu.t[o])[1]))


@kernel('@BlockZap')
def k_block_zap(emu, node):
    fill, header, *rest = node['framework_vector']
    count = header & 0x7fff
    assert header & 0x8000 and header & 0xc00000 == 0xc00000
    nzero = ((header >> 16) & 7) * 256
    mem = emu.mem.pools[1]
    for src, dst in zip(rest[:2 * count:2], rest[1:2 * count:2]):
        tile = mem[src:src + BLOCK].copy()
        tile[BLOCK - nzero:] = fill & 255
        mem[dst:dst + BLOCK] = tile
