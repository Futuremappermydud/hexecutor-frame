"""Convolution-to-HVX lowering; graph storage and requantization stay on ARM."""

from hexecutor.paths import workspace
PATHS = workspace()

import time

import numpy as np


class HvxConvolution:
    def __init__(self, transport, verify=False, direct=False):
        self.transport = transport
        self.direct = direct
        self.verify = verify
        self.calls = 0
        self.quantized_calls = 0
        self.macs = 0
        self.seconds = 0.0

    def accumulate(self, x, w, zero_point):
        if self.direct:
            return self.accumulate_direct(x, w, zero_point)
        start = time.monotonic()
        kh, kw, ci, co = w.shape
        oh, ow = x.shape[0] - kh + 1, x.shape[1] - kw + 1
        if co % 32:
            raise ValueError('HVX output depth must be padded to 32')
        # u16 activations need their top bit removed to fit signed halfwords.
        # Reintroduce that exact, per-channel correction in int64 afterwards.
        center = 32768 if x.dtype == np.uint16 else zero_point
        pieces = [x[ky:ky + oh, kx:kx + ow, :ci] for ky in range(kh) for kx in range(kw)]
        columns = np.concatenate(pieces, axis=-1).reshape(oh * ow, -1).astype(np.int32) - center
        if np.any(columns < -32768) or np.any(columns > 32767):
            raise ValueError('activation does not fit the signed-halfword lowering')
        a = columns.astype(np.int16)
        b = np.ascontiguousarray(w.reshape(-1, co), dtype=np.int32)
        # 256 * 32768 * 128 = 2**30: every DSP partial fits signed int32.
        acc = self.transport.matmul(a, b)
        if self.verify:
            expected = a.astype(np.int64) @ b.astype(np.int64)
            if not np.array_equal(acc, expected):
                raise RuntimeError(f'HVX dot product mismatch at convolution {self.calls}')
        acc += (center - zero_point) * b.sum(axis=0, dtype=np.int64)
        self.calls += 1
        self.macs += a.shape[0] * a.shape[1] * b.shape[1]
        self.seconds += time.monotonic() - start
        return acc.reshape(oh, ow, co)

    def accumulate_quantized(self,x,w,quantization):
        if not self.direct or not hasattr(self.transport,'conv_quantized'):return None
        start=time.monotonic()
        kh,kw,ci,co=w.shape;oh,ow=x.shape[0]-kh+1,x.shape[1]-kw+1
        x=x[:,:,:ci]
        if kh==kw==1:x=x.reshape(1,-1,ci)
        result=self.transport.conv_quantized(np.ascontiguousarray(x),w,quantization).reshape(oh,ow,co)
        if self.verify:
            xs=x.reshape(oh+kh-1,ow+kw-1,ci)
            acc=sum(xs[ky:ky+oh,kx:kx+ow].astype(np.int64) @ w[ky,kx].astype(np.int64)
                    for ky in range(kh) for kx in range(kw))
            expected=np.clip(((((acc+quantization[:,0])>>quantization[:,2])*quantization[:,1])+8192)>>14,0,255).astype(np.uint8)
            if not np.array_equal(result,expected):
                from pathlib import Path
                dump=(PATHS.native / 'build/hmx-failure.npz')
                np.savez(dump,x=xs,w=w,quantization=quantization,result=result,expected=expected)
                raise RuntimeError(f'DSP quantized convolution {self.calls} differs at {np.count_nonzero(result!=expected)} values; x={xs.shape}, w={w.shape}; saved {dump}')
        self.calls+=1;self.quantized_calls+=1;self.macs+=oh*ow*kh*kw*ci*co;self.seconds+=time.monotonic()-start
        return result

    def accumulate_direct(self, x, w, zero_point):
        """Whole convolution on the DSP from the raw u8 tensor (no im2col).
        u16 activations are split into exact high/low byte planes."""
        start = time.monotonic()
        kh, kw, ci, co = w.shape
        oh, ow = x.shape[0] - kh + 1, x.shape[1] - kw + 1
        x = x[:, :, :ci]
        if kh == kw == 1:
            x = x.reshape(1, -1, ci)  # pointwise: one long row of pixels
        if x.dtype == np.uint8:
            acc = np.asarray(self.transport.conv(np.ascontiguousarray(x), w),dtype=np.int64)
        elif x.dtype == np.uint16:
            hi = np.asarray(self.transport.conv((x >> 8).astype(np.uint8), w),dtype=np.int64)
            lo = np.asarray(self.transport.conv((x & 255).astype(np.uint8), w),dtype=np.int64)
            acc = hi * 256 + lo
        else:
            raise ValueError(f'unsupported activation type {x.dtype}')
        acc = acc.reshape(oh, ow, co)
        if self.verify:
            expected = np.zeros_like(acc)
            xs = x.reshape(oh + kh - 1, ow + kw - 1, ci) if kh == kw == 1 else x
            for ky in range(kh):
                for kx in range(kw):
                    expected += xs[ky:ky + oh, kx:kx + ow].astype(np.int64) @ w[ky, kx].astype(np.int64)
            if not np.array_equal(acc, expected):
                raise RuntimeError(f'HVX convolution mismatch at convolution {self.calls}')
        acc -= zero_point * w.sum(axis=(0, 1, 2), dtype=np.int64)
        self.calls += 1
        self.macs += oh * ow * kh * kw * ci * co
        self.seconds += time.monotonic() - start
        return acc
