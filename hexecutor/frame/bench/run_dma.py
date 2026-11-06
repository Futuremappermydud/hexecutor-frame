#!/usr/bin/env python3
"""DDR -> VTCM bandwidth: HVX loads vs. user DMA (bench_dma.c)."""
import json
from pathlib import Path
import struct
import sys


def mode_is_dma(a):
    return a[0] == 10

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from frame_transport import FrameTransport
from run_bench import ARENA_CACHED, ARENA_UNCACHED, DATA, bench


def main():
    build = HERE.parent / 'build'
    rows = []
    with FrameTransport(build / 'qd_loader.elf', build / 'bench_dma.elf', 30) as t:
        t.arena[DATA:DATA + 0x600000] = bytes(range(256)) * (0x600000 // 256)

        def run(label, moved, *a, **k):
            r = bench(t, *a, **k)
            r['label'] = label
            outs = [struct.unpack_from('<8I', t.arena, 2048 + i * 32) for i in range(3)]
            r['r0'] = [hex(o[6]) for o in outs]; r['r1'] = [hex(o[7]) for o in outs]
            r['landed_cycles'] = [(o[7] >> 8) << 6 for o in outs]
            cyc = max(r['landed_cycles']) if mode_is_dma(a) else max(r['cycles'])
            r['bytes_per_cycle'] = moved / cyc if cyc else 0
            r['gbps_at_1.9ghz'] = r['bytes_per_cycle'] * 1.9
            rows.append(r)
            print(json.dumps(r), flush=True)

        mb = 1 << 20
        if len(sys.argv) > 1 and sys.argv[1] in ('sweep', 'big', 'mix', 'stress', 'tlbdump', 'shape', 'llcc'):
            pass
        else:
          run('hvx cached 1t', mb, 6, 1, ARENA_CACHED + DATA, mb, threads=1)
        run('hvx cached 3t', 3 * mb, 6, 1, ARENA_CACHED + DATA, mb, threads=3)
        run('hvx uncached 3t', 3 * mb, 6, 1, ARENA_UNCACHED + DATA, mb, threads=3)
        if len(sys.argv) > 1 and sys.argv[1] == 'tlb':
            flags = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            t.arena[0x3000:0x3100] = bytes(256)
            run('dma tlb', 1, 12, flags, threads=1)
            dump = struct.unpack_from('<64I', t.arena, 0x3000)
            print('ssr', hex(dump[0]), 'dma_jtlb', dump[1], 'dma_version', dump[2], 'paused', hex(dump[3]))
            for i, va in enumerate((0x88800000, 0xf1800000, 0x31000000, 0x31400000)):
                print(hex(va), 'idx', hex(dump[4 + i * 4]), 'lo', hex(dump[5 + i * 4]), 'hi', hex(dump[6 + i * 4]))
            print('dma part', [(hex(dump[32 + 2 * i]), hex(dump[33 + 2 * i])) for i in range(8)])
            return
        if len(sys.argv) > 1 and sys.argv[1] == 'dm':
            t.arena[0x3000:0x3100] = bytes(256)
            run('dm regs', 1, 14, 0, threads=1)
            dump = struct.unpack_from('<32I', t.arena, 0x3000)
            print('DM0..5', [hex(v) for v in dump[:6]])
            return
        if len(sys.argv) > 1 and sys.argv[1] == 'shape':
            t.arena[0x3000:0x3100] = bytes(256)
            run('map big', 1, 16, 1, threads=1)
            if struct.unpack_from('<I', t.arena, 0x3000)[0] != 1:
                print('map failed', hex(struct.unpack_from('<I', t.arena, 0x3000)[0])); return
            struct.pack_into('<I', t.arena, 32, 0x2e3)
            big = 0xa0000000
            cases = [('2t 1MBx2 separate', 2, 1 << 21, 1 << 20, 2),
                     ('2t 64KBx16 adjacent', 16, 1 << 20, 65536, 2),
                     ('2t 1MBx1 adjacent', 1, 1 << 20, 1 << 20, 2),
                     ('2t 64KBx32 separate', 32, 1 << 22, 65536, 2),
                     ('1t 1MBx2', 2, 1 << 21, 1 << 20, 1),
                     ('2t 256KBx8 adjacent', 8, 1 << 21, 262144, 2)]
            for label, n, region, desc, th in cases:
                for rep in range(3):
                    run(label, th * n * desc, 10, n, big, region, desc, threads=th)
                    r = rows[-1]
                    print(label, [round(c / 1.8e3, 2) for c in r['landed_cycles'][:th]], 'ms', round(r['gbps_at_1.9ghz'] * 1.8 / 1.9, 2), 'GB/s', flush=True)
            run('unmap big', 1, 16, 0, threads=1)
            return
        if len(sys.argv) > 1 and sys.argv[1] == 'llcc':
            t.arena[0x3000:0x3100] = bytes(256)
            run('map big', 1, 16, 1, threads=1)
            if struct.unpack_from('<I', t.arena, 0x3000)[0] != 1:
                print('map failed'); return
            struct.pack_into('<I', t.arena, 32, 0x2e3)
            big = 0xa0000000
            for rep in range(4):
                base = big + 0x4000000 + rep * 0x400000      # fresh 4 MB each time
                run('fresh', 2 << 20, 10, 1, base, 1 << 20, 1 << 20, threads=2)
                f = rows[-1]
                run('reread', 2 << 20, 10, 1, base, 1 << 20, 1 << 20, threads=2)
                g = rows[-1]
                print('fresh', [round(c / 1.8e3) for c in f['landed_cycles'][:2]], 'us  reread',
                      [round(c / 1.8e3) for c in g['landed_cycles'][:2]], 'us', flush=True)
            run('unmap big', 1, 16, 0, threads=1)
            return
        if len(sys.argv) > 1 and sys.argv[1] == 'tlbdump':
            for rep in range(int(sys.argv[2]) if len(sys.argv) > 2 else 1):
                t.arena[0x3000:0x3400] = bytes(1024)
                run('tlbdump', 1, 18, 0, threads=1)
                d = struct.unpack_from('<256I', t.arena, 0x3000)
                used = [(i, hex(d[2 * i + 1]), hex(d[2 * i])) for i in range(128) if d[2 * i + 1] >> 31]
                print(rep, len(used), used)
            return
        if len(sys.argv) > 1 and sys.argv[1] == 'stress':
            run('map big', 1, 16, 1, threads=1)
            if struct.unpack_from('<I', t.arena, 0x3000)[0] != 1:
                print('map failed; assuming qwen_dsp entries (slots 64..111) map the buffer')
            struct.pack_into('<I', t.arena, 32, int(sys.argv[2], 0))
            src = 0xa0000000 if sys.argv[3] == 'big' else ARENA_CACHED + DATA
            times = []
            for i in range(int(sys.argv[4])):
                run(f'stress {i}', 2 * 16 * 65536, 10, 16, src, 1 << 21, 65536, threads=2)
                times.append([c / 1.8e6 for c in rows[-1]['landed_cycles'][:2]])
                if any(int(v, 16) & 0xff not in (0, 1) for v in rows[-1]['r1']):
                    print('STALL', rows[-1]['r1'], rows[-1]['r0']); break
            print('ms per 1 MB (t0, t1):', [[round(a, 2), round(b, 2)] for a, b in times])
            return
        if len(sys.argv) > 1 and sys.argv[1] == 'mix':
            struct.pack_into('<I', t.arena, 32, 0x2e7)
            run('dma 64kx16 1t', mb, 10, 16, ARENA_CACHED + DATA, mb, 65536, threads=1)
            run('mix dma t0 + hvx t1,t2', 3 * mb, 17, 16, ARENA_CACHED + DATA, mb, 65536, threads=3)
            run('mix dma t0 + hvx t1', 2 * mb, 17, 16, ARENA_CACHED + DATA, mb, 65536, threads=2)
            return
        if len(sys.argv) > 1 and sys.argv[1] == 'big':
            t.arena[0x3000:0x3100] = bytes(256)
            run('map big', 1, 16, 1, threads=1)
            print('map', [hex(v) for v in struct.unpack_from('<3I', t.arena, 0x3000)])
            if struct.unpack_from('<I', t.arena, 0x3000)[0] == 1:
                big = 0xa0000000
                struct.pack_into('<I', t.arena, 32, 0x2e7)
                run('big hvx 1t', mb, 6, 1, big, mb, threads=1)
                run('big hvx 3t', 3 * mb, 6, 1, big, mb, threads=3)
                run('big dma 64kx64 1t', 64 * 65536, 10, 64, big, 1 << 22, 65536, threads=1)
                run('big dma 64kx16 2t', 2 * 16 * 65536, 10, 16, big, 1 << 21, 65536, threads=2)
                run('big dma 64kx16 3t', 3 * 16 * 65536, 10, 16, big, 1 << 21, 65536, threads=3)
                run('big dma 1Mx4 1t', 4 << 20, 10, 4, big, 1 << 22, 1 << 20, threads=1)
                run('unmap big', 1, 16, 0, threads=1)
                print('unmap', [hex(v) for v in struct.unpack_from('<3I', t.arena, 0x3000)])
            return
        if len(sys.argv) > 1 and sys.argv[1] == 'dm2':
            value = int(sys.argv[2], 0); keep = int(sys.argv[3]) if len(sys.argv) > 3 else 0
            t.arena[0x3000:0x3100] = bytes(256)
            run('dm2 dma', 65536, 15, keep, ARENA_CACHED + DATA, value, 65536, threads=1)
            dump = struct.unpack_from('<16I', t.arena, 0x3000)
            print('saved DM2', hex(dump[0]), 'pause', hex(dump[1]), 'DM2 set', hex(dump[2]), 'DM0 pre', hex(dump[3]),
                  'cycles', dump[4], 'DM0..5 after', [hex(v) for v in dump[8:14]], 'desc w1', hex(dump[14]),
                  'pause2', hex(dump[15]), 'DM2 final', hex(dump[5]), 'result', hex(rows[-1]['result'][0]), 'r1', rows[-1]['r1'][0])
            return
        if len(sys.argv) > 1 and sys.argv[1] == 'alias':
            flags = int(sys.argv[2])
            t.arena[0x3000:0x3100] = bytes(256)
            run('dma alias', 65536, 13, flags, ARENA_CACHED + DATA, 0, 65536, threads=1)
            dump = struct.unpack_from('<16I', t.arena, 0x3000)
            print('alias', [hex(v) for v in dump[:11]], 'dmpoll_after', rows[-1]['r0'][0], 'result', hex(rows[-1]['result'][0]))
            return
        values = [int(v, 0) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 and sys.argv[1] == 'sweep' else [0x2e3]
        for dm2 in values:
            struct.pack_into('<I', t.arena, 32, dm2)
            run(f'dm2={dm2:#x} 64kx64 1t', 64 * 65536, 10, 64, ARENA_CACHED + DATA, 1 << 22, 65536, threads=1)
            run(f'dm2={dm2:#x} 64kx16 2t', 2 * 16 * 65536, 10, 16, ARENA_CACHED + DATA, 1 << 21, 65536, threads=2)
            run(f'dm2={dm2:#x} 64kx16 3t', 3 * 16 * 65536, 10, 16, ARENA_CACHED + DATA, 1 << 21, 65536, threads=3)
    (build / 'bench-dma.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
