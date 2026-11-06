#!/usr/bin/env python3
"""Extract the Turing clock map from a QuRT CDSP image (CDSP.HT.3.0 layout).

Accepts a complete ELF (cdsp.mbn) or a split image directory (cdsp.mbn header
plus cdsp.bNN segments). For each turing_* clock it reports the controller,
branch offset, RCG offset and frequency table. QuRT fills controller bases at
run time, so they are labelled by anchor clocks: q6_cc from the HMX clock,
turing_cc from q6ss_q6_axim, and gcc (QuRT's GCC base is GCC+0x10000; the
reported offsets add that back) from nspnoc's RCG domain.

  python3 qurt_clock_map.py IMAGE [OUT.json]
"""
from pathlib import Path
import json
import re
import struct
import sys

LEVELS = {16: 'RET', 48: 'MIN_SVS', 52: 'LOW_SVS_D2', 56: 'LOW_SVS_D1', 64: 'LOW_SVS', 128: 'SVS',
          192: 'SVS_L1', 256: 'NOM', 320: 'NOM_L1', 384: 'TURBO', 416: 'TURBO_L1', 448: 'TURBO_L2'}


def load(path):
    path = Path(path)
    hdr_path = path / 'cdsp.mbn' if path.is_dir() else path
    d = bytearray(hdr_path.read_bytes())
    phoff, = struct.unpack_from('<I', d, 0x1c)
    n, = struct.unpack_from('<H', d, 0x2c)
    segs = [struct.unpack_from('<8I', d, phoff + 32 * i) for i in range(n)]
    end = max(p[1] + p[4] for p in segs)
    if len(d) < end:  # split image: fill segments from cdsp.bNN
        d.extend(bytes(end - len(d)))
        for i, p in enumerate(segs):
            if p[4]:
                part = (hdr_path.parent / f'{hdr_path.stem}.b{i:02d}').read_bytes()
                if len(part) != p[4]:
                    raise SystemExit(f'segment {i}: {len(part)} bytes, header says {p[4]}')
                d[p[1]:p[1] + p[4]] = part
    return bytes(d), [p for p in segs if p[0] == 1]


class Image:
    def __init__(self, path):
        self.d, self.segs = load(path)

    def off(self, va):
        for _, o, v, _, fs, *_ in self.segs:
            if v <= va < v + fs:
                return o + va - v
        return None

    def va(self, off):
        for _, o, v, _, fs, *_ in self.segs:
            if o <= off < o + fs:
                return v + off - o
        return None

    def words(self, va, n):
        o = self.off(va)
        return struct.unpack_from(f'<{n}I', self.d, o) if o is not None else None

    def table(self, va, count):
        rows = []
        for i in range(count):
            hz, src, cfg, _, lvl, *_ = self.words(va + 40 * i, 10)
            rows.append({'mhz': round(hz / 1e6, 3), 'cfg': f'{cfg:#x}', 'src_sel': cfg >> 8 & 7,
                         'div': ((cfg & 0x1f) + 1) / 2, 'level': LEVELS.get(lvl, lvl)})
        return rows

    def ext_list(self, va, key):  # external (GCC) domains: mux map 7-word, OPP 5-word records
        rows = []
        for i in range(16):
            w = self.words(va + (28 if key == 'mux' else 20) * i, 3)
            if w is None or (key == 'mux' and w[0] == 0xff) or (key == 'opp' and w[0] == 0):
                break
            rows.append({'mux': w[0], 'mhz': round(w[1] / 1e6, 3)} if key == 'mux'
                        else {'mhz': round(w[0] / 1e6, 3), 'level': LEVELS.get(w[1], w[1])})
        return rows


def main():
    img = Image(sys.argv[1])
    d = img.d
    ver = re.search(rb'QC_IMAGE_VERSION_STRING=([^\0]+)', d)
    clocks = {}
    for m in re.finditer(rb'turing_(?:q6_)?cc_[a-z0-9_]+_clk\0', d):
        name, nva = m.group()[:-1].decode(), img.va(m.start())
        for r in re.finditer(re.escape(struct.pack('<I', nva)), d):
            if r.start() % 4:
                continue
            dom, _, base, branch = img.words(img.va(r.start()) + 4, 4)
            if base >> 28 == 0xf and img.off(dom) is not None and branch < 0x40000:
                clocks[name] = {'domain': dom, 'base': base, 'branch': branch}
                break
    domains = {}
    for c in clocks.values():
        tbl, count, _, _, _, _, base, rcg = img.words(c['domain'], 8)
        dom = {'base': base, 'rcg': rcg}
        if tbl and 0 < count < 40:
            dom['table'] = img.table(tbl, count)
        elif not tbl:  # external domain: OPP list and mux map precede it
            opp, mux = img.words(c['domain'] - 0x14, 2)
            if img.off(opp) is not None and img.off(mux) is not None:
                dom['opp'], dom['mux'] = img.ext_list(opp, 'opp'), img.ext_list(mux, 'mux')
        domains[c['domain']] = dom
    anchors = {'q6_cc': ('turing_q6_cc_mxu_cc_core_clk', 'base'),
               'turing_cc': ('turing_cc_q6ss_q6_axim_clk', 'base')}
    label = {clocks[c][k]: n for n, (c, k) in anchors.items() if c in clocks}
    gcc_base = domains[clocks['turing_cc_nspnoc_clk']['domain']]['base'] if 'turing_cc_nspnoc_clk' in clocks else None
    phys = {'q6_cc': 0x32348000, 'turing_cc': 0x32008000, 'gcc': 0x100000}

    def where(base, off):
        name = 'gcc' if base == gcc_base else label.get(base, f'base@{base:#x}')
        off += 0x10000 if name == 'gcc' else 0
        return {'controller': name, 'offset': f'{off:#x}',
                'phys_guess': f'{phys[name] + off:#x}' if name in phys else None}

    out = {'image_version': ver.group(1).decode() if ver else None, 'clocks': {}}
    for name, c in sorted(clocks.items()):
        dom = domains[c['domain']]
        entry = {'branch': where(c['base'], c['branch']), 'rcg': where(dom['base'], dom['rcg'])}
        entry.update({k: v for k, v in dom.items() if k in ('table', 'opp', 'mux')})
        out['clocks'][name] = entry
    text = json.dumps(out, indent=1)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text + '\n')
    for name, e in out['clocks'].items():
        freqs = [r['mhz'] for r in e.get('table', e.get('opp', []))]
        print(f"{name:44s} branch {e['branch']['controller']}+{e['branch']['offset']:>7s}  "
              f"rcg {e['rcg']['controller']}+{e['rcg']['offset']:>7s}  {freqs}")


if __name__ == '__main__':
    main()
