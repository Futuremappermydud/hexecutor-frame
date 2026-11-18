#!/usr/bin/env python3
"""Generate fp16_depthwise.S: hand-scheduled interior depthwise row loops.

A hardware thread issues one packet every three cycles, so packets are the cost.
Each tap takes two: {vmpy; vadd lo; slot} and {vadd hi; shuffle; row load}. A
packet holds one vector load, so a 5x5 filter loads weights with the multiply
(`.cur`) while a 3x3 keeps its nine weights in registers and uses that slot for
row vectors. The first row's setup overlaps the previous pair's conversion and
clamp. Summation order matches dw_border in fp16_depthwise_hvx.h, so both
produce identical bits. Usage: gen_depthwise.py > fp16_depthwise.S
"""

ROW = ["r6", "r7", "r8", "r9", "r12"]
VEC = ["v4", "v5", "v6", "v10"]
WEIGHT_REGS = [f"v{i}" for i in range(18, 27)]


def taps(K, sw, odd):
    """Tap sources for one row: (kind, a, b) over the row's vectors."""
    out = []
    for kx in range(K):
        e = kx + odd
        i = e // 2
        if sw == 1:
            out.append(("v", i, i) if e % 2 == 0 else ("e", i, i + 1))
        else:
            out.append(("s", i, i + 1) if e % 2 == 0 else ("o", i, i + 1))
    return out


class Row:
    """Greedy supply of one row's vector loads and shuffles into free slots."""

    def __init__(self, reg, row_taps, vectors, advance=True):
        self.reg, self.taps, self.vectors, self.advance = reg, row_taps, vectors, advance
        self.loaded, self.next_load, self.tmp_of, self.kx = set(), 0, None, 0
        self.ready = False

    def load_op(self):
        if self.next_load >= self.vectors:
            return []
        i = self.next_load
        self.next_load += 1
        ops = [f"{VEC[i]} = vmem({self.reg}+#{i})"]
        if self.next_load == self.vectors and self.advance:
            ops.append(f"{self.reg} = add({self.reg},r13)")
        return ops

    def shuffle_op(self):
        """The next shuffle whose inputs arrived in an earlier packet."""
        if self.kx < len(self.taps) and not self.ready:
            kind, a, b = self.taps[self.kx]
            if kind == "v":
                if a in self.loaded:
                    self.ready = True
            elif kind in "so" and a in self.loaded and b in self.loaded:
                self.ready = True
                name = "vshuffe" if kind == "s" else "vshuffo"
                return f"v7.h = {name}({VEC[b]}.h,{VEC[a]}.h)"
            elif kind == "e":
                if self.tmp_of == a and b in self.loaded:
                    self.ready = True
                    return f"v7.h = vshuffe({VEC[b]}.h,v9.h)"
                if self.tmp_of != a and a in self.loaded:
                    self.tmp_of = a
                    return f"v9.h = vshuffo({VEC[a]}.h,{VEC[a]}.h)"
        # A plain tap frees the slot for the next tap's first half. v7 and the
        # final shuffle must wait until the current tap has consumed v7.
        if self.ready and self.kx + 1 < len(self.taps):
            kind, a, _ = self.taps[self.kx + 1]
            if kind == "e" and self.tmp_of != a and a in self.loaded:
                self.tmp_of = a
                return f"v9.h = vshuffo({VEC[a]}.h,{VEC[a]}.h)"
        return None

    def support(self, want_load=True):
        """Ops for one packet with a shuffle slot and optionally a load slot."""
        arrived = set()
        ops = []
        shuffle = self.shuffle_op()
        if shuffle:
            ops.append(shuffle)
        if want_load:
            before = self.next_load
            ops += self.load_op()
            if self.next_load != before:
                arrived.add(before)
        self.loaded |= arrived
        return ops

    def consumed(self):
        self.kx += 1
        self.ready = False


def setup(row):
    """Packets that make a row's first tap ready, with no other work in them."""
    packets = []
    while True:
        ops = row.support()
        if ops:
            packets.append(ops)
        if row.ready:
            return packets
        if not ops and row.next_load >= row.vectors and not row.ready:
            # One more evaluation with the last load now visible.
            ops = row.support()
            if ops:
                packets.append(ops)
            if row.ready:
                return packets
            raise AssertionError("row setup cannot make progress")


def loop(K, sw, odd):
    name = f"hx_dw{K}s{sw}{'o' if odd else 'e'}"
    row_taps = taps(K, sw, odd)
    vectors = max(b for _, _, b in row_taps) + 1
    in_registers = K == 3
    p = [f"    .globl {name}", f"{name}:"]
    p.append("    { r3 = asl(r3,#7); r13 = #%d; r10 = #%d }" % (128 * sw, K * K * 128))
    p.append("    { r6 = memw(r1+#0); r7 = memw(r1+#4); r10 = add(r2,r10) }")
    p.append("    { r8 = memw(r1+#8); r6 = add(r6,r3); r7 = add(r7,r3) }")
    if K == 5:
        p.append("    { r9 = memw(r1+#12); r12 = memw(r1+#16); r8 = add(r8,r3) }")
        p.append("    { r9 = add(r9,r3); r12 = add(r12,r3) }")
    else:
        p.append("    { r8 = add(r8,r3) }")
    p.append("    { v13 = vmem(r10+#0) }")
    p.append("    { v14 = vmem(r5+#0) }")
    p.append("    { v15 = vmem(r5+#1) }")
    if in_registers:
        p.append("    { r14 = r2 }")
        for reg in WEIGHT_REGS:
            p.append(f"    {{ {reg} = vmem(r14++#1) }}")
    else:
        p.append("    { r14 = r2 }")
    p.append("    { v0 = v13; v1 = v13 }")
    # The first row's pointer advances in the conversion packet, so its setup
    # can overlap the clamp and store of the pair before it.
    first = Row(ROW[0], row_taps, vectors, advance=False)
    prefix = setup(first)
    convert = ["v11.hf = v1:0.qf32", f"{ROW[0]} = add({ROW[0]},r13)"]
    if in_registers:
        convert += ["v0 = v13", "v1 = v13"]
        clamp = ["v11.h = vmax(v11.h,v14.h)"]
    else:
        convert += ["r14 = r2"]
        clamp = ["v0 = v13", "v1 = v13", "v11.h = vmax(v11.h,v14.h)"]
    store = ["v11.h = vmin(v11.h,v15.h)", "vmem(r0++#1) = v11.new"]
    overlapped = 0
    for packet in (clamp, store):
        if overlapped < len(prefix) and len(packet) + len(prefix[overlapped]) <= 4:
            packet += prefix[overlapped]
            overlapped += 1
        else:
            break
    for ops in prefix[:overlapped]:
        p.append("    { " + "; ".join(ops) + " }")
    p.append("    loop0(1f,r4)")
    p.append("    .p2align 4")
    p.append("1:")
    for ops in prefix[overlapped:]:
        p.append("    { " + "; ".join(ops) + " }")
    n = 0
    rows = [first] + [Row(ROW[i], row_taps, vectors) for i in range(1, K)]
    for ky in range(K):
        row = rows[ky]
        for kx in range(K):
            pair, prev_hi, prev_lo = ("v3:2", "v17", "v16") if n % 2 == 0 else ("v17:16", "v3", "v2")
            # Skip the very first add pair: nothing has been multiplied yet.
            if n:
                ops = row.support()
                p.append("    { " + "; ".join([f"v1.qf32 = vadd(v1.qf32,{prev_hi}.qf32)"] + ops) + " }")
                while not row.ready:
                    ops = row.support()
                    p.append("    { " + "; ".join(ops or ["nop"]) + " }")
            kind, a, _ = row_taps[kx]
            source = VEC[a] if kind == "v" else "v7"
            weight = WEIGHT_REGS[n] if in_registers else "v8"
            m = [f"{pair}.qf32 = vmpy({source}.hf,{weight}.hf)"]
            if not in_registers:
                m.insert(1, "v8.cur = vmem(r14++#1)")
            if n:
                m.append(f"v0.qf32 = vadd(v0.qf32,{prev_lo}.qf32)")
            row.consumed()
            if in_registers:
                # This packet's free slot: a row vector, else a shuffle.
                target = row if kx + 1 < K else (rows[ky + 1] if ky + 1 < K else None)
                if target is not None:
                    if target.next_load < target.vectors:
                        before = target.next_load
                        m += target.load_op()
                        target.loaded.add(before)
                    else:
                        shuffle = target.shuffle_op()
                        if shuffle:
                            m.append(shuffle)
            p.append("    { " + "; ".join(m) + " }")
            n += 1
    last_hi, last_lo = ("v3", "v2") if (n - 1) % 2 == 0 else ("v17", "v16")
    p.append(f"    {{ v0.qf32 = vadd(v0.qf32,{last_lo}.qf32) }}")
    p.append(f"    {{ v1.qf32 = vadd(v1.qf32,{last_hi}.qf32) }}")
    p.append("    { " + "; ".join(convert) + " }")
    p.append("    { " + "; ".join(clamp) + " }")
    p.append("    { " + "; ".join(store) + " }:endloop0")
    p.append("    { jumpr r31 }")
    return "\n".join(p)


print("/* Generated by gen_depthwise.py; do not edit.")
print(" * r0 out, r1 row pointers, r2 weights then bias, r3 first pair j, r4 pairs,")
print(" * r5 { floor, ceiling } halfword limits, which implement ReLU, ReLU6 or none. */")
print("    .text")
for K in (3, 5):
    for sw in (1, 2):
        for odd in (0, 1):
            print(loop(K, sw, odd))
