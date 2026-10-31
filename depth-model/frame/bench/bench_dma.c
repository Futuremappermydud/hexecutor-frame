/* DDR -> VTCM bandwidth on the Frame CDSP: HVX loads vs. user DMA.
 * Same command/result layout as bench_hvx.c (run with bench/run_dma.py). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#define ARENA_UNCACHED 0xf1800000u
typedef struct { uint32_t magic, sequence, mode, iters, addr, bytes, stride, threads; } Cmd;
typedef struct { uint32_t done, cyc_lo, cyc_hi, result, pk_lo, pk_hi, r0, r1; } Out;

/* V75 2D descriptor (type 9), as in llama.cpp's ggml-hexagon dma-queue.h. */
typedef struct {
    uint32_t next;
    uint32_t dst_stride:24, desc_size:2, dst_comp:1, src_comp:1,
             dst_bypass:1, src_bypass:1, order:1, done:1;
    uint32_t src, dst;
    uint32_t desc_type:8, reserved0:24;
    uint32_t row_size:24, nrows_lo:8;
    uint32_t nrows_hi:8, src_stride:24;
    uint32_t offset:24, reserved1:8;
} Desc;

static uint32_t physread(uint32_t lo, uint32_t hi) {
    uint32_t value;
    __asm__ volatile("%0 = memw_phys(%1,%2)" : "=r"(value) : "r"(lo), "r"(hi) : "memory");
    return value;
}

static uint64_t tlb_read(uint32_t index) {
    uint64_t entry;
    __asm__ volatile("%0 = tlbr(%1)" : "=r"(entry) : "r"(index) : "memory");
    return entry;
}

static uint32_t tlb_probe(uint32_t va, uint32_t asid) {
    uint32_t index;
    __asm__ volatile("%0 = tlbp(%1)" : "=r"(index) : "r"((asid << 20) | (va >> 12)) : "memory");
    return index;
}

/* r0 = dmcfgrd(r1) and dmcfgwr(r1, r0), encodings from the stock QuRT image. */
static uint32_t dm_read(uint32_t index) {
    uint32_t value;
    __asm__ volatile("r1 = %1\n .word 0xa801c0a0\n %0 = r0" : "=r"(value) : "r"(index) : "r0", "r1", "memory");
    return value;
}

static void dm_write(uint32_t index, uint32_t value) {
    __asm__ volatile("r1 = %0\n r0 = %1\n .word 0xa801c0c0" :: "r"(index), "r"(value) : "r0", "r1", "memory");
}

static inline uint64_t cycles(void) {
    uint64_t c; __asm__ volatile("%0 = upcycle" : "=r"(c)); return c;
}

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid; __asm__ volatile("%0 = htid" : "=r"(tid));
    volatile Cmd *cmd = (volatile Cmd *)ARENA_UNCACHED;
    volatile Out *out = (volatile Out *)(ARENA_UNCACHED + 2048 + tid * 32);
    if (tid >= 3) return;
    uint32_t seq = cmd->sequence, mode = cmd->mode, iters = cmd->iters;
    uint32_t addr = cmd->addr, bytes = cmd->bytes, stride = cmd->stride;
    uint32_t result = 0, r0 = 0, r1 = 0;
    uint32_t dst = 0x31000000u + tid * 0xF0000u;
    uint64_t c0 = cycles();
    if (mode == 17) { mode = tid ? 6 : 10; if (tid) iters = 1; }
    if (cmd->magic != 0x48434e42u || tid >= cmd->threads) {
        /* idle */
    } else if (mode == 6) { /* HVX copy, 8 loads in flight */
        for (uint32_t p = 0; p < iters; p++)
            for (uint32_t o = 0; o < bytes; o += 1024) {
                const HVX_Vector *s8 = (const HVX_Vector *)(addr + tid * bytes + o);
                HVX_Vector *d8 = (HVX_Vector *)(dst + (o & 0x7ffffu));
                HVX_Vector v0 = s8[0], v1 = s8[1], v2 = s8[2], v3 = s8[3];
                HVX_Vector v4 = s8[4], v5 = s8[5], v6 = s8[6], v7 = s8[7];
                d8[0] = v0; d8[1] = v1; d8[2] = v2; d8[3] = v3;
                d8[4] = v4; d8[5] = v5; d8[6] = v6; d8[7] = v7;
            }
        result = Q6_R_vextract_VR(*(HVX_Vector *)dst, 0);
    } else if (mode == 10) {
        /* User DMA: `iters & 255` descriptors of `stride` bytes each, chained,
         * from addr + tid*bytes into this thread's VTCM slice (wrapping).
         * Variant (iters >> 8): 1 = 1D descriptors, 2 = descriptors in L2-TCM,
         * 4 = descriptors in the cached arena, 8 = no cache bypass,
         * 16 = VTCM source. */
        uint32_t variant = iters >> 8, n = iters & 255u;
        if (n > 64u) n = 64u;
        uint32_t base = variant & 2u ? 0x30080000u + tid * 0x2000u :
                        variant & 4u ? 0x88808000u + tid * 0x2000u :
                        ARENA_UNCACHED + 0x8000u + tid * 0x2000u;
        if (variant & 16u) addr = 0x31000000u + 0x600000u;
        volatile Desc *d = (volatile Desc *)base;
        volatile uint32_t *d1 = (volatile uint32_t *)base;
        uint32_t pre = 0, saved_dm2 = dm_read(2);
        __asm__ volatile("%0 = dmpoll" : "=r"(pre) :: "memory");
        /* dmpause clears an error (state 2) but blocks on a stalled run (1). */
        if ((pre & 3u) == 1u) { r1 = 0xb5; goto dma10_out; }
        if (pre & 3u) { uint32_t p0; __asm__ volatile("%0 = dmpause" : "=r"(p0) :: "memory"); }
        uint32_t dm2 = ((volatile uint32_t *)ARENA_UNCACHED)[8];
        dm_write(2, dm2 ? dm2 : 0x2e3u);
        uint32_t last_dst = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t o = i * stride;
            uint32_t src = addr + tid * bytes + (o & (bytes - 1u));
            uint32_t to = dst + (o & 0x7ffffu);
            uint32_t bypass = variant & 8u ? 0u : 1u;
            last_dst = to;
            if (variant & 1u) {
                d1[i * 4 + 0] = i + 1 < n ? base + (i + 1) * 16u : 0u;
                d1[i * 4 + 1] = (stride & 0xffffffu) | (bypass << 28) | (bypass << 29);
                d1[i * 4 + 2] = src;
                d1[i * 4 + 3] = to;
            } else {
                Desc x = {0};
                x.next = i + 1 < n ? (uint32_t)&d[i + 1] : 0u;
                x.desc_size = 1; x.src_bypass = bypass; x.dst_bypass = bypass;
                x.src = src; x.dst = to;
                x.desc_type = 9;
                x.row_size = stride; x.nrows_lo = 1; x.nrows_hi = 0;
                x.src_stride = stride; x.dst_stride = stride;
                d[i] = x;
            }
        }
        if (variant & 4u)
            for (uint32_t p = base; p < base + 0x2000u; p += 32u)
                __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
        volatile uint32_t *tail = (volatile uint32_t *)(last_dst + stride - 4u);
        *tail = 0xa5a5a5a5u;
        __asm__ volatile("barrier" ::: "memory");
        uint64_t start = cycles();
        __asm__ volatile("release(%0):at" :: "r"(base) : "memory");
        __asm__ volatile("dmstart(%0)" :: "r"(base) : "memory");
        uint32_t st = 0, first_st = 0;
        __asm__ volatile("%0 = dmpoll" : "=r"(first_st) :: "memory");
        volatile uint32_t *last_word1 = variant & 1u ? &d1[(n - 1) * 4 + 1] : &((volatile uint32_t *)&d[n - 1])[1];
        (void)tail;
        while (!(*last_word1 >> 31)) {
            if (cycles() - start > 1800000000u) { r1 = 0xdead; break; }
        }
        uint64_t landed = cycles() - start;
        __asm__ volatile("%0 = dmpoll" : "=r"(st) :: "memory");
        r0 = (st & 0xffffu) | (first_st << 16);
        r1 |= ((uint32_t)(landed >> 6) & 0xffffffu) << 8;
        r1 |= (variant & 1u ? d1[(n - 1) * 4 + 1] >> 31 : d[n - 1].done);
        out->pk_lo = pre;
        /* Never dmpause a stalled engine: it blocks the thread forever. */
        dm_write(2, saved_dm2);
        result = *(volatile uint32_t *)dst;
dma10_out:;
    } else if (mode == 12 && tid == 0) {
        /* DMA TLB: report (and with iters & 1, populate) the DMA partition
         * at JTLB index 512 from the main entries covering our buffers.
         * iters & 2 sets the entry's hsv39 bit (low word bit 27). */
        volatile uint32_t *dump = (volatile uint32_t *)(ARENA_UNCACHED + 0x3000u);
        uint32_t ssr, cfg;
        __asm__ volatile("%0 = ssr" : "=r"(ssr));
        __asm__ volatile("%0 = cfgbase" : "=r"(cfg));
        uint32_t asid = (ssr >> 8) & 127u;
        dump[0] = ssr;
        dump[1] = physread(0xf0, cfg << 5);
        dump[2] = physread(0x68, cfg << 5);
        static const uint32_t vas[4] = {0x88800000u, 0xf1800000u, 0x31000000u, 0x31400000u};
        uint64_t found[4];
        for (uint32_t i = 0; i < 4; i++) {
            uint32_t index = tlb_probe(vas[i], asid);
            dump[4 + i * 4] = index;
            found[i] = index & 0x80000000u ? 0 : tlb_read(index & 0x3ffu);
            dump[5 + i * 4] = (uint32_t)found[i];
            dump[6 + i * 4] = (uint32_t)(found[i] >> 32);
        }
        uint32_t dma_entries = dump[1];
        if (dma_entries && dma_entries <= 256u) {
            if (iters & 1u) {
                for (uint32_t i = 0; i < 4; i++) if (found[i]) {
                    uint64_t entry = found[i] | (iters & 2u ? (1ull << 27) : 0ull);
                    uint32_t index = 512u + i;
                    __asm__ volatile("tlbw(%0,%1)\n isync" :: "r"(entry), "r"(index) : "memory");
                }
            }
            for (uint32_t i = 0; i < 8; i++) {
                uint64_t e = tlb_read(512u + i);
                dump[32 + i * 2] = (uint32_t)e;
                dump[33 + i * 2] = (uint32_t)(e >> 32);
            }
        }
        __asm__ volatile("%0 = dmpoll" : "=r"(r0) :: "memory");
        if (iters & 4u) {
            /* Restart the engine stuck on the first descriptor. */
            uint32_t paused;
            __asm__ volatile("%0 = dmpause" : "=r"(paused) :: "memory");
            dump[3] = paused;
            __asm__ volatile("dmresume(%0)" :: "r"(paused & ~15u) : "memory");
            uint64_t start = cycles();
            while (cycles() - start < 10000000u) { }
            __asm__ volatile("%0 = dmpoll" : "=r"(r1) :: "memory");
        }
    } else if (mode == 13 && tid == 0) {
        /* DMA through temporary alias TLB entries: 0x8a000000 -> the DMA
         * buffer (PA 0x88000000, 16 MB) and 0x3a000000 -> VTCM (4 MB).
         * iters & 2: set entry bit 27 (H2 "hsv39"). iters & 8: real VTCM VA
         * for the destination. Aliases are removed before returning. */
        volatile uint32_t *dump = (volatile uint32_t *)(ARENA_UNCACHED + 0x3000u);
        uint32_t ssr;
        __asm__ volatile("%0 = ssr" : "=r"(ssr));
        uint32_t asid = (ssr >> 8) & 127u;
        uint32_t slot = 0;
        for (uint32_t i = 32; i < 63; i++)
            if (!(tlb_read(i) >> 63) && !(tlb_read(i + 1) >> 63)) { slot = i; break; }
        dump[0] = slot;
        dump[1] = tlb_probe(0x8a000000u, asid);
        dump[2] = tlb_probe(0x3a000000u, asid);
        if (!slot || !(dump[1] & 0x80000000u) || !(dump[2] & 0x80000000u)) { r1 = 0xbad; goto dma13_out; }
        uint64_t bit = iters & 2u ? (1ull << 27) : 0ull;
        uint64_t ddr = (0xc008a000ull << 32) | 0xf7110040ull | bit;
        uint64_t tcm = (0xc003a000ull << 32) | 0xf7062020ull | bit;
        __asm__ volatile("tlbw(%0,%1)\n isync" :: "r"(ddr), "r"(slot) : "memory");
        __asm__ volatile("tlbw(%0,%1)\n isync" :: "r"(tcm), "r"(slot + 1u) : "memory");
        dump[3] = tlb_probe(0x8a000000u, asid);
        dump[4] = tlb_probe(0x3a000000u, asid);
        uint32_t st0, paused = 0;
        __asm__ volatile("%0 = dmpoll" : "=r"(st0) :: "memory");
        dump[5] = st0;
        if (st0 & 3u) {
            __asm__ volatile("%0 = dmpause" : "=r"(paused) :: "memory");
            __asm__ volatile("%0 = dmpoll" : "=r"(st0) :: "memory");
        }
        dump[6] = paused; dump[7] = st0;
        {
            uint32_t alias_off = 0x8a000000u - 0x88000000u;
            volatile uint32_t *dv = (volatile uint32_t *)(ARENA_UNCACHED + 0xa000u);
            uint32_t desc = 0x88800000u + 0xa000u + alias_off;
            uint32_t src = addr + alias_off;
            uint32_t to = iters & 8u ? 0x31000000u : 0x3a000000u;
            volatile uint32_t *tail = (volatile uint32_t *)(0x31000000u + stride - 4u);
            *tail = 0xa5a5a5a5u;
            Desc x = {0};
            x.next = 0; x.desc_size = 1; x.src_bypass = 1; x.dst_bypass = 1;
            x.src = src; x.dst = to; x.desc_type = 9;
            x.row_size = stride; x.nrows_lo = 1; x.src_stride = stride; x.dst_stride = stride;
            const uint32_t *xs = (const uint32_t *)&x;
            for (uint32_t i = 0; i < 8; i++) dv[i] = xs[i];
            __asm__ volatile("barrier" ::: "memory");
            uint64_t start = cycles();
            __asm__ volatile("release(%0):at" :: "r"(desc) : "memory");
            __asm__ volatile("dmstart(%0)" :: "r"(desc) : "memory");
            while (*tail == 0xa5a5a5a5u) {
                if (cycles() - start > 50000000u) { r1 = 0xdead; break; }
            }
            dump[8] = (uint32_t)(cycles() - start);
            __asm__ volatile("%0 = dmpoll" : "=r"(r0) :: "memory");
            dump[9] = dv[1];
            result = *(volatile uint32_t *)0x31000000u;
        }
        {
            uint64_t zero = 0;
            __asm__ volatile("tlbw(%0,%1)\n isync" :: "r"(zero), "r"(slot) : "memory");
            __asm__ volatile("tlbw(%0,%1)\n isync" :: "r"(zero), "r"(slot + 1u) : "memory");
        }
dma13_out:
        dump[10] = r1;
    } else if (mode == 14 && tid == 0) {
        /* Read DM0..DM5 (and DM7 for DM6 selections given in addr/bytes). */
        volatile uint32_t *dump = (volatile uint32_t *)(ARENA_UNCACHED + 0x3000u);
        for (uint32_t i = 0; i < 6; i++) dump[i] = dm_read(i);
        if (iters & 1u) {
            uint32_t saved = dm_read(6);
            for (uint32_t i = 0; i < 16; i++) { dm_write(6, addr + i * stride); dump[16 + i] = dm_read(7); }
            dm_write(6, saved);
        }
        if (iters & 2u) dm_write(bytes >> 24, bytes & 0xffffffu);
    } else if (mode == 18 && tid == 0) {
        volatile uint32_t *dump = (volatile uint32_t *)(ARENA_UNCACHED + 0x3000u);
        for (uint32_t i = 0; i < 128u; i++) {
            uint64_t e = tlb_read(i);
            dump[2 * i] = (uint32_t)e; dump[2 * i + 1] = (uint32_t)(e >> 32);
        }
    } else if (mode == 16 && tid == 0) {
        /* iters = 1: map 32 x 16 MB at VA = IOVA 0xa0000000 (cdsp_mem) into
         * JTLB slots 96..127; iters = 0: remove them. dump[0] = status. */
        volatile uint32_t *dump = (volatile uint32_t *)(ARENA_UNCACHED + 0x3000u);
        uint32_t bad = 0;
        for (uint32_t i = 0; i < 32; i++) {
            uint64_t want = ((uint64_t)(0xc0000000u | ((0xa0000000u >> 12) + i * 0x1000u)) << 32) |
                            (0xf7000000u | ((((0xa0000000u >> 12) + i * 0x1000u) << 1) | 0x40u));
            uint64_t have = tlb_read(96u + i);
            if (have && have != want) { bad = 96u + i; break; }
        }
        if (bad) { dump[0] = 0xbad00000u | bad; }
        else {
            for (uint32_t i = 0; i < 32; i++) {
                uint32_t pn = (0xa0000000u >> 12) + i * 0x1000u;
                uint64_t entry = iters ? (((uint64_t)(0xc0000000u | pn) << 32) | (0xf7000000u | (pn << 1) | 0x40u)) : 0ull;
                uint32_t index = 96u + i;
                __asm__ volatile("tlbw(%0,%1)\n isync" :: "r"(entry), "r"(index) : "memory");
            }
            dump[0] = iters ? 1u : 2u;
            dump[1] = tlb_probe(0xa0000000u, 0);
            dump[2] = tlb_probe(0xbf000000u, 0);
        }
    } else if (mode == 15 && tid == 0) {
        /* One DMA with DM2 = `bytes`, restoring DM2 afterwards unless iters & 1. */
        volatile uint32_t *dump = (volatile uint32_t *)(ARENA_UNCACHED + 0x3000u);
        uint32_t saved = dm_read(2), st;
        dump[0] = saved;
        __asm__ volatile("%0 = dmpause" : "=r"(st) :: "memory");
        dump[1] = st;
        dm_write(2, bytes);
        dump[2] = dm_read(2);
        dump[3] = dm_read(0);
        volatile uint32_t *dv = (volatile uint32_t *)(ARENA_UNCACHED + 0xa000u);
        uint32_t desc = ARENA_UNCACHED + 0xa000u;
        volatile uint32_t *tail = (volatile uint32_t *)(0x31000000u + stride - 4u);
        *tail = 0xa5a5a5a5u;
        Desc x = {0};
        x.next = 0; x.desc_size = 1; x.src_bypass = 1; x.dst_bypass = 1;
        x.src = addr; x.dst = 0x31000000u; x.desc_type = 9;
        x.row_size = stride; x.nrows_lo = 1; x.src_stride = stride; x.dst_stride = stride;
        const uint32_t *xs = (const uint32_t *)&x;
        for (uint32_t i = 0; i < 8; i++) dv[i] = xs[i];
        __asm__ volatile("barrier" ::: "memory");
        uint64_t start = cycles();
        __asm__ volatile("release(%0):at" :: "r"(desc) : "memory");
        __asm__ volatile("dmstart(%0)" :: "r"(desc) : "memory");
        while (*tail == 0xa5a5a5a5u) {
            if (cycles() - start > 50000000u) { r1 = 0xdead; break; }
        }
        dump[4] = (uint32_t)(cycles() - start);
        for (uint32_t i = 0; i < 6; i++) dump[8 + i] = dm_read(i);
        dump[14] = dv[1];
        if (!(iters & 1u)) dm_write(2, saved);
        dump[5] = dm_read(2);
        result = *(volatile uint32_t *)0x31000000u;
    }
    uint64_t c1 = cycles();
    out->cyc_lo = (uint32_t)(c1 - c0); out->cyc_hi = (uint32_t)((c1 - c0) >> 32);
    out->result = result; out->r0 = r0; out->r1 = r1;
    if (mode != 10) out->pk_lo = 0;
    __asm__ volatile("barrier" ::: "memory");
    out->done = seq;
    *(volatile uint32_t *)(ARENA_UNCACHED + 128 + tid * 16) = seq;
    __asm__ volatile("barrier" ::: "memory");
}
