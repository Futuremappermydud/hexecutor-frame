/* Focused check of the data-movement fast paths in prepared_datamove.inc and
 * prepared_vcopy.h against the generic reference kernels, for the host and
 * the V75 simulator.
 *
 *   test_prepared_datamove_sim [cycles]
 *
 * Every case builds random blocked tensors over one pool (absent blocks, slice
 * offsets, shuffled block placement, misaligned pool bases, aliasing blocks),
 * runs the kernel once with the fast paths disabled (the generic reference)
 * and once enabled, each on its own copy of the pool image, and requires the
 * whole pool, the dry-pass scratch measure and the result code to agree.
 * "cycles" also prints pcycles (simulator) or clock ticks (host) for the
 * real-size shapes, reference and fast. Run via run_prepared_datamove_sim.sh. */
static int dm_on = 1;
#define DM_ENABLED dm_on
#include "prepared_reference.c"
#include <stdio.h>
#ifdef __hexagon__
#include <hexagon_sim_timer.h>
#define TICKS() hexagon_sim_read_pcycles()
#else
#include <time.h>
#define TICKS() ((uint64_t)clock())
#endif

enum { POOL_BYTES = 1 << 20, SLOTS = POOL_BYTES / 2048, PI = 2, PC = 5, MAXB = 1024 };
static uint32_t rs = 0x1234567;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

static uint8_t image[POOL_BYTES + 256] __attribute__((aligned(128)));
static uint8_t memA[POOL_BYTES + 256] __attribute__((aligned(128)));
static uint8_t memB[POOL_BYTES + 256] __attribute__((aligned(128)));
static uint8_t konst[256] __attribute__((aligned(128)));
static uint8_t scratch[1 << 20] __attribute__((aligned(128)));
static uint8_t tile[4096];
static uint32_t slot_order[SLOTS];
static uint32_t used_slots, lim = 160;   /* slots in play: only these bytes are filled and compared */
static uint32_t tabs[8][MAXB];
static int bad, cases, fast_taken;

static void shuffle_slots(void) {
    for (uint32_t i = 0; i < lim; i++) slot_order[i] = i;
    for (uint32_t i = lim - 1; i > 0; i--) { uint32_t j = rnd() % (i + 1), t = slot_order[i]; slot_order[i] = slot_order[j]; slot_order[j] = t; }
    used_slots = 0;
}
static void fill_image(void) {
    for (size_t i = 0; i < sizeof tile; i += 4) { uint32_t v = rnd(); memcpy(tile + i, &v, 4); }
    for (size_t at = 0; at < (size_t)lim * 2048; at += 2048) memcpy(image + at, tile + (rnd() & 2047), 2048);
}
/* A blocked tensor of element size es with dims d inside m at origin p. absent: percent of absent blocks. */
static uint32_t *mk_blocked(T *t, uint32_t es, const int32_t *d, const int32_t *m, const int32_t *p, uint32_t slice, int absent, uint32_t *tab) {
    memset(t, 0, sizeof *t);
    t->dtype = es == 1 ? DT_U8 : DT_U16; t->esize = es; t->flags = F_SHAPE | F_BLOCKED | F_CROUTON; t->crouton = 1;
    for (int i = 0; i < 4; i++) { t->d[i] = d[i]; t->m[i] = m[i]; t->p[i] = p[i]; }
    int64_t cw = es == 1 ? 8 : 4;
    uint32_t n = (uint32_t)(((m[0] * ((m[1] + 7) / 8)) * ((m[2] + cw - 1) / cw)) * ((m[3] + 31) / 32)) + slice;
    if (n > MAXB) { fprintf(stderr, "table too big %lu\n", (unsigned long)n); exit(1); }
    for (uint32_t i = 0; i < n; i++) {
        if (i < slice || (absent && (int)(rnd() % 100) < absent)) { tab[i] = 0; continue; }
        if (used_slots >= lim) { fprintf(stderr, "out of slots\n"); exit(1); }
        tab[i] = (slot_order[used_slots++] * 2048u) | PI;
    }
    t->blocks = tab; t->nblocks = n; t->slice = slice;
    return tab;
}
static void mk_g(G *g, uint8_t *mem, int dry, uint8_t *scr, size_t cap, int measure) {
    memset(g, 0, sizeof *g);
    g->dry = dry; g->measure_only = (uint8_t)measure; g->scratch = scr; g->cap = cap;
    g->pool[PI] = mem; g->pool_bytes[PI] = POOL_BYTES; g->write_limit[PI] = POOL_BYTES;
    g->pool[PC] = konst; g->pool_bytes[PC] = sizeof konst; g->write_limit[PC] = 0;
}
typedef int (*Kernel)(G *);
typedef struct { T in[CACHED_INPUTS], out; uint32_t nin; } Ops;
static void load_ops(G *g, const Ops *o) { memcpy(g->in, o->in, sizeof g->in); g->out = o->out; g->nin = o->nin; g->nout = 1; }

/* Runs the kernel in reference and fast mode; compares everything. Returns 1 when the fast mode took a path
 * (scratch need differs from the reference's). */
static int compare(Kernel k, const Ops *ops, int base_shift, const char *name, int expect_fast) {
    uint8_t *ma = memA + base_shift, *mb = memB + base_shift;
    memcpy(ma, image, (size_t)lim * 2048); memcpy(mb, image, (size_t)lim * 2048);
    size_t need[2]; int rc[2], rcd[2];
    for (int f = 0; f < 2; f++) {
        G g; dm_on = f;
        mk_g(&g, f ? mb : ma, 1, NULL, SIZE_MAX, 1); load_ops(&g, ops);
        rcd[f] = k(&g); need[f] = g.used;
        mk_g(&g, f ? mb : ma, 0, scratch, need[f], 0); load_ops(&g, ops);
        /* the wet pass must fit the dry measure; a dry failure means no wet run */
        rc[f] = rcd[f] ? rcd[f] : k(&g);
    }
    dm_on = 1;
    cases++;
    int ok = rc[0] == rc[1] && rcd[0] == rcd[1] && !memcmp(ma, mb, (size_t)lim * 2048) && need[1] <= need[0];
    int taken = need[1] != need[0] || (rcd[0] == 0 && expect_fast == 2);
    if (!ok) { bad++; printf("FAIL %s rc %d/%d dry %d/%d need %lu/%lu same=%d\n", name, rc[0], rc[1], rcd[0], rcd[1], (unsigned long)need[0], (unsigned long)need[1], !memcmp(ma, mb, (size_t)lim * 2048)); }
    if (expect_fast == 1 && !taken && rcd[0] == 0) { bad++; printf("FAIL %s: fast path not taken\n", name); }
    if (expect_fast == 0 && taken && rcd[0] == 0) { bad++; printf("FAIL %s: fast path taken unexpectedly\n", name); }
    fast_taken += taken;
    return taken;
}

/* ---- constants tensor (pool 5, direct, i32 shape (1,1,1,n)) */
static void mk_const(T *t, uint32_t off, const int32_t *v, int n) {
    memset(t, 0, sizeof *t);
    t->dtype = DT_I32; t->esize = 4; t->flags = F_SHAPE | F_DIRECT; t->pool = PC; t->offset = off;
    int32_t d[4] = {1, 1, 1, n};
    for (int i = 0; i < 4; i++) { t->d[i] = d[i]; t->m[i] = d[i]; }
    memcpy(konst + off, v, (size_t)n * 4);
}

/* ---- space to depth */
static int s2d_case(int ci, int oh, int ow, int ph, int pw, int absent, int slice_i, int slice_o, int shift, int alias, int expect_fast) {
    int oc = 4 * ci;
    int32_t hrows = ph + 2 * oh + 1 + (int32_t)(rnd() % 8);
    int32_t im[4] = {1, hrows, pw + 2 * ow + (int32_t)(rnd() % 2), ci}, id[4] = {1, hrows - ph, im[2] - pw, ci}, ip[4] = {0, ph, pw, 0};
    int32_t om[4] = {1, (oh + 7) / 8 * 8, ow + (int32_t)(rnd() % 8), oc}, od[4] = {1, oh, ow, oc}, op[4] = {0, 0, 0, 0};
    /* reference ow = (im2 - pw) / 2 capped at om2 */
    shuffle_slots(); fill_image();
    Ops ops; memset(&ops, 0, sizeof ops);
    mk_blocked(&ops.in[0], 1, id, im, ip, (uint32_t)slice_i, absent, tabs[0]);
    int32_t sv[4] = {2, 2, 2, 2};
    memset(&ops.in[1], 0, sizeof(T)); ops.in[1].flags = F_SV | F_SHAPE; (void)sv;
    mk_blocked(&ops.out, 1, od, om, op, (uint32_t)slice_o, absent, tabs[1]);
    ops.nin = 2;
    if (alias) { /* the output reuses input slots */
        for (uint32_t i = 0; i < ops.out.nblocks; i++) { uint32_t j = (i * 7) % ops.in[0].nblocks, tries = 0; while (!tabs[0][j] && tries++ < ops.in[0].nblocks) j = (j + 1) % ops.in[0].nblocks; if (tabs[0][j] && i % 3 == 0) tabs[1][i] = tabs[0][j]; }
    }
    char name[96]; snprintf(name, sizeof name, "s2d ci%d oh%d ow%d ph%d pw%d abs%d sl%d/%d sh%d al%d", ci, oh, ow, ph, pw, absent, slice_i, slice_o, shift, alias);
    return compare(k_space_to_depth, &ops, shift, name, expect_fast);
}

/* ---- resize near */
static int resize_case(int c, int h_in, int w_in, int h_out, int w_out, int hoff, int absent, int shift, int v4, int expect_fast, int pin2, int po2) {
    int32_t im[4] = {1, h_in, w_in + pin2, c}, id[4] = {1, h_in, w_in, c}, ip[4] = {0, 0, pin2, 0};
    int32_t om[4] = {1, h_out, w_out + po2, c}, od[4] = {1, h_out, w_out, c}, op[4] = {0, 0, po2, 0};
    shuffle_slots(); fill_image();
    Ops ops; memset(&ops, 0, sizeof ops);
    mk_blocked(&ops.in[0], 1, id, im, ip, 0, absent, tabs[0]);
    int32_t hp[7] = {h_in, h_out, (int32_t)0x80000000, 0, v4, hoff, h_out}, wp[7] = {w_in, w_out, (int32_t)0x80000000, 0, v4, 0, w_out};
    mk_const(&ops.in[1], 0, hp, 7); mk_const(&ops.in[2], 32, wp, 7);
    mk_blocked(&ops.out, 1, od, om, op, 0, absent, tabs[1]);
    ops.nin = 3;
    char name[96]; snprintf(name, sizeof name, "resize c%d %dx%d->%dx%d off%d abs%d sh%d v4=%d", c, h_in, w_in, h_out, w_out, hoff, absent, shift, v4);
    return compare(k_resize, &ops, shift, name, expect_fast);
}

/* ---- transpose H<->W, gold = t_read / permute / t_write */
static int gold_transpose(G *g, const T *x, const T *o) {
    A a; int e = t_read(g, x, &a, 0); if (e) return e;
    int32_t n[4] = {a.n[0], a.n[2], a.n[1], a.n[3]};
    A r; e = anew(g, &r, n, a.dt); if (e) return e;
    if (!g->dry) {
        const uint16_t *s = a.p; uint16_t *d = r.p;
        for (int b = 0; b < n[0]; b++) for (int h = 0; h < n[1]; h++) for (int w = 0; w < n[2]; w++) for (int c = 0; c < n[3]; c++)
            d[(((size_t)b * n[1] + h) * n[2] + w) * n[3] + c] = s[(((size_t)b * a.n[1] + w) * a.n[2] + h) * a.n[3] + c];
    }
    return t_write(g, o, &r);
}
static T tr_x, tr_o;
static int tr_kernel(G *g) {
    if (dm_on) { int r = tr_hw_fast(g, &tr_x, &tr_o); if (r >= 0) return r; }
    return gold_transpose(g, &tr_x, &tr_o);
}
static int tr_case(int h, int w, int c, int absent, int shift, int expect_fast, int pw_in, int po_h) {
    int32_t im[4] = {1, h + po_h, w + pw_in, c}, id[4] = {1, h, w, c}, ip[4] = {0, po_h, pw_in, 0};
    int32_t om[4] = {1, w, h, c}, od[4] = {1, w, h, c}, op[4] = {0, 0, 0, 0};
    shuffle_slots(); fill_image();
    Ops ops; memset(&ops, 0, sizeof ops);
    mk_blocked(&tr_x, 2, id, im, ip, 0, absent, tabs[0]);
    mk_blocked(&tr_o, 2, od, om, op, 0, absent, tabs[1]);
    ops.nin = 1; ops.in[0] = tr_x; ops.out = tr_o;
    char name[96]; snprintf(name, sizeof name, "transpose hw %dx%dx%d abs%d sh%d pw%d po%d", h, w, c, absent, shift, pw_in, po_h);
    return compare(tr_kernel, &ops, shift, name, expect_fast);
}


/* ---- flat (direct) tensors */
static void mk_flat(T *t, uint32_t es, uint32_t off, const int32_t *d, const int32_t *m, const int32_t *p) {
    memset(t, 0, sizeof *t);
    t->dtype = es == 1 ? DT_U8 : es == 2 ? DT_U16 : DT_I32; t->esize = es; t->flags = F_SHAPE | F_DIRECT; t->pool = PI; t->offset = off;
    for (int i = 0; i < 4; i++) { t->d[i] = d[i]; t->m[i] = m[i]; t->p[i] = p[i]; }
}
static void mk_sv(T *t, const int32_t *v) { memset(t, 0, sizeof *t); t->flags = F_SV; t->dtype = DT_I32; memcpy(t->sv, v, 16); }
static uint32_t flat_bytes(const int32_t *m, uint32_t es) { return (uint32_t)(m[0] * m[1] * m[2] * m[3]) * es; }
static void flat_pair(T *x, T *o, uint32_t es, const int32_t *xd, const int32_t *xm, const int32_t *xp, const int32_t *od, const int32_t *om, const int32_t *op, int alias) {
    fill_image();
    uint32_t xo = 128 * (rnd() % 8), ob = flat_bytes(xm, es);
    uint32_t oo = alias ? xo + (rnd() % (ob + 1)) / 2 : (xo + ob + 256 + 128 * (rnd() % 4));
    mk_flat(x, es, xo, xd, xm, xp); mk_flat(o, es, oo, od, om, op);
    if (oo + flat_bytes(om, es) > POOL_BYTES) { fprintf(stderr, "flat case too big\n"); exit(1); }
}
static int slice_case(uint32_t es, const int32_t *xm, const int32_t *xp, const int32_t *xd, const int32_t *st, const int32_t *sz, const int32_t *om, const int32_t *op, int alias, int shift, int expect) {
    Ops ops; memset(&ops, 0, sizeof ops);
    flat_pair(&ops.in[0], &ops.out, es, xd, xm, xp, sz, om, op, alias);
    mk_sv(&ops.in[1], st); mk_sv(&ops.in[2], sz); ops.nin = 3;
    char name[64]; snprintf(name, sizeof name, "slice es%lu al%d sh%d", (unsigned long)es, alias, shift);
    return compare(k_slice, &ops, shift, name, expect);
}
static int slicepad_case(uint32_t es, const int32_t *xm, const int32_t *xd, const int32_t *pad, const int32_t *st, const int32_t *sz, int64_t fill, int alias, int expect) {
    static const int32_t z[4] = {0, 0, 0, 0};
    Ops ops; memset(&ops, 0, sizeof ops);
    flat_pair(&ops.in[0], &ops.out, es, xd, xm, z, sz, sz, z, alias);
    mk_sv(&ops.in[1], pad); mk_sv(&ops.in[2], st); mk_sv(&ops.in[3], sz);
    int32_t f = (int32_t)fill; mk_const(&ops.in[4], 64, &f, 1); ops.in[4].d[3] = 1; ops.in[4].m[3] = 1; ops.nin = 5;
    return compare(k_slice_pad, &ops, 0, "slicepad", expect);
}
static int gather_case(uint32_t es, const int32_t *xd, int axis, const int32_t *idx, int ni, int alias, int expect) {
    static const int32_t z[4] = {0, 0, 0, 0};
    int32_t od[4]; memcpy(od, xd, 16); od[axis] = ni;
    Ops ops; memset(&ops, 0, sizeof ops);
    flat_pair(&ops.in[0], &ops.out, es, xd, xd, z, od, od, z, alias);
    int32_t ax = axis; mk_const(&ops.in[1], 64, &ax, 1); mk_const(&ops.in[2], 96, idx, ni); mk_const(&ops.in[3], 160, &ax, 1); ops.nin = 4;
    return compare(k_gather, &ops, 0, "gather", expect);
}
static T ft_x, ft_o; static int32_t ft_perm[4];
static int ft_kernel(G *g) {
    if (dm_on) { int r = tr_flat_fast(g, &ft_x, &ft_o, ft_perm); if (r >= 0) return r; }
    A a; int e = t_read(g, &ft_x, &a, 0); if (e) return e;
    int32_t n[4]; for (int i = 0; i < 4; i++) n[i] = a.n[ft_perm[i]];
    A r; e = anew(g, &r, n, a.dt); if (e) return e;
    if (!g->dry) {
        size_t es = dsize(a.dt), st[4] = {(size_t)a.n[1] * a.n[2] * a.n[3], (size_t)a.n[2] * a.n[3], (size_t)a.n[3], 1}, k = 0;
        for (int i0 = 0; i0 < n[0]; i0++) for (int i1 = 0; i1 < n[1]; i1++) for (int i2 = 0; i2 < n[2]; i2++) for (int i3 = 0; i3 < n[3]; i3++, k++)
            memcpy((uint8_t *)r.p + k * es, (uint8_t *)a.p + (i0 * st[ft_perm[0]] + i1 * st[ft_perm[1]] + i2 * st[ft_perm[2]] + i3 * st[ft_perm[3]]) * es, es);
    }
    return t_write(g, &ft_o, &r);
}
static int ftr_case(uint32_t es, const int32_t *xd, const int32_t *xm, const int32_t *xp, const int *perm, int alias, int expect) {
    static const int32_t z[4] = {0, 0, 0, 0};
    int32_t od[4]; for (int i = 0; i < 4; i++) { od[i] = xd[perm[i]]; ft_perm[i] = perm[i]; }
    Ops ops; memset(&ops, 0, sizeof ops);
    flat_pair(&ft_x, &ft_o, es, xd, xm, xp, od, od, z, alias);
    ops.nin = 1; ops.in[0] = ft_x; ops.out = ft_o;
    return compare(ft_kernel, &ops, 0, "flat transpose", expect);
}

/* ---- vector copy */
static uint8_t cp_src[8192 + 512] __attribute__((aligned(128))), cp_a[8192 + 512] __attribute__((aligned(128))), cp_b[8192 + 512] __attribute__((aligned(128)));
static void vcopy_tests(void) {
    for (int it = 0; it < 120; it++) {
        size_t n = it < 40 ? (size_t)it * 37 : it < 80 ? 400 + rnd() % 3000 : 4096 + rnd() % 3000;
        size_t so = rnd() % 200, dof = rnd() % 200;
        int overlap = it % 7 == 6;
        for (size_t i = 0; i < sizeof cp_src; i++) cp_src[i] = (uint8_t)rnd();
        memcpy(cp_a, cp_src, sizeof cp_a); memcpy(cp_b, cp_src, sizeof cp_b);
        if (overlap) { /* within one buffer, both directions */
            size_t shift = 1 + rnd() % 300; if (it & 1) { so = dof + shift; } else { dof = so + shift; }
            if (n + (so > dof ? so : dof) > sizeof cp_a - 8) n = sizeof cp_a - 8 - (so > dof ? so : dof);
            memmove(cp_a + dof, cp_a + so, n); qd_vcopy(cp_b + dof, cp_b + so, n);
        } else {
            memmove(cp_a + dof, cp_src + so, n); qd_vcopy(cp_b + dof, cp_src + so, n);
        }
        cases++;
        if (memcmp(cp_a, cp_b, sizeof cp_a)) { bad++; printf("FAIL vcopy n=%lu so=%lu do=%lu overlap=%d\n", (unsigned long)n, (unsigned long)so, (unsigned long)dof, overlap); }
    }
}

static void cycles_tests(void) {
    static const char *const hdr = "%-34s %12s %12s\n";
    printf(hdr, "case (pcycles or ticks)", "reference", "fast");
    lim = SLOTS;
    uint64_t t[2];
    /* real shapes */
    for (int which = 0; which < 4; which++) {
        for (int f = 0; f < 2; f++) {
            rs = 99;
            shuffle_slots(); fill_image();
            Ops ops; memset(&ops, 0, sizeof ops);
            Kernel k = NULL; const char *nm = "";
            if (which == 0) { /* s2d: in (1,16,130,32 @ 7,7) m 24x144 -> out (1,8,66,128) m 8x72 */
                int32_t im[4] = {1, 24, 144, 32}, id[4] = {1, 17, 137, 32}, ip[4] = {0, 7, 7, 0}, om[4] = {1, 8, 72, 128}, od[4] = {1, 8, 66, 128}, op[4] = {0, 0, 0, 0};
                mk_blocked(&ops.in[0], 1, id, im, ip, 0, 0, tabs[0]); mk_blocked(&ops.out, 1, od, om, op, 0, 0, tabs[1]); ops.nin = 2; k = k_space_to_depth; nm = "space_to_depth 8x68x128 (ci 32)";
            } else if (which == 1) {
                int32_t im[4] = {1, 16, 128, 32}, id[4] = {1, 16, 128, 32}, ip[4] = {0, 0, 0, 0}, om[4] = {1, 8, 256, 32}, od[4] = {1, 8, 256, 32}, op[4] = {0, 0, 0, 0};
                mk_blocked(&ops.in[0], 1, id, im, ip, 0, 0, tabs[0]); mk_blocked(&ops.out, 1, od, om, op, 0, 0, tabs[1]);
                int32_t hp[7] = {16, 8, (int32_t)0x80000000, 0, 0x40000000, 4, 8}, wp[7] = {128, 256, (int32_t)0x80000000, 0, 0x40000000, 0, 256};
                mk_const(&ops.in[1], 0, hp, 7); mk_const(&ops.in[2], 32, wp, 7); ops.nin = 3; k = k_resize; nm = "resize 8x256x32 from 16x128x32";
            } else if (which == 2) {
                int32_t im[4] = {1, 80, 32, 64}, id[4] = {1, 80, 32, 64}, ip[4] = {0, 0, 0, 0}, om[4] = {1, 32, 80, 64}, od[4] = {1, 32, 80, 64}, op[4] = {0, 0, 0, 0};
                mk_blocked(&tr_x, 2, id, im, ip, 0, 0, tabs[0]); mk_blocked(&tr_o, 2, od, om, op, 0, 0, tabs[1]);
                ops.nin = 1; ops.in[0] = tr_x; ops.out = tr_o; k = tr_kernel; nm = "transpose hw u16 80x32x64";
            } else {
                nm = "vcopy 512 KiB (misaligned src)";
            }
            uint64_t c0;
            if (which < 3) {
                dm_on = f;
                memcpy(memA, image, (size_t)lim * 2048);
                G g; mk_g(&g, memA, 1, NULL, SIZE_MAX, 1); load_ops(&g, &ops); int r = k(&g); size_t need = g.used;
                if (r) { printf("cycles case %s dry failed %d\n", nm, r); bad++; continue; }
                mk_g(&g, memA, 0, scratch, need, 0); load_ops(&g, &ops);
                c0 = TICKS(); r = k(&g); t[f] = TICKS() - c0;
                if (r) { printf("cycles case %s run failed %d\n", nm, r); bad++; }
            } else {
                dm_on = f;
                c0 = TICKS();
                if (f) qd_vcopy(cp_b + 5, cp_src + 3, 8192 - 8); else memmove(cp_a + 5, cp_src + 3, 8192 - 8);
                t[f] = (TICKS() - c0) * 64;
            }
        }
        char nm[64];
        snprintf(nm, sizeof nm, "%s", which == 0 ? "space_to_depth 8x68x128 (ci 32)" : which == 1 ? "resize 8x256x32 from 16x128x32" : which == 2 ? "transpose hw u16 80x32x64" : "vcopy 512 KiB (x64 of 8 KiB)");
        printf("%-34s %12llu %12llu\n", nm, (unsigned long long)t[0], (unsigned long long)t[1]);
    }
    dm_on = 1;
}

int main(int argc, char **argv) {
    int cycles = argc > 1 && !strcmp(argv[1], "cycles");
    (void)cycles;
    vcopy_tests();
    /* space to depth: real-like, edges, variants that must decline */
    s2d_case(32, 8, 68, 7, 7, 0, 0, 0, 0, 0, 1);
    s2d_case(32, 8, 36, 7, 7, 0, 0, 0, 0, 0, 1);
    for (int it = 0; it < 24; it++) {
        int ci = it % 3 == 0 ? 64 : 32, oh = 1 + (int)(rnd() % 3), ow = 4 * (1 + (int)(rnd() % 4)), ph = (int)(rnd() % 10), pw = it % 4 == 0 ? 15 : 7;
        int absent = it % 2 ? 15 : 0, si = it % 5 == 0 ? 3 : 0, so = it % 7 == 0 ? 2 : 0, shift = it % 6 == 0 ? 8 : 0;
        s2d_case(ci, oh, ow, ph, pw, absent, si, so, shift, it % 11 == 10, it % 11 == 10 || shift ? 0 : 1);
    }
    s2d_case(32, 2, 6, 7, 7, 0, 0, 0, 0, 0, 0);   /* ow not a multiple of 4 */
    s2d_case(32, 2, 8, 7, 5, 0, 0, 0, 0, 0, 0);   /* pw not 7 mod 8 */
    /* resize */
    resize_case(32, 16, 128, 8, 256, 4, 0, 0, 0x40000000, 1, 0, 0);
    resize_case(64, 8, 64, 8, 128, 0, 0, 0, 0, 1, 0, 0);
    for (int it = 0; it < 20; it++) {
        int c = it % 3 == 0 ? 64 : 32, win = 8 << (it % 3), hin = 2 + (int)(rnd() % 6), hout = 1 + (int)(rnd() % 8);
        int hoff = (int)(rnd() % 3), absent = it % 2 ? 20 : 0;
        if (hin <= (hout - 1) / 2 + hoff) hin = (hout - 1) / 2 + hoff + 1;
        resize_case(c, hin, win, hout, 2 * win, hoff, absent, it % 5 == 0 ? 8 : 0, it % 4 ? 0x40000000 : 0x7fffffff, it % 5 != 0 ? 1 : 0, it % 6 == 0 ? 8 : 0, it % 6 == 0 ? 8 : 0);
    }
    resize_case(32, 4, 16, 4, 32, 0, 0, 0, (int)0x80000000, 0, 0, 0);   /* v4 too large: generic */
    /* transpose */
    tr_case(8, 8, 32, 0, 0, 1, 0, 0);
    tr_case(16, 8, 64, 10, 0, 1, 0, 0);
    for (int it = 0; it < 12; it++) {
        int h = 2 * (1 + (int)(rnd() % 6)), w = 2 * (1 + (int)(rnd() % 6)), c = it % 3 == 0 ? 64 : 32;
        tr_case(h, w, c, it % 2 ? 15 : 0, it % 5 == 0 ? 8 : 0, it % 5 == 0 ? 0 : 1, it % 4 == 1 ? 2 : 0, it % 4 == 2 ? 2 : 0);
    }
    tr_case(3, 4, 32, 0, 0, 0, 0, 0);   /* odd H */
    /* flat slice / slice_pad / gather / transpose */
    {
        int32_t m1[4] = {1, 12, 16, 2}, z[4] = {0, 0, 0, 0}, st[4] = {0, 3, 0, 0}, sz[4] = {1, 4, 16, 2}, o1[4] = {1, 4, 16, 2};
        slice_case(2, m1, z, m1, st, sz, o1, z, 0, 0, 1);
        slice_case(2, m1, z, m1, st, sz, o1, z, 1, 0, 0);
        int32_t xp[4] = {0, 1, 2, 0}, xd[4] = {1, 9, 13, 2}, st2[4] = {0, 2, 3, 0}, sz2[4] = {1, 5, 7, 2}, om[4] = {1, 6, 9, 2}, op[4] = {0, 1, 1, 0};
        slice_case(2, m1, xp, xd, st2, sz2, om, op, 0, 8, 1);
        slice_case(1, m1, xp, xd, st2, sz2, om, op, 0, 0, 1);
        int32_t xm[4] = {1, 5, 7, 40}, xd2[4] = {1, 5, 7, 40}, pad[4] = {0, 0, 0, 0}, st3[4] = {0, 0, 0, 12}, sz3[4] = {1, 5, 7, 16};
        slicepad_case(1, xm, xd2, pad, st3, sz3, 0, 0, 1);
        int32_t pad2[4] = {0, 1, 2, 3}, st4[4] = {0, 0, 0, 0}, sz4[4] = {1, 7, 9, 50};
        slicepad_case(1, xm, xd2, pad2, st4, sz4, 7, 0, 1);
        slicepad_case(2, xm, xd2, pad2, st4, sz4, 300, 0, 1);
        slicepad_case(2, xm, xd2, pad2, st4, sz4, 300, 1, 0);
        int32_t gd[4] = {1, 1, 40, 80}, gi[8] = {5, 39, -1, 0, 7, 7, 20, -40};
        gather_case(1, gd, 2, gi, 8, 0, 1);
        gather_case(1, gd, 2, gi, 8, 1, 0);
        gather_case(2, gd, 2, gi, 8, 0, 1);
        int32_t gj[8] = {1, 0, 1, -2, 0, 0, 1, 1}; int32_t gd2[4] = {2, 2, 6, 5};
        gather_case(1, gd2, 0, gj, 3, 0, 1);
        gather_case(1, gd2, 1, gj, 8, 0, 1);
        int32_t bad_idx[3] = {1, 99, 0};
        gather_case(1, gd, 2, bad_idx, 3, 0, 1);   /* out of range: both fail the same way */
        int32_t td[4] = {1, 1, 37, 21}, tz[4] = {0, 0, 0, 0};
        int p0132[4] = {0, 1, 3, 2}, p0231[4] = {0, 2, 3, 1}, p1023[4] = {1, 0, 2, 3}, p3210[4] = {3, 2, 1, 0};
        ftr_case(1, td, td, tz, p0132, 0, 1);
        ftr_case(2, td, td, tz, p0132, 0, 1);
        int32_t td2[4] = {2, 3, 19, 5}, tp[4] = {0, 1, 2, 0}, tm[4] = {3, 4, 21, 5};
        ftr_case(2, td2, tm, tp, p0231, 0, 1);
        ftr_case(1, td2, tm, tp, p1023, 0, 1);
        ftr_case(1, td2, td2, tz, p3210, 0, 1);
        ftr_case(1, td, td, tz, p0132, 1, 0);
    }
    printf("%s: %d cases, %d took a fast path, %d mismatches\n", bad ? "FAIL" : "PASS", cases, fast_taken, bad);
    if (cycles) cycles_tests();
    return bad != 0;
}
