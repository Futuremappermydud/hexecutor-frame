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
#define DM_STATS 1
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
static int bad, cases, fast_taken, movers;
static int want_id = -1;   /* dm_stat counter the case must (expect 1) or must not (expect 0) advance */
enum { FLAT_BASE = 768 * 1024 };
static uint32_t ext_len;   /* bytes at FLAT_BASE that are also copied, compared and filled (flat tensors) */
static void fill_ext(uint32_t len) { ext_len = len; for (uint32_t i = 0; i < len; i += 4) { uint32_t v = rnd(); memcpy(image + FLAT_BASE + i, &v, 4); } }

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
static Nprg2ExecCall tcall;
static Nprg2Pool tpools[7];
static int tmemo = 0;   /* 1: the harness call carries a program, so the per-command memos are live */
static uint32_t tcmd = 0;
static void mk_g(G *g, uint8_t *mem, int dry, uint8_t *scr, size_t cap, int measure) {
    memset(g, 0, sizeof *g);
    tcall.pools = tpools; tcall.program = tmemo ? (const void *)&tcall : NULL; tcall.command_index = tcmd;
    g->call = &tcall;
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
    memcpy(ma + FLAT_BASE, image + FLAT_BASE, ext_len); memcpy(mb + FLAT_BASE, image + FLAT_BASE, ext_len);
    size_t need[2]; int rc[2], rcd[2];
    unsigned long long st0 = 0, st1 = 0;
    for (int f = 0; f < 2; f++) {
        G g; dm_on = f;
        mk_g(&g, f ? mb : ma, 1, NULL, SIZE_MAX, 1); load_ops(&g, ops);
        rcd[f] = k(&g); need[f] = g.used;
        mk_g(&g, f ? mb : ma, 0, scratch, need[f], 0); load_ops(&g, ops);
        /* the wet pass must fit the dry measure; a dry failure means no wet run */
        if (f && want_id >= 0) st0 = dm_stat[want_id];
        rc[f] = rcd[f] ? rcd[f] : k(&g);
        if (f && want_id >= 0) st1 = dm_stat[want_id];
        if (tmemo && !rc[f]) { /* the second wet run of a command takes the per-command memo (Memo.dmv); the reference runs twice too */
            mk_g(&g, f ? mb : ma, 0, scratch, need[f], 0); load_ops(&g, ops);
            rc[f] = k(&g);
        }
    }
    dm_on = 1;
    cases++; tcmd = (uint32_t)cases % 8000;
    int same = !memcmp(ma, mb, (size_t)lim * 2048) && !memcmp(ma + FLAT_BASE, mb + FLAT_BASE, ext_len);
    int ok = rc[0] == rc[1] && rcd[0] == rcd[1] && same && need[1] <= need[0];
    int taken = need[1] != need[0] || (rcd[0] == 0 && expect_fast == 2);
    if (want_id >= 0) { taken = st1 > st0; if (taken && !(rc[1] == 0)) taken = 0; }
    if (!ok) { bad++; printf("FAIL %s rc %d/%d dry %d/%d need %lu/%lu same=%d\n", name, rc[0], rc[1], rcd[0], rcd[1], (unsigned long)need[0], (unsigned long)need[1], same); }
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
    if (dm_on) { int r = tr_fast_dispatch(g, &ft_x, &ft_o, ft_perm); if (r >= 0) return r; }
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

static int ftr_case2(uint32_t es, const int32_t *xd, const int32_t *xm, const int32_t *xp, const int *perm, const int32_t *om, const int32_t *op, int alias, int expect) {
    int32_t od[4]; for (int i = 0; i < 4; i++) { od[i] = xd[perm[i]]; ft_perm[i] = perm[i]; }
    Ops ops; memset(&ops, 0, sizeof ops);
    flat_pair(&ft_x, &ft_o, es, xd, xm, xp, od, om, op, alias);
    ops.nin = 1; ops.in[0] = ft_x; ops.out = ft_o;
    return compare(ft_kernel, &ops, 0, "flat transpose (strided)", expect);
}

/* ---- blocked transposes (any permutation, gold = t_read / permute / t_write) */
static int round32(int v) { return (v + 31) / 32 * 32; }
static int trb_case(uint32_t es, const int *perm, const int32_t *xd, const int32_t *xp, const int32_t *op, int xsl, int osl, int absent, int shift, int alias, int expect, const char *tag) {
    int32_t n[4], xm[4], om[4];
    for (int i = 0; i < 4; i++) { n[i] = xd[perm[i]]; ft_perm[i] = perm[i]; }
    for (int i = 0; i < 3; i++) { xm[i] = xd[i] + xp[i]; om[i] = n[i] + op[i]; }
    xm[3] = round32(xd[3] + xp[3]); om[3] = round32(n[3] + op[3]);
    int cw = es == 1 ? 8 : 4;
    int nb = xm[0] * ((xm[1] + 7) / 8) * ((xm[2] + cw - 1) / cw) * (xm[3] / 32) + om[0] * ((om[1] + 7) / 8) * ((om[2] + cw - 1) / cw) * (om[3] / 32) + xsl + osl;
    lim = nb + 8 > 160 ? (uint32_t)(nb + 8) : 160; if (lim > SLOTS) lim = SLOTS;
    shuffle_slots(); fill_image();
    Ops ops; memset(&ops, 0, sizeof ops);
    mk_blocked(&ft_x, es, xd, xm, xp, (uint32_t)xsl, absent, tabs[0]);
    mk_blocked(&ft_o, es, n, om, op, (uint32_t)osl, absent, tabs[1]);
    if (alias) for (uint32_t i = 0; i < ft_o.nblocks; i += 3) if (i < ft_x.nblocks && tabs[0][i]) tabs[1][i] = tabs[0][i];
    ops.nin = 1; ops.in[0] = ft_x; ops.out = ft_o;
    char name[120]; snprintf(name, sizeof name, "blocked transpose es%lu %d%d%d%d d%d,%d,%d,%d xp%d,%d,%d,%d op%d,%d,%d,%d abs%d sl%d/%d sh%d al%d %s", (unsigned long)es, perm[0], perm[1], perm[2], perm[3], (int)xd[0], (int)xd[1], (int)xd[2], (int)xd[3], (int)xp[0], (int)xp[1], (int)xp[2], (int)xp[3], (int)op[0], (int)op[1], (int)op[2], (int)op[3], absent, xsl, osl, shift, alias, tag);
    want_id = perm[1] == 1 ? (es == 2 ? 7 : 8) : (es == 2 ? 2 : 9);
    int r = compare(ft_kernel, &ops, shift, name, expect);
    want_id = -1;
    lim = 160;
    return r;
}

/* ---- flat <-> padded blocked flow tensors (COPY_VALUES), gold = the generic kernel */
static int copyc2_case(int to_blocked, int H, int W, int p1, int p2, int slice, int absent, int shift, int fmstride, int overlap, int expect) {
    int32_t bd[4] = {1, H, W, 2}, bm[4] = {1, H + p1, W + p2, 32}, bp[4] = {0, p1, p2, 0};
    int32_t fd[4] = {1, H, W, 2}, fm[4] = {1, H, W + fmstride, 2}, fp[4] = {0, 0, 0, 0};
    { int nbk = (H + p1 + 7) / 8 * ((W + p2 + 3) / 4) + slice; lim = nbk + 8 > 160 ? (uint32_t)(nbk + 8) : 160; }
    shuffle_slots(); fill_image();
    uint32_t foff = overlap ? 2048u * 3 + 4 * (rnd() % 8) : FLAT_BASE + 4 * (rnd() % 64);
    fill_ext(4 * (uint32_t)(H * (W + fmstride)) + 512 + (overlap ? 0 : 0));
    Ops ops; memset(&ops, 0, sizeof ops);
    T bt, ft;
    mk_blocked(&bt, 2, bd, bm, bp, (uint32_t)slice, absent, tabs[0]);
    mk_flat(&ft, 2, foff, fd, fm, fp);
    if (overlap) for (uint32_t i = 0; i < bt.nblocks; i++) if (tabs[0][i]) { ft.offset = (tabs[0][i] & ~127u) + 4 * (rnd() % 8); break; }
    if (to_blocked) { ops.in[0] = ft; ops.out = bt; } else { ops.in[0] = bt; ops.out = ft; }
    ops.nin = 1;
    char name[100]; snprintf(name, sizeof name, "copy_values %s %dx%d p%d,%d sl%d abs%d sh%d fs%d ov%d", to_blocked ? "flat->blocked" : "blocked->flat", H, W, p1, p2, slice, absent, shift, fmstride, overlap);
    want_id = to_blocked ? 10 : 11;
    int r = compare(k_copy_values, &ops, shift, name, expect);
    want_id = -1; ext_len = 0; lim = 160;
    return r;
}

/* ---- u8 blocked -> flat COPY_VALUES (reshape: the pixel count and depth D are kept) */
static int cvb2f_align;
static int cvb2f_case(int H, int W, int D, int p1, int p2, int slice, int absent, int shift, int oh, int ow, int fstride, int overlap, int expect) {
    int32_t bd[4] = {1, H, W, D}, bm[4] = {1, H + p1, W + p2, (D + 31) / 32 * 32}, bp[4] = {0, p1, p2, 0};
    int32_t fd[4] = {1, oh, ow, D}, fm[4] = {1, oh, ow + fstride, D}, fp[4] = {0, 0, 0, 0};
    { int nbk = (H + p1 + 7) / 8 * ((W + p2 + 7) / 8) * ((D + 31) / 32) + slice; lim = nbk + 8 > 160 ? (uint32_t)(nbk + 8) : 160; if (lim > SLOTS) lim = SLOTS; }
    shuffle_slots(); fill_image();
    uint32_t foff = overlap ? 2048u * 3 + 4 * (rnd() % 8) : FLAT_BASE + (cvb2f_align ? 0 : 4 * (rnd() % 64));
    fill_ext((uint32_t)(oh * (ow + fstride) * D) + 512);
    Ops ops; memset(&ops, 0, sizeof ops);
    T bt, ft;
    mk_blocked(&bt, 1, bd, bm, bp, (uint32_t)slice, absent, tabs[0]);
    mk_flat(&ft, 1, foff, fd, fm, fp);
    if (overlap) for (uint32_t i = 0; i < bt.nblocks; i++) if (tabs[0][i]) { ft.offset = (tabs[0][i] & ~127u) + 4 * (rnd() % 8); break; }
    ops.in[0] = bt; ops.out = ft; ops.nin = 1;
    char name[100]; snprintf(name, sizeof name, "copy_values u8 blocked->flat %dx%dx%d -> %dx%d p%d,%d sl%d abs%d sh%d fs%d ov%d", H, W, D, oh, ow, p1, p2, slice, absent, shift, fstride, overlap);
    want_id = 5;
    int r = compare(k_copy_values, &ops, shift, name, expect);
    want_id = -1; ext_len = 0; lim = 160;
    return r;
}

/* ---- flat -> flat COPY_VALUES */
static int cvf2f_case(uint32_t es, uint32_t dt, const int32_t *xd, const int32_t *xm, const int32_t *xp, const int32_t *od, const int32_t *om, const int32_t *op, int alias, int expect) {
    Ops ops; memset(&ops, 0, sizeof ops);
    flat_pair(&ops.in[0], &ops.out, es, xd, xm, xp, od, om, op, alias);
    ops.in[0].dtype = dt; ops.out.dtype = dt; ops.nin = 1;
    want_id = 5;
    int r = compare(k_copy_values, &ops, 0, "copy_values flat->flat", expect);
    want_id = -1;
    return r;
}

/* ---- IM2COL */
static int im2col_case(int u16, int oh, int ow, int po1, int po2, int slice, int absent, int shift, int ipad1, int ipad2, int stride, int overlap, int expect) {
    uint32_t es = u16 ? 2 : 1;
    int32_t ks = u16 ? 4 : 3, cin = u16 ? 2 : 1, st = u16 ? 4 : stride;
    int32_t ih = u16 ? 4 * oh : st * (oh - 1) + ks, iw = u16 ? 4 * ow : st * (ow - 1) + ks;
    int32_t fd[4] = {1, ih, iw, cin}, fm[4] = {1, ih + ipad1, iw + ipad2, cin}, fp[4] = {0, ipad1 ? 1 : 0, ipad2 ? 2 : 0, 0};
    if (ipad1) fm[1] = ih + ipad1 + 1;
    if (ipad2) fm[2] = iw + ipad2 + 2;
    int32_t od[4] = {1, oh, ow, u16 ? 32 : 9}, om[4] = {1, oh + po1, ow + po2, 32}, op[4] = {0, po1, po2, 0};
    lim = 160; shuffle_slots(); fill_image();
    fill_ext(es * (uint32_t)(fm[1] * fm[2] * fm[3]) + 512);
    Ops ops; memset(&ops, 0, sizeof ops);
    mk_flat(&ops.in[0], es, FLAT_BASE + (rnd() % 200), fd, fm, fp);
    int32_t kv[4] = {ks, ks, cin, 32}, sv[4] = {1, st, st, 1};
    mk_sv(&ops.in[1], kv); mk_sv(&ops.in[2], sv);
    mk_blocked(&ops.out, es, od, om, op, (uint32_t)slice, absent, tabs[0]);
    if (overlap) for (uint32_t i = 0; i < ops.out.nblocks; i++) if (tabs[0][i]) { ops.in[0].offset = (tabs[0][i] & ~127u) + (rnd() % 8) * 128; break; }
    ops.nin = 3;
    char name[100]; snprintf(name, sizeof name, "im2col %s %dx%d po%d,%d sl%d abs%d sh%d ip%d,%d st%d ov%d", u16 ? "u16" : "u8", oh, ow, po1, po2, slice, absent, shift, ipad1, ipad2, (int)st, overlap);
    want_id = u16 ? 13 : 12;
    int r = compare(k_im2col, &ops, shift, name, expect);
    want_id = -1; ext_len = 0;
    return r;
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

/* Times one kernel in reference and fast mode (the caller has built the tensors over the shared pool image). */
static void cyc_case(const char *nm, Kernel k, const Ops *ops, uint32_t ext) {
    uint64_t t[2] = {0, 0};
    for (int f = 0; f < 2; f++) {
        dm_on = f;
        memcpy(memA, image, (size_t)lim * 2048); memcpy(memA + FLAT_BASE, image + FLAT_BASE, ext);
        G g; mk_g(&g, memA, 1, NULL, SIZE_MAX, 1); load_ops(&g, ops); int r = k(&g); size_t need = g.used;
        if (r) { printf("cycles case %s dry failed %d\n", nm, r); bad++; return; }
        mk_g(&g, memA, 0, scratch, need, 0); load_ops(&g, ops);
        uint64_t c0 = TICKS(); r = k(&g); t[f] = TICKS() - c0;
        if (r) { printf("cycles case %s run failed %d\n", nm, r); bad++; }
    }
    dm_on = 1;
    printf("%-34s %12llu %12llu\n", nm, (unsigned long long)t[0], (unsigned long long)t[1]);
}
static void cyc_trb(const char *nm, uint32_t es, const int *perm, const int32_t *xd, const int32_t *xm) {
    int32_t n[4], om[4], z4[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++) { n[i] = xd[perm[i]]; ft_perm[i] = perm[i]; }
    for (int i = 0; i < 3; i++) om[i] = n[i];
    om[3] = round32(n[3]);
    lim = SLOTS; rs = 99; shuffle_slots(); fill_image();
    Ops ops; memset(&ops, 0, sizeof ops);
    mk_blocked(&ft_x, es, xd, xm, z4, 0, 0, tabs[0]); mk_blocked(&ft_o, es, n, om, z4, 0, 0, tabs[1]);
    ops.nin = 1; ops.in[0] = ft_x; ops.out = ft_o;
    cyc_case(nm, ft_kernel, &ops, 0);
    lim = 160;
}
static void cyc_flat(const char *nm, uint32_t es, const int32_t *xd, const int *perm) {
    static const int32_t z[4] = {0, 0, 0, 0};
    int32_t od[4]; for (int i = 0; i < 4; i++) { od[i] = xd[perm[i]]; ft_perm[i] = perm[i]; }
    lim = 160; rs = 99; shuffle_slots(); fill_image();
    Ops ops; memset(&ops, 0, sizeof ops);
    flat_pair(&ft_x, &ft_o, es, xd, xd, z, od, od, z, 0);
    ops.nin = 1; ops.in[0] = ft_x; ops.out = ft_o;
    cyc_case(nm, ft_kernel, &ops, 0);
}
static void cycles_new(void) {
    static const int pwc[4] = {0, 1, 3, 2}, phw[4] = {0, 2, 1, 3}, p0231[4] = {0, 2, 3, 1};
    int32_t xa[4] = {1, 80, 64, 32}, xd2[4] = {1, 80, 32, 64}, xe[4] = {1, 32, 256, 2}, xem[4] = {1, 32, 256, 32};
    cyc_trb("wc16 u16 80x64x32 (18/56)", 2, pwc, xa, xa);
    cyc_trb("wc16 u16 80x32x64 (18/56)", 2, pwc, xd2, xd2);
    cyc_trb("wc16 u16 32x256x2/32 (18/56 E1/10)", 2, pwc, xe, xem);
    int32_t xu[4] = {1, 1, 64, 128}, xum[4] = {1, 8, 64, 128}, xv[4] = {1, 32, 256, 2}, xvm[4] = {1, 32, 256, 32};
    cyc_trb("wc8 u8 1x64x128 (18/55)", 1, pwc, xu, xum);
    cyc_trb("wc8 u8 32x256x2/32 (18/55 big/10)", 1, pwc, xv, xvm);
    int32_t xh[4] = {1, 32, 2, 256}, xhm[4] = {1, 32, 8, 256};
    cyc_trb("hw8 u8 32x2x256 (18/55 big/10)", 1, phw, xh, xhm);
    int32_t xh16[4] = {1, 32, 2, 256}, xh16m[4] = {1, 32, 4, 256};
    cyc_trb("hw16 u16 32x2x256 (18/56 E2/10)", 2, phw, xh16, xh16m);
    int32_t f1[4] = {1, 1, 80, 4096}, f2[4] = {1, 1, 256, 80}, f3[4] = {1, 2, 320, 256};
    cyc_flat("flat u8 80x4096 (18/57)", 1, f1, pwc);
    cyc_flat("flat u8 256x80 (18/57)", 1, f2, pwc);
    cyc_flat("flat u16 2x320x256 (18/58)", 2, f3, p0231);
    /* COPY_VALUES, IM2COL */
    for (int w = 0; w < 4; w++) {
        lim = 300; rs = 99; shuffle_slots(); fill_image();
        Ops ops; memset(&ops, 0, sizeof ops);
        Kernel k = NULL; const char *nm = ""; uint32_t ext = 0;
        if (w < 2) {
            int32_t bd[4] = {1, 32, 256, 2}, bm[4] = {1, 32, 256, 32}, z4[4] = {0, 0, 0, 0}, fm[4] = {1, 32, 256, 2};
            T bt, ft; mk_blocked(&bt, 2, bd, bm, z4, 0, 0, tabs[0]); mk_flat(&ft, 2, FLAT_BASE, bd, fm, z4);
            fill_ext(4 * 32 * 256 + 512); ext = ext_len; ext_len = 0;
            if (w == 0) { ops.in[0] = ft; ops.out = bt; nm = "copy_values 32x256x2 flat->CH (16/43)"; } else { ops.in[0] = bt; ops.out = ft; nm = "copy_values 32x256x2 CH->flat (16/45)"; }
            ops.nin = 1; k = k_copy_values;
        } else if (w == 2) {
            lim = 160;
            int32_t fd[4] = {1, 10, 258, 1}, od[4] = {1, 8, 256, 9}, om[4] = {1, 8, 256, 32}, z4[4] = {0, 0, 0, 0};
            mk_flat(&ops.in[0], 1, FLAT_BASE, fd, fd, z4); int32_t kv[4] = {3, 3, 1, 32}, sv[4] = {1, 1, 1, 1};
            mk_sv(&ops.in[1], kv); mk_sv(&ops.in[2], sv); mk_blocked(&ops.out, 1, od, om, z4, 0, 0, tabs[0]);
            fill_ext(10 * 258 + 512); ext = ext_len; ext_len = 0; ops.nin = 3; k = k_im2col; nm = "im2col 3x3 u8 8x256x9 (5/24)";
        } else {
            lim = 160;
            int32_t fd[4] = {1, 32, 256, 2}, od[4] = {1, 8, 64, 32}, z4[4] = {0, 0, 0, 0};
            mk_flat(&ops.in[0], 2, FLAT_BASE, fd, fd, z4); int32_t kv[4] = {4, 4, 2, 32}, sv[4] = {1, 4, 4, 1};
            mk_sv(&ops.in[1], kv); mk_sv(&ops.in[2], sv); mk_blocked(&ops.out, 2, od, od, z4, 0, 0, tabs[0]);
            fill_ext(4 * 32 * 256 + 512); ext = ext_len; ext_len = 0; ops.nin = 3; k = k_im2col; nm = "im2col 4x4 u16 8x64x32 (5/25)";
        }
        cyc_case(nm, k, &ops, ext);
        lim = 160;
    }
}

/* Warm timing of the data movers the Frame profile ranks lowest (kernel only, not setup): reference
 * (fast paths off) and fast, the fast run after one untimed run so the per-command memos are live. */
static void cyc_warm(const char *nm, Kernel k, const Ops *ops, uint32_t ext) {
    uint64_t t[2] = {0, 0};
    for (int f = 0; f < 2; f++) {
        dm_on = f; tcmd = (tcmd + 1) % 8000;
        memcpy(memA, image, (size_t)lim * 2048); memcpy(memA + FLAT_BASE, image + FLAT_BASE, ext);
        G g; mk_g(&g, memA, 1, NULL, SIZE_MAX, 1); load_ops(&g, ops); int r = k(&g); size_t need = g.used;
        if (r) { printf("cycles case %s dry failed %d (f=%d line %u)\n", nm, r, f, (unsigned)reference_error_line); bad++; return; }
        mk_g(&g, memA, 0, scratch, need, 0); load_ops(&g, ops); r = k(&g);
        if (r) { printf("cycles case %s warm run failed %d\n", nm, r); bad++; return; }
        mk_g(&g, memA, 0, scratch, need, 0); load_ops(&g, ops);
        uint64_t c0 = TICKS(); r = k(&g); t[f] = TICKS() - c0;
        if (r) { printf("cycles case %s run failed %d\n", nm, r); bad++; }
    }
    dm_on = 1;
    printf("%-40s %12llu %12llu\n", nm, (unsigned long long)t[0], (unsigned long long)t[1]);
}
static void cyc_resize(const char *nm, int c, int win, int h) {
    int32_t im[4] = {1, h, win, c}, om[4] = {1, h, 2 * win, c}, z4[4] = {0, 0, 0, 0};
    lim = 300; rs = 99; shuffle_slots(); fill_image();
    Ops ops; memset(&ops, 0, sizeof ops);
    mk_blocked(&ops.in[0], 1, im, im, z4, 0, 0, tabs[0]); mk_blocked(&ops.out, 1, om, om, z4, 0, 0, tabs[1]);
    int32_t hp[7] = {h, h, (int32_t)0x80000000, 0, 0x40000000, 0, h}, wp[7] = {win, 2 * win, (int32_t)0x80000000, 0, 0x40000000, 0, 2 * win};
    mk_const(&ops.in[1], 0, hp, 7); mk_const(&ops.in[2], 32, wp, 7); ops.nin = 3;
    cyc_warm(nm, k_resize, &ops, 0);
    lim = 160;
}
static void cyc_tr(const char *nm, uint32_t es, int flat, const int32_t *xd, const int32_t *xm, const int32_t *od, const int32_t *om) {
    static const int32_t perm[4] = {0, 1, 3, 2}, z4[4] = {0, 0, 0, 0};
    lim = SLOTS; rs = 99; shuffle_slots(); fill_image();
    Ops ops; memset(&ops, 0, sizeof ops);
    uint32_t ext = 0;
    if (flat) {
        flat_pair(&ops.in[0], &ops.out, es, xd, xm, z4, od, om, z4, 0);
    } else {
        mk_blocked(&ops.in[0], es, xd, xm, z4, 0, 0, tabs[0]); mk_blocked(&ops.out, es, od, om, z4, 0, 0, tabs[1]);
    }
    for (int i = 0; i < 4; i++) ft_perm[i] = perm[i];
    ft_x = ops.in[0]; ft_o = ops.out; ops.nin = 1;   /* k_transpose itself needs the executor's tensor views: time the fast dispatch */
    cyc_warm(nm, ft_kernel, &ops, ext);
    lim = 160;
}
static void cyc_cv(const char *nm, int kind) {
    static const int32_t z4[4] = {0, 0, 0, 0};
    lim = 300; rs = 99; shuffle_slots(); fill_image();
    Ops ops; memset(&ops, 0, sizeof ops);
    uint32_t ext = 0;
    T bt, ft;
    if (kind == 0) {   /* 16/44 */
        int32_t bd[4] = {1, 1, 128, 64}, bm[4] = {1, 8, 128, 64};
        mk_blocked(&bt, 1, bd, bm, z4, 0, 0, tabs[0]); mk_flat(&ft, 1, FLAT_BASE, bd, bd, z4); fill_ext(8192 + 512); ext = ext_len; ext_len = 0;
        ops.in[0] = bt; ops.out = ft;
    } else if (kind == 1) {   /* 16/50 */
        int32_t bd[4] = {1, 8, 8, 64}, fd[4] = {1, 1, 64, 64};
        mk_blocked(&bt, 1, bd, bd, z4, 0, 0, tabs[0]); mk_flat(&ft, 1, FLAT_BASE, fd, fd, z4); fill_ext(4096 + 512); ext = ext_len; ext_len = 0;
        ops.in[0] = bt; ops.out = ft;
    } else if (kind == 2) {   /* 16/65 */
        int32_t fd[4] = {1, 1, 1, 64};
        mk_flat(&ops.in[0], 4, FLAT_BASE, fd, fd, z4); mk_flat(&ops.out, 4, FLAT_BASE + 1024, fd, fd, z4);
        ops.in[0].dtype = DT_F32; ops.out.dtype = DT_F32; fill_ext(2048); ext = ext_len; ext_len = 0;
    } else {   /* 16/43 (flat -> CH) and 16/45 (CH -> flat) */
        int32_t bd[4] = {1, 32, 256, 2}, bm[4] = {1, 32, 256, 32};
        mk_blocked(&bt, 2, bd, bm, z4, 0, 0, tabs[0]); mk_flat(&ft, 2, FLAT_BASE, bd, bd, z4); fill_ext(4 * 32 * 256 + 512); ext = ext_len; ext_len = 0;
        if (kind == 3) { ops.in[0] = ft; ops.out = bt; } else { ops.in[0] = bt; ops.out = ft; }
    }
    ops.nin = 1;
    cyc_warm(nm, k_copy_values, &ops, ext);
    lim = 160;
}
static void cycles_movers(void) {
    printf("%-40s %12s %12s\n", "warm kernel pcycles", "reference", "fast");
    cyc_resize("24/51 resize 8x128x32 -> 8x256x32", 32, 128, 8);
    cyc_resize("24/51 resize 8x64x64 -> 8x128x64", 64, 64, 8);
    cyc_resize("24/51 resize 8x32x128 -> 8x64x128", 128, 32, 8);
    { int32_t xd[4] = {1, 1, 64, 128}, xm[4] = {1, 8, 64, 128}, od[4] = {1, 1, 128, 64}, om[4] = {1, 8, 128, 64};
      cyc_tr("18/55 transpose wc8 1x64x128", 1, 0, xd, xm, od, om); }
    { int32_t xd[4] = {1, 80, 64, 32}, od[4] = {1, 80, 32, 64};
      cyc_tr("18/56 transpose wc16 80x64x32", 2, 0, xd, xd, od, od); }
    { int32_t xd[4] = {1, 1, 256, 80}, od[4] = {1, 1, 80, 256};
      cyc_tr("18/57 transpose flat u8 256x80", 1, 1, xd, xd, od, od); }
    { int32_t xd[4] = {1, 1, 80, 4096}, od[4] = {1, 1, 4096, 80};
      cyc_tr("18/57 transpose flat u8 80x4096", 1, 1, xd, xd, od, od); }
    cyc_cv("16/44 copy_values CB->FB 128x64", 0);
    cyc_cv("16/50 copy_values CB 8x8x64 -> FB", 1);
    cyc_cv("16/65 copy_values f32 flat 64", 2);
    cyc_cv("16/43 copy_values FH -> CH 32x256x2", 3);
    cyc_cv("16/45 copy_values CH -> FH 32x256x2", 4);
}

int main(int argc, char **argv) {
    int cycles = argc > 1 && !strcmp(argv[1], "cycles");
    tmemo = argc > 1 && (!strcmp(argv[1], "memo") || !strcmp(argv[1], "movers"));
    movers = argc > 1 && !strcmp(argv[1], "movers");
    (void)cycles;
    if (!movers) {
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
    }
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
    if (!movers) {
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
    /* blocked W<->C transposes */
    {
        static const int pwc[4] = {0, 1, 3, 2}, phw[4] = {0, 2, 1, 3};
        int32_t z4[4] = {0, 0, 0, 0};
        int32_t a1[4] = {1, 8, 32, 32}, a2[4] = {1, 10, 64, 32}, a3[4] = {1, 16, 32, 64}, a4[4] = {2, 5, 32, 34}, a5[4] = {1, 12, 64, 2}, a6[4] = {1, 3, 96, 16};
        trb_case(2, pwc, a1, z4, z4, 0, 0, 0, 0, 0, 1, "");
        trb_case(2, pwc, a2, z4, z4, 0, 0, 0, 0, 0, 1, "");
        trb_case(2, pwc, a3, z4, z4, 0, 0, 0, 0, 0, 1, "");
        trb_case(2, pwc, a4, z4, z4, 3, 5, 12, 0, 0, 1, "partial group");
        trb_case(2, pwc, a5, z4, z4, 0, 0, 0, 0, 0, 1, "c=2");
        trb_case(2, pwc, a6, z4, z4, 0, 0, 10, 0, 0, 1, "c=16");
        int32_t xo1[4] = {0, 3, 2, 0}, oo1[4] = {0, 5, 6, 32}, xo2[4] = {0, 0, 0, 32}, oo2[4] = {0, 1, 2, 0};
        trb_case(2, pwc, a2, xo1, oo1, 2, 0, 5, 0, 0, 1, "origins");
        trb_case(2, pwc, a1, xo2, oo2, 0, 3, 0, 0, 0, 1, "x depth origin");
        trb_case(2, pwc, a1, z4, z4, 0, 0, 0, 8, 0, 0, "misaligned pool");
        trb_case(2, pwc, a2, z4, z4, 0, 0, 0, 0, 1, 0, "aliased");
        int32_t b1[4] = {1, 8, 32, 33}, b2[4] = {1, 8, 48, 32}, xo3[4] = {0, 0, 1, 0}, xo4[4] = {0, 0, 0, 16};
        trb_case(2, pwc, b1, z4, z4, 0, 0, 0, 0, 0, 0, "odd c");
        trb_case(2, pwc, b2, z4, z4, 0, 0, 0, 0, 0, 0, "w not 32");
        trb_case(2, pwc, a1, xo3, z4, 0, 0, 0, 0, 0, 0, "odd w origin");
        trb_case(2, pwc, a1, xo4, z4, 0, 0, 0, 0, 0, 0, "c origin not 32");
        /* u8 */
        int32_t u1[4] = {1, 1, 32, 32}, u2[4] = {1, 3, 64, 128}, u3[4] = {1, 8, 32, 34}, u4[4] = {1, 5, 64, 2}, u5[4] = {1, 2, 64, 5}, u6[4] = {2, 1, 32, 96};
        trb_case(1, pwc, u1, z4, z4, 0, 0, 0, 0, 0, 1, "");
        trb_case(1, pwc, u2, z4, z4, 0, 0, 0, 0, 0, 1, "");
        trb_case(1, pwc, u2, z4, z4, 2, 3, 12, 0, 0, 1, "");
        trb_case(1, pwc, u3, z4, z4, 0, 0, 0, 0, 0, 1, "partial group 34");
        trb_case(1, pwc, u4, z4, z4, 0, 0, 0, 0, 0, 1, "c=2");
        trb_case(1, pwc, u5, z4, z4, 0, 0, 7, 0, 0, 1, "c=5");
        trb_case(1, pwc, u6, z4, z4, 0, 0, 0, 0, 0, 1, "");
        int32_t xo5[4] = {0, 1, 4, 32}, oo5[4] = {0, 7, 8, 0}, xo6[4] = {0, 0, 2, 0};
        trb_case(1, pwc, u2, xo5, oo5, 0, 0, 5, 0, 0, 1, "origins");
        trb_case(1, pwc, u2, xo6, z4, 0, 0, 0, 0, 0, 0, "w origin not 4");
        trb_case(1, pwc, u1, z4, z4, 0, 0, 0, 8, 0, 0, "misaligned pool");
        trb_case(1, pwc, u2, z4, z4, 0, 0, 0, 0, 1, 0, "aliased");
        /* u8 H<->W */
        int32_t h1[4] = {1, 8, 8, 32}, h2[4] = {1, 12, 2, 64}, h3[4] = {1, 4, 5, 32}, h4[4] = {2, 16, 12, 32}, h5[4] = {1, 6, 8, 32};
        trb_case(1, phw, h1, z4, z4, 0, 0, 0, 0, 0, 1, "");
        trb_case(1, phw, h2, z4, z4, 0, 0, 0, 0, 0, 1, "w=2");
        trb_case(1, phw, h3, z4, z4, 1, 2, 10, 0, 0, 1, "");
        trb_case(1, phw, h4, z4, z4, 0, 0, 0, 0, 0, 1, "");
        int32_t xo7[4] = {0, 4, 4, 0}, oo7[4] = {0, 3, 8, 0};
        trb_case(1, phw, h4, xo7, oo7, 0, 0, 6, 0, 0, 1, "origins");
        trb_case(1, phw, h5, z4, z4, 0, 0, 0, 0, 0, 0, "h not 4");
        trb_case(1, phw, h1, z4, z4, 0, 0, 0, 8, 0, 0, "misaligned pool");
        trb_case(1, phw, h4, z4, z4, 0, 0, 0, 0, 1, 0, "aliased");
        /* u16 H<->W with many blocks (extent-based overlap test) */
        int32_t w1[4] = {1, 40, 4, 64};
        trb_case(2, phw, w1, z4, z4, 0, 0, 0, 0, 0, 1, "w=4");
        int32_t w2[4] = {1, 24, 2, 64};
        trb_case(2, phw, w2, z4, z4, 0, 0, 0, 0, 0, 1, "w=2");
    }
    }
    /* flat u8 matrix transposes, flat interleave */
    {
        int32_t td1[4] = {1, 1, 32, 200}, td2[4] = {2, 2, 16, 129}, td3[4] = {1, 1, 48, 19}, tz[4] = {0, 0, 0, 0};
        int p0132[4] = {0, 1, 3, 2}, p0231[4] = {0, 2, 3, 1};
        want_id = 14;
        ftr_case(1, td1, td1, tz, p0132, 0, 1);
        ftr_case(1, td2, td2, tz, p0132, 0, 1);
        ftr_case(1, td3, td3, tz, p0132, 0, 1);
        int32_t tm[4] = {3, 3, 20, 140}, tp[4] = {1, 1, 3, 5}, om4[4] = {3, 3, 149, 40}, op4[4] = {1, 0, 4, 7};
        ftr_case2(1, td2, tm, tp, p0132, om4, op4, 0, 1);
        int32_t tb1[4] = {1, 1, 256, 80}, tb2[4] = {1, 2, 80, 300}, tb3[4] = {2, 1, 208, 130}, tb4[4] = {1, 1, 144, 8}, tb5[4] = {1, 1, 16, 1};
        ftr_case(1, tb1, tb1, tz, p0132, 0, 1);
        ftr_case(1, tb2, tb2, tz, p0132, 0, 1);
        ftr_case(1, tb3, tb3, tz, p0132, 0, 1);
        ftr_case(1, tb4, tb4, tz, p0132, 0, 1);
        ftr_case(1, tb5, tb5, tz, p0132, 0, 1);
        int32_t tb6[4] = {1, 1, 80, 4096};
        ftr_case(1, tb6, tb6, tz, p0132, 0, 1);
        int32_t tm2[4] = {2, 2, 300, 150}, tp2[4] = {1, 1, 5, 7}, om5[4] = {2, 2, 150, 299}, op5[4] = {0, 1, 3, 11};
        ftr_case2(1, tb1, tm2, tp2, p0132, om5, op5, 0, 1);
        int32_t te[4] = {1, 1, 20, 100};
        ftr_case(1, te, te, tz, p0132, 0, 0);   /* rows not a multiple of 16: generic tiles */
        ftr_case(1, td3, td3, tz, p0132, 1, 0);
        ftr_case(2, td1, td1, tz, p0132, 0, 0);   /* u16 matrix: the tiled generic path */
        want_id = 15;
        int32_t il1[4] = {1, 2, 5, 70}, il2[4] = {2, 2, 3, 128}, il3[4] = {1, 2, 1, 3};
        int32_t ilp[4] = {1, 2, 0, 0}, ilm[4] = {3, 4, 5, 70};
        int32_t olm[4] = {3, 7, 70, 2}, olp[4] = {1, 2, 0, 0};
        for (uint32_t es = 1; es <= 4; es *= 2) {
            ftr_case(es, il1, il1, tz, p0231, 0, 1);
            ftr_case(es, il2, il2, tz, p0231, 0, 1);
            ftr_case(es, il3, il3, tz, p0231, 0, 1);
            ftr_case(es, il1, il1, tz, p0231, 1, 0);
        }
        ftr_case2(2, il1, ilm, ilp, p0231, olm, olp, 0, 1);
        int32_t ilbad[4] = {1, 3, 5, 7};
        ftr_case(2, ilbad, ilbad, tz, p0231, 0, 0);   /* three planes: generic */
        want_id = -1;
    }
    /* COPY_VALUES flat <-> padded blocked */
    for (int dir = 0; dir < 2; dir++) {
        copyc2_case(dir, 32, 256, 0, 0, 0, 0, 0, 0, 0, 1);
        copyc2_case(dir, 5, 64, 0, 0, 0, 0, 0, 0, 0, 1);
        copyc2_case(dir, 1, 32, 3, 2, 1, 0, 0, 0, 0, 1);
        copyc2_case(dir, 9, 96, 6, 6, 3, 15, 0, 0, 0, 1);
        copyc2_case(dir, 17, 64, 1, 0, 0, 25, 0, 0, 0, 1);
        copyc2_case(dir, 3, 64, 0, 0, 0, 0, 8, 0, 0, 0);    /* misaligned pool */
        copyc2_case(dir, 3, 64, 0, 0, 0, 0, 0, 4, 0, 1);    /* flat rows with a stride */
        copyc2_case(dir, 3, 64, 0, 0, 0, 0, 0, 0, 1, 0);    /* flat inside the blocks */
        copyc2_case(dir, 3, 48, 0, 0, 0, 0, 0, 0, 0, 0);    /* width not a multiple of 32 */
        copyc2_case(dir, 3, 64, 0, 1, 0, 0, 0, 0, 0, 0);    /* odd pixel origin */
    }
    /* u8 blocked -> flat */
    cvb2f_case(1, 128, 64, 0, 0, 0, 0, 0, 1, 128, 0, 0, 1);      /* 16/44 */
    cvb2f_case(8, 8, 64, 0, 0, 0, 0, 0, 1, 64, 0, 0, 1);         /* 16/50 */
    for (int it = 0; it < 48; it++) {
        cvb2f_align = it >= 24;
        int D = it % 3 == 0 ? 32 : it % 3 == 1 ? 64 : 128;
        int H = 1 + (int)(rnd() % 5), W = 8 * (1 + (int)(rnd() % 3)) + (it % 4 == 1 ? 3 : 0);
        int oh = it % 2 ? H * W : H, ow = it % 2 ? 1 : W;
        if (it % 5 == 2) { oh = 1; ow = H * W; }
        cvb2f_case(H, W, D, it % 4 ? (int)(rnd() % 9) : 0, it % 3 ? (int)(rnd() % 9) : 0, it % 5 == 0 ? 3 : 0, it % 2 ? 15 : 0, it % 7 == 0 ? 8 : 0, oh, ow, it % 6 == 0 ? 5 : 0, 0, it % 7 == 0 ? 0 : 1);
    }
    cvb2f_align = 1;   /* 128-byte aligned flat output with whole block columns: the dense run */
    cvb2f_case(1, 128, 64, 0, 0, 0, 0, 0, 1, 128, 0, 0, 1);
    cvb2f_case(8, 8, 64, 0, 0, 0, 0, 0, 1, 64, 0, 0, 1);
    cvb2f_case(3, 16, 128, 5, 8, 0, 20, 0, 3, 16, 0, 0, 1);
    cvb2f_case(2, 24, 32, 0, 16, 1, 10, 0, 6, 8, 0, 0, 1);
    cvb2f_case(2, 24, 64, 3, 8, 0, 30, 0, 2, 24, 0, 0, 1);
    cvb2f_case(2, 8, 128, 0, 0, 0, 0, 0, 16, 1, 0, 0, 1);
    cvb2f_align = 0;
    cvb2f_case(2, 16, 64, 0, 0, 0, 0, 0, 2, 16, 0, 1, 0);        /* flat inside the blocks */
    cvb2f_case(2, 16, 96, 0, 0, 0, 0, 0, 2, 16, 0, 0, 0);        /* D=96 stays on copy_runs */
    {
        int32_t z[4] = {0, 0, 0, 0}, a[4] = {1, 1, 1, 64}, b[4] = {1, 1, 8, 16}, c[4] = {2, 3, 4, 5}, c2[4] = {2, 1, 12, 5}, c3[4] = {1, 6, 10, 5}, cm[4] = {3, 3, 4, 5}, cp[4] = {1, 0, 0, 0};
        cvf2f_case(4, DT_F32, a, a, z, a, a, z, 0, 1);
        cvf2f_case(1, DT_U8, a, a, z, b, b, z, 0, 1);
        cvf2f_case(2, DT_U16, b, b, z, b, b, z, 0, 1);
        cvf2f_case(4, DT_F32, c, c, z, c2, c2, z, 0, 1);
        cvf2f_case(4, DT_F32, c, c, z, c3, c3, z, 0, 1);
        cvf2f_case(1, DT_U8, c, cm, cp, c2, c2, z, 0, 1);       /* batch origin: still one run */
        cvf2f_case(1, DT_U8, c, c, z, c2, c2, z, 1, 0);          /* aliased */
        int32_t d1[4] = {1, 4, 8, 16}, d1m[4] = {1, 4, 9, 16};
        cvf2f_case(1, DT_U8, d1, d1m, z, b, b, z, 0, 0);         /* source rows with a stride */
    }
    if (!movers) {
    /* IM2COL */
    for (int u16 = 0; u16 < 2; u16++) {
        int ows[4] = {u16 ? 8 : 128, u16 ? 16 : 256, u16 ? 64 : 128, u16 ? 24 : 256};
        for (int it = 0; it < 12; it++) {
            int oh = it < 4 ? (int[]){1, 3, 8, 9}[it] : 1 + (int)(rnd() % 9), ow = ows[it & 3];
            int po1 = it % 3 == 1 ? 3 : it % 3 == 2 ? 8 : 0, po2 = it % 4 == 1 ? (u16 ? 2 : 8) : it % 4 == 2 ? (u16 ? 6 : 16) : 0;
            im2col_case(u16, oh, ow, po1, po2, it % 5 == 0 ? 3 : 0, it % 2 ? 12 : 0, 0, it % 3, it % 4 == 3 ? 2 : 0, 1, 0, 1);
        }
        im2col_case(u16, 2, ows[1], 0, 0, 0, 0, 8, 0, 0, 1, 0, 0);   /* misaligned pool */
        im2col_case(u16, 2, ows[1], 0, 0, 0, 0, 0, 0, 0, 1, 1, 0);   /* source inside the blocks */
    }
    im2col_case(0, 2, 64, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0);    /* width not a multiple of 128 */
    im2col_case(0, 2, 128, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0);   /* stride 2 */
    im2col_case(0, 2, 128, 0, 4, 0, 0, 0, 0, 0, 1, 0, 0);   /* pixel origin not a multiple of 8 */
    }
    printf("%s: %d cases, %d took a fast path, %d mismatches\n", bad ? "FAIL" : "PASS", cases, fast_taken, bad);
    if (cycles) { cycles_tests(); cycles_new(); }
    if (movers) cycles_movers();
    return bad != 0;
}
