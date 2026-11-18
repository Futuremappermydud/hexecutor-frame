/* HMX leaves for QNN FP16 run-list handlers (see qnn16.h). V75. */
    .text

    .globl qnn16_hmx_reset
qnn16_hmx_reset:
    { mxclracc.hf }
    { jumpr r31 }

/* r0 = completion tile. */
    .globl qnn16_hmx_finish
qnn16_hmx_finish:
    { r1 = #2; r2 = #0 }
    { mxclracc.hf }
    { cvt.hf = acc(r1) }
    { mxmem(r0,r2) = cvt }
    { syncht }
    { v0 = vmem(r0+#0) }
    { syncht }
    { jumpr r31 }

/* r0 = Qnn16Conv1x1 { a, w, fi, c, m, k, n }.
 * For each output block m and channel group n: accumulate the K activation
 * tiles of block m (each through its own block-table pointer, one 2 KiB load
 * each, like QNN's hmx_v73_convf16_1x1_stride1) against weight tiles
 * [n][0..K), convert with scale/bias table n and store through the output
 * table. One deep load over contiguous tiles rounds identically on the ISS but
 * would require contiguous inputs.
 * Each output takes `tables` conversion steps (bias = mxmem2 of the next
 * 256-byte table, cvt.hf = acc(acc[i])) before one store, as in QNN's loop.
 * Register use (caller-saved only): r0 params, r1 a row, r2 w, r3 fi,
 * r4 c cursor, r5 m left, r6 k, r7 n, r8 n left, r9 w cursor, r10 fi cursor,
 * r11 a/acc cursor, r12 k/table left, r13 tile, r14 0x7ff, r28 0. */
    .globl qnn16_conv1x1
qnn16_conv1x1:
    { r1 = memw(r0+#0); r2 = memw(r0+#4) }
    { r3 = memw(r0+#8); r4 = memw(r0+#12) }
    { r5 = memw(r0+#16); r6 = memw(r0+#20) }
    { r7 = memw(r0+#24); r14 = #2047; r28 = #0 }
    { p0 = cmp.eq(r5,#0); if (p0.new) jumpr:nt r31 }
.Lc1_m:
    { r9 = r2; r10 = r3; r8 = r7 }
.Lc1_n:
    { r11 = r1; r12 = r6 }
.Lc1_k:
    { r13 = memw(r11++#4); r12 = add(r12,#-1) }
    { activation.hf = mxmem(r13,r14)
      weight.hf = mxmem(r9,r14) }
    { r9 = add(r9,#2048); p0 = cmp.gt(r12,#0); if (p0.new) jump:t .Lc1_k }
    { r12 = memw(r0+#28); r11 = memw(r0+#32) }
.Lc1_t:
    { bias = mxmem2(r10) }
    { r13 = memw(r11++#4) }
    { cvt.hf = acc(r13) }
    { r10 = add(r10,#256); r12 = add(r12,#-1) }
    { p0 = cmp.gt(r12,#0); if (p0.new) jump:t .Lc1_t }
    { r13 = memw(r4++#4) }
    { mxmem(r13,r28) = cvt }
    { r8 = add(r8,#-1) }
    { p0 = cmp.gt(r8,#0); if (p0.new) jump:t .Lc1_n }
    { r5 = add(r5,#-1); r13 = asl(r6,#2) }
    { r1 = add(r1,r13); p0 = cmp.gt(r5,#0) }
    { if (p0) jump .Lc1_m }
    { jumpr r31 }

/* Same as qnn16_conv1x1, except that p->w is a table of n * k pointers to
 * weight tiles ([n][k], each 2 KiB and tile-aligned), so the tiles need not
 * be contiguous. Used by the PW1X1 HMX form (qnn16_tile_pw_hmx.c). */
    .globl qnn16_conv1x1_wt
qnn16_conv1x1_wt:
    { r1 = memw(r0+#0); r2 = memw(r0+#4) }
    { r3 = memw(r0+#8); r4 = memw(r0+#12) }
    { r5 = memw(r0+#16); r6 = memw(r0+#20) }
    { r7 = memw(r0+#24); r14 = #2047; r28 = #0 }
    { p0 = cmp.eq(r5,#0); if (p0.new) jumpr:nt r31 }
.Lw_m:
    { r9 = r2; r10 = r3; r8 = r7 }
.Lw_n:
    { r11 = r1; r12 = r6 }
.Lw_k:
    { r13 = memw(r11++#4); r12 = add(r12,#-1) }
    { r15 = memw(r9++#4) }
    { activation.hf = mxmem(r13,r14); weight.hf = mxmem(r15,r14) }
    { p0 = cmp.gt(r12,#0); if (p0.new) jump:t .Lw_k }
    { r12 = memw(r0+#28); r11 = memw(r0+#32) }
.Lw_t:
    { bias = mxmem2(r10) }
    { r13 = memw(r11++#4) }
    { cvt.hf = acc(r13) }
    { r10 = add(r10,#256); r12 = add(r12,#-1) }
    { p0 = cmp.gt(r12,#0); if (p0.new) jump:t .Lw_t }
    { r13 = memw(r4++#4) }
    { mxmem(r13,r28) = cvt }
    { r8 = add(r8,#-1) }
    { p0 = cmp.gt(r8,#0); if (p0.new) jump:t .Lw_n }
    { r5 = add(r5,#-1); r13 = asl(r6,#2) }
    { r1 = add(r1,r13); p0 = cmp.gt(r5,#0) }
    { if (p0) jump .Lw_m }
    { jumpr r31 }
