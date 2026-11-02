/* Qualcomm assembler, -mv75 -mhmx. All inputs and outputs reside in VTCM.
 * r0=A, r1=B, r2=Fi, r3=C, r4=number of 32-channel depth tiles. */
    .text
    .globl qd_hmx_wide
qd_hmx_wide:
    { r8 = r4; r10 = #32 }
    { mxclracc }
    { bias = mxmem2(r2) }
1:
    { r9 = min(r8,r10) }
    { r5 = asl(r9,#11); r6 = asl(r9,#10) }
    { r5 = add(r5,#-1); r6 = add(r6,#-1) }
    { activation.ub = mxmem(r0,r5):deep
      weight.b = mxmem(r1,r6) }
    { r5 = add(r5,#1); r6 = add(r6,#1) }
    { r0 = add(r0,r5); r1 = add(r1,r6) }
    { r8 = sub(r8,r9) }
    { p0 = cmp.gt(r8,#0) }
    { if (p0) jump 1b }
    { r7 = #0 }
    { cvt.uh = acc(r7):2x1 }
    { mxmem(r3,r7) = cvt }
    { jumpr r31 }
    .globl qd_hmx_finish
qd_hmx_finish:
    { mxclracc }
    { bias = mxmem2(r0) }
    { r2 = #0 }
    { cvt.uh = acc(r2):2x1 }
    { mxmem(r1,r2) = cvt }
    { jumpr r31 }
/* One output block (N tile) over mt M tiles: the bias is loaded once, then
 * per tile mxclracc, the deep loads, acc(0) conversion and store, which is
 * qd_hmx_wide per tile without the per-call bias load. (acc(2), the tuned fp16
 * GEMM's transfer-and-clear, is not exact in this integer :2x1 mode: V75 sim.)
 * r0=first A tile, r1=B of this N block, r2=Fi of this N block, r3=first C
 * tile, r4=KB | NB<<16 (C tiles are NB*2048 apart, A tiles KB*2048), r5=mt. */
    .globl qd_hmx_wide_n
qd_hmx_wide_n:
    { r6 = zxth(r4); r10 = lsr(r4,#16) }
    { r10 = asl(r10,#11); r11 = asl(r6,#11) }
    { r7 = #0; r12 = #32 }
    { mxclracc }
    { bias = mxmem2(r2) }
2:
    { mxclracc }
    { r8 = r6; r13 = r0; r14 = r1 }
3:
    { r9 = min(r8,r12) }
    { r15 = asl(r9,#11); r28 = asl(r9,#10) }
    { r15 = add(r15,#-1); r28 = add(r28,#-1) }
    { activation.ub = mxmem(r13,r15):deep
      weight.b = mxmem(r14,r28) }
    { r15 = add(r15,#1); r28 = add(r28,#1) }
    { r13 = add(r13,r15); r14 = add(r14,r28) }
    { r8 = sub(r8,r9) }
    { p0 = cmp.gt(r8,#0) }
    { if (p0) jump 3b }
    { r9 = #0 }
    { cvt.uh = acc(r7):2x1 }
    { mxmem(r3,r9) = cvt }
    { r3 = add(r3,r10); r0 = add(r0,r11); r5 = add(r5,#-1) }
    { p0 = cmp.gt(r5,#0) }
    { if (p0) jump 2b }
    { jumpr r31 }
/* Native int8 (HW-rounding builds): one N block over mt output tiles of 64
 * pixels. A tiles are crouton blocks (8 rows x 8 px x 32 ch, :deep:cm), KB
 * of them per tile, KB*2048 apart; conversion cvt.ub (silicon midpoint
 * rounding, HMX_NATIVE_ROUNDING.md) stored :cm as a whole output block at
 * out[t]. Bias loaded once; mxclracc and acc(0) per tile, as qd_hmx_wide_n.
 * r0=first A tile, r1=B of this N block, r2=Fi of this N block, r3=table of
 * mt output block addresses, r4=KB, r5=mt. */
    .globl qd_hmx_native_n
qd_hmx_native_n:
    { r11 = asl(r4,#11); r7 = #0; r12 = #32 }
    { mxclracc }
    { bias = mxmem2(r2) }
2:
    { mxclracc }
    { r8 = r4; r13 = r0; r14 = r1 }
3:
    { r9 = min(r8,r12) }
    { r15 = asl(r9,#11); r28 = asl(r9,#10) }
    { r15 = add(r15,#-1); r28 = add(r28,#-1) }
    { activation.ub = mxmem(r13,r15):deep:cm
      weight.b = mxmem(r14,r28) }
    { r15 = add(r15,#1); r28 = add(r28,#1) }
    { r13 = add(r13,r15); r14 = add(r14,r28) }
    { r8 = sub(r8,r9) }
    { p0 = cmp.gt(r8,#0) }
    { if (p0) jump 3b }
    { r10 = memw(r3++#4) }
    { cvt.ub = acc(r7) }
    { mxmem(r10,r7):cm = cvt }
    { r0 = add(r0,r11); r5 = add(r5,#-1) }
    { p0 = cmp.gt(r5,#0) }
    { if (p0) jump 2b }
    { jumpr r31 }
