/* Qualcomm assembler, -mv75 -mhmx. All inputs and outputs reside in VTCM.
 * r0=A, r1=B, r2=Fi, r3=C, r4=number of 32-channel depth tiles. */
    .text
    .globl qd_hmx_native
qd_hmx_native:
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
    { cvt.ub = acc(r7) }
    { mxmem(r3,r7) = cvt }
    { jumpr r31 }

/* Qualcomm assembler, -mv75 -mhmx. All inputs and outputs reside in VTCM.
 * r0=A, r1=B, r2=Fi, r3=C, r4=number of 32-channel depth tiles. */
    .text
    .globl qd_hmx_native_cm
qd_hmx_native_cm:
    { r8 = r4; r10 = #32 }
    { mxclracc }
    { bias = mxmem2(r2) }
1:
    { r9 = min(r8,r10) }
    { r5 = asl(r9,#11); r6 = asl(r9,#10) }
    { r5 = add(r5,#-1); r6 = add(r6,#-1) }
    { activation.ub = mxmem(r0,r5):deep:cm
      weight.b = mxmem(r1,r6) }
    { r5 = add(r5,#1); r6 = add(r6,#1) }
    { r0 = add(r0,r5); r1 = add(r1,r6) }
    { r8 = sub(r8,r9) }
    { p0 = cmp.gt(r8,#0) }
    { if (p0) jump 1b }
    { r7 = #0 }
    { cvt.ub = acc(r7) }
    { mxmem(r3,r7):cm = cvt }
    { jumpr r31 }
