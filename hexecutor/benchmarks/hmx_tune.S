/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Qualcomm assembler: -mv75 -mhmx. Arguments follow the Hexagon ABI. */
    .text
    .globl hmx_tune_init
hmx_tune_init:
    { mxclracc.hf }
    { bias = mxmem2(r0) }
    { jumpr r31 }

/* A, B, C, byte range, flags. flag 1 = clear per output; flag 2 = sync.
 * acc(2) transfers and clears the accumulator for the next output tile. */
    .globl hmx_tune_deep
hmx_tune_deep:
    { p0 = tstbit(r4,#0); r5 = #2; r6 = #0 }
    { if (!p0) jump .Ldeep }
    { mxclracc.hf }
.Ldeep:
    { activation.hf = mxmem(r0,r3):deep
      weight.hf = mxmem(r1,r3) }
    { cvt.hf = acc(r5) }
    { mxmem(r2,r6) = cvt }
    { p0 = tstbit(r4,#1) }
    { if (!p0) jumpr r31 }
    { syncht }
    { v0 = vmem(r2+#0) }
    { syncht }
    { jumpr r31 }

    .globl hmx_tune_finish
hmx_tune_finish:
    { r1 = #2; r2 = #0 }
    { mxclracc.hf }
    { cvt.hf = acc(r1) }
    { mxmem(r0,r2) = cvt }
    { syncht }
    { v0 = vmem(r0+#0) }
    { syncht }
    { jumpr r31 }

/* A, B, C, bytes per K strip, N tiles, M tiles. No stack traffic. */
    .globl hmx_tune_gemm
hmx_tune_gemm:
    { r6 = add(r3,#-1); r7 = #2; r8 = #0 }
    loop1(.Lm,r5)
.Lm:
    { r9 = r1 }
    loop0(.Ln,r4)
.Ln:
    { activation.hf = mxmem(r0,r6):deep
      weight.hf = mxmem(r9,r6) }
    { cvt.hf = acc(r7) }
    { mxmem(r2,r8) = cvt }
    { r9 = add(r9,r3); r2 = add(r2,#2048) }:endloop0
    { r0 = add(r0,r3) }:endloop1
    { jumpr r31 }

    .globl hmx_tune_gemm_after
hmx_tune_gemm_after:
    { r6 = add(r3,#-1); r8 = #0 }
    loop1(.Lam,r5)
.Lam:
    { r9 = r1 }
    loop0(.Lan,r4)
.Lan:
    { activation.hf = mxmem(r0,r6):deep
      weight.hf = mxmem(r9,r6) }
    { mxmem(r2,r8):after.hf = acc }
    { r9 = add(r9,r3); r2 = add(r2,#2048) }:endloop0
    { r0 = add(r0,r3) }:endloop1
    { jumpr r31 }
