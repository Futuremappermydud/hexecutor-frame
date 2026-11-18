/* V75 fp16 GEMM. Tiles are [16 row pairs][32 columns][2 rows].
 * A [M/32,K/32,1024 halves], B [N/32,K/32,1024 halves], C [M/32,N/32,...].
 * A deep load can cover at most 32 K tiles (1024 channels). Split larger
 * K strips without clearing the accumulator between loads. */
    .text
    .globl hx_hmx_init
hx_hmx_init:
    { mxclracc.hf }
    { bias = mxmem2(r0) }
    { jumpr r31 }
/* r0 A, r1 B, r2 C, r3 bytes per K strip, r4 N tiles, r5 M tiles. */
    .globl hx_hmx_gemm
hx_hmx_gemm:
    { r6 = #2; r7 = #0; r8 = #65536 }
    loop1(.Lm,r5)
.Lm:
    { r9 = r1 }
    loop0(.Ln,r4)
.Ln:
    { r10 = r0; r11 = r9; r12 = r3 }
.Lk:
    { r13 = min(r12,r8) }
    { r14 = add(r13,#-1) }
    { activation.hf = mxmem(r10,r14):deep
      weight.hf = mxmem(r11,r14) }
    { r10 = add(r10,r13); r11 = add(r11,r13); r12 = sub(r12,r13) }
    { p0 = cmp.gt(r12,#0) }
    { if (p0) jump .Lk }
    { cvt.hf = acc(r6) }
    { mxmem(r2,r7) = cvt }
    { r9 = add(r9,r3); r2 = add(r2,#2048) }:endloop0
    { r0 = add(r0,r3) }:endloop1
    { jumpr r31 }
/* Ordered sentinel store drains the matrix stream before HVX reuse. */
    .globl hx_hmx_finish
hx_hmx_finish:
    { r1 = #2; r2 = #0 }
    { mxclracc.hf }
    { cvt.hf = acc(r1) }
    { mxmem(r0,r2) = cvt }
    { syncht }
    { v0 = vmem(r0+#0) }
    { syncht }
    { jumpr r31 }
/* One M tile, per-output-tile packed FP16 scale/additive bias tables.
 * r0 A, r1 B, r2 C, r3 K bytes, r4 N tiles, r5 256-byte parameter tables. */
    .globl hx_hmx_gemm_bias
hx_hmx_gemm_bias:
    { r6 = #2; r7 = #0; r8 = #65536 }
    loop0(.Lbias_n,r4)
.Lbias_n:
    { r10 = r0; r11 = r1; r12 = r3 }
.Lbias_k:
    { r13 = min(r12,r8) }
    { r14 = add(r13,#-1) }
    { activation.hf = mxmem(r10,r14):deep
      weight.hf = mxmem(r11,r14) }
    { r10 = add(r10,r13); r11 = add(r11,r13); r12 = sub(r12,r13) }
    { p0 = cmp.gt(r12,#0) }
    { if (p0) jump .Lbias_k }
    { bias = mxmem2(r5) }
    { cvt.hf = acc(r6) }
    { mxmem(r2,r7) = cvt }
    { r1 = add(r1,r3); r2 = add(r2,#2048); r5 = add(r5,#256) }:endloop0
    { jumpr r31 }
/* In-place pointwise strip. A: M tiles, each with K tiles at a_kstep; B: [N][K]
 * contiguous; C: N planes at c_plane, M tiles contiguous in each. Every deep
 * load covers exactly one tile-aligned tile, so no load can cross the 4 MB VTCM
 * TLB page boundary. One ordered completion by the caller covers the strip.
 * r0 A, r1 B, r2 C, r3 HxHmxStrip { m, n, k, a_kstep, a_mstep, c_plane, bias }. */
    .globl hx_hmx_strip
hx_hmx_strip:
    { r4 = memw(r3+#0); r5 = memw(r3+#24) }
    { r6 = #2; r7 = #0; r14 = #2047 }
.Lstrip_m:
    { r8 = memw(r3+#4); r9 = r1; r12 = r2 }
    { r13 = memw(r3+#12); r15 = memw(r3+#24) }
    loop1(.Lstrip_n,r8)
.Lstrip_n:
    { r8 = memw(r3+#8); r10 = r0 }
    loop0(.Lstrip_k,r8)
.Lstrip_k:
    { activation.hf = mxmem(r10,r14):deep
      weight.hf = mxmem(r9,r14) }
    { r10 = add(r10,r13); r9 = add(r9,#2048) }:endloop0
    { bias = mxmem2(r15) }
    { cvt.hf = acc(r6) }
    { mxmem(r12,r7) = cvt }
    { r8 = memw(r3+#20); r15 = add(r15,#256) }
    { r12 = add(r12,r8) }:endloop1
    { r8 = memw(r3+#16); r2 = add(r2,#2048); r4 = add(r4,#-1) }
    { r0 = add(r0,r8); p0 = cmp.gt(r4,#0) }
    { if (p0) jump .Lstrip_m }
    { jumpr r31 }
