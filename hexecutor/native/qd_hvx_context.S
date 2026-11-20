/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Preserve borrowed context 3: 32 vectors and four predicates (0/1 byte masks).
 * Caller must hold the parked firmware manager's context for the whole lease.
 * R0 returns saved SSR from enter; leave takes (save_area, saved_ssr). */
    .text
    .p2align 4
    .globl qds_hvx_context_enter
    .type qds_hvx_context_enter,@function
qds_hvx_context_enter:
    { r1 = r0 }
    { r5 = r0 }
    { r0 = ssr }
    { r2 = and(r0, ##0x47ffffff) }
    { r2 = or(r2, ##0x98000000) }
    { ssr = r2 }
    { isync }
    { vmem(r1++#1) = v0 }
    { vmem(r1++#1) = v1 }
    { vmem(r1++#1) = v2 }
    { vmem(r1++#1) = v3 }
    { vmem(r1++#1) = v4 }
    { vmem(r1++#1) = v5 }
    { vmem(r1++#1) = v6 }
    { vmem(r1++#1) = v7 }
    { vmem(r1++#1) = v8 }
    { vmem(r1++#1) = v9 }
    { vmem(r1++#1) = v10 }
    { vmem(r1++#1) = v11 }
    { vmem(r1++#1) = v12 }
    { vmem(r1++#1) = v13 }
    { vmem(r1++#1) = v14 }
    { vmem(r1++#1) = v15 }
    { vmem(r1++#1) = v16 }
    { vmem(r1++#1) = v17 }
    { vmem(r1++#1) = v18 }
    { vmem(r1++#1) = v19 }
    { vmem(r1++#1) = v20 }
    { vmem(r1++#1) = v21 }
    { vmem(r1++#1) = v22 }
    { vmem(r1++#1) = v23 }
    { vmem(r1++#1) = v24 }
    { vmem(r1++#1) = v25 }
    { vmem(r1++#1) = v26 }
    { vmem(r1++#1) = v27 }
    { vmem(r1++#1) = v28 }
    { vmem(r1++#1) = v29 }
    { vmem(r1++#1) = v30 }
    { vmem(r1++#1) = v31 }
    { barrier }
    { r2 = r1 }
    { r3 = ##0x01010101 }
    { r4 = #0 }
    { v30 = vsplat(r3) }
    { v29 = vsplat(r4) }
    { v31 = vmux(Q0, v30, v29) }
    { vmem(r2+#0) = v31 }
    { v31 = vmux(Q1, v30, v29) }
    { vmem(r2+#1) = v31 }
    { v31 = vmux(Q2, v30, v29) }
    { vmem(r2+#2) = v31 }
    { v31 = vmux(Q3, v30, v29) }
    { vmem(r2+#3) = v31 }
    { barrier }
    { r2 = add(r5, ##3712) }
    { v29 = vmem(r2+#0) }
    { v30 = vmem(r2+#1) }
    { v31 = vmem(r2+#2) }
    { barrier }
    { jumpr r31 }
    .size qds_hvx_context_enter,.-qds_hvx_context_enter

    .p2align 4
    .globl qds_hvx_context_leave
    .type qds_hvx_context_leave,@function
qds_hvx_context_leave:
    { r4 = r1 }
    { r1 = r0 }
    { r2 = add(r0, ##4096) }
    { r3 = ##0x01010101 }
    { v31 = vmem(r2+#0) }
    { Q0 = vand(v31, r3) }
    { v31 = vmem(r2+#1) }
    { Q1 = vand(v31, r3) }
    { v31 = vmem(r2+#2) }
    { Q2 = vand(v31, r3) }
    { v31 = vmem(r2+#3) }
    { Q3 = vand(v31, r3) }
    { v0 = vmem(r1++#1) }
    { v1 = vmem(r1++#1) }
    { v2 = vmem(r1++#1) }
    { v3 = vmem(r1++#1) }
    { v4 = vmem(r1++#1) }
    { v5 = vmem(r1++#1) }
    { v6 = vmem(r1++#1) }
    { v7 = vmem(r1++#1) }
    { v8 = vmem(r1++#1) }
    { v9 = vmem(r1++#1) }
    { v10 = vmem(r1++#1) }
    { v11 = vmem(r1++#1) }
    { v12 = vmem(r1++#1) }
    { v13 = vmem(r1++#1) }
    { v14 = vmem(r1++#1) }
    { v15 = vmem(r1++#1) }
    { v16 = vmem(r1++#1) }
    { v17 = vmem(r1++#1) }
    { v18 = vmem(r1++#1) }
    { v19 = vmem(r1++#1) }
    { v20 = vmem(r1++#1) }
    { v21 = vmem(r1++#1) }
    { v22 = vmem(r1++#1) }
    { v23 = vmem(r1++#1) }
    { v24 = vmem(r1++#1) }
    { v25 = vmem(r1++#1) }
    { v26 = vmem(r1++#1) }
    { v27 = vmem(r1++#1) }
    { v28 = vmem(r1++#1) }
    { v29 = vmem(r1++#1) }
    { v30 = vmem(r1++#1) }
    { v31 = vmem(r1++#1) }
    { barrier }
    { ssr = r4 }
    { isync }
    { jumpr r31 }
    .size qds_hvx_context_leave,.-qds_hvx_context_leave

