/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Call extended code on a private L2-TCM stack; preserve Valve's stack and
 * callee-saved registers exactly. r16 survives the entire C call by the ABI. */
    .text
    .globl qds_on_stack
qds_on_stack:
    { allocframe(#16) }
    { memd(r29+#0)=r17:16; r17=r0 }
    { r16=r29; r29=r2; r0=r1 }
    { callr r17 }
    { r29=r16 }
    { r17:16=memd(r29+#0) }
    { dealloc_return }
