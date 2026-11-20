/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Called by worker 0 only, with SSR.XE2 set. No QuRT/HAP dependency.
 * HMX words were assembled/disassembled with Qualcomm Clang 19.0.07,
 * -mv75 -mhmx. Encoding them explicitly also permits open LLVM assemblers
 * to build this fixed-register probe. Packet termination bits are included;
 * the two loads below form one packet. Changing registers requires reassembly.
 */
    .text
    .globl hmx_clear
hmx_clear:
    .word 0xa6e0c013              /* { mxclracc.hf } */
    { jumpr r31 }

/* r0 = activation tile, r1 = weight tile, r2 = output, r3 = scales.
 * FP16 32x32 tiles, 2,048 bytes each, all in VTCM. */
    .globl hmx_dot
hmx_dot:
    { r4 = #2047; r5 = #0 }
    { r6 = #2 }
    .word 0xa6e0c013              /* { mxclracc.hf } */
    .word 0x9203c3fe              /* { bias = mxmem2(r3) } */
    .word 0x920044e4, 0x9201e4ef  /* { activation.hf = mxmem(r0,r4)
                                    weight.hf = mxmem(r1,r4) } */
    .word 0xa6e6da10              /* { cvt.hf = acc(r6) } */
    .word 0xa6e2c518              /* { mxmem(r2,r5) = cvt } */
    { syncht }
    { jumpr r31 }
