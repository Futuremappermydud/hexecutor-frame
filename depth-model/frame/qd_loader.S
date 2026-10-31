/* Fixed 4 KB trampoline registered as the worker callback. It invalidates the
 * rewritable payload region in the data and instruction caches, then tail-jumps
 * to it, so new payloads can reuse one address without stale instructions.
 * This page itself must never change once a DSP session has executed it. */
#define PAYLOAD_BASE  0x87610000
#define PAYLOAD_LINES (0x8000 / 32)
    .section .text.entry,"ax",@progbits
    .globl loader
loader:
    { r0 = ##PAYLOAD_BASE; r1 = #PAYLOAD_LINES }
    { r28 = ##PAYLOAD_BASE }
    loop0(1f, r1)
1:  { dcinva(r0) }
    { icinva(r0) }
    { r0 = add(r0, #32) }:endloop0
    { isync }
    { jumpr r28 }
