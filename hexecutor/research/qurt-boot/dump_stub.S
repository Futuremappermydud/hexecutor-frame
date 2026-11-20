/* SPDX-License-Identifier: LGPL-3.0-or-later */
    .section .text_hook, "ax"
    .globl hook
hook:
    { jump cave }
    .section .text_cave, "ax"
    .globl cave
cave:
    { r28 = ##0xfe01a000 }
    { memw(r28+#0) = r0 ; memw(r28+#4) = r1 }
    { memw(r28+#8) = r2 ; memw(r28+#12) = r3 }
    { memw(r28+#16) = r4 ; memw(r28+#20) = r5 }
    { memw(r28+#24) = r6 ; memw(r28+#28) = r7 }
    { memw(r28+#32) = r16 ; memw(r28+#36) = r17 }
    { memw(r28+#40) = r18 ; memw(r28+#44) = r19 }
    { memw(r28+#48) = r31 ; memw(r28+#52) = r29 }
    { memw(r28+#56) = r30 ; memw(r28+#60) = r20 }
    { r3 = ##0xc0ffee01 }
    { memw(r28+#68) = r3 }
    { dccleaninva(r28) }
    { r2 = add(r28,#32) }
    { dccleaninva(r2) }
    { r2 = add(r28,#64) }
    { dccleaninva(r2) }
    { syncht }
spin:
    { jump spin }
