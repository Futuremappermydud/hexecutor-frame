/* SPDX-License-Identifier: LGPL-3.0-or-later */
    .text
cave2:
    { r28 = ##0x2000000 }
loop:
    { r28 = add(r28,#-1); p0 = cmp.eq(r28,#1); if (!p0.new) jump:t loop }
    { r28 = #0 }
    { jump cave2 }
