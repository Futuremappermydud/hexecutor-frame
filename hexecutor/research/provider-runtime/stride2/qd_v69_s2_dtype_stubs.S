/* SPDX-License-Identifier: LGPL-3.0-or-later */
.text
.p2align 4
.globl qd_v69_s2_dtype_in
qd_v69_s2_dtype_in:
  memw(r0+#0)=#1
#ifdef S2_UNIT_SCALES
  memw(r0+#4)=##0x3f800000
#else
  memw(r0+#4)=##0x3d3c0bea
#endif
  memw(r0+#8)=#0
  jumpr r31
.globl qd_v69_s2_dtype_out
qd_v69_s2_dtype_out:
  memw(r0+#0)=#1
#ifdef S2_UNIT_SCALES
  memw(r0+#4)=##0x3f800000
#else
  memw(r0+#4)=##0x3d96f504
#endif
  memw(r0+#8)=#0
  jumpr r31
.globl qd_v69_s2_dtype_qint8
qd_v69_s2_dtype_qint8:
  memw(r0+#0)=#7
#ifdef S2_UNIT_SCALES
  memw(r0+#4)=##0x3f800000
#else
  memw(r0+#4)=##0x3c18421d
#endif
  memw(r0+#8)=#0
  jumpr r31
