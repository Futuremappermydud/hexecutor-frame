/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
typedef unsigned u32;
void callback(void) {
    u32 tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    volatile u32 *out = (volatile u32 *)0xf0b12100;
    out[tid*4] = 0xc0de0000 | tid;
    out[tid*4+3] = 0x100; /* reached callback before the first HVX instruction */
    HVX_Vector a = Q6_V_vsplat_R(54 + tid);
    HVX_Vector b = Q6_V_vsplat_R(9);
    HVX_Vector c = Q6_Vw_vadd_VwVw(a,b);
    out[tid*4+1] = Q6_R_vextract_VR(c,0);
    __asm__ volatile("barrier" ::: "memory");
    out[tid*4+2] = 1;
}
