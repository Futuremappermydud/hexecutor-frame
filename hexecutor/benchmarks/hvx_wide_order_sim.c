/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdint.h>
#include <stdio.h>
#include <hexagon_standalone.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
int main(void) {
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    int16_t a[64] __attribute__((aligned(128))),b[64] __attribute__((aligned(128)));
    int32_t o[64] __attribute__((aligned(128)));
    for(unsigned i=0;i<64;i++){a[i]=i;b[i]=1;}
    HVX_VectorPair w=Q6_Ww_vmpy_VhVh(*(HVX_Vector *)a,*(HVX_Vector *)b);
    *(HVX_Vector *)o=Q6_V_lo_W(w);*(HVX_Vector *)(o+32)=Q6_V_hi_W(w);
    for(unsigned i=0;i<64;i++)printf("%d ",(int)o[i]);printf("\n");
    SIM_RELEASE_HVX;return 0;
}
