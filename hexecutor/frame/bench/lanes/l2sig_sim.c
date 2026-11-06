/* ISS check: the QDS_L2_SIG signature must change for any single flipped byte, and for
 * paired flips and swaps that a plain XOR would miss. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#define BYTES 16384u
static uint8_t mem[BYTES] __attribute__((aligned(128)));
static uint64_t sig(uint64_t h,uint32_t base,uint32_t bytes){
    const HVX_Vector *a=(const HVX_Vector *)base;
    HVX_Vector x=Q6_V_vzero(),t=Q6_V_vzero();
    for(uint32_t i=0;i<bytes/128u;i++){
        HVX_Vector v=a[i];
        x=Q6_V_vxor_VV(x,v);t=Q6_Vw_vadd_VwVw(Q6_Vw_vsub_VwVw(t,x),v);
    }
    for(uint32_t i=0;i<32u;i++){
        h=(h^(uint32_t)Q6_R_vextract_VR(x,i*4u))*0x100000001b3ull;
        h=(h^(uint32_t)Q6_R_vextract_VR(t,i*4u))*0x100000001b3ull;
    }
    return h;
}
int main(void){
    for(uint32_t i=0;i<BYTES;i++)mem[i]=(uint8_t)(i*2654435761u>>24);
    const uint64_t H=0xcbf29ce484222325ull,base=sig(H,(uint32_t)mem,BYTES);
    uint32_t miss=0,tests=0;
    if(sig(H,(uint32_t)mem,BYTES)!=base){puts("FAIL not deterministic");return 1;}
    for(uint32_t i=0;i<BYTES;i+=37){ /* single bits */
        uint8_t b=1u<<(i&7);mem[i]^=b;tests++;miss+=sig(H,(uint32_t)mem,BYTES)==base;mem[i]^=b;
    }
    printf("single-bit: %u tests, %u missed\n",tests,miss);
    uint32_t pm=0,pt=0;
    for(uint32_t i=0;i+128<BYTES;i+=211){ /* the same bit flipped in two vectors: XOR alone cancels */
        uint8_t b=1u<<(i&7);mem[i]^=b;mem[i+128]^=b;pt++;pm+=sig(H,(uint32_t)mem,BYTES)==base;mem[i]^=b;mem[i+128]^=b;
    }
    printf("paired-bit: %u tests, %u missed\n",pt,pm);
    uint32_t sm=0,st=0;static uint8_t tmp[128] __attribute__((aligned(128)));
    for(uint32_t i=0;i+256<=BYTES;i+=128*7){ /* two vectors swapped */
        memcpy(tmp,mem+i,128);memcpy(mem+i,mem+i+128,128);memcpy(mem+i+128,tmp,128);st++;
        sm+=sig(H,(uint32_t)mem,BYTES)==base;
        memcpy(tmp,mem+i,128);memcpy(mem+i,mem+i+128,128);memcpy(mem+i+128,tmp,128);
    }
    printf("swapped vectors: %u tests, %u missed\n",st,sm);
    if(sig(H,(uint32_t)mem,BYTES)!=base){puts("FAIL restore");return 1;}
    puts(miss||pm||sm?"FAIL":"PASS");
    return miss||pm||sm;
}
