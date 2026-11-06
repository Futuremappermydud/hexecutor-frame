/* Exhaustive ISS proof of the cheaper plane merge. Old: clamp 4096..8192, subtract, double, s1 mul.
 * New: high plane offset so it clips at q>=4096, one saturating subtract, s1 mul by 2M.
 * Models the planes exactly: lo = q mod 256, hi_old = clip(floor(q/256)+16,0,255),
 * hi_new = clip(floor(q/256)+239,0,255). Reference: clip(clip(q,0,4096)*M+8192 >> 14,0,255),
 * which is the engine's exact rule for every q (q>4096 saturates to 255 for M>=1024). */
#include <stdint.h>
#include <stdio.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <hexagon_sim_timer.h>
static inline HVX_Vector rq_old(HVX_Vector lo,HVX_Vector hi,HVX_Vector mv){
    const HVX_Vector c4096=Q6_Vh_vsplat_R(4096),c8192=Q6_Vh_vsplat_R(8192);
    HVX_VectorPair w=Q6_W_vshuff_VVR(hi,lo,-1);
    HVX_Vector a=Q6_Vh_vsub_VhVh(Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vmax_VuhVuh(Q6_V_lo_W(w),c4096),c8192),c4096),
               b=Q6_Vh_vsub_VhVh(Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vmax_VuhVuh(Q6_V_hi_W(w),c4096),c8192),c4096);
    a=Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_Vh_vadd_VhVh(a,a),mv);
    b=Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_Vh_vadd_VhVh(b,b),mv);
    return Q6_Vub_vpack_VhVh_sat(b,a);
}
static inline HVX_Vector rq_new(HVX_Vector lo,HVX_Vector hi,HVX_Vector mv2){
    const HVX_Vector off=Q6_Vh_vsplat_R(239<<8);
    HVX_VectorPair w=Q6_W_vshuff_VVR(hi,lo,-1);
    HVX_Vector a=Q6_Vuh_vsub_VuhVuh_sat(Q6_V_lo_W(w),off),b=Q6_Vuh_vsub_VuhVuh_sat(Q6_V_hi_W(w),off);
    a=Q6_Vh_vmpy_VhVh_s1_rnd_sat(a,mv2);
    b=Q6_Vh_vmpy_VhVh_s1_rnd_sat(b,mv2);
    return Q6_Vub_vpack_VhVh_sat(b,a);
}
static uint8_t L[128] __attribute__((aligned(128))),H0[128] __attribute__((aligned(128))),H1[128] __attribute__((aligned(128)));
static int16_t MV[64] __attribute__((aligned(128))),MV2[64] __attribute__((aligned(128)));
static uint8_t O0[128] __attribute__((aligned(128))),O1[128] __attribute__((aligned(128)));
static int clip8(int v){return v<0?0:v>255?255:v;}
int main(void){
    unsigned long long tests=0,bad0=0,bad1=0;
    /* every q from -70000..70000 for a set of M including extremes; lane-mixed */
    static const int Ms[]={1024,1025,1100,1365,1536,1707,1924,2000,2046,2047};
    for(unsigned mi=0;mi<sizeof Ms/sizeof Ms[0];mi++){
        int M=Ms[mi];
        for(int l=0;l<64;l++){MV[l]=(int16_t)M;MV2[l]=(int16_t)(2*M);}
        for(int q0=-70000;q0<70000;q0+=128){
            int q[128];
            for(int i=0;i<128;i++){
                q[i]=q0+i;
                int fl=q[i]>=0?q[i]/256:-((-q[i]+255)/256);
                L[i]=(uint8_t)(q[i]&255);
                H0[i]=(uint8_t)(fl+16<0?0:fl+16>255?255:fl+16);
                H1[i]=(uint8_t)(fl+239<0?0:fl+239>255?255:fl+239);
            }
            *(HVX_Vector*)O0=rq_old(*(HVX_Vector*)L,*(HVX_Vector*)H0,*(HVX_Vector*)MV);
            *(HVX_Vector*)O1=rq_new(*(HVX_Vector*)L,*(HVX_Vector*)H1,*(HVX_Vector*)MV2);
            for(int i=0;i<128;i++){
                int qc=q[i]<0?0:q[i]>4096?4096:q[i];
                int want=clip8((qc*M+8192)>>14);
                tests++;bad0+=O0[i]!=want;bad1+=O1[i]!=want;
            }
        }
    }
    printf("lanes %llu old mismatches %llu new mismatches %llu\n",tests,bad0,bad1);
    /* relative packet cost over a block-sized loop */
    static uint8_t BL[2048] __attribute__((aligned(128))),BH[2048] __attribute__((aligned(128))),BO[2048] __attribute__((aligned(128)));
    for(int i=0;i<64;i++){MV[i]=1500;MV2[i]=3000;}
    HVX_Vector mv=*(HVX_Vector*)MV,mv2=*(HVX_Vector*)MV2;
    unsigned long long a=hexagon_sim_read_pcycles();
    for(int r=0;r<200;r++)for(int i=0;i<16;i++)((HVX_Vector*)BO)[i]=rq_old(((HVX_Vector*)BL)[i],((HVX_Vector*)BH)[i],mv);
    unsigned long long b=hexagon_sim_read_pcycles();
    for(int r=0;r<200;r++)for(int i=0;i<16;i++)((HVX_Vector*)BO)[i]=rq_new(((HVX_Vector*)BL)[i],((HVX_Vector*)BH)[i],mv2);
    unsigned long long c=hexagon_sim_read_pcycles();
    printf("sim pcycles per 2 KiB block: old %.1f new %.1f\n",(double)(b-a)/200,(double)(c-b)/200);
    return bad1!=0;
}
