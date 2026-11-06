/* V75 simulator test of qd_linearclip_flat_u8 (HVX) against the reference
 * expression of prepared_reference.c k_linearclip, written independently in
 * int64: clip(((((x*m)+128)>>8)+32)>>6, 0, 255). Every input byte, many
 * multipliers (the model's 16620 included, and the extremes), unaligned
 * source and destination, every length class around the vector size, and the
 * identical in-place case. Guard bytes around the destination must survive. */
#include <hexagon_standalone.h>
#include <stdio.h>
#include <string.h>
#include "linearclip_flat_u8.h"

enum { N=20480+512, GUARD=64 };
static uint8_t srcbuf[N+256] __attribute__((aligned(128)));
static uint8_t dstbuf[N+256+2*GUARD] __attribute__((aligned(128)));
static uint8_t want[N];

static uint8_t oracle(uint8_t x,int64_t m) {
    int64_t v=(((int64_t)x*m+128)>>8)+32;v>>=6;
    return (uint8_t)(v<0?0:v>255?255:v);
}
static int check(const uint8_t *src,uint8_t *dst,size_t n,uint32_t m,const char *what) {
    for(size_t i=0;i<n;i++)want[i]=oracle(src[i],m);
    qd_linearclip_flat_u8(src,dst,n,m);
    for(size_t i=0;i<n;i++)if(dst[i]!=want[i]){
        printf("FAIL %s m=%u n=%u i=%u x=%u got=%u want=%u\n",what,(unsigned)m,(unsigned)n,(unsigned)i,(unsigned)src[i],(unsigned)dst[i],(unsigned)want[i]);
        return 0;
    }
    return 1;
}
int main(void) {
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    static const uint32_t ms[]={0,1,2,255,256,257,16384,16620,16621,32767,32768,40000,65534,65535,1234,777,43210};
    static const size_t ns[]={0,1,2,7,63,64,127,128,129,255,256,257,300,1000,4096,N};
    uint32_t seed=12345;
    for(size_t i=0;i<sizeof srcbuf;i++){seed=seed*1664525u+1013904223u;srcbuf[i]=(uint8_t)(seed>>24);}
    for(unsigned mi=0;mi<sizeof ms/sizeof ms[0];mi++) {
        /* all 256 byte values at every lane of a vector */
        for(unsigned rot=0;rot<128;rot+=37) {
            for(int i=0;i<512;i++)srcbuf[i]=(uint8_t)(i+rot);
            memset(dstbuf,0xcd,sizeof dstbuf);
            if(!check(srcbuf,dstbuf+GUARD,512,ms[mi],"all-values"))return 1;
        }
        for(size_t k=0;k<sizeof ns/sizeof ns[0];k++)
            for(unsigned so=0;so<3;so++)for(unsigned dof=0;dof<3;dof++) {
                size_t n=ns[k];unsigned s_off=so*5,d_off=dof*3;
                for(size_t i=0;i<sizeof srcbuf;i++){seed=seed*1664525u+1013904223u;srcbuf[i]=(uint8_t)(seed>>24);}
                memset(dstbuf,0xcd,sizeof dstbuf);
                if(!check(srcbuf+s_off,dstbuf+GUARD+d_off,n,ms[mi],"disjoint"))return 1;
                for(unsigned g=0;g<GUARD+d_off;g++)if(dstbuf[g]!=0xcd){printf("FAIL guard before n=%u\n",(unsigned)n);return 2;}
                for(size_t g=GUARD+d_off+n;g<sizeof dstbuf;g++)if(dstbuf[g]!=0xcd){printf("FAIL guard after n=%u\n",(unsigned)n);return 2;}
            }
        /* in place, aligned and not */
        for(unsigned off=0;off<2;off++) {
            size_t n=1000;
            for(size_t i=0;i<sizeof srcbuf;i++){seed=seed*1664525u+1013904223u;srcbuf[i]=(uint8_t)(seed>>24);}
            for(size_t i=0;i<n;i++)want[i]=oracle(srcbuf[off*7+i],ms[mi]);
            qd_linearclip_flat_u8(srcbuf+off*7,srcbuf+off*7,n,ms[mi]);
            for(size_t i=0;i<n;i++)if(srcbuf[off*7+i]!=want[i]){printf("FAIL in-place m=%u i=%u\n",(unsigned)ms[mi],(unsigned)i);return 3;}
        }
    }
    printf("PASS linearclip flat u8: %u multipliers x lengths x alignments, all 256 values, in place, guards intact\n",
           (unsigned)(sizeof ms/sizeof ms[0]));
    return 0;
}
