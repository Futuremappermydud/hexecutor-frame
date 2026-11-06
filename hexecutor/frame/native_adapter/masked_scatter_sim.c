#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "qd_native_1x1_adapter.h"
#define FIXTURE "/src/native_adapter/fixture-qdrx-1195.bin"
extern void qd_hmx_finish(void *,void *);
static uint8_t pool1[0x8000] __attribute__((aligned(128)));
static uint32_t tables[512] __attribute__((aligned(128)));
static uint32_t abi[30] __attribute__((aligned(128)));
static uint32_t pool_sizes[7]={0,sizeof pool1,0,0,0,0,0};
static int guard_reduction_lane_test(QdN1x1Prepared *p) {
    for(uint32_t lane=0;lane<QD_N1X1_BLOCK_BYTES;lane+=128u) {
        const uint32_t locations[2]={lane,lane+127u};
        for(uint32_t edge=0;edge<2;edge++) {
            uint32_t at=locations[edge];
            uint8_t saved=p->guard_block[at];p->guard_block[at]^=1u;
            int status=qd_native_1x1_finish(p);p->guard_block[at]=saved;
            if(status!=QD_N1X1_GUARD)return 1;
            saved=p->input_guard_block[at];p->input_guard_block[at]^=1u;
            status=qd_native_1x1_finish(p);p->input_guard_block[at]=saved;
            if(status!=QD_N1X1_GUARD)return 2;
        }
    }
    return 0;
}
int main(void){
 FILE *f=fopen(FIXTURE,"rb");uint32_t h[9];if(!f||fread(h,sizeof h,1,f)!=1||h[0]!=0x31585151u)return 1;
 if(fseek(f,36+(long)h[6]*2048,SEEK_SET))return 2;
 uint8_t weights[16384] __attribute__((aligned(128)));int32_t fi[256] __attribute__((aligned(128)));
 if(fread(weights,h[7],1,f)!=1||fread(fi,h[8],1,f)!=1)return 3;fclose(f);
 uint8_t *base=(uint8_t *)(__rdcfg(__vtcm_base)<<16);for(uint32_t i=0;i<0x80000;i+=4096)add_translation(base+i,base+i,6);memset(base,0xa5,0x80000);
 memset(pool1,0,sizeof pool1);memset(pool1+0x4000,0xa5,4u*2048u);
 for(uint32_t b=0;b<4;b++)for(uint32_t i=0;i<2048;i++)pool1[b*2048u+i]=(uint8_t)(b*41u+i*19u+3u);
 uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
 uint32_t xb[16],yb[10];for(uint32_t i=0;i<16;i++)xb[i]=0x801u;for(uint32_t i=0;i<10;i++)yb[i]=0x801u;for(uint32_t i=0;i<4;i++){xb[12+i]=i*2048u|1u;yb[6+i]=0x4000u+i*2048u|1u;}QdN1x1Request r;memset(&r,0,sizeof r);r.pool_count=7;r.pool[1]=pool1;r.pool_bytes=pool_sizes;
 r.input=(QdN1x1Blocked){.dims={1,2,8,128},.max_dims={1,2,8,128},.slice_offset=12,.block_count=16,.blocks=xb};
 r.output=(QdN1x1Blocked){.dims={1,2,8,128},.max_dims={1,2,8,128},.slice_offset=6,.block_count=10,.blocks=yb};
 r.weights=weights;r.weights_bytes=h[7];r.weight_ci=128;r.weight_co=128;r.weight_kernel_h=1;r.weight_kernel_w=1;
 r.input_dtype=QD_N1X1_DTYPE_U8;r.output_dtype=QD_N1X1_DTYPE_U8;r.weight_dtype=QD_N1X1_DTYPE_I8;r.fi_dtype=QD_N1X1_DTYPE_I32;r.weight_format=QD_N1X1_WEIGHT_XPB;r.fi=fi;r.fi_bytes=h[8];
 r.workspace=base;r.workspace_bytes=QD_N1X1_WORKSPACE_MAX;r.table_storage=tables;r.table_storage_bytes=sizeof tables;r.abi_storage=abi;r.abi_storage_bytes=sizeof abi;
 QdN1x1Prepared p;int e=qd_native_1x1_prepare(&r,&p);if(e){printf("FAIL prepare %d\n",e);return 4;}
 e=guard_reduction_lane_test(&p);if(e){printf("FAIL guard lane reduction %d\n",e);return 5;}
 __asm__ volatile("mxclracc":::"memory");qd_qnn_hmx_1x1(p.output_desc,p.input_desc,p.weights,p.fi,p.cfg,p.mode);
 uint8_t *done=base+0x52000;*(volatile uint32_t*)done=0xffffffffu;qd_hmx_finish(p.zero_words,done);uint32_t tries=0;while(*(volatile uint32_t*)done&&tries++<100000)__asm__ volatile("nop");if(*(volatile uint32_t*)done)return 5;
 e=qd_native_1x1_finish(&p);if(e){printf("FAIL finish %d\n",e);return 6;}
 for(uint32_t cb=0;cb<4;cb++)for(uint32_t y=2;y<8;y++)for(uint32_t x=0;x<8;x++)for(uint32_t c=0;c<32;c++)if(pool1[0x4000+cb*2048u+y*256+x*32+c]!=0xa5){printf("FAIL outside write y=%u x=%u c=%u\n",y,x,c);return 7;}
 printf("PASS masked scatter: effective 2x8 preserved rows 2..7; HVX guard scan detected both endpoints in all 16 128-byte lanes\n");return 0;
}
