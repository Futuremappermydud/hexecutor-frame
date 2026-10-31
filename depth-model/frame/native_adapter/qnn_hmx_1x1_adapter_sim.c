#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "qd_native_1x1_adapter.h"

#define FIXTURE "/src/native_adapter/fixture-qdrx-1195.bin"
#define OUTPUT  "/src/native_adapter/output-qdrx-1195.bin"
#define MAGIC 0x31585151u
extern void qd_hmx_finish(void *, void *);

static uint8_t pool1[0x210000] __attribute__((aligned(128)));
static uint32_t block_tables[512] __attribute__((aligned(128)));
static uint32_t abi_words[30] __attribute__((aligned(128)));
static uint32_t pool_sizes[7]={0,sizeof pool1,0,0,0,0,0};

int main(void) {
    FILE *f=fopen(FIXTURE,"rb");
    if(!f){puts("FAIL fixture open");return 1;}
    uint32_t h[9];
    if(fread(h,sizeof h,1,f)!=1 || h[0]!=MAGIC || h[1]!=1195 ||
       h[2]!=8 || h[3]!=64 || h[4]!=128 || h[5]!=128 || h[6]!=32 ||
       h[7]!=16384 || h[8]!=1024){puts("FAIL fixture header");return 2;}
    uint8_t *base=(uint8_t *)(__rdcfg(__vtcm_base)<<16);
    for(uint32_t i=0;i<0x80000u;i+=4096)add_translation(base+i,base+i,6);
    memset(base,0,0x80000u);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;
    __asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *src=pool1+0x100000u;
    uint8_t *weights=pool1+0x200000u,*fi=pool1+0x204000u,*done=base+0x52000u;
    if(fread(src,2048,32,f)!=32 || fread(weights,h[7],1,f)!=1 ||
       fread(fi,h[8],1,f)!=1){puts("FAIL fixture payload");return 3;}
    fclose(f);
    memset(pool1+0x140000u,0xa5,32u*2048u);
    for(uintptr_t a=(uintptr_t)src;a<(uintptr_t)src+32u*2048u;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    for(uintptr_t a=(uintptr_t)weights;a<(uintptr_t)weights+h[7];a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    for(uintptr_t a=(uintptr_t)fi;a<(uintptr_t)fi+h[8];a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    __asm__ volatile("barrier" ::: "memory");
    uint32_t xb[32],yb[32];
    for(uint32_t i=0;i<32;i++){xb[i]=0x100000u+i*2048u|1u;yb[i]=0x140000u+i*2048u|1u;}
    QdN1x1Request r;memset(&r,0,sizeof r);
    r.pool_count=7;r.pool[1]=pool1;r.pool_bytes=pool_sizes;
    r.input=(QdN1x1Blocked){.dims={1,8,64,128},.max_dims={1,8,64,128},
        .block_count=32,.blocks=xb};
    r.output=(QdN1x1Blocked){.dims={1,8,64,128},.max_dims={1,8,64,128},
        .block_count=32,.blocks=yb};
    r.weights=weights;r.weights_bytes=h[7];r.weight_ci=128;r.weight_co=128;
    r.weight_kernel_h=1;r.weight_kernel_w=1;
    r.input_dtype=QD_N1X1_DTYPE_U8;r.output_dtype=QD_N1X1_DTYPE_U8;
    r.weight_dtype=QD_N1X1_DTYPE_I8;r.fi_dtype=QD_N1X1_DTYPE_I32;
    r.weight_format=QD_N1X1_WEIGHT_XPB;r.fi=(const int32_t *)fi;r.fi_bytes=h[8];
    r.table_storage=block_tables;r.table_storage_bytes=sizeof block_tables;
    r.abi_storage=abi_words;r.abi_storage_bytes=sizeof abi_words;
    /* Simulator VTCM is 512 KiB; hardware integration places this exact
     * workspace at the independently leased upper 512 KiB of pool1. */
    r.workspace=base;r.workspace_bytes=QD_N1X1_WORKSPACE_MAX;
    QdN1x1Prepared p;int e=qd_native_1x1_prepare(&r,&p);
    if(e){printf("FAIL prepare %d\n",e);return 4;}
    uint32_t in_desc[6],out_desc[6],cfg[16],mode[2];
    memcpy(in_desc,p.input_desc,sizeof in_desc);memcpy(out_desc,p.output_desc,sizeof out_desc);
    memcpy(cfg,p.cfg,sizeof cfg);memcpy(mode,p.mode,sizeof mode);
    __asm__ volatile("mxclracc" ::: "memory");
    qd_qnn_hmx_1x1(out_desc,in_desc,p.weights,p.fi,cfg,mode);
    *(volatile uint32_t *)done=0xffffffffu;qd_hmx_finish(p.zero_words,done);
    uint32_t tries=0;while(*(volatile uint32_t *)done && tries++<100000)__asm__ volatile("nop");
    if(*(volatile uint32_t *)done){printf("FAIL HMX completion done=%08x tries=%u\n",*(volatile uint32_t *)done,(unsigned)tries);return 5;}
    e=qd_native_1x1_finish(&p);if(e){printf("FAIL finish %d\n",e);return 6;}
    f=fopen(OUTPUT,"wb");if(!f){puts("FAIL output open");return 7;}
    if(fwrite(h,sizeof h,1,f)!=1){puts("FAIL output header");return 8;}
    for(uint32_t i=0;i<p.output_blocks;i++)
        if(fwrite((const void *)(uintptr_t)p.output_table[i],2048,1,f)!=1){puts("FAIL output blocks");return 9;}
    fclose(f);
    printf("PASS command 1195 adapter + original leaf: %u input / %u output croutons, workspace %u B\n",
           p.input_blocks,p.output_blocks,p.workspace_used);
    return 0;
}
