/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "qd_native_direct.h"

#ifndef QD_FIXTURE_PATH
#define QD_FIXTURE_PATH "/src/native_kernels/xwb_ci64_qdrx247_cpu_prefix_fixture.bin"
#endif
extern void qd_hmx_finish(void *,void *);
#define PAGE 2048u
#define POOL_CAP (4u*1024u*1024u)
#define FIX_CAP (6u*1024u*1024u)
static uint8_t fixture[FIX_CAP] __attribute__((aligned(128)));
static uint8_t graph_pool[POOL_CAP] __attribute__((aligned(128)));
static uint8_t graph_snapshot[POOL_CAP];
static uint32_t xblocks[2048],yblocks[2048],direct_xblocks[2048],direct_yblocks[2048];
static uint32_t graph_sizes[7]={0,POOL_CAP,0,0,0,0,0},direct_sizes[7];
static uint32_t direct_tables[512] __attribute__((aligned(128)));
static uint32_t direct_abi[30] __attribute__((aligned(16)));
static uint8_t active_snapshot[QD_DIRECT_ACTIVE_BYTES];

int main(void){
    FILE*f=fopen(QD_FIXTURE_PATH,"rb");if(!f){puts("FAIL fixture open");return 1;}
    uint32_t h[32];if(fread(h,sizeof h,1,f)!=1||h[1]!=247u){puts("FAIL fixture header");return 2;}
    uint32_t xb=h[7]*PAGE,yb=h[13]*PAGE,wb=h[17],fb=h[18];
    uint64_t total=(uint64_t)xb+yb+wb+fb;if(total>sizeof fixture||fread(fixture,(size_t)total,1,f)!=1||fgetc(f)!=EOF){puts("FAIL fixture payload");return 3;}fclose(f);
    const uint8_t*x=fixture,*y=x+xb,*w=y+yb,*fi=w+wb;
    memcpy(graph_pool,x,xb);memcpy(graph_pool+xb,y,yb);memcpy(graph_pool+xb+yb,w,wb);memcpy(graph_pool+xb+yb+wb,fi,fb);
    uint8_t*srcx=graph_pool,*srcy=graph_pool+xb,*srcw=graph_pool+xb+yb,*srcfi=srcw+wb;
    for(uint32_t i=0;i<h[7];i++)xblocks[i]=i*PAGE|1u;
    for(uint32_t i=0;i<h[13];i++)yblocks[i]=(xb+i*PAGE)|1u;
    for(uintptr_t a=(uintptr_t)srcx;a<(uintptr_t)srcx+xb;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    for(uintptr_t a=(uintptr_t)srcy;a<(uintptr_t)srcy+yb;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    for(uintptr_t a=(uintptr_t)srcw;a<(uintptr_t)srcw+wb;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    for(uintptr_t a=(uintptr_t)srcfi;a<(uintptr_t)srcfi+fb;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    __asm__ volatile("barrier":::"memory");memcpy(graph_snapshot,graph_pool,sizeof graph_pool);
    uint8_t*vtcm=(uint8_t*)(__rdcfg(__vtcm_base)<<16);
    for(uint32_t i=0;i<QD_DIRECT_VTCM_BYTES;i+=4096u)add_translation(vtcm+i,vtcm+i,6);
    memset(vtcm,0,QD_DIRECT_VTCM_BYTES);

    QdN3x3Request staged;memset(&staged,0,sizeof staged);staged.pool_count=7;staged.pool[1]=graph_pool;staged.pool_bytes=graph_sizes;
    staged.input=(QdN3x3Blocked){.dims={1,h[2],h[3],h[14]},.max_dims={1,h[4],h[5],h[14]},.padding={0,h[19],h[20],0},.slice_offset=h[6],.block_count=h[7],.blocks=xblocks};
    staged.output=(QdN3x3Blocked){.dims={1,h[8],h[9],h[24]?h[24]:h[15]},.max_dims={1,h[10],h[11],h[15]},.padding={0,h[21],h[22],0},.slice_offset=h[12],.block_count=h[13],.blocks=yblocks};
    staged.weights=srcw;staged.weights_bytes=wb;staged.weight_ci=h[14];staged.weight_co=h[15];staged.weight_kernel_h=3;staged.weight_kernel_w=3;
    staged.input_dtype=1;staged.output_dtype=1;staged.weight_dtype=2;staged.fi_dtype=5;staged.weight_format=h[16];
    staged.fi=(const int32_t*)srcfi;staged.fi_bytes=fb;staged.input_zero_point=h[23];staged.output_zero_point=h[25];
    staged.workspace=vtcm;staged.workspace_bytes=QD_N3X3_WORKSPACE_MAX;
    QdN3x3Prepared p;int e=qd_native_3x3_prepare(&staged,&p);if(e){printf("FAIL reference prepare=%d\n",e);return 4;}
    memcpy(active_snapshot,vtcm,QD_DIRECT_ACTIVE_BYTES);
    for(uint32_t i=0;i<h[7];i++)direct_xblocks[i]=i*PAGE|1u;
    for(uint32_t q=0;q<p.input_pages;q++){
        uint32_t ty=q/(p.input_page_cols*(staged.weight_ci/32u));
        uint32_t rem=q%(p.input_page_cols*(staged.weight_ci/32u));
        uint32_t tx=rem/(staged.weight_ci/32u),cb=rem%(staged.weight_ci/32u);
        uint32_t ix=((ty*p.input_page_cols+tx)*(staged.weight_ci/32u))+staged.input.slice_offset+cb;
        direct_xblocks[ix]=(uint32_t)((p.input_data+q*PAGE)-vtcm)|1u;
    }
    for(uint32_t i=0;i<h[13];i++)direct_yblocks[i]=i*PAGE|1u;
    for(uint32_t q=0;q<p.output_pages;q++){
        uint32_t ty=q/(p.output_page_cols*(staged.weight_co/32u));
        uint32_t rem=q%(p.output_page_cols*(staged.weight_co/32u));
        uint32_t tx=rem/(staged.weight_co/32u),cb=rem%(staged.weight_co/32u);
        uint32_t iy=((ty*p.output_page_cols+tx)*(staged.weight_co/32u))+staged.output.slice_offset+cb;
        direct_yblocks[iy]=(uint32_t)((p.output_data+q*PAGE)-vtcm)|1u;
    }
    direct_sizes[1]=QD_DIRECT_ACTIVE_BYTES;
    QdN3x3Request direct=staged;direct.pool[1]=vtcm;direct.pool_bytes=direct_sizes;
    direct.input.blocks=direct_xblocks;direct.output.blocks=direct_yblocks;
    direct.weights=p.weights;direct.fi=(const int32_t*)p.fi;direct.workspace=0;direct.workspace_bytes=0;
    QdDirectBinding b;e=qd_direct_validate_3x3(&direct,vtcm,QD_DIRECT_ACTIVE_BYTES,direct_tables,sizeof direct_tables,
        direct_abi,sizeof direct_abi,vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,0);
    if(e){printf("FAIL direct validate=%d\n",e);return 5;}
    e=qd_direct_bind_3x3(&direct,vtcm,QD_DIRECT_ACTIVE_BYTES,direct_tables,sizeof direct_tables,
        direct_abi,sizeof direct_abi,vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,&b);
    if(e){printf("FAIL direct bind=%d\n",e);return 6;}
    uint32_t mismatch=0;
    for(uint32_t i=0;i<p.input_pages;i++)mismatch+=direct_tables[i]!=p.input_table[i];
    for(uint32_t i=0;i<p.output_pages;i++)mismatch+=direct_tables[QD_N3X3_TABLE_ENTRIES+i]!=p.output_table[i];
    for(uint32_t i=1;i<6;i++)mismatch+=direct_abi[i]!=p.input_desc[i];
    for(uint32_t i=1;i<6;i++)mismatch+=direct_abi[6+i]!=p.output_desc[i];
    for(uint32_t i=0;i<16;i++)mismatch+=direct_abi[12+i]!=p.cfg[i];
    mismatch+=direct_abi[28]!=p.mode[0]||direct_abi[29]!=p.mode[1];
    if(mismatch){printf("FAIL staged/direct ABI mismatch=%u\n",mismatch);return 7;}
    if(memcmp(vtcm,active_snapshot,QD_DIRECT_ACTIVE_BYTES)){puts("FAIL bind wrote active pool pages");return 8;}
    for(uintptr_t a=(uintptr_t)direct_tables;a<(uintptr_t)direct_tables+sizeof direct_tables;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    for(uintptr_t a=(uintptr_t)direct_abi;a<(uintptr_t)direct_abi+sizeof direct_abi;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    __asm__ volatile("barrier":::"memory");
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    __asm__ volatile("barrier\n mxclracc":::"memory");
    qd_qnn_hmx_stride1_generic(b.out_desc,b.in_desc,b.weights,b.fi,b.cfg,b.mode);
    *b.done=0xffffffffu;qd_hmx_finish(b.zero_fi,(void*)b.done);
    uint32_t tries=0;while(*b.done&&tries++<100000u)__asm__ volatile("nop");
    if(*b.done){puts("FAIL HMX drain");return 9;}
    if(memcmp(graph_pool,graph_snapshot,sizeof graph_pool)){puts("FAIL graph-pool source changed");return 10;}
    uint32_t yoff=(uint32_t)(p.output_data-vtcm),outbytes=p.output_pages*PAGE;
    for(uint32_t i=0;i<QD_DIRECT_ACTIVE_BYTES;i++)if((i<yoff||i>=yoff+outbytes)&&vtcm[i]!=active_snapshot[i]){printf("FAIL VTCM outside Y changed at %x\n",i);return 11;}
    for(uint32_t i=0;i<PAGE;i++)if(((uint8_t*)b.output_guard)[i]!=0xa5||((uint8_t*)b.input_guard)[i]!=0){puts("FAIL guards changed");return 12;}
    if(((uint32_t*)b.zero_fi)[0]!=0x404000u||((uint32_t*)b.zero_fi)[32]!=0){puts("FAIL zero FI changed");return 13;}
    FILE*out=fopen("/src/native_runtime/direct-3x3-output.bin","wb");if(!out){puts("FAIL output open");return 14;}
    if(fwrite(h,sizeof h,1,out)!=1){puts("FAIL output header");return 15;}
    for(uint32_t i=0;i<b.out_pages;i++)if(fwrite((const void*)(uintptr_t)b.out_table[i],PAGE,1,out)!=1){puts("FAIL output gather");return 16;}
    fclose(out);
    printf("PASS direct S1 3x3 sequence 247: %u/%u input/output pages; ABI matched; no tensor staging\n",b.in_pages,b.out_pages);
    return 0;
}
