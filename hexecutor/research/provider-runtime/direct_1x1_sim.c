/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "qd_native_direct.h"

#ifndef QD_FIXTURE_PATH
#define QD_FIXTURE_PATH "/src/native_adapter/fixture-qdrx-1195.bin"
#endif
extern void qd_hmx_finish(void *,void *);
static uint8_t graph_pool[0x210000] __attribute__((aligned(128)));
static uint32_t src_xb[32],src_yb[32],direct_xb[32],direct_yb[32];
static uint32_t adapter_tables[512] __attribute__((aligned(128)));
static uint32_t adapter_abi[30] __attribute__((aligned(16)));
static uint32_t graph_sizes[7]={0,sizeof graph_pool,0,0,0,0,0};
static uint32_t direct_sizes[7];
static uint8_t saved_output[32u*2048u];
static uint8_t saved_active[QD_DIRECT_ACTIVE_BYTES];
static uint8_t saved_graph[0x210000];

int main(void){
    FILE*f=fopen(QD_FIXTURE_PATH,"rb");if(!f){puts("FAIL open");return 1;}
    uint32_t h[9];if(fread(h,sizeof h,1,f)!=1||h[0]!=0x31585151u||h[6]!=32||h[7]!=16384||h[8]!=1024){puts("FAIL header");return 2;}
    uint8_t*vtcm=(uint8_t*)(__rdcfg(__vtcm_base)<<16);
    for(uint32_t i=0;i<QD_DIRECT_VTCM_BYTES;i+=4096u)add_translation(vtcm+i,vtcm+i,6);
    memset(vtcm,0,QD_DIRECT_VTCM_BYTES);
    uint8_t*src=graph_pool+0x100000u,*src_y=graph_pool+0x140000u;
    uint8_t*src_w=graph_pool+0x200000u,*src_fi=graph_pool+0x204000u;
    if(fread(src,2048,32,f)!=32||fread(src_w,h[7],1,f)!=1||fread(src_fi,h[8],1,f)!=1){puts("FAIL payload");return 3;}fclose(f);
    for(uintptr_t a=(uintptr_t)src;a<(uintptr_t)src+32u*2048u;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    for(uintptr_t a=(uintptr_t)src_w;a<(uintptr_t)src_w+h[7];a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    for(uintptr_t a=(uintptr_t)src_fi;a<(uintptr_t)src_fi+h[8];a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    __asm__ volatile("barrier":::"memory");
    memset(src_y,0x5a,32u*2048u);memcpy(saved_output,src_y,sizeof saved_output);
    memcpy(saved_graph,graph_pool,sizeof graph_pool);
    for(uint32_t i=0;i<32;i++){src_xb[i]=(0x100000u+i*2048u)|1u;src_yb[i]=(0x140000u+i*2048u)|1u;}
    QdN1x1Request staged;memset(&staged,0,sizeof staged);staged.pool_count=7;staged.pool[1]=graph_pool;staged.pool_bytes=graph_sizes;
    staged.input=(QdN1x1Blocked){.dims={1,8,64,128},.max_dims={1,8,64,128},.block_count=32,.blocks=src_xb};
    staged.output=(QdN1x1Blocked){.dims={1,8,64,128},.max_dims={1,8,64,128},.block_count=32,.blocks=src_yb};
    staged.weights=src_w;staged.weights_bytes=h[7];staged.weight_ci=128;staged.weight_co=128;
    staged.weight_kernel_h=1;staged.weight_kernel_w=1;staged.input_dtype=1;staged.output_dtype=1;
    staged.weight_dtype=2;staged.fi_dtype=5;staged.weight_format=QD_N1X1_WEIGHT_XPB;
    staged.fi=(const int32_t*)src_fi;staged.fi_bytes=h[8];staged.table_storage=adapter_tables;
    staged.table_storage_bytes=sizeof adapter_tables;staged.abi_storage=adapter_abi;staged.abi_storage_bytes=sizeof adapter_abi;
    staged.workspace=vtcm;staged.workspace_bytes=QD_N1X1_WORKSPACE_MAX;
    QdN1x1Prepared p;int e=qd_native_1x1_prepare(&staged,&p);if(e){printf("FAIL reference prepare=%d\n",e);return 4;}
    memcpy(saved_active,vtcm,QD_DIRECT_ACTIVE_BYTES);
    uint32_t*in_desc_expected=(uint32_t*)adapter_abi,*out_desc_expected=in_desc_expected+6;
    uint32_t*cfg_expected=out_desc_expected+6,*mode_expected=cfg_expected+16;
    uint32_t table_expected[512];memcpy(table_expected,adapter_tables,sizeof table_expected);
    uint32_t desc_expected[30];memcpy(desc_expected,adapter_abi,sizeof desc_expected);

    /* Present the staging adapter's exact VTCM input/output/W/FI addresses as
     * an already-bound pool1 map. Reverse each page index to match its proven
     * table order; the new binder must emit byte-identical leaf pointer ABI. */
    for(uint32_t i=0;i<32;i++){direct_xb[i]=((31u-i)*2048u)|1u;direct_yb[i]=(0x20000u+(31u-i)*2048u)|1u;}
    direct_sizes[1]=QD_DIRECT_ACTIVE_BYTES;
    QdN1x1Request direct=staged;direct.pool[1]=vtcm;direct.pool_bytes=direct_sizes;
    direct.input.blocks=direct_xb;direct.output.blocks=direct_yb;
    direct.weights=vtcm+0x40000u;direct.fi=(const int32_t*)(vtcm+0x50000u);
    direct.workspace=0;direct.workspace_bytes=0;direct.table_storage=0;direct.abi_storage=0;
    QdDirectBinding b;e=qd_direct_bind_1x1(&direct,vtcm,QD_DIRECT_ACTIVE_BYTES,adapter_tables,sizeof adapter_tables,
        adapter_abi,sizeof adapter_abi,vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,&b);
    if(e){printf("FAIL direct bind=%d\n",e);return 5;}
    uint32_t mismatch=0;
    for(uint32_t i=0;i<32;i++)mismatch+=adapter_tables[i]!=table_expected[i];
    for(uint32_t i=0;i<32;i++)mismatch+=adapter_tables[QD_N1X1_TABLE_ENTRIES+i]!=table_expected[QD_N1X1_TABLE_ENTRIES+i];
    for(uint32_t i=0;i<30;i++)mismatch+=adapter_abi[i]!=desc_expected[i];
    if(b.weights!=p.weights||b.fi!=p.fi||b.in_pages!=p.input_blocks||b.out_pages!=p.output_blocks)mismatch++;
    if(mismatch){printf("FAIL staged/direct ABI differs count=%u\n",mismatch);
        for(uint32_t i=0;i<30;i++)if(adapter_abi[i]!=desc_expected[i])printf("abi[%u] %08x != %08x\n",i,adapter_abi[i],desc_expected[i]);
        for(uint32_t i=0;i<32;i++)if(adapter_tables[i]!=table_expected[i])printf("it[%u] %08x != %08x\n",i,adapter_tables[i],table_expected[i]);
        for(uint32_t i=0;i<32;i++)if(adapter_tables[QD_N1X1_TABLE_ENTRIES+i]!=table_expected[QD_N1X1_TABLE_ENTRIES+i])printf("ot[%u] %08x != %08x\n",i,adapter_tables[QD_N1X1_TABLE_ENTRIES+i],table_expected[QD_N1X1_TABLE_ENTRIES+i]);
        printf("cfg direct/current %08x/%08x %08x/%08x mode %08x/%08x\n",cfg_expected[3],adapter_abi[15],cfg_expected[6],adapter_abi[18],mode_expected[0],adapter_abi[28]);return 6;}
    if(memcmp(vtcm,saved_active,QD_DIRECT_ACTIVE_BYTES)){puts("FAIL bind modified active VTCM tensor bytes");return 7;}
    for(uintptr_t a=(uintptr_t)adapter_tables;a<(uintptr_t)adapter_tables+sizeof adapter_tables;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    for(uintptr_t a=(uintptr_t)adapter_abi;a<(uintptr_t)adapter_abi+sizeof adapter_abi;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    __asm__ volatile("barrier":::"memory");
    uint32_t in_desc[6],out_desc[6],cfg[16],mode[2];
    memcpy(in_desc,b.in_desc,sizeof in_desc);memcpy(out_desc,b.out_desc,sizeof out_desc);
    memcpy(cfg,b.cfg,sizeof cfg);memcpy(mode,b.mode,sizeof mode);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;
    __asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    __asm__ volatile("barrier\n mxclracc":::"memory");
    qd_qnn_hmx_1x1(out_desc,in_desc,b.weights,b.fi,cfg,mode);
    *b.done=0xffffffffu;qd_hmx_finish(b.zero_fi,(void*)b.done);
    uint32_t tries=0;while(*b.done&&tries++<100000u)__asm__ volatile("nop");
    if(*b.done){puts("FAIL completion");return 8;}
    if(memcmp(graph_pool,saved_graph,sizeof graph_pool)){puts("FAIL graph-pool input/weight/FI/output source changed");return 9;}
    for(uint32_t i=0;i<QD_DIRECT_ACTIVE_BYTES;i++)
        if(i<0x20000u||i>=0x30000u)if(vtcm[i]!=saved_active[i]){printf("FAIL outside active output changed at %x\n",i);return 10;}
    for(uint32_t i=0;i<QD_DIRECT_PAGE_BYTES;i++)if(((uint8_t*)b.output_guard)[i]!=0xa5||((uint8_t*)b.input_guard)[i]!=0){puts("FAIL guard changed");return 11;}
    if(((uint32_t*)b.zero_fi)[0]!=0x404000u||((uint32_t*)b.zero_fi)[32]!=0){puts("FAIL zero FI changed");return 12;}
    FILE*out=fopen("/src/native_runtime/direct-1x1-output.bin","wb");
    if(!out||fwrite(h,sizeof h,1,out)!=1){puts("FAIL output header");return 13;}
    for(uint32_t i=0;i<b.out_pages;i++)if(fwrite((const void*)(uintptr_t)b.out_table[i],2048,1,out)!=1){puts("FAIL output table gather");return 14;}
    fclose(out);
    uint32_t changed=0;for(uint32_t i=0;i<sizeof saved_output;i++)changed+=((vtcm+0x20000u)[i]!=saved_output[i]);
    printf("PASS direct 1x1: staged/direct ABI identical; %u pages; output bytes changed=%u; outside active pages and guards preserved\n",b.out_pages,changed);
    return 0;
}
