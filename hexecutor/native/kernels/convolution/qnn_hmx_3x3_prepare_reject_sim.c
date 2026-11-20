/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Negative-path proof: unsupported geometry, bad block address and workspace
 * alias all reject before the VTCM workspace is written. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "qd_native_3x3_adapter.h"

#define POOL_BYTES 65536u
static uint8_t pool1[POOL_BYTES] __attribute__((aligned(128)));
static uint8_t workspace[QD_N3X3_WORKSPACE_MAX] __attribute__((aligned(128)));
static int8_t weights[9216] __attribute__((aligned(128)));
static int32_t fi[64] __attribute__((aligned(128)));
static uint32_t in_blocks[9],out_blocks[1],pool_sizes[7]={0,POOL_BYTES,0,0,0,0,0};

static QdN3x3Request valid_request(void) {
    QdN3x3Request r;memset(&r,0,sizeof r);
    for(uint32_t i=0;i<9;i++)in_blocks[i]=i*QD_N3X3_PAGE_BYTES|1u;
    out_blocks[0]=20u*QD_N3X3_PAGE_BYTES|1u;
    r.pool_count=7;r.pool[1]=pool1;r.pool_bytes=pool_sizes;
    r.input=(QdN3x3Blocked){.dims={1,10,10,32},.max_dims={1,24,24,32},
        .padding={0,7,7,0},.slice_offset=0,.block_count=9,.blocks=in_blocks};
    r.output=(QdN3x3Blocked){.dims={1,8,8,32},.max_dims={1,8,8,32},
        .padding={0,0,0,0},.slice_offset=0,.block_count=1,.blocks=out_blocks};
    r.weights=(const uint8_t*)weights;r.weights_bytes=sizeof weights;
    r.weight_ci=32;r.weight_co=32;r.weight_kernel_h=3;r.weight_kernel_w=3;
    r.input_dtype=QD_N3X3_DTYPE_U8;r.output_dtype=QD_N3X3_DTYPE_U8;
    r.weight_dtype=QD_N3X3_DTYPE_I8;r.fi_dtype=QD_N3X3_DTYPE_I32;
    r.weight_format=QD_N3X3_WEIGHT_XWB;r.fi=fi;r.fi_bytes=sizeof fi;
    r.workspace=workspace;r.workspace_bytes=sizeof workspace;
    return r;
}

static int untouched(void) {
    for(uint32_t i=0;i<sizeof workspace;i++)if(workspace[i]!=0x5au)return 0;
    return 1;
}
static int reject_untouched(const char*name,QdN3x3Request*r,int expected) {
    memset(workspace,0x5a,sizeof workspace);QdN3x3Prepared p;
    int e=qd_native_3x3_prepare(r,&p);
    if(e!=expected||!untouched()) {
        printf("FAIL %s status=%d expected=%d untouched=%d\n",name,e,expected,untouched());
        return 1;
    }
    printf("PASS reject %s status=%d workspace_untouched=1\n",name,e);return 0;
}

int main(void) {
    QdN3x3Prepared p;QdN3x3Request r=valid_request();memset(workspace,0x5a,sizeof workspace);
    int e=qd_native_3x3_prepare(&r,&p);
    if(e!=QD_N3X3_OK||untouched()) {printf("FAIL valid baseline status=%d\n",e);return 1;}
    puts("PASS valid baseline prepare");
    r=valid_request();r.weight_kernel_w=1;
    if(reject_untouched("wrong kernel geometry",&r,QD_N3X3_UNSUPPORTED))return 2;
    r=valid_request();in_blocks[8]=(POOL_BYTES+1u)|1u;
    if(reject_untouched("input block pool bounds",&r,QD_N3X3_RANGE))return 3;
    r=valid_request();r.weights=workspace;
    if(reject_untouched("weight/workspace alias",&r,QD_N3X3_ALIAS))return 4;
    puts("PASS all prepare rejection paths leave workspace unchanged");return 0;
}
