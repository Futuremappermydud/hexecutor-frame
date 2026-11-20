/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "im2col_u8_crouton_direct.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    uint8_t *const *pools;
    uint32_t fail;
    int fail_enabled;
} Access;

static uint8_t *access_block(uint32_t e,int store,void *opaque) {
    Access *a=(Access *)opaque;(void)store;
    if (a->fail_enabled && e==a->fail) return NULL;
    uint32_t p=e&127u,off=e&~127u;
    return p && a->pools[p] ? a->pools[p]+off : NULL;
}

static void dense_reference(uint8_t *out,const uint8_t *src) {
    for (int h=0;h<8;h++) for (int w=0;w<256;w++) {
        uint8_t *d=out+((size_t)h*256+w)*9;int c=0;
        for (int ky=0;ky<3;ky++) for (int kx=0;kx<3;kx++)
            d[c++]=src[(size_t)(h+ky)*258+w+kx];
    }
}

static void blocked_reference(uint8_t *blocks,const uint8_t *dense) {
    memset(blocks,0,32*2048);
    for (int h=0;h<8;h++) for (int w=0;w<256;w++) {
        uint8_t *d=blocks+(size_t)(w/8)*2048+h*256+(w%8)*32;
        memcpy(d,dense+((size_t)h*256+w)*9,9);
    }
}

int main(void) {
    static uint8_t pool_mem[40*2048];
    static uint8_t src[2580], dense[8*256*9], expected[32*2048], before[40*2048];
    uint8_t *pools[QD_LAYOUT_POOL_COUNT]={0};
    size_t pool_bytes[QD_LAYOUT_POOL_COUNT]={0};uint32_t blocks[40];
    int32_t in[4]={1,10,258,1},out[4]={1,8,256,9},max[4]={1,8,256,32};
    int32_t kernel[4]={3,3,1,32},stride[4]={1,1,1,1};
    pools[1]=pool_mem;pool_bytes[1]=sizeof pool_mem;
    for (uint32_t b=0;b<40;b++) blocks[b]=1u | (b*2048u);
    Access access={pools,0,0};

    for (unsigned seed=0;seed<32;seed++) {
        for (size_t i=0;i<sizeof src;i++) src[i]=(uint8_t)(i*31u+seed*47u+(i>>3));
        memset(pool_mem,0xa5,sizeof pool_mem);
        dense_reference(dense,src);blocked_reference(expected,dense);
        if (!qd_layout_im2col_u8_crouton_direct(src,sizeof src,in,out,max,kernel,stride,
                blocks,40,4,pools,pool_bytes,access_block,&access)) return 1;
        if (memcmp(pool_mem+4*2048,expected,sizeof expected)) {
            fprintf(stderr,"blocked output mismatch at seed %u\n",seed);return 2;
        }
        for (size_t i=0;i<4*2048;i++) if (pool_mem[i]!=0xa5) return 9;
        for (size_t i=36*2048;i<sizeof pool_mem;i++) if (pool_mem[i]!=0xa5) return 10;
    }

    /* Reject callback failure before writes. */
    memset(pool_mem,0x5a,sizeof pool_mem);memcpy(before,pool_mem,sizeof before);
    access.fail=blocks[11];access.fail_enabled=1;
    if (qd_layout_im2col_u8_crouton_direct(src,sizeof src,in,out,max,kernel,stride,
            blocks,40,4,pools,pool_bytes,access_block,&access)) return 3;
    if (memcmp(pool_mem,before,sizeof before)) return 4;
    access.fail_enabled=0;

    /* Reject aliasing and a repeated output block without mutation. */
    if (qd_layout_im2col_u8_crouton_direct(pool_mem+4*2048,sizeof src,in,out,max,kernel,stride,
            blocks,40,4,pools,pool_bytes,access_block,&access)) return 5;
    if (memcmp(pool_mem,before,sizeof before)) return 6;
    blocks[35]=blocks[34];
    if (qd_layout_im2col_u8_crouton_direct(src,sizeof src,in,out,max,kernel,stride,
            blocks,40,4,pools,pool_bytes,access_block,&access)) return 7;
    if (memcmp(pool_mem,before,sizeof before)) return 8;
    blocks[35]=1u | (35u*2048u);

    /* Encoded pool 7 and 127 must reject before indexing the seven-entry
     * arrays. The same test is useful under ASan to catch any out-of-bounds. */
    for (uint32_t bad_pool_i=0;bad_pool_i<2;bad_pool_i++) {
        blocks[4]=(bad_pool_i?127u:7u) | (4u*2048u);
        if (qd_layout_im2col_u8_crouton_direct(src,sizeof src,in,out,max,kernel,stride,
                blocks,40,4,pools,pool_bytes,access_block,&access)) return 11;
        if (memcmp(pool_mem,before,sizeof before)) return 12;
    }
    blocks[4]=1u | (4u*2048u);

    /* A 128-byte aligned but non-2048-aligned offset must reject. */
    blocks[4]=1u | (4u*2048u+128u);
    if (qd_layout_im2col_u8_crouton_direct(src,sizeof src,in,out,max,kernel,stride,
            blocks,40,4,pools,pool_bytes,access_block,&access)) return 13;
    if (memcmp(pool_mem,before,sizeof before)) return 14;
    blocks[4]=1u | (4u*2048u);

    /* Different encoded pools can still point into overlapping physical
     * ranges; validate actual addresses, not only duplicate IDs. */
    pools[2]=pool_mem+5*2048;pool_bytes[2]=sizeof pool_mem-5*2048;
    blocks[4]=2u;
    if (qd_layout_im2col_u8_crouton_direct(src,sizeof src,in,out,max,kernel,stride,
            blocks,40,4,pools,pool_bytes,access_block,&access)) return 15;
    if (memcmp(pool_mem,before,sizeof before)) return 16;

    printf("PASS fused im2col u8 geometry, slice, zero padding, pool/bounds/alias guards\n");
    return 0;
}
