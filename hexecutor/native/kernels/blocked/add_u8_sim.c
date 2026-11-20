/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "add_u8.h"
#include <hexagon_types.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint8_t amem[65664] __attribute__((aligned(128)));
static uint8_t bmem[65664] __attribute__((aligned(128)));
static uint8_t omem[65664] __attribute__((aligned(128)));
static uint8_t *apools[128],*bpools[128],*opools[128];
static uint32_t ablocks[32],bblocks[32],oblocks[32];
static uint8_t ci_mem[45056] __attribute__((aligned(128))),bi_mem[45056] __attribute__((aligned(128))),co_mem[45056] __attribute__((aligned(128)));
static uint8_t *cip[128],*bip[128],*cop[128];
static uint32_t cib[5],bib[5],cob[20];
static QdBlockedU8 full(uint8_t **p,uint32_t *blocks) {
    QdBlockedU8 t={.pools=p,.blocks=blocks,.block_count=32,.slice=0,
        .meta={1,8,8,1024},.dims={1,8,8,1024},.origin={0,0,0,0}};return t;
}
int main(void) {
    apools[1]=amem;bpools[1]=bmem;opools[1]=omem;
    for(unsigned i=0;i<32;i++)ablocks[i]=bblocks[i]=oblocks[i]=1u+i*2048u;
    for(unsigned i=0;i<65536;i++){amem[i]=(uint8_t)i;bmem[i]=(uint8_t)(i>>8);}
    memset(omem,0xa5,sizeof omem);
    QdBlockedU8 a=full(apools,ablocks),b=full(bpools,bblocks),o=full(opools,oblocks);
    QdBlockedAddU8 prep;QdAddU8Params p={16,1151,13};
    if(!qd_blocked_add_u8_prepare(&a,&b,&o,&prep)||!qd_blocked_add_u8_apply(&prep,&p))return 1;
    for(unsigned i=0;i<65536;i++)if(omem[i]!=qd_add_u8_oracle(amem[i],bmem[i],&p))return 2;
    for(unsigned i=65536;i<sizeof omem;i++)if(omem[i]!=0xa5)return 3;

    /* Independent metadata, origins and slices, W/depth tails, missing
     * source/output blocks, and exact preservation of every other byte. */
    cip[1]=ci_mem;bip[1]=bi_mem;cop[1]=co_mem;
    QdBlockedU8 ca={.pools=cip,.blocks=cib,.block_count=5,.slice=0,
        .meta={1,8,16,64},.dims={1,3,7,33},.origin={0,3,3,17}};
    QdBlockedU8 cb={.pools=bip,.blocks=bib,.block_count=5,.slice=1,
        .meta={1,8,16,64},.dims={1,3,7,33},.origin={0,3,3,17}};
    QdBlockedU8 co={.pools=cop,.blocks=cob,.block_count=20,.slice=2,
        .meta={1,16,24,96},.dims={1,3,7,33},.origin={0,9,7,49}};
    memset(cib,0,sizeof cib);memset(bib,0,sizeof bib);
    cib[0]=bib[1]=1; /* Depth block1 and W-block1 are absent from both sources. */
    for(unsigned j=0;j<20;j++)cob[j]=1u+j*2048u;
    cob[16]=0; /* touched output table index for W-block1, D-block2 */
    memset(ci_mem,0,sizeof ci_mem);memset(bi_mem,0,sizeof bi_mem);memset(co_mem,0xa5,sizeof co_mem);
    for(unsigned i=0;i<2048;i++){ci_mem[i]=(uint8_t)(i*37u+9u);bi_mem[i]=(uint8_t)(i*19u+5u);}
    QdBlockedAddU8 cp;QdAddU8Params pc={32,2052,13};
    if(!qd_blocked_add_u8_prepare(&ca,&cb,&co,&cp)||!qd_blocked_add_u8_apply(&cp,&pc))return 4;
    uint8_t written[40960];memset(written,0,sizeof written);
    for(unsigned h=0;h<3;h++)for(unsigned w=0;w<7;w++)for(unsigned d=0;d<33;d++) {
        unsigned H=9+h,W=7+w,D=49+d,ix=((H/8)*3+W/8)*3+D/32+2;
        unsigned loc=(H%8)*256+(W%8)*32+D%32,off=ix*2048+loc;
        unsigned srcw=3+w,asrc=(3+h)%8*256+(srcw%8)*32+(17+d)%32;
        unsigned bsrc=asrc;
        uint8_t av=(srcw<8&&d<15)?ci_mem[asrc]:0,bv=(srcw<8&&d<15)?bi_mem[bsrc]:0;
        if(ix!=16){written[off]=1;if(co_mem[off]!=qd_add_u8_oracle(av,bv,&pc))return 5;}
    }
    for(unsigned i=0;i<sizeof co_mem;i++)if(!written[i]&&co_mem[i]!=0xa5)return 6;
    uint32_t repeated[32];for(unsigned i=0;i<32;i++)repeated[i]=1;
    QdBlockedU8 bad=full(opools,repeated);
    if(qd_blocked_add_u8_prepare(&a,&b,&bad,&prep))return 7;
    puts("PASS blocked u8 add: exhaustive byte pairs, crops, slices, holes, aliases, guards");
    return 0;
}
