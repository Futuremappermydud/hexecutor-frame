#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "linearclip_u8_u16.h"

static uint8_t src[8192] __attribute__((aligned(128)));
static uint8_t dst[32768] __attribute__((aligned(128)));
static uint32_t sb[8],db[64];
static uint8_t *sp[2]={0,src},*dp[2]={0,dst};

static uint8_t *src_block(int Dblock) {return src+(2048+Dblock*2048);}
static uint8_t read_src(const QdBlockedU8 *t,int b,int h,int w,int d) {
    int B=b+t->origin[0],H=h+t->origin[1],W=w+t->origin[2],D=d+t->origin[3];
    int nbh=(t->meta[1]+7)/8,nbw=(t->meta[2]+7)/8,nbd=(t->meta[3]+31)/32;
    int index=(((B*nbh+H/8)*nbw+W/8)*nbd)+t->slice+D/32;
    uint32_t e=t->blocks[index];if(!(e&127))return 0;
    return t->pools[e&127][(e&~127u)+(H%8)*256+(W%8)*32+D%32];
}
static uint16_t *read_dst(const QdBlockedU16Out *t,int b,int h,int w,int d) {
    int B=b+t->origin[0],H=h+t->origin[1],W=w+t->origin[2],D=d+t->origin[3];
    int nbh=(t->meta[1]+7)/8,nbw=(t->meta[2]+3)/4,nbd=(t->meta[3]+31)/32;
    int index=(((B*nbh+H/8)*nbw+W/4)*nbd)+t->slice+D/32;
    uint32_t e=t->blocks[index];
    return (uint16_t *)(t->pools[e&127]+(e&~127u)+(H%8)*256+((W%4)/2)*128+(W%2)*2+(D%32)*4);
}
int main(void) {
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    QdBlockedU8 in={sp,sb,8,1,{1,8,8,72},{1,5,5,47},{0,1,1,21}};
    QdBlockedU16Out out={dp,db,64,2,{2,16,12,104},{1,5,5,47},{1,5,5,53}};
    for(int i=0;i<8;i++)sb[i]=0;for(int i=0;i<64;i++)db[i]=0;
    for(int i=0;i<3;i++)sb[1+i]=(2048+i*2048)|1;
    memset(src,0x73,sizeof src);memset(dst,0xa5,sizeof dst);
    uint32_t seed=0x921ec47u;
    for(int h=0;h<8;h++)for(int w=0;w<8;w++)for(int dbi=0;dbi<3;dbi++)
      for(int d=0;d<32;d++) {seed=seed*1664525u+1013904223u;src_block(dbi)[h*256+w*32+d]=(uint8_t)(seed>>24);}
    int blockno=0;
    for(int hg=0;hg<2;hg++)for(int wg=1;wg<=2;wg++)for(int dep=1;dep<=3;dep++) {
        int idx=(((1*2+hg)*3+wg)*4)+out.slice+dep;
        db[idx]=(2048+(11-blockno++)*2048)|1;
    }
    QdBlockedLinearclipU8U16 prep;
    if(!qd_blocked_linearclip_u8_u16_prepare(&in,&out,&prep))return 1;
    QdLinearclipU8U16Params p={30000,13};
    if(!qd_blocked_linearclip_u8_u16_apply(&prep,&p))return 2;
    for(int h=0;h<5;h++)for(int w=0;w<5;w++)for(int d=0;d<47;d++) {
        uint32_t want=qd_linearclip_u8_u16_oracle(read_src(&in,0,h,w,d),&p);
        if(*read_dst(&out,0,h,w,d)!=want){printf("oracle mismatch h%d w%d d%d got%u want%u\n",h,w,d,(unsigned)*read_dst(&out,0,h,w,d),(unsigned)want);return 3;}
    }
    uint8_t touched[sizeof dst]={0};
    for(int h=0;h<5;h++)for(int w=0;w<5;w++)for(int d=0;d<47;d++) {
        size_t off=(size_t)((uint8_t *)read_dst(&out,0,h,w,d)-dst);touched[off]=touched[off+1]=1;
    }
    for(size_t i=0;i<sizeof dst;i++)if(!touched[i]&&dst[i]!=0xa5){printf("guard changed %lu\n",(unsigned long)i);return 4;}
    /* Include saturation and a different runtime parameter pair. */
    memset(dst,0xa5,sizeof dst);p.multiplier=65535;p.shift=1;
    if(!qd_blocked_linearclip_u8_u16_apply(&prep,&p))return 5;
    for(int h=0;h<5;h++)for(int w=0;w<5;w++)for(int d=0;d<47;d++)
        if(*read_dst(&out,0,h,w,d)!=qd_linearclip_u8_u16_oracle(read_src(&in,0,h,w,d),&p))return 6;
    uint32_t aliased[64];memcpy(aliased,db,sizeof db);aliased[32]=aliased[31];out.blocks=aliased;
    if(qd_blocked_linearclip_u8_u16_prepare(&in,&out,&prep)){puts("accepted repeated output alias");return 7;}
    puts("PASS u8-to-u16 blocked linearclip: independent oracle, parity/tails, shifted metadata/slices, guards, saturation, alias reject");
    SIM_RELEASE_HVX;return 0;
}
