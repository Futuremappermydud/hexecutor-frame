/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Generic spatial leaf 32x32 channel-index proof for Xwb. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
extern void qd_qnn_hmx_stride1_generic(uint32_t *, uint32_t *, void *, void *, uint32_t *, uint32_t *);
extern void qd_hmx_finish(void *, void *);
extern void qd_set_hmx_params_stride1(uint32_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
#define PAGE 2048u
#define IH 10u
#define IW 258u
#define OH 8u
#define OW 256u
#define CI 32u
#define CO 32u
#define WB 34u
#define IB (3u*34u)
#define OB 32u
static uint32_t offw(uint32_t ky,uint32_t kx,uint32_t ci,uint32_t co) {
    return ((((co/32u*(CI/32u)+ci/32u)*3u+ky)*3u+kx)*1024u)+((ci/4u)%8u)*128u+(co%32u)*4u+ci%4u;
}
int main(void) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x80000u;i+=4096u)add_translation((void*)(base+i),(void*)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    memset((void*)base,0,0x80000u);
    uint8_t *in=(uint8_t*)base,*ig=(uint8_t*)(base+0x33000u),*out=(uint8_t*)(base+0x38000u),*og=(uint8_t*)(base+0x48000u),*w=(uint8_t*)(base+0x4a000u);
    int32_t *fi=(int32_t*)(base+0x4d000u),*z=(int32_t*)(base+0x4e000u);
    uint32_t *it=(uint32_t*)(base+0x4f000u),*ot=(uint32_t*)(base+0x4f400u);volatile uint32_t *done=(volatile uint32_t*)(base+0x4f800u);uint8_t *tg=og+PAGE;
    memset(out,0xa5,OB*PAGE);memset(og,0xa5,PAGE);memset(ig,0xa5,0x5000u);
    for(uint32_t i=0;i<256u;i++){it[i]=(uint32_t)ig;ot[i]=(uint32_t)tg;}
    for(uint32_t b=0;b<IB;b++)it[b]=(uint32_t)(in+b*PAGE);
    for(uint32_t b=0;b<OB;b++)ot[b]=(uint32_t)(out+b*PAGE);
    for(uint32_t y=0;y<IH;y++)for(uint32_t x=0;x<IW;x++)for(uint32_t c=0;c<CI;c++){
      uint32_t py=y+7u,px=x+7u,b=(py/8u)*WB+px/8u,of=(py%8u)*256u+(px%8u)*CI+c;
      in[b*PAGE+of]=(uint8_t)(1u+((y*29u+x*13u+c*7u)%250u));
    }
    memset(w,0,9u*1024u);for(uint32_t c=0;c<32;c++)w[offw(1,1,c,c)]=1;
    for(uint32_t c=0;c<32;c++){fi[c]=0x406000;fi[32+c]=0;}z[0]=0x404000;
    uint32_t id[]={ (uint32_t)it,1u,WB,IW+7u,IH+7u,CI },od[]={ (uint32_t)ot,1u,OW/8u,OW,OH,CO },cfg[16]={0},mode[]={1u,0u};
    qd_set_hmx_params_stride1(cfg,0x700u,3u,3u,CI,7u,1u,12u);
    *done=0xffffffffu;__asm__ volatile("mxclracc":::"memory");qd_qnn_hmx_stride1_generic(od,id,w,fi,cfg,mode);qd_hmx_finish(z,(void*)done);
    uint32_t spins=0;while(*done&&spins++<100001u)__asm__ volatile("nop");if(*done){puts("FAIL completion");return 1;}
    uint32_t bad=0;for(uint32_t y=0;y<OH;y++)for(uint32_t x=0;x<OW;x++)for(uint32_t c=0;c<CO;c++){
      uint32_t b=(y/8u)*(OW/8u)+x/8u,of=(y%8u)*256u+(x%8u)*CO+c,got=out[b*PAGE+of];
      uint32_t want=1u+(((y+1u)*29u+(x+1u)*13u+c*7u)%250u);
      if(got!=want){if(bad<12)printf("MISMATCH y=%u x=%u c=%u got=%u want=%u\n",y,x,c,got,want);bad++;}
    }
    printf("diagonal_channelmap mismatches=%u/%u completion_spins=%u\n",bad,OH*OW*CO,spins);
    for(uint32_t i=0;i<0x5000u;i++)if(ig[i]!=0xa5){printf("FAIL input guard %u\n",i);return 2;}
    for(uint32_t i=0;i<PAGE;i++)if(og[i]!=0xa5){printf("FAIL output guard %u\n",i);return 3;}
    for(uint32_t i=IB;i<256;i++)if(it[i]!=(uint32_t)ig){printf("FAIL input tail %u\n",i);return 4;}
    for(uint32_t i=OB;i<256;i++)if(ot[i]!=(uint32_t)tg){printf("FAIL output tail %u\n",i);return 5;}
    if(bad){puts("FAIL channel mapping");return 6;}puts("PASS 32x32 diagonal channel map, full output, and guards.");return 0;
}
