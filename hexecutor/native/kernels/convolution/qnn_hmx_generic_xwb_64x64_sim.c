/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Generic V75 spatial leaf CI=CO=64, two depth pages per crouton. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
extern void qd_qnn_hmx_stride1_generic(uint32_t*,uint32_t*,void*,void*,uint32_t*,uint32_t*);
extern void qd_hmx_finish(void*,void*);
extern void qd_set_hmx_params_stride1(uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
#define PAGE 2048u
#define IH 10u
#define IW 130u
#define OH 8u
#define OW 128u
#define CI 64u
#define CO 64u
#define WB 18u
#define DB 2u
#define ROWSTRIDE (WB*DB)
#define IB (3u*ROWSTRIDE)
#define OB (16u*DB)
#define INBYTES (IB*PAGE)
#define OUTBYTES (OB*PAGE)
static uint32_t woff(uint32_t y,uint32_t x,uint32_t ci,uint32_t co){return ((((co/32u*(CI/32u)+ci/32u)*3u+y)*3u+x)*1024u)+((ci/4u)%8u)*128u+(co%32u)*4u+ci%4u;}
int main(void){
 uint32_t base=__rdcfg(__vtcm_base)<<16;for(uint32_t i=0;i<0x80000u;i+=4096u)add_translation((void*)(base+i),(void*)(base+i),6);uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");memset((void*)base,0,0x80000u);
 uint8_t*in=(uint8_t*)base,*ig=(uint8_t*)(base+INBYTES),*out=(uint8_t*)(base+0x3a000u),*og=(uint8_t*)(base+0x4a000u),*w=(uint8_t*)(base+0x4c000u);int32_t*fi=(int32_t*)(base+0x56000u),*z=(int32_t*)(base+0x57000u);uint32_t*it=(uint32_t*)(base+0x58000u),*ot=(uint32_t*)(base+0x58400u);volatile uint32_t*done=(volatile uint32_t*)(base+0x58800u);uint8_t*tg=og+PAGE;
 if(INBYTES!=221184u||OUTBYTES!=65536u||3u*3u*CI*CO!=36864u){puts("FAIL static byte geometry");return 10;}
 memset(out,0xa5,OUTBYTES);memset(ig,0xa5,0x4000u);memset(og,0xa5,PAGE);for(uint32_t i=0;i<256;i++){it[i]=(uint32_t)ig;ot[i]=(uint32_t)tg;}for(uint32_t b=0;b<IB;b++)it[b]=(uint32_t)(in+b*PAGE);for(uint32_t b=0;b<OB;b++)ot[b]=(uint32_t)(out+b*PAGE);
 for(uint32_t y=0;y<IH;y++)for(uint32_t x=0;x<IW;x++)for(uint32_t c=0;c<CI;c++){uint32_t py=y+7u,px=x+7u,b=((py/8u)*WB+px/8u)*DB+c/32u,off=(py%8u)*256u+(px%8u)*32u+c%32u;in[b*PAGE+off]=(uint8_t)(1u+((y*29u+x*13u+c*7u)%250u));}
 memset(w,0,3u*3u*CI*CO);for(uint32_t c=0;c<CI;c++)w[woff(1,1,c,c)]=1;for(uint32_t c=0;c<CO;c++){fi[c]=0x406000;fi[CO+c]=0;}z[0]=0x404000;
 uint32_t id[]={(uint32_t)it,1u,ROWSTRIDE,IW+7u,IH+7u,CI},od[]={(uint32_t)ot,1u,OW/8u*DB,OW,OH,CO},cfg[16]={0},mode[]={1u,0u};qd_set_hmx_params_stride1(cfg,0x700u,3u,3u,CI,7u,1u,12u);
 printf("CI=CO=64 input=%ux%u max=144x24 pad=7,7 blocks=18x3x2 rowstride=%u; output=%ux%u blocks=16x1x2; weights=36KiB cfg=",IW,IH,ROWSTRIDE,OW,OH);for(uint32_t i=0;i<13;i++)printf("%s%x",i?",":"",cfg[i]);puts("");
 *done=0xffffffffu;__asm__ volatile("mxclracc":::"memory");qd_qnn_hmx_stride1_generic(od,id,w,fi,cfg,mode);qd_hmx_finish(z,(void*)done);uint32_t spin=0;while(*done&&spin++<100001u)__asm__ volatile("nop");if(*done){puts("FAIL completion");return 1;}
 uint32_t bad=0;for(uint32_t y=0;y<OH;y++)for(uint32_t x=0;x<OW;x++)for(uint32_t c=0;c<CO;c++){uint32_t b=((y/8u)*(OW/8u)+x/8u)*DB+c/32u,off=(y%8u)*256u+(x%8u)*32u+c%32u,got=out[b*PAGE+off];uint32_t iy=y+1u+7u,ix=x+1u+7u,ib=((iy/8u)*WB+ix/8u)*DB+c/32u,io=(iy%8u)*256u+(ix%8u)*32u+c%32u,want=in[ib*PAGE+io];if(got!=want){if(bad<8)printf("MISMATCH y=%u x=%u c=%u got=%u want=%u\n",y,x,c,got,want);bad++;}}
 printf("ci64_co64_identity mismatches=%u/%u completion_spins=%u\n",bad,OH*OW*CO,spin);for(uint32_t i=0;i<0x4000u;i++)if(ig[i]!=0xa5u){printf("FAIL input guard=%u\n",i);return 2;}for(uint32_t i=0;i<PAGE;i++)if(og[i]!=0xa5u){printf("FAIL output guard=%u\n",i);return 3;}for(uint32_t i=IB;i<256;i++)if(it[i]!=(uint32_t)ig)return 4;for(uint32_t i=OB;i<256;i++)if(ot[i]!=(uint32_t)tg)return 5;if(bad){puts("FAIL CI64/CO64 output oracle");return 6;}puts("PASS CI64/CO64 identity, full output, completion and guards/tails.");return 0;
}
