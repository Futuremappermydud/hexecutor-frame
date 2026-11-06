/* Probe distinct CI/CO Xwb lanes, both weight signs, each spatial tap. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
extern void qd_qnn_hmx_stride1_generic(uint32_t*,uint32_t*,void*,void*,uint32_t*,uint32_t*);
extern void qd_hmx_finish(void*,void*);
extern void qd_set_hmx_params_stride1(uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
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
static uint32_t woff(uint32_t y,uint32_t x,uint32_t ci,uint32_t co){return ((((co/32u*(CI/32u)+ci/32u)*3u+y)*3u+x)*1024u)+((ci/4u)%8u)*128u+(co%32u)*4u+ci%4u;}
int main(void){
 uint32_t base=__rdcfg(__vtcm_base)<<16;for(uint32_t i=0;i<0x80000u;i+=4096u)add_translation((void*)(base+i),(void*)(base+i),6);uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");memset((void*)base,0,0x80000u);
 uint8_t *in=(uint8_t*)base,*ig=(uint8_t*)(base+0x33000u),*out=(uint8_t*)(base+0x38000u),*og=(uint8_t*)(base+0x48000u),*w=(uint8_t*)(base+0x4a000u);int32_t *fi=(int32_t*)(base+0x4d000u),*z=(int32_t*)(base+0x4e000u);uint32_t *it=(uint32_t*)(base+0x4f000u),*ot=(uint32_t*)(base+0x4f400u);volatile uint32_t *done=(volatile uint32_t*)(base+0x4f800u);uint8_t*tg=og+PAGE;
 memset(out,0xa5,OB*PAGE);memset(og,0xa5,PAGE);memset(ig,0xa5,0x5000u);for(uint32_t i=0;i<256;i++){it[i]=(uint32_t)ig;ot[i]=(uint32_t)tg;}for(uint32_t b=0;b<IB;b++)it[b]=(uint32_t)(in+b*PAGE);for(uint32_t b=0;b<OB;b++)ot[b]=(uint32_t)(out+b*PAGE);
 for(uint32_t y=0;y<IH;y++)for(uint32_t x=0;x<IW;x++)for(uint32_t c=0;c<CI;c++){uint32_t py=y+7u,px=x+7u,b=(py/8u)*WB+px/8u,off=(py%8u)*256u+(px%8u)*CI+c;in[b*PAGE+off]=(uint8_t)(1u+((y*29u+x*13u+c*7u)%250u));}
 uint32_t id[]={(uint32_t)it,1u,WB,IW+7u,IH+7u,CI},od[]={(uint32_t)ot,1u,OW/8u,OW,OH,CO},cfg[16]={0},mode[]={1u,0u};qd_set_hmx_params_stride1(cfg,0x700u,3u,3u,CI,7u,1u,12u);for(uint32_t c=0;c<CO;c++){fi[c]=0x406000;fi[32u+c]=0;}z[0]=0x404000;
 const uint32_t ins[]={0,1,7,19};const uint32_t outs[]={1,0,19,7};uint32_t totalbad=0,runs=0;
 for(uint32_t sign=0;sign<2;sign++)for(uint32_t p=0;p<4;p++)for(uint32_t ky=0;ky<3;ky++)for(uint32_t kx=0;kx<3;kx++){
   uint32_t ci=ins[p],co=outs[p];memset(w,0,9u*1024u);w[woff(ky,kx,ci,co)]=(uint8_t)(sign?0xff:1);for(uint32_t c=0;c<CO;c++)fi[32u+c]=0;fi[32u+co]=sign?256:0;memset(out,0xa5,OB*PAGE);*done=0xffffffffu;__asm__ volatile("mxclracc":::"memory");qd_qnn_hmx_stride1_generic(od,id,w,fi,cfg,mode);qd_hmx_finish(z,(void*)done);uint32_t spin=0;while(*done&&spin++<100001u)__asm__ volatile("nop");if(*done){printf("FAIL timeout case=%u\n",runs);return 1;}
   uint32_t bad=0;for(uint32_t y=0;y<OH;y++)for(uint32_t x=0;x<OW;x++)for(uint32_t c=0;c<CO;c++){uint32_t b=(y/8u)*(OW/8u)+x/8u,off=(y%8u)*256u+(x%8u)*CO+c,got=out[b*PAGE+off];uint32_t want=0;if(c==co){uint32_t sx=x+2u-kx,sy=y+ky,iv=1u+((sy*29u+sx*13u+ci*7u)%250u);want=sign?256u-iv:iv;}if(got!=want){if(bad<1)printf("MISMATCH sign=%u ci=%u co=%u tap=%u,%u y=%u x=%u got=%u want=%u\n",sign,ci,co,ky,kx,y,x,got,want);bad++;}}
   totalbad+=bad;runs++;if(bad)printf("case_done sign=%u ci=%u co=%u tap=%u,%u bad=%u\n",sign,ci,co,ky,kx,bad);
 }
 printf("offdiag_signed_probe runs=%u mismatches=%u/%u\n",runs,totalbad,runs*OH*OW*CO);for(uint32_t i=0;i<0x5000;i++)if(ig[i]!=0xa5)return 2;for(uint32_t i=0;i<PAGE;i++)if(og[i]!=0xa5)return 3;for(uint32_t i=IB;i<256;i++)if(it[i]!=(uint32_t)ig)return 4;for(uint32_t i=OB;i<256;i++)if(ot[i]!=(uint32_t)tg)return 5;if(totalbad)return 6;puts("PASS distinct input/output lanes, signed weights, all 9 spatial taps, full output and guards.");return 0;
}
