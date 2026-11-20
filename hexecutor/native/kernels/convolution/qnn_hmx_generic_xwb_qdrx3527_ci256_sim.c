/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Actual CPU-prefix sequence 3527, CI256/CO128 Xwb 3x3 stride-1 HMX simulation. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
extern void qd_qnn_hmx_stride1_generic(uint32_t*,uint32_t*,void*,void*,uint32_t*,uint32_t*);
extern void qd_hmx_finish(void*,void*);
extern void qd_set_hmx_params_stride1(uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
#define FIXTURE "/src/native_kernels/xwb_ci256_qdrx3527_cpu_prefix_fixture.bin"
#define MAGIC 0x354b5748u
#define PAGE 2048u
#define VTCM_BYTES 0x100000u
#define PAYLOAD_CAP (1024u*1024u)
static uint8_t payload[PAYLOAD_CAP] __attribute__((aligned(128)));
static uint32_t in_table[256] __attribute__((aligned(128))),out_table[256] __attribute__((aligned(128)));
static uint64_t fnv64(const uint8_t*p,uint32_t n){uint64_t h=UINT64_C(0xcbf29ce484222325);for(uint32_t i=0;i<n;i++)h=(h^p[i])*UINT64_C(0x100000001b3);return h;}
static uint32_t up(uint32_t x,uint32_t a){return (x+a-1u)&~(a-1u);}
int main(void){
 FILE*f=fopen(FIXTURE,"rb");if(!f){puts("FAIL fixture open");return 1;}uint32_t h[32];
 if(fread(h,sizeof h,1,f)!=1||h[0]!=MAGIC||h[1]!=3527u){puts("FAIL header");fclose(f);return 2;}
 uint32_t ci=h[14],co=h[15],ib=h[7],yb=h[13],wb=h[17],fb=h[18];
 uint32_t icb=ci/32u,ocb=co/32u,active_y=((h[10]+7u)/8u)*((h[11]+7u)/8u)*ocb;
 uint32_t xb=ib*PAGE,ybytes=yb*PAGE;uint64_t total=(uint64_t)xb+ybytes+wb+fb;
 if(h[2]!=10||h[3]!=66||h[4]!=24||h[5]!=80||h[6]!=0||ib!=240||
    h[8]!=8||h[9]!=64||h[10]!=8||h[11]!=64||h[12]!=84||yb!=120||
    ci!=256||co!=128||h[16]!=2||wb!=294912||fb!=1024||active_y!=32||total>sizeof payload){
   puts("FAIL unsupported QDRX3527 shape");fclose(f);return 3;}
 if(fread(payload,(size_t)total,1,f)!=1||fgetc(f)!=EOF){puts("FAIL payload length");fclose(f);return 4;}fclose(f);
 const uint8_t*xdata=payload,*ydata=xdata+xb,*wdata=ydata+ybytes,*fidata=wdata+wb;
 uint8_t*ref=(uint8_t*)payload+xb+h[12]*PAGE; /* Normalize active Y slice into table slot zero. */
 uint32_t base=__rdcfg(__vtcm_base)<<16;
 for(uint32_t i=0;i<VTCM_BYTES;i+=4096u)add_translation((void*)(base+i),(void*)(base+i),6);
 uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");memset((void*)base,0xa5,VTCM_BYTES);
 uint32_t inbytes=xb,outbytes=active_y*PAGE,inp=0,in_guard=up(inbytes,4096u),outp=up(in_guard+0x2000u,4096u),out_guard=outp+outbytes,wp=up(out_guard+PAGE,4096u),fip=up(wp+wb,4096u),zp=up(fip+fb,4096u),donep=up(zp+256u,4096u);
 if(donep+4096u>VTCM_BYTES){puts("FAIL VTCM range");return 5;}
 uint8_t*input=(uint8_t*)(base+inp),*ig=(uint8_t*)(base+in_guard),*output=(uint8_t*)(base+outp),*og=(uint8_t*)(base+out_guard),*weights=(uint8_t*)(base+wp);
 int32_t*fi=(int32_t*)(base+fip),*zero=(int32_t*)(base+zp);volatile uint32_t*done=(volatile uint32_t*)(base+donep);uint8_t*tg=og+PAGE;
 memcpy(input,xdata,inbytes);memcpy(weights,wdata,wb);memcpy(fi,fidata,fb);memset(ig,0xa5,0x2000u);memset(output,0xa5,outbytes);memset(og,0xa5,PAGE);
 memset(zero,0,64u*sizeof(uint32_t));for(uint32_t c=0;c<32u;c++)zero[c]=0x404000;
 for(uint32_t i=0;i<256u;i++){in_table[i]=(uint32_t)ig;out_table[i]=(uint32_t)tg;}
 for(uint32_t i=0;i<ib;i++)in_table[i]=(uint32_t)(input+i*PAGE);
 for(uint32_t i=0;i<active_y;i++)out_table[i]=(uint32_t)(output+i*PAGE);
 for(uintptr_t p=(uintptr_t)input;p<(uintptr_t)input+inbytes;p+=32)__asm__ volatile("dccleaninva(%0)"::"r"(p):"memory");
 for(uintptr_t p=(uintptr_t)weights;p<(uintptr_t)weights+wb;p+=32)__asm__ volatile("dccleaninva(%0)"::"r"(p):"memory");
 for(uintptr_t p=(uintptr_t)fi;p<(uintptr_t)fi+fb;p+=32)__asm__ volatile("dccleaninva(%0)"::"r"(p):"memory");__asm__ volatile("barrier":::"memory");
 uint32_t row_in=((h[5]+7u)/8u)*icb,row_out=((h[11]+7u)/8u)*ocb;
 uint32_t input_desc[]={(uint32_t)in_table,icb,row_in,h[3]+h[20],h[2]+h[19],ci};
 uint32_t output_desc[]={(uint32_t)out_table,ocb,row_out,h[9]+h[22],h[8]+h[21],co};
 uint32_t cfg[16]={0},mode[]={1u,0u};qd_set_hmx_params_stride1(cfg,0x700u,3u,(h[4]+7u)/8u,ci,7u,1u,12u);
 printf("QDRX3527 CI256/CO128 X=%ux%u max=%ux%u pages=%u descW/H=%u/%u row=%u; Y=%ux%u pages=%u slice=%u row=%u cfg=",h[2],h[3],h[4],h[5],ib,input_desc[3],input_desc[4],row_in,h[8],h[9],active_y,h[12],row_out);
 for(uint32_t i=0;i<13;i++)printf("%s%x",i?",":"",cfg[i]);
 printf("\nstaged_fnv64 X=%016llx W=%016llx FI=%016llx Yref=%016llx\n",(unsigned long long)fnv64(input,inbytes),(unsigned long long)fnv64(weights,wb),(unsigned long long)fnv64((const uint8_t*)fi,fb),(unsigned long long)fnv64(ref,outbytes));
 *done=0xffffffffu;__asm__ volatile("mxclracc":::"memory");qd_qnn_hmx_stride1_generic(output_desc,input_desc,weights,fi,cfg,mode);qd_hmx_finish(zero,(void*)done);uint32_t spins=0;while(*done&&spins++<100001u)__asm__ volatile("nop");if(*done){puts("FAIL completion");return 6;}
 uint32_t bad=0,plus=0,minus=0,other=0;for(uint32_t i=0;i<outbytes;i++){int d=(int)output[i]-(int)ref[i];if(d){bad++;if(d==1)plus++;else if(d==-1)minus++;else other++;if(bad==1)printf("FIRST_DIFF byte=%u got=%u ref=%u delta=%d\n",i,output[i],ref[i],d);}}
 printf("CPU physical compare mismatches=%u/%u +1=%u -1=%u other=%u spins=%u\n",bad,outbytes,plus,minus,other,spins);
 for(uint32_t i=0;i<0x2000u;i++)if(ig[i]!=0xa5)return 7;for(uint32_t i=0;i<PAGE;i++)if(og[i]!=0xa5)return 8;
 for(uint32_t i=ib;i<256;i++)if(in_table[i]!=(uint32_t)ig){printf("FAIL input table tail %u\n",i);return 9;}for(uint32_t i=active_y;i<256;i++)if(out_table[i]!=(uint32_t)tg){printf("FAIL output table tail %u\n",i);return 10;}
 puts("RAW_OUTPUT_BEGIN");for(uint32_t i=0;i<outbytes;i++){printf("%02x",output[i]);if((i&255u)==255u)putchar('\n');}puts("RAW_OUTPUT_END");
 if(minus||other){puts("FAIL CPU comparison outside 0/+1");return 11;}puts("PASS QDRX3527 CI256/CO128 physical output CPU equal or +1; guards and table tails intact.");return 0;
}
