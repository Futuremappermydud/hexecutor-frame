/* Actual QDRX5019 CI32/CO2 tail proof with pre/post physical padded Y oracle. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
extern void qd_qnn_hmx_stride1_generic(uint32_t*,uint32_t*,void*,void*,uint32_t*,uint32_t*);
extern void qd_hmx_finish(void*,void*);
extern void qd_set_hmx_params_stride1(uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
#define FIXTURE "/src/native_kernels/xwb_ci32_co2_qdrx5019_cpu_prefix_fixture.bin"
#define MAGIC 0x394f5748u
#define PAGE 2048u
#define ACTIVE_PAGES 32u
#define ACTIVE_BYTES (ACTIVE_PAGES*PAGE)
#define WEIGHT_PAD_BYTES 9216u
#define PAYLOAD_CAP (6u*1024u*1024u)
static uint8_t payload[PAYLOAD_CAP] __attribute__((aligned(128)));
static uint32_t in_table[256] __attribute__((aligned(128))),out_table[256] __attribute__((aligned(128)));
static uint64_t fnv64(const uint8_t*p,uint32_t n){uint64_t h=UINT64_C(0xcbf29ce484222325);for(uint32_t i=0;i<n;i++)h=(h^p[i])*UINT64_C(0x100000001b3);return h;}
int main(void){
 FILE*f=fopen(FIXTURE,"rb");if(!f){puts("FAIL fixture open");return 1;}uint32_t h[32];
 if(fread(h,sizeof h,1,f)!=1||h[0]!=MAGIC||h[1]!=5019u){puts("FAIL header");fclose(f);return 2;}
 uint32_t xb=h[7]*PAGE,yb=h[13]*PAGE,wb=h[17],fb=h[18];uint64_t total=(uint64_t)xb+2ull*yb+wb+fb;
 if(h[2]!=10||h[3]!=258||h[4]!=24||h[5]!=272||h[6]!=0||h[7]!=102||
    h[8]!=8||h[9]!=256||h[10]!=8||h[11]!=256||h[12]!=0||h[13]!=1280||
    h[14]!=32||h[15]!=32||h[24]!=2||h[16]!=2||wb!=9216||fb!=256||total>sizeof payload){puts("FAIL unsupported QDRX5019 shape");fclose(f);return 3;}
 if(fread(payload,(size_t)total,1,f)!=1||fgetc(f)!=EOF){puts("FAIL fixture length");fclose(f);return 4;}fclose(f);
 const uint8_t*xdata=payload,*ypre=xdata+xb,*ypost=ypre+yb,*wdata=ypost+yb,*fidata=wdata+wb;
 const uint8_t*pre=ypre+h[12]*PAGE,*ref=ypost+h[12]*PAGE;
 uint32_t base=__rdcfg(__vtcm_base)<<16;for(uint32_t i=0;i<0x80000u;i+=4096u)add_translation((void*)(base+i),(void*)(base+i),6);
 uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");memset((void*)base,0xa5,0x80000u);
 uint8_t*in=(uint8_t*)base,*ig=(uint8_t*)(base+0x33000u),*out=(uint8_t*)(base+0x36000u),*og=(uint8_t*)(base+0x46000u),*w=(uint8_t*)(base+0x48000u),*wg=(uint8_t*)(base+0x4a400u);
 int32_t*fi=(int32_t*)(base+0x4b000u),*zero=(int32_t*)(base+0x4c000u);volatile uint32_t*done=(volatile uint32_t*)(base+0x4d000u);uint8_t*tg=og+PAGE;
 memcpy(in,xdata,xb);memcpy(out,pre,ACTIVE_BYTES);memset(w,0,WEIGHT_PAD_BYTES);memcpy(w,wdata,wb);memcpy(fi,fidata,fb);memset(ig,0xa5,0x2000u);memset(og,0xa5,PAGE);memset(wg,0xa5,0xc00u);
 memset(zero,0,64u*sizeof(uint32_t));for(uint32_t c=0;c<32;c++)zero[c]=0x404000;
 for(uint32_t i=0;i<256;i++){in_table[i]=(uint32_t)ig;out_table[i]=(uint32_t)tg;}for(uint32_t i=0;i<102;i++)in_table[i]=(uint32_t)(in+i*PAGE);for(uint32_t i=0;i<ACTIVE_PAGES;i++)out_table[i]=(uint32_t)(out+i*PAGE);
 for(uintptr_t p=(uintptr_t)in;p<(uintptr_t)in+xb;p+=32)__asm__ volatile("dccleaninva(%0)"::"r"(p):"memory");for(uintptr_t p=(uintptr_t)w;p<(uintptr_t)w+WEIGHT_PAD_BYTES;p+=32)__asm__ volatile("dccleaninva(%0)"::"r"(p):"memory");for(uintptr_t p=(uintptr_t)fi;p<(uintptr_t)fi+fb;p+=32)__asm__ volatile("dccleaninva(%0)"::"r"(p):"memory");__asm__ volatile("barrier":::"memory");
 uint32_t id[]={(uint32_t)in_table,1u,34u,265u,17u,32u},od[]={(uint32_t)out_table,1u,32u,256u,8u,32u},cfg[16]={0},mode[]={1u,0u};qd_set_hmx_params_stride1(cfg,0x700u,3u,3u,32u,7u,1u,12u);
 printf("QDRX5019 X=10x258 max=24x272 pages=102 ->Y=8x256 logical-C2/max-C32 pages=%u W=%u FI=%u cfg=",ACTIVE_PAGES,wb,fb);for(uint32_t i=0;i<13;i++)printf("%s%x",i?",":"",cfg[i]);
 printf("\nstaged_fnv64 X=%016llx W=%016llx FI=%016llx Ypre=%016llx Ypost=%016llx\n",(unsigned long long)fnv64(in,xb),(unsigned long long)fnv64(w,wb),(unsigned long long)fnv64((const uint8_t*)fi,fb),(unsigned long long)fnv64(pre,ACTIVE_BYTES),(unsigned long long)fnv64(ref,ACTIVE_BYTES));
 *done=0xffffffffu;__asm__ volatile("mxclracc":::"memory");qd_qnn_hmx_stride1_generic(od,id,w,fi,cfg,mode);qd_hmx_finish(zero,(void*)done);uint32_t spins=0;while(*done&&spins++<100001u)__asm__ volatile("nop");if(*done){puts("FAIL completion");return 5;}
 uint32_t logical_bad=0,logical_plus=0,logical_minus=0,logical_other=0,padded_changed=0,cpu_padded_changed=0;
 for(uint32_t i=0;i<ACTIVE_BYTES;i++){uint32_t c=i&31u;if(c<2u){int d=(int)out[i]-(int)ref[i];if(d){logical_bad++;if(d==1)logical_plus++;else if(d==-1)logical_minus++;else logical_other++;}}else{if(out[i]!=ref[i])padded_changed++;if(pre[i]!=ref[i])cpu_padded_changed++;}}
 printf("logical_CO2 compare mismatches=%u +1=%u -1=%u other=%u nonlogical_lanes_different_HMX_vs_CPU=%u CPU_pre_vs_post=%u spins=%u\n",logical_bad,logical_plus,logical_minus,logical_other,padded_changed,cpu_padded_changed,spins);
 for(uint32_t i=0;i<0x2000;i++)if(ig[i]!=0xa5)return 6;for(uint32_t i=0;i<PAGE;i++)if(og[i]!=0xa5)return 7;for(uint32_t i=0;i<0xc00;i++)if(wg[i]!=0xa5)return 8;
 for(uint32_t i=102;i<256;i++)if(in_table[i]!=(uint32_t)ig)return 9;for(uint32_t i=ACTIVE_PAGES;i<256;i++)if(out_table[i]!=(uint32_t)tg)return 10;
 puts("RAW_OUTPUT_BEGIN");for(uint32_t i=0;i<ACTIVE_BYTES;i++){printf("%02x",out[i]);if((i&255u)==255u)putchar('\n');}puts("RAW_OUTPUT_END");
 if(logical_minus||logical_other||padded_changed){puts("FAIL logical or physical output mismatch");return 11;}puts("PASS QDRX5019 logical CO2 plus physical maxC32 output exact/+1; pre/post nonlogical lane changes are accounted for.");return 0;
}
