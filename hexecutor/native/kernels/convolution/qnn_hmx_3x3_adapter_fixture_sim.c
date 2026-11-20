/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Device-shaped staging/leaf/drain/scatter path for actual CPU-prefix fixtures. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "qd_native_3x3_adapter.h"
#ifndef QD_FIXTURE_PATH
#define QD_FIXTURE_PATH "/src/native_kernels/xwb_ci64_qdrx247_cpu_prefix_fixture.bin"
#endif
#define PAGE 2048u
#define FIXTURE_CAP (6u*1024u*1024u)
#define POOL_CAP (4u*1024u*1024u)
static uint8_t fixture[FIXTURE_CAP] __attribute__((aligned(128)));
static uint8_t pool1[POOL_CAP] __attribute__((aligned(128)));
static uint32_t xblocks[256],yblocks[2048],pool_sizes[7]={0,POOL_CAP,0,0,0,0,0};
static uint64_t fnv64(const uint8_t*p,uint32_t n){uint64_t h=UINT64_C(0xcbf29ce484222325);for(uint32_t i=0;i<n;i++)h=(h^p[i])*UINT64_C(0x100000001b3);return h;}
int main(void){
 FILE*f=fopen(QD_FIXTURE_PATH,"rb");if(!f){puts("FAIL fixture open");return 1;}uint32_t h[32];
 if(fread(h,sizeof h,1,f)!=1||h[1]==0){puts("FAIL fixture header");fclose(f);return 2;}
 uint32_t xb=h[7]*PAGE,yb=h[13]*PAGE,wb=h[17],fb=h[18],tail_case=(h[1]==5019u);
 uint64_t total=(uint64_t)xb+(tail_case?2ull*yb:yb)+wb+fb;
 if(total>sizeof fixture||fread(fixture,(size_t)total,1,f)!=1||fgetc(f)!=EOF){puts("FAIL fixture payload");fclose(f);return 3;}fclose(f);
 const uint8_t*x=fixture,*y_pre=x+xb,*y=tail_case?y_pre+yb:y_pre,*w=y+yb,*fi=w+wb;
 uint32_t inpages=((h[4]+7u)/8u)*((h[5]+7u)/8u)*(h[14]/32u);
 uint32_t outpages=((h[10]+7u)/8u)*((h[11]+7u)/8u)*(h[15]/32u);
 if(xb+yb>sizeof pool1||h[7]>256||h[6]+inpages>h[7]||h[12]+outpages>h[13]){puts("FAIL fixture capacity");return 4;}
 memcpy(pool1,x,xb);memcpy(pool1+xb,y_pre,yb);
 for(uint32_t i=0;i<h[7];i++)xblocks[i]=i*PAGE|1u;
 if(h[13]>2048u){puts("FAIL oversized QDRX Y table");return 4;}
 for(uint32_t i=0;i<h[13];i++)yblocks[i]=(xb+i*PAGE)|1u;
 QdN3x3Request r;memset(&r,0,sizeof r);r.pool_count=7;r.pool[1]=pool1;r.pool_bytes=pool_sizes;
 r.input=(QdN3x3Blocked){.dims={1,h[2],h[3],h[14]},.max_dims={1,h[4],h[5],h[14]},.padding={0,h[19],h[20],0},.slice_offset=h[6],.block_count=h[7],.blocks=xblocks};
 r.output=(QdN3x3Blocked){.dims={1,h[8],h[9],h[24]?h[24]:h[15]},.max_dims={1,h[10],h[11],h[15]},.padding={0,h[21],h[22],0},.slice_offset=h[12],.block_count=h[13],.blocks=yblocks};
 r.weights=w;r.weights_bytes=wb;r.weight_ci=h[14];r.weight_co=h[15];r.weight_kernel_h=3;r.weight_kernel_w=3;
 r.input_dtype=QD_N3X3_DTYPE_U8;r.output_dtype=QD_N3X3_DTYPE_U8;r.weight_dtype=QD_N3X3_DTYPE_I8;r.fi_dtype=QD_N3X3_DTYPE_I32;r.weight_format=h[16];
 r.fi=(const int32_t*)fi;r.fi_bytes=fb;r.input_zero_point=h[23];r.output_zero_point=h[25];
 uint32_t vtcm=__rdcfg(__vtcm_base)<<16;for(uint32_t i=0;i<0x101000u;i+=4096u)add_translation((void*)(vtcm+i),(void*)(vtcm+i),6);
 uint8_t*workspace=(uint8_t*)vtcm,*high=(uint8_t*)(vtcm+QD_N3X3_WORKSPACE_MAX);
 memset(workspace,0x3c,QD_N3X3_WORKSPACE_MAX);memset(high,0x5a,4096u);
 r.workspace=workspace;r.workspace_bytes=QD_N3X3_WORKSPACE_MAX;
 QdN3x3Prepared p;int e=qd_native_3x3_prepare(&r,&p);if(e){printf("FAIL prepare=%d\n",e);return 5;}
 printf("adapter sequence=%u CI=%u CO=%u Xpages=%u Ypages=%u slice=%u workspace_used=%u cfg=",h[1],h[14],h[15],p.input_pages,p.output_pages,h[12],p.workspace_used);
 for(uint32_t i=0;i<13;i++)printf("%s%x",i?",":"",p.cfg[i]);
 printf("\nstaged_fnv64 X=%016llx W=%016llx FI=%016llx\n",(unsigned long long)fnv64(p.input_data,p.input_pages*PAGE),(unsigned long long)fnv64(p.weights,wb),(unsigned long long)fnv64((const uint8_t*)p.fi,fb));
 if((e=qd_native_3x3_execute(&p))){printf("FAIL execute=%d\n",e);return 6;}
 if((e=qd_native_3x3_finish(&p))){printf("FAIL adapter_finish=%d\n",e);return 7;}
 uint32_t pages=((h[10]+7u)/8u)*((h[11]+7u)/8u)*(h[15]/32u),mismatch=0,plus=0,minus=0,other=0;
 const uint8_t*expected=y+h[12]*PAGE;uint32_t active_bytes=pages*PAGE;
 for(uint32_t i=0;i<active_bytes;i++){int d=(int)pool1[xb+h[12]*PAGE+i]-(int)expected[i];if(d){mismatch++;if(d==1)plus++;else if(d==-1)minus++;else other++;}}
 printf("CPU active physical compare mismatches=%u/%u +1=%u -1=%u other=%u\n",mismatch,active_bytes,plus,minus,other);
 for(uint32_t i=0;i<4096u;i++)if(high[i]!=0x5a){puts("FAIL workspace boundary canary");return 8;}
 if(minus||other){puts("FAIL output not CPU-equal/+1");return 9;}
 puts("RAW_OUTPUT_BEGIN");for(uint32_t i=0;i<active_bytes;i++){printf("%02x",pool1[xb+h[12]*PAGE+i]);if((i&255u)==255u)putchar('\n');}puts("RAW_OUTPUT_END");
 puts("PASS bounded generic 3x3 adapter prepare/execute/drain/guard/scatter");return 0;
}
