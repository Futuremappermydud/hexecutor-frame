/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "qd_native_1x1_adapter.h"
#define FIXTURE "/src/native_adapter/fixture-qdrx-2380.bin"
#define OUTPUT "/src/native_adapter/output-qdrx-2380.bin"
extern void qd_hmx_finish(void *,void *);
static uint8_t pool1[0x100000] __attribute__((aligned(128)));
static uint32_t tables[512] __attribute__((aligned(128)));
static uint32_t abi[30] __attribute__((aligned(128)));
static uint32_t pool_sizes[7]={0,sizeof pool1,0,0,0,0,0};
static uint8_t payload[0x100000] __attribute__((aligned(128)));
int main(void){
 FILE*f=fopen(FIXTURE,"rb");if(!f){puts("FAIL fixture open");return 1;}
 uint32_t h[32];if(fread(h,sizeof h,1,f)!=1||h[0]!=0x31585151u||!h[1]){puts("FAIL fixture header");return 2;}
 uint32_t ih=h[7],oh=h[13],ci=h[14],co=h[15],wb=h[17],fb=h[18];
 uint32_t input_bytes=ih*2048u,output_bytes=oh*2048u;
 uint64_t payload_bytes=(uint64_t)input_bytes+output_bytes+wb+fb;
 if(!ih||!oh||ih>256||oh>256||payload_bytes>sizeof payload||input_bytes+output_bytes>sizeof pool1||
    !ci||!co||ci>256||co>256||wb!=ci*co||fb!=co*8u||h[6]+ci/32u>ih||h[12]+co/32u>oh){puts("FAIL fixture sizes");fclose(f);return 3;}
 uint8_t*base=(uint8_t*)(__rdcfg(__vtcm_base)<<16);for(uint32_t i=0;i<0x80000u;i+=4096)add_translation(base+i,base+i,6);memset(base,0xa5,0x80000u);
 if(fread(payload,(size_t)payload_bytes,1,f)!=1||fgetc(f)!=EOF){puts("FAIL fixture payload length");fclose(f);return 4;}fclose(f);
 const uint8_t*xdata=payload;const uint8_t*ydata=xdata+input_bytes;const uint8_t*weights=ydata+output_bytes;const uint8_t*fi=weights+wb;
 memcpy(pool1,xdata,input_bytes);memcpy(pool1+input_bytes,ydata,output_bytes);
 /* Copy W/FI into aligned static operands; fixture addresses need not be aligned. */
 static uint8_t aligned_weights[16384] __attribute__((aligned(128)));
 static uint8_t aligned_fi[1024] __attribute__((aligned(128)));
 memcpy(aligned_weights,weights,wb);memcpy(aligned_fi,fi,fb);
 for(uintptr_t p=(uintptr_t)pool1;p<(uintptr_t)pool1+input_bytes+output_bytes;p+=32)__asm__ volatile("dccleaninva(%0)"::"r"(p):"memory");
 for(uintptr_t p=(uintptr_t)aligned_weights;p<(uintptr_t)aligned_weights+wb;p+=32)__asm__ volatile("dccleaninva(%0)"::"r"(p):"memory");
 for(uintptr_t p=(uintptr_t)aligned_fi;p<(uintptr_t)aligned_fi+fb;p+=32)__asm__ volatile("dccleaninva(%0)"::"r"(p):"memory");__asm__ volatile("barrier":::"memory");
 uint32_t xb[256],yb[256];for(uint32_t i=0;i<256;i++){xb[i]=1u;yb[i]=1u|(input_bytes&~127u);}for(uint32_t i=0;i<ih;i++)xb[i]=i*2048u|1u;for(uint32_t i=0;i<oh;i++)yb[i]=(input_bytes+i*2048u)|1u;
 QdN1x1Request r;memset(&r,0,sizeof r);r.pool_count=7;r.pool[1]=pool1;r.pool_bytes=pool_sizes;
 r.input=(QdN1x1Blocked){.dims={1,h[2],h[3],ci},.max_dims={1,h[4],h[5],ci},.padding={0,h[19],h[20],0},.slice_offset=h[6],.block_count=ih,.blocks=xb};
 r.output=(QdN1x1Blocked){.dims={1,h[8],h[9],co},.max_dims={1,h[10],h[11],co},.padding={0,h[21],h[22],0},.slice_offset=h[12],.block_count=oh,.blocks=yb};
 r.weights=aligned_weights;r.weights_bytes=wb;r.weight_ci=ci;r.weight_co=co;r.weight_kernel_h=1;r.weight_kernel_w=1;r.input_dtype=1;r.output_dtype=1;r.weight_dtype=2;r.fi_dtype=5;r.weight_format=h[16];r.fi=(const int32_t*)aligned_fi;r.fi_bytes=fb;
 r.workspace=base;r.workspace_bytes=QD_N1X1_WORKSPACE_MAX;r.table_storage=tables;r.table_storage_bytes=sizeof tables;r.abi_storage=abi;r.abi_storage_bytes=sizeof abi;
 QdN1x1Prepared p;int e=qd_native_1x1_prepare(&r,&p);if(e){printf("FAIL prepare %d\n",e);return 5;}
 uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");__asm__ volatile("mxclracc":::"memory");qd_qnn_hmx_1x1(p.output_desc,p.input_desc,p.weights,p.fi,p.cfg,p.mode);
 uint8_t*done=base+0x52000u;*(volatile uint32_t*)done=0xffffffffu;qd_hmx_finish(p.zero_words,done);uint32_t tries=0;while(*(volatile uint32_t*)done&&tries++<100000)__asm__ volatile("nop");if(*(volatile uint32_t*)done){puts("FAIL HMX drain");return 6;}
 e=qd_native_1x1_finish(&p);if(e){printf("FAIL finish %d\n",e);return 7;}
 FILE*out=fopen(OUTPUT,"wb");if(!out){puts("FAIL output open");return 8;}uint32_t hdr[32];memcpy(hdr,h,sizeof hdr);if(fwrite(hdr,sizeof hdr,1,out)!=1||fwrite(pool1+input_bytes,2048,oh,out)!=oh){puts("FAIL output write");fclose(out);return 8;}fclose(out);
 printf("PASS actual command %u adapter+leaf: %u/%u croutons; workspace %u B\n",h[1],p.input_blocks,p.output_blocks,p.workspace_used);return 0;
}
