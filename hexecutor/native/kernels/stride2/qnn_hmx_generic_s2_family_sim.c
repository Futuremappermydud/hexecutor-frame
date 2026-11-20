/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>

#define PAGE 2048u
#define VTCM_BYTES 0x100000u
#ifndef FIXTURE_BYTES
#define FIXTURE_BYTES 700000u
#endif
#ifndef SEQUENCE
#define SEQUENCE 224u
#endif
#ifndef CI
#define CI 128u /* lowered input depth (4 * source weight CI) */
#endif
#ifndef CO
#define CO 64u
#endif
#ifndef IN_H
#define IN_H 9u
#endif
#ifndef IN_W
#define IN_W 130u
#endif
#ifndef IN_MAX_W
#define IN_MAX_W 136u
#endif
#ifndef OUT_H
#define OUT_H 8u
#endif
#ifndef OUT_W
#define OUT_W 128u
#endif
#define IN_ROW_BLOCKS ((OUT_W+16u)/8u)
#define ACTIVE_H_BLOCKS ((OUT_H+15u)/8u)
#define ACTIVE_W_BLOCKS ((OUT_W+15u)/8u)
#define INPUT_PAGES (ACTIVE_H_BLOCKS*ACTIVE_W_BLOCKS*(CI/32u))
#define INPUT_TABLE_ENTRIES (3u*IN_ROW_BLOCKS*(CI/32u))
#define OUTPUT_PAGES (((OUT_H+7u)/8u)*((OUT_W+7u)/8u)*(CO/32u))
#define LOWERED_BYTES (2u*2u*CI*CO)
#define KERNEL_BYTES (3u*3u*CI*CO)
#define TABLE_ENTRIES 512u
#define FIXTURE_MAGIC 0x3253324eu

static uint8_t fixture[FIXTURE_BYTES] __attribute__((aligned(128)));
static uint32_t xblocks[INPUT_PAGES], yblocks[OUTPUT_PAGES];
extern void qd_qnn_hmx_stride1_generic(uint32_t *,uint32_t *,void *,void *,uint32_t *,uint32_t *);
extern void qd_set_hmx_params_stride1(uint32_t *,uint32_t,uint32_t,uint32_t,
                                      uint32_t,uint32_t,uint32_t,uint32_t);
extern void qd_hmx_finish(void *,void *);

static void unpack_xwb(int8_t *out,const uint8_t *raw,uint32_t kh,uint32_t kw,
                       uint32_t ci,uint32_t co) {
    size_t n=0;
    for(uint32_t h=0;h<kh;h++)for(uint32_t w=0;w<kw;w++)
        for(uint32_t i=0;i<ci;i++)for(uint32_t o=0;o<co;o++) {
            size_t off=((((size_t)(o/32)*(ci/32)+i/32)*kh+h)*kw+w)*1024
                      +(size_t)((i/4)%8)*128+(o%32)*4+i%4;
            out[n++]=(int8_t)raw[off];
        }
}
static void lower_and_pack_s2(uint8_t *packed,const uint8_t *raw,
                              uint32_t ci,uint32_t co) {
    static const uint8_t taps[9][2]={{0,2},{0,0},{2,2},{2,0},{0,1},{2,1},{1,2},{1,0},{1,1}};
    static int8_t unpacked[9u*128u*128u];
    static int8_t tr[9u*128u*128u];
    static int8_t lowered[4u*512u*128u];
    unpack_xwb(unpacked,raw,3,3,ci,co);
    memset(tr,0,sizeof tr);memset(lowered,0,sizeof lowered);memset(packed,0,KERNEL_BYTES);
    for(uint32_t k=0;k<9;k++)memcpy(tr+((size_t)taps[k][0]*3+taps[k][1])*ci*co,
                                     unpacked+(size_t)k*ci*co,(size_t)ci*co);
    for(uint32_t a=0;a<2;a++)for(uint32_t b=0;b<2;b++)
        for(uint32_t dy=0;dy<2;dy++)for(uint32_t dx=0;dx<2;dx++) {
            uint32_t ky=2*a+dy,kx=2*b-dx;
            if(ky<3 && kx<3)for(uint32_t cb=0;cb<ci/32;cb++) {
                uint32_t j=cb*4+2*dy+dx;
                for(uint32_t ch=0;ch<32;ch++)
                    memcpy(lowered+(((size_t)(a*2+b)*4*ci+32*j+ch)*co),
                           tr+(((size_t)ky*3+kx)*ci+32*cb+ch)*co,co);
            }
        }
    /* Lift logical 2x2 lowered weights into the upper-left of a 3x3
     * generic-leaf kernel. The leaf addresses x+2-raw_kx, so logical dx
     * is stored at raw_kx=2-dx; unused bottom row and left column stay zero. */
    for(uint32_t h=0;h<2;h++)for(uint32_t w=1;w<3;w++)
        for(uint32_t i=0;i<4*ci;i++)for(uint32_t o=0;o<co;o++) {
            uint32_t dx=2u-w;
            size_t src=(((size_t)(h*2+dx)*4*ci+i)*co+o);
            size_t dst=((((size_t)(o/32)*(4*ci/32)+i/32)*3+h)*3+w)*1024
                      +(size_t)((i/4)%8)*128+(o%32)*4+i%4;
            packed[dst]=(uint8_t)lowered[src];
        }
}
static uint64_t fnv64(const uint8_t*p,uint32_t n){
    uint64_t h=UINT64_C(0xcbf29ce484222325);
    for(uint32_t i=0;i<n;i++)h=(h^p[i])*UINT64_C(0x100000001b3);
    return h;
}

int main(void) {
    FILE*f=fopen("/src/native_stride2/s2_cpu_prefix_fixture.bin","rb");
    if(!f){puts("FAIL fixture open");return 1;}
    uint32_t h[48];
    if(fread(h,sizeof h,1,f)!=1||h[0]!=FIXTURE_MAGIC){puts("FAIL fixture header");fclose(f);return 2;}
    uint32_t xbytes=h[15]*PAGE,wb=h[35],fb=h[36],ybytes=h[29]*PAGE;
    uint64_t payload=(uint64_t)xbytes+wb+fb+ybytes;
    if(payload>FIXTURE_BYTES||fread(fixture,(size_t)payload,1,f)!=1||fgetc(f)!=EOF){puts("FAIL fixture payload");fclose(f);return 3;}
    fclose(f);
    if(h[1]!=SEQUENCE||h[15]*PAGE!=xbytes||h[29]<OUTPUT_PAGES||h[30]!=3||h[31]!=3||
       h[3]!=IN_H||h[4]!=IN_W||h[5]!=CI||h[17]!=OUT_H||h[18]!=OUT_W||
       h[32]*4u!=CI||h[33]!=CO||wb!=9u*h[32]*h[33]||fb!=CO*8u){puts("FAIL pinned S2 geometry");return 4;}
    const uint8_t*x=fixture,*w=x+xbytes,*fi=w+wb,*y=fi+fb;
    uint32_t vtcm=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<VTCM_BYTES;i+=4096u)add_translation((void*)(vtcm+i),(void*)(vtcm+i),6);
    uint8_t *base=(uint8_t*)vtcm;
    uint8_t *input=base+0x00000,*input_guard=base+0x50000;
    uint8_t *output=base+0x51000,*output_guard=base+0x61000;
    uint8_t *zero_page=base+0x62000;
    uint8_t *weights=base+0x64000,*weight_guard=base+0xf4000;
    int32_t *fi_stage=(int32_t*)(base+0xf5000),*finish_fi=(int32_t*)(base+0xf6000);
    uint32_t *it=(uint32_t*)(base+0x63000),*ot=it+TABLE_ENTRIES;
    volatile uint32_t *done=(volatile uint32_t*)(base+0xf7000);
    uint32_t *input_desc=(uint32_t*)(base+0xf8000),*output_desc=input_desc+6;
    uint32_t *cfg=output_desc+6,*mode=cfg+16;
    if((uintptr_t)(mode+2)>(uintptr_t)(base+VTCM_BYTES)){puts("FAIL VTCM layout");return 5;}

    memset(base,0x3c,VTCM_BYTES);memset(input,0,INPUT_PAGES*PAGE);memset(zero_page,0,PAGE);
    for(uint32_t sy=0;sy<OUT_H+1u;sy++)for(uint32_t sx=0;sx<OUT_W+1u;sx++)for(uint32_t ch=0;ch<CI;ch++) {
        uint32_t srcb=h[14]+((sy/8u*((IN_MAX_W+7u)/8u)+sx/8u)*(CI/32u)+ch/32u);
        size_t so=(size_t)srcb*PAGE+(sy%8u)*256u+(sx%8u)*32u+ch%32u;
        uint32_t dy=sy+7u,dx=sx+7u;
        uint32_t bh=dy/8u,bw=dx/8u,bd=ch/32u;
        uint32_t dstb=(bh*ACTIVE_W_BLOCKS+bw)*(CI/32u)+bd;
        size_t doff=(size_t)dstb*PAGE+(dy%8u)*256u+(dx%8u)*32u+ch%32u;
        input[doff]=x[so];
    }
    memset(output,0xa5,OUTPUT_PAGES*PAGE);
    memset(input_guard,0xa5,PAGE);memset(output_guard,0xa5,PAGE);memset(weight_guard,0xa5,128);
    memset(weights,0,KERNEL_BYTES);memset(fi_stage,0,CO*8);memset(finish_fi,0,64*4);
    for(uint32_t i=0;i<32;i++)finish_fi[i]=0x404000u;
    memcpy(fi_stage,fi,fb);lower_and_pack_s2(weights,w,CI/4u,CO);

    for(uint32_t i=0;i<TABLE_ENTRIES;i++)it[i]=(uint32_t)(uintptr_t)zero_page;
    for(uint32_t i=0;i<TABLE_ENTRIES;i++)ot[i]=(uint32_t)(uintptr_t)output_guard;
    for(uint32_t bh=0;bh<3u;bh++)for(uint32_t bw=0;bw<IN_ROW_BLOCKS;bw++)for(uint32_t bd=0;bd<CI/32u;bd++){uint32_t k=(bh*IN_ROW_BLOCKS+bw)*(CI/32u)+bd;if(bh<ACTIVE_H_BLOCKS&&bw<ACTIVE_W_BLOCKS){uint32_t p=(bh*ACTIVE_W_BLOCKS+bw)*(CI/32u)+bd;xblocks[p]=p*PAGE|1u;it[k]=(uint32_t)(uintptr_t)(input+p*PAGE);}}
    for(uint32_t i=0;i<OUTPUT_PAGES;i++){yblocks[i]=i*PAGE|1u;ot[i]=(uint32_t)(uintptr_t)(output+i*PAGE);}
    memset(cfg,0,16u*4);qd_set_hmx_params_stride1(cfg,0x700u,3u,3u,CI,7u,1u,12u);
    input_desc[0]=(uint32_t)(uintptr_t)it;input_desc[1]=CI/32;input_desc[2]=IN_ROW_BLOCKS*(CI/32);
    input_desc[3]=OUT_W+9u;input_desc[4]=17;input_desc[5]=CI;
    output_desc[0]=(uint32_t)(uintptr_t)ot;output_desc[1]=CO/32;output_desc[2]=((OUT_W+7u)/8u)*(CO/32);
    output_desc[3]=OUT_W;output_desc[4]=OUT_H;output_desc[5]=CO;
    mode[0]=1u;mode[1]=0u;*done=0xffffffffu;
    printf("S2 generic leaf cfg=");for(uint32_t i=0;i<13;i++)printf("%s%x",i?",":"",cfg[i]);
    printf("\nHMX-stage hashes X=%016llx Wlowered=%016llx FI=%016llx CPUY=%016llx\n",
        (unsigned long long)fnv64(input,xbytes),(unsigned long long)fnv64(weights,KERNEL_BYTES),
        (unsigned long long)fnv64((uint8_t*)fi_stage,fb),(unsigned long long)fnv64(y,OUTPUT_PAGES*PAGE));

    uint32_t old_ssr;__asm__ volatile("%0=ssr":"=r"(old_ssr));
    __asm__ volatile("ssr=%0\n isync"::"r"(old_ssr|(1u<<26)):"memory");
    *done=0xffffffffu;__asm__ volatile("mxclracc":::"memory");
    for(uintptr_t q=(uintptr_t)base;q<(uintptr_t)(base+VTCM_BYTES);q+=32)__asm__ volatile("dccleaninva(%0)"::"r"(q):"memory");__asm__ volatile("barrier":::"memory");
        qd_qnn_hmx_stride1_generic(output_desc,input_desc,weights,fi_stage,cfg,mode);
    qd_hmx_finish(finish_fi,(void*)done);
    uint32_t spins=0;while(*done&&spins++<100001u)__asm__ volatile("nop");
    __asm__ volatile("ssr=%0\n isync"::"r"(old_ssr):"memory");
    if(*done){puts("FAIL HMX timeout");return 6;}

    uint32_t bad=0,plus=0,minus=0,other=0;
    const uint8_t*expected=y+(size_t)h[28]*PAGE;
    for(uint32_t i=0;i<OUTPUT_PAGES*PAGE;i++) {
        int d=(int)output[i]-(int)expected[i];
        if(d){if(bad<24)printf("DIFF i=%u y=%u x=%u ch=%u got=%u exp=%u d=%d\n",i,i/(OUT_W*CO),(i/CO)%OUT_W,i%CO,output[i],expected[i],d);bad++;if(d==1)plus++;else if(d==-1)minus++;else other++;}
    }
    for(uint32_t i=0;i<PAGE;i++)if(input_guard[i]!=0xa5||output_guard[i]!=0xa5){puts("FAIL page guard");return 7;}
    for(uint32_t i=0;i<128;i++)if(weight_guard[i]!=0xa5){puts("FAIL weight guard");return 8;}
    for(uint32_t i=OUTPUT_PAGES;i<TABLE_ENTRIES;i++)if(ot[i]!=(uint32_t)(uintptr_t)output_guard){puts("FAIL output table tail");return 9;}
    for(uint32_t i=INPUT_TABLE_ENTRIES;i<TABLE_ENTRIES;i++)if(it[i]!=(uint32_t)(uintptr_t)zero_page){puts("FAIL input table tail");return 10;}
    printf("first output actual/expected:");for(uint32_t i=0;i<32;i++)printf(" %u/%u",output[i],expected[i]);putchar('\n');
    printf("CPU full active physical output compare mismatches=%u/%u +1=%u -1=%u other=%u\n",
           bad,OUTPUT_PAGES*PAGE,plus,minus,other);
    if(minus||other){puts("FAIL candidate differs from CPU by more than +1");return 11;}
#ifdef RAW_DUMP
    puts("RAW_OUTPUT_BEGIN");for(uint32_t i=0;i<OUTPUT_PAGES*PAGE;i++){printf("%02x",output[i]);if((i&255u)==255u)putchar('\n');}puts("RAW_OUTPUT_END");
#endif
    puts("PASS actual S2 lowered 2x2 generic leaf, full physical output and guards");return 0;
}
