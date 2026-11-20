/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_HMX_NATIVE_KERNEL_H
#define QD_HMX_NATIVE_KERNEL_H
#include <stddef.h>
/* Version 7 is an opt-in native integer conversion experiment. It preserves
 * the arena layout, with measured numerical differences from version 6. */
extern void qd_hmx_native(void *,void *,void *,void *,uint32_t);
extern void qd_hmx_native_cm(void *,void *,void *,void *,uint32_t);
static HVX_Vector hmx_native_a4(HVX_Vector v) {HmxBytes b=(HmxBytes)v;return (HVX_Vector)__builtin_shufflevector(b,b,0,32,64,96,1,33,65,97,2,34,66,98,3,35,67,99,4,36,68,100,5,37,69,101,6,38,70,102,7,39,71,103,8,40,72,104,9,41,73,105,10,42,74,106,11,43,75,107,12,44,76,108,13,45,77,109,14,46,78,110,15,47,79,111,16,48,80,112,17,49,81,113,18,50,82,114,19,51,83,115,20,52,84,116,21,53,85,117,22,54,86,118,23,55,87,119,24,56,88,120,25,57,89,121,26,58,90,122,27,59,91,123,28,60,92,124,29,61,93,125,30,62,94,126,31,63,95,127);}
static HVX_Vector hmx_native_dense4(HVX_Vector v) {HmxBytes b=(HmxBytes)v;return (HVX_Vector)__builtin_shufflevector(b,b,0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77,81,85,89,93,97,101,105,109,113,117,121,125,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78,82,86,90,94,98,102,106,110,114,118,122,126,3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83,87,91,95,99,103,107,111,115,119,123,127);}
static HVX_Vector hmx_native_g4(HVX_Vector v) {HmxBytes b=(HmxBytes)v;return (HVX_Vector)__builtin_shufflevector(b,b,0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15,16,20,24,28,17,21,25,29,18,22,26,30,19,23,27,31,32,36,40,44,33,37,41,45,34,38,42,46,35,39,43,47,48,52,56,60,49,53,57,61,50,54,58,62,51,55,59,63,64,68,72,76,65,69,73,77,66,70,74,78,67,71,75,79,80,84,88,92,81,85,89,93,82,86,90,94,83,87,91,95,96,100,104,108,97,101,105,109,98,102,106,110,99,103,107,111,112,116,120,124,113,117,121,125,114,118,122,126,115,119,123,127);}
static inline __attribute__((always_inline)) void hmx_native_tile32(HVX_Vector *dest,const uint8_t *const *p,int cm) {
    HVX_Vector a0=hmx_native_g4(*(const HVX_UVector *)p[0]),a1=hmx_native_g4(*(const HVX_UVector *)p[1]);
    HVX_Vector a2=hmx_native_g4(*(const HVX_UVector *)p[2]),a3=hmx_native_g4(*(const HVX_UVector *)p[3]);
    HVX_Vector a4=hmx_native_g4(*(const HVX_UVector *)p[4]),a5=hmx_native_g4(*(const HVX_UVector *)p[5]);
    HVX_Vector a6=hmx_native_g4(*(const HVX_UVector *)p[6]),a7=hmx_native_g4(*(const HVX_UVector *)p[7]);
    HVX_Vector b0,b1,b2,b3,b4,b5,b6,b7;
    HMX_STAGE(a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7);
    HMX_STAGE(b0,b1,b2,b3,b4,b5,b6,b7,a0,a1,a2,a3,a4,a5,a6,a7);
    HMX_STAGE(a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7);
    if(cm){b0=hmx_native_dense4(b0);b1=hmx_native_dense4(b1);b2=hmx_native_dense4(b2);b3=hmx_native_dense4(b3);
           b4=hmx_native_dense4(b4);b5=hmx_native_dense4(b5);b6=hmx_native_dense4(b6);b7=hmx_native_dense4(b7);}
    dest[0]=b0;dest[1]=b1;dest[2]=b2;dest[3]=b3;dest[4]=b4;dest[5]=b5;dest[6]=b6;dest[7]=b7;
}
static inline __attribute__((always_inline)) void hmx_native_copy_a(HVX_Vector *dest,const uint8_t *src,uint32_t vectors,int cm) {
    if(cm)for(uint32_t q=0;q<vectors;q++)dest[q]=*(const HVX_UVector *)(src+q*128);
    else for(uint32_t q=0;q<vectors;q++)dest[q]=hmx_native_a4(*(const HVX_UVector *)(src+q*128));
}
static int hmx_native_prepare(const QdConv *c,const QdHmxLayout *l,uint8_t *vtcm,uint32_t *table,uint32_t tid) {
    uint32_t depth=c->kh*c->kw*c->g;
    if(depth>QDC_PREPARE_TABLE-8)return 0;
    uint32_t k=0,x32=c->reserved[0]&QDC_X32,cbs=c->g>>3;
    if(x32) {
        if((c->g&7) || l->kb!=c->kh*c->kw*cbs)return 0;
        for(uint32_t ky=0;ky<c->kh;ky++)for(uint32_t kx=0;kx<c->kw;kx++)for(uint32_t cb=0;cb<cbs;cb++,k++)
            table[k]=(ky*cbs+cb)*c->wp*32+kx*32;
    } else {
        for(uint32_t ky=0;ky<c->kh;ky++)for(uint32_t kx=0;kx<c->kw;kx++)for(uint32_t g=0;g<c->g;g++,k++)
            table[k]=(ky*c->g+g)*c->wp*4+kx*4;
        for(;k<l->kb*8;k++)table[k]=0;
    }
    uint32_t rowbytes=c->g*c->wp*4,mb=0,kb=tid,row=0,column=0;
    while(kb>=l->kb){kb-=l->kb;mb++;column+=64;while(column>=c->owp){column-=c->owp;row++;}}
    const uint8_t *zero=vtcm+c->h*rowbytes;
    while(mb<l->mb) {
        HVX_Vector *dest=(HVX_Vector *)(vtcm+l->a+((size_t)mb*l->kb+kb)*2048);
        uint32_t y=row,x=column;
        if(x32) {
            int cm=c->version==QDC_NATIVE_CM_VERSION;
            const uint8_t *src=vtcm+y*rowbytes+x*32+table[kb];
            if(c->owp-column>=64)hmx_native_copy_a(dest,src,16,cm);
            else {
                hmx_native_copy_a(dest,src,8,cm);
                if(y+1<c->oh)hmx_native_copy_a(dest+8,vtcm+(y+1)*rowbytes+table[kb],8,cm);
                else for(uint32_t q=8;q<16;q++)dest[q]=Q6_V_vzero();
            }
        } else for(uint32_t half=0;half<2;half++) {
            const uint8_t *base=vtcm+y*rowbytes+x*4,*p[8];
            for(uint32_t j=0;j<8;j++)p[j]=(y<c->oh && kb*8+j<depth)?base+table[kb*8+j]:zero;
            hmx_native_tile32(dest+half*8,p,c->version==QDC_NATIVE_CM_VERSION);
            x+=32;if(x>=c->owp){x=0;y++;}
        }
        kb+=3;
        while(kb>=l->kb){kb-=l->kb;mb++;column+=64;while(column>=c->owp){column-=c->owp;row++;}}
    }
    return 1;
}
static int hmx_native_calculate(const QdConv *c,const QdHmxLayout *l,uint8_t *vtcm,const QdQuant *q) {
    int32_t *fi=(int32_t *)(vtcm+l->fi);
    for(uint32_t n=0;n<c->n;n++) {
        fi[(n>>5)*64+(n&31)]=0x400000u|((20-q[n].preshift)<<10)|(q[n].multiplier-1024);
        fi[(n>>5)*64+32+(n&31)]=q[n].correction;
    }
    int32_t *zero=(int32_t *)(vtcm+l->zero);
    for(uint32_t n=0;n<64;n++)zero[n]=n<32?0x404000:0;
    uint8_t *completion=vtcm+l->completion;
    for(uint32_t i=0;i<2048;i+=128)*(HVX_Vector *)(completion+i)=Q6_V_vsplat_R(-1);
    barrier();
    for(uint32_t m=0;m<l->mb;m++)for(uint32_t n=0;n<l->nb;n++)
        if(c->version==QDC_NATIVE_CM_VERSION)
            qd_hmx_native_cm(vtcm+l->a+(size_t)m*l->kb*2048,vtcm+l->b+(size_t)n*l->kb*1024,
                            fi+n*64,vtcm+l->c+((size_t)m*l->nb+n)*2048,l->kb);
        else qd_hmx_native(vtcm+l->a+(size_t)m*l->kb*2048,vtcm+l->b+(size_t)n*l->kb*1024,
                          fi+n*64,vtcm+l->c+((size_t)m*l->nb+n)*2048,l->kb);
    qd_hmx_finish(zero,completion);
    uint64_t end=cycles()+1000000000u;
    while((uint32_t)(*(volatile HVX_Vector *)completion)[0])if(cycles()>end)return QDC_BARRIER_TIMEOUT;
    barrier();return 0;
}
static void hmx_native_collect(const QdConv *c,const QdHmxLayout *l,uint8_t *vtcm,uint8_t *output,uint32_t tid) {
    uint32_t nb=tid,mb=0;while(nb>=l->nb){nb-=l->nb;mb++;}
    uint32_t blocked=c->reserved[0]&QDC_CROUTON8,height=(c->reserved[0]>>8)&255,width=c->reserved[0]>>16;
    uint32_t old_mb=c->oh*(c->owp>>5);
    for(uint32_t tile=tid;tile<l->mb*l->nb;tile+=3) {
        const HVX_Vector *src=(const HVX_Vector *)(vtcm+l->c+(size_t)tile*2048);
        uint32_t y=0,x=mb*64,stride=blocked?(c->oh==1?width:c->owp):c->owp;
        while(x>=stride){x-=stride;y++;}
        for(uint32_t i=0;i<16;i++) {
            HVX_Vector v=c->version==QDC_NATIVE_CM_VERSION?src[i]:hmx_native_dense4(src[i]);
            if(blocked) {
                if(y<height && x<width) {
                    uint32_t block=((y/8)*(width/8)+x/8)*l->nb+nb;
                    *(HVX_Vector *)(output+(size_t)block*2048+(y%8)*256+(x%8)*32)=v;
                }
            } else {
                uint32_t m32=mb*2+(i>>3);
                if(m32<old_mb)*(HVX_Vector *)(output+((size_t)m32*l->nb+nb)*1024+(i&7)*128)=v;
            }
            x+=4;if(x>=stride){x=0;y++;}
        }
        nb+=3;while(nb>=l->nb){nb-=l->nb;mb++;}
    }
}
#endif
