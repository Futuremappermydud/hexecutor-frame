/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Native accessors for the recovered blocked tensors. No precomputed address
 * arrays: walk contiguous channel runs and derive each 2 KB block directly. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static inline __attribute__((always_inline)) void transfer_impl(uint8_t *restrict values, uint8_t *const *pools, const uint32_t *bp,
                  const uint32_t *bo, const int32_t *layout, const int32_t *dims,
                  const int32_t *origin, uint32_t size, uint32_t store) {
    int64_t maxh=layout[0], maxw=layout[1], maxd=layout[2], slice=layout[3], count=layout[4];
    int64_t cw=size==1?8:size==2?4:2;
    int64_t nbh=(maxh+7)/8, nbw=(maxw+cw-1)/cw, nbd=(maxd+31)/32;
    size_t pos=0;
    for(int64_t b=0;b<dims[0];b++) for(int64_t h=0;h<dims[1];h++)
        for(int64_t w=0;w<dims[2];w++) {
            int64_t B=b+origin[0], H=h+origin[1], W=w+origin[2];
            for(int64_t d=0;d<dims[3];) {
                int64_t D=d+origin[3];
                int64_t n=32-D%32;
                if(n>dims[3]-d)n=dims[3]-d;
                int64_t index=((B*nbh+H/8)*nbw+W/cw)*nbd+D/32+slice;
                uint32_t pool=(index>=0 && index<count)?bp[index]:0;
                uint8_t *base=pool?pools[pool]:NULL;
                if(base) {
                    size_t within=size==1?(H%8)*256+(W%8)*32+D%32:
                                  size==2?(H%8)*128+((W%4)/2)*64+(D%32)*2+W%2:
                                          (H%8)*64+(W%2)*32+D%32;
                    uint8_t *ptr=base+bo[index]+within*size;
                    if(size!=2) {
                        if(store)memcpy(ptr,values+pos,n*size);else memcpy(values+pos,ptr,n*size);
                    } else {
                        for(int64_t c=0;c<n;c++) {
                            if(store)memcpy(ptr+c*4,values+pos+c*2,2);
                            else memcpy(values+pos+c*2,ptr+c*4,2);
                        }
                    }
                } else if(!store)memset(values+pos,0,n*size);
                pos+=n*size;d+=n;
            }
        }
}

void qdm_transfer(uint8_t *values,uint8_t *const *pools,const uint32_t *bp,const uint32_t *bo,
                  const int32_t *layout,const int32_t *dims,const int32_t *origin,uint32_t size,uint32_t store) {
#define RUN(S,W) transfer_impl(values,pools,bp,bo,layout,dims,origin,S,W)
    if(size==1){if(store)RUN(1,1);else RUN(1,0);}
    else if(size==2){if(store)RUN(2,1);else RUN(2,0);}
    else {if(store)RUN(4,1);else RUN(4,0);}
#undef RUN
}

void qdm_unpack_weights(int8_t *out,const int8_t *raw,uint32_t H,uint32_t W,uint32_t I,uint32_t O,uint32_t packed) {
    size_t index=0;
    for(uint32_t h=0;h<H;h++)for(uint32_t w=0;w<W;w++)for(uint32_t i=0;i<I;i++)for(uint32_t o=0;o<O;o++) {
        size_t offset;
        if(packed)offset=(((o/32*H+h)*W+w)*(I/4)+i/4)*128+(o%32)*4+i%4;
        else offset=((((o/32*(I/32)+i/32)*H+h)*W+w)*1024)+(i/4%8)*128+(o%32)*4+i%4;
        out[index++]=raw[offset];
    }
}

void qdm_requant(uint8_t *out,const int64_t *acc,const int32_t *mult,const int32_t *bias,
                 uint32_t pixels,uint32_t channels,uint32_t sixteen) {
    int32_t M[1024],pre[1024],post[1024];
    for(uint32_t c=0;c<channels;c++) {
        uint32_t h=(uint32_t)mult[c]&65535, shift=34-((h>>10)&31);
        M[c]=1024+(h&1023);pre[c]=sixteen?8:(int32_t)shift-14;post[c]=(int32_t)shift-pre[c];
    }
    for(uint32_t p=0;p<pixels;p++)for(uint32_t c=0;c<channels;c++) {
        size_t j=(size_t)p*channels+c;
        int64_t a=acc[j]+(int64_t)bias[c]*(sixteen?256:1);
        int64_t q=(((a>>pre[c])*M[c])+((int64_t)1<<(post[c]-1)))>>post[c];
        int64_t max=sixteen?65535:255;
        if(q<0)q=0; if(q>max)q=max;
        if(sixteen)((uint16_t *)out)[j]=(uint16_t)q;else out[j]=(uint8_t)q;
    }
}

void qdm_pack_x(uint8_t *out,const uint8_t *x,uint32_t H,uint32_t W,uint32_t C,uint32_t G,uint32_t WP,uint32_t bytes) {
    memset(out,0,bytes);
    for(uint32_t h=0;h<H;h++)for(uint32_t g=0;g<G;g++)for(uint32_t w=0;w<W;w++) {
        uint8_t *d=out+((size_t)h*G*WP+g*WP+w)*4;
        const uint8_t *s=x+((size_t)h*W+w)*C+g*4;
        uint32_t n=C-g*4;if(n>4)n=4;
        memcpy(d,s,n);
    }
}
void qdm_pack_w(int8_t *out,const int8_t *w,uint32_t KH,uint32_t KW,uint32_t C,uint32_t N,uint32_t G) {
    for(uint32_t k=0;k<KH*KW;k++)for(uint32_t g=0;g<G;g++)for(uint32_t n=0;n<N;n++)
        for(uint32_t c=0;c<4;c++)out[(((size_t)k*G+g)*N+n)*4+c]=g*4+c<C?w[((size_t)k*C+g*4+c)*N+n]:0;
}
void qdm_collect(int64_t *out,const int32_t *raw,uint32_t H,uint32_t W,uint32_t N,uint32_t WP) {
    for(uint32_t h=0;h<H;h++)for(uint32_t n=0;n<N;n++)for(uint32_t w=0;w<W;w++)
        out[((size_t)h*W+w)*N+n]=raw[((size_t)h*N+n)*WP+w];
}
void qdm_copy(void *out,const void *in,size_t bytes) {memcpy(out,in,bytes);}

/* Bounded fast path for the frequent short DSP requests. The Python transport
 * handles timeout, cancellation, reset detection and cleanup after this returns. */
int qdm_poll(const volatile uint32_t *arena,uint32_t sequence,uint32_t spins) {
    const volatile uint32_t *status=arena+128/4;
    for(uint32_t i=0;i<spins;i++) {
        if(status[0]==sequence && status[4]==sequence && status[8]==sequence)return 0;
        if(status[3]==0x54495845u && status[7]==0x54495845u && status[11]==0x54495845u)return 2;
    }
    return 1;
}

/* Prove that the vector epilogue's correction fits signed 32 bits for every
 * possible u8 input, then prepare its exact integer parameters. */
int qdm_quant_params(int32_t *out,const int8_t *w,const int32_t *mult,const int32_t *bias,
                     uint32_t terms,uint32_t channels,int32_t zp) {
    for(uint32_t c=0;c<channels;c++) {
        int64_t sum=0,pos=0,neg=0;
        for(uint32_t i=0;i<terms;i++) {int32_t v=w[(size_t)i*channels+c];sum+=v;if(v>0)pos+=v;else neg+=v;}
        int64_t correction=(int64_t)bias[c]-(int64_t)zp*sum;
        int32_t h=mult[c]&65535,pre=20-((h>>10)&31);
        if(pre<0 || pre>31 || correction+255*neg<INT32_MIN || correction+255*pos>INT32_MAX)return 1;
        out[c*3]=(int32_t)correction;out[c*3+1]=1024+(h&1023);out[c*3+2]=pre;
    }
    return 0;
}
void qdm_collect_u8(uint8_t *out,const uint8_t *raw,uint32_t H,uint32_t W,uint32_t N,uint32_t WP) {
    for(uint32_t h=0;h<H;h++)for(uint32_t n=0;n<N;n++)for(uint32_t w=0;w<W;w++)
        out[((size_t)h*W+w)*N+n]=raw[(((size_t)h*(N/4)+n/4)*(WP/32)+w/32)*128+(n%4)*32+w%32];
}

void qdm_pack_w_hmx(int8_t *out,const int8_t *w,uint32_t KH,uint32_t KW,uint32_t CI,uint32_t N,uint32_t G,uint32_t KB) {
    uint32_t cp=G*4,depth=KH*KW*cp;
    for(uint32_t nb=0;nb<N/32;nb++)for(uint32_t kb=0;kb<KB;kb++)for(uint32_t group=0;group<8;group++)
        for(uint32_t n=0;n<32;n++)for(uint32_t c=0;c<4;c++) {
            uint32_t k=kb*32+group*4+c,tap=k/cp,channel=k%cp;
            out[((((size_t)nb*KB+kb)*8+group)*32+n)*4+c]=
                k<depth && channel<CI?w[((size_t)tap*CI+channel)*N+nb*32+n]:0;
        }
}
void qdm_collect_hmx_pre(uint8_t *out,const uint16_t *raw,const int32_t *params,uint32_t H,uint32_t W,uint32_t N,uint32_t WP) {
    for(uint32_t h=0;h<H;h++)for(uint32_t w=0;w<W;w++)for(uint32_t n=0;n<N;n++) {
        uint32_t p=h*WP+w;
        size_t offset=((((size_t)(p/32)*(N/32)+n/32)*16+(p%32)/2)*32+n%32)*2+p%2;
        uint32_t value=((uint32_t)raw[offset]*(uint32_t)params[n*3+1]+8192)>>14;
        out[((size_t)h*W+w)*N+n]=(uint8_t)(value>255?255:value);
    }
}

/* HMX's fused u8 epilogue returns contiguous 32-channel pixel rows. */
void qdm_collect_hmx_u8(uint8_t *out,const uint8_t *raw,uint32_t H,uint32_t W,uint32_t N,uint32_t WP) {
    for(uint32_t h=0;h<H;h++)for(uint32_t w=0;w<W;w++)for(uint32_t nb=0;nb<N/32;nb++) {
        uint32_t p=h*WP+w;
        size_t offset=(((size_t)(p/32)*(N/32)+nb)*32+p%32)*32;
        memcpy(out+((size_t)h*W+w)*N+nb*32,raw+offset,32);
    }
}

/* Fused ARM integer elementwise arithmetic; unsupported ranges fall back to
 * the independent emulator. Bounds prove that all supported sums fit u32. */
int qdm_add_u8(uint8_t *out,const uint8_t *a,const uint8_t *b,size_t count,int32_t ma,int32_t mb,int32_t shift) {
    int32_t s=shift+8;
    if(ma<0 || mb<0 || s<1 || s>31)return 1;
    uint32_t round=1u<<(s-1);
    if(((uint64_t)(uint32_t)ma+(uint32_t)mb)*255+round>UINT32_MAX)return 1;
    for(size_t i=0;i<count;i++) {
        uint32_t v=((uint32_t)a[i]*(uint32_t)ma+(uint32_t)b[i]*(uint32_t)mb+round)>>s;
        out[i]=(uint8_t)(v>255?255:v);
    }
    return 0;
}
int qdm_requant_u8(uint8_t *out,const uint8_t *x,size_t count,uint32_t packed) {
    uint32_t m=packed&65535;int32_t s=15-(packed>>24);
    if(s<1 || s>31)return 1;
    uint32_t round=1u<<(s-1);
    for(size_t i=0;i<count;i++) {
        uint32_t v=((uint32_t)x[i]*m+round)>>s;
        out[i]=(uint8_t)(v>255?255:v);
    }
    return 0;
}
