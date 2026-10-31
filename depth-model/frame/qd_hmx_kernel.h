#ifndef QD_HMX_KERNEL_H
#define QD_HMX_KERNEL_H
extern void qd_hmx_wide(void *,void *,void *,void *,uint32_t);
extern void qd_hmx_finish(void *,void *);
typedef unsigned char HmxBytes __attribute__((vector_size(128)));
static HVX_Vector hmx_transpose(HVX_Vector value) {
    HmxBytes b=(HmxBytes)value;
    return (HVX_Vector)__builtin_shufflevector(b,b,0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15,16,17,24,25,18,19,26,27,20,21,28,29,22,23,30,31,32,33,40,41,34,35,42,43,36,37,44,45,38,39,46,47,48,49,56,57,50,51,58,59,52,53,60,61,54,55,62,63,64,65,72,73,66,67,74,75,68,69,76,77,70,71,78,79,80,81,88,89,82,83,90,91,84,85,92,93,86,87,94,95,96,97,104,105,98,99,106,107,100,101,108,109,102,103,110,111,112,113,120,121,114,115,122,123,116,117,124,125,118,119,126,127);
}
static void hmx_zero(uint8_t *a,uint32_t bytes,uint32_t tid) {
    for(uint32_t offset=tid*128;offset<bytes;offset+=3*128)*(HVX_Vector *)(a+offset)=Q6_V_vzero();
}
static void hmx_prepare_scatter(const QdConv *c,const QdHmxLayout *l,uint8_t *vtcm,uint32_t tid) {
    const uint8_t *x=vtcm;uint8_t *a=vtcm+l->a;
    int32_t indices[32] __attribute__((aligned(128)));
    for(uint32_t i=0;i<32;i++)indices[i]=(i/4)*128+(i%4)*4;
    HVX_Vector offset=*(HVX_Vector *)indices;
    uint32_t turn=0,vecs=c->owp>>5;
    for(uint32_t oy=0;oy<c->oh;oy++)for(uint32_t vx=0;vx<vecs;vx++) {
        uint32_t k4=0,mb=oy*vecs+vx;
        for(uint32_t ky=0;ky<c->kh;ky++)for(uint32_t kx=0;kx<c->kw;kx++)for(uint32_t g=0;g<c->g;g++,k4++) {
            if(turn==tid) {
                const uint8_t *src=x+((oy+ky)*c->g+g)*c->wp*4+(vx*32+kx)*4;
                HVX_Vector v=*(const HVX_UVector *)src;
                HVX_VectorPair wide=Q6_Wuh_vunpack_Vub(v);
                HVX_Vector lo=hmx_transpose(Q6_Vh_vasl_VhR(Q6_V_lo_W(wide),8));
                HVX_Vector hi=hmx_transpose(Q6_Vh_vasl_VhR(Q6_V_hi_W(wide),8));
                uint8_t *dest=a+(mb*l->kb+(k4>>3))*2048;
                HVX_Vector off=Q6_Vw_vadd_VwVw(offset,Q6_V_vsplat_R((k4&7)*16));
                Q6_vscatter_RMVwV((uint32_t)dest,2047,off,lo);
                Q6_vscatter_RMVwV((uint32_t)(dest+1024),1023,off,hi);
            }
            if(++turn==3)turn=0;
        }
    }
    /* Drain this worker's scatter queues once per destination region after
     * issuing all independent writes, before HMX consumes the activation tiles. */
    for(uint32_t tile=0;tile<l->mb*l->kb;tile++) {
        (void)*(volatile HVX_Vector *)(a+tile*2048);
        (void)*(volatile HVX_Vector *)(a+tile*2048+1024);
    }
}
/* Activation tiles without scatter. Lane i of a transformed source vector goes
 * to row i/4, column (i%4)*4 + group*16 of its tile, so a tile's 16 rows are an
 * 8x8 transpose of 16-byte chunks across the tile's eight channel groups. Three
 * shuffle stages perform it; absent groups (k4 >= depth) are zero, so the tiles
 * need no separate clearing. Per-k4 source offsets come from `table`, built by
 * each worker (no hardware divide). Returns 0 if the table does not fit. */
#define QDC_PREPARE_TABLE 4096u
/* Source halves are kept in named registers (an array would live on the stack,
 * which is uncached DDR here). Each of the two passes reloads the eight sources. */
#define HMX_SRC(p,high) hmx_transpose(Q6_Vh_vasl_VhR((high)?Q6_V_hi_W(Q6_Wuh_vunpack_Vub(*(const HVX_UVector *)(p))):\
                                                    Q6_V_lo_W(Q6_Wuh_vunpack_Vub(*(const HVX_UVector *)(p))),8))
#define HMX_ZIP(a,b,lo,hi) do { HVX_VectorPair z_=Q6_W_vshuff_VVR(b,a,-16); lo=Q6_V_lo_W(z_); hi=Q6_V_hi_W(z_); } while(0)
#define HMX_STAGE(v0,v1,v2,v3,v4,v5,v6,v7,w0,w1,w2,w3,w4,w5,w6,w7) do { \
    HMX_ZIP(v0,v4,w0,w1); HMX_ZIP(v1,v5,w2,w3); HMX_ZIP(v2,v6,w4,w5); HMX_ZIP(v3,v7,w6,w7); } while(0)
static inline __attribute__((always_inline)) void hmx_tile_half(HVX_Vector *dest,const uint8_t *const *p,int high) {
    HVX_Vector a0=HMX_SRC(p[0],high),a1=HMX_SRC(p[1],high),a2=HMX_SRC(p[2],high),a3=HMX_SRC(p[3],high);
    HVX_Vector a4=HMX_SRC(p[4],high),a5=HMX_SRC(p[5],high),a6=HMX_SRC(p[6],high),a7=HMX_SRC(p[7],high);
    HVX_Vector b0,b1,b2,b3,b4,b5,b6,b7;
    HMX_STAGE(a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7);
    HMX_STAGE(b0,b1,b2,b3,b4,b5,b6,b7,a0,a1,a2,a3,a4,a5,a6,a7);
    HMX_STAGE(a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7);
    dest[0]=b0;dest[1]=b1;dest[2]=b2;dest[3]=b3;dest[4]=b4;dest[5]=b5;dest[6]=b6;dest[7]=b7;
}
/* `zero` must address 128 zero bytes: the host clears the pad after the packet. */
static int hmx_prepare_fast(const QdConv *c,const QdHmxLayout *l,uint8_t *vtcm,uint32_t *table,uint32_t tid) {
    uint32_t depth=c->kh*c->kw*c->g;
    if(depth>QDC_PREPARE_TABLE-8) return 0;
    uint32_t k4=0;
    for(uint32_t ky=0;ky<c->kh;ky++)for(uint32_t kx=0;kx<c->kw;kx++)for(uint32_t g=0;g<c->g;g++,k4++)
        table[k4]=(ky*c->g+g)*c->wp*4+kx*4;
    for(;k4<l->kb*8;k4++)table[k4]=0;
    const uint8_t *x=vtcm,*zero=vtcm+c->h*c->g*c->wp*4;
    uint8_t *a=vtcm+l->a;
    uint32_t vecs=c->owp>>5,row=c->g*c->wp*4;
    uint32_t mb=0,kb=tid,oy=0,vx=0;
    while(kb>=l->kb){kb-=l->kb;mb++;if(++vx==vecs){vx=0;oy++;}}
    while(mb<l->mb) {
        const uint8_t *base=x+oy*row+vx*128;
        const uint32_t *t=table+kb*8;
        uint32_t k=kb*8;
        const uint8_t *p[8];
        for(uint32_t j=0;j<8;j++)p[j]=k+j<depth?base+t[j]:zero;
        HVX_Vector *dest=(HVX_Vector *)(a+(mb*l->kb+kb)*2048);
        hmx_tile_half(dest,p,0);
        hmx_tile_half(dest+8,p,1);
        kb+=3;
        while(kb>=l->kb){kb-=l->kb;mb++;if(++vx==vecs){vx=0;oy++;}}
    }
    return 1;
}
/* Activations packed [h][g/8][wp][32] (QDC_X32, g a multiple of 8): a tile row
 * is a pixel pair with its 32 K lanes interleaved by pixel, as the high byte of
 * each halfword. One 128-byte load (four pixels) yields two rows: a byte zip with
 * zero widens it, and a halfword shuffle interleaves the two pixels of each row.
 * K block kb is tap*(g/8)+cb, the same ordering as the weights. */
static int hmx_prepare_x32(const QdConv *c,const QdHmxLayout *l,uint8_t *vtcm,uint32_t *table,uint32_t tid) {
    uint32_t cbs=c->g>>3;
    if((c->g&7) || l->kb!=c->kh*c->kw*cbs || l->kb>QDC_PREPARE_TABLE) return 0;
    uint32_t k=0;
    for(uint32_t ky=0;ky<c->kh;ky++)for(uint32_t kx=0;kx<c->kw;kx++)for(uint32_t cb=0;cb<cbs;cb++,k++)
        table[k]=(ky*cbs+cb)*c->wp*32+kx*32;
    const uint8_t *x=vtcm;uint8_t *a=vtcm+l->a;
    uint32_t vecs=c->owp>>5,row=cbs*c->wp*32;
    uint32_t mb=0,kb=tid,oy=0,vx=0;
    while(kb>=l->kb){kb-=l->kb;mb++;if(++vx==vecs){vx=0;oy++;}}
    HVX_Vector zero=Q6_V_vzero();
    while(mb<l->mb) {
        const uint8_t *src=x+oy*row+vx*1024+table[kb];
        HVX_Vector *dest=(HVX_Vector *)(a+(mb*l->kb+kb)*2048);
        for(uint32_t q=0;q<8;q++) {
            HVX_VectorPair w=Q6_W_vshuff_VVR(*(const HVX_UVector *)(src+q*128),zero,-1);
            dest[2*q]=Q6_Vh_vshuff_Vh(Q6_V_lo_W(w));
            dest[2*q+1]=Q6_Vh_vshuff_Vh(Q6_V_hi_W(w));
        }
        kb+=3;
        while(kb>=l->kb){kb-=l->kb;mb++;if(++vx==vecs){vx=0;oy++;}}
    }
    return 1;
}
static int hmx_calculate(const QdConv *c,const QdHmxLayout *l,uint8_t *vtcm,const QdQuant *q) {
    int32_t *fi=(int32_t *)(vtcm+l->fi);
    for(uint32_t n=0;n<c->n;n++) {
        uint32_t p=q[n].preshift;
        fi[(n>>5)*64+(n&31)]=0x400000u|((16-p)<<10);
        fi[(n>>5)*64+32+(n&31)]=q[n].correction-(p?(1u<<(p-1)):0);
    }
    int32_t *zero=(int32_t *)(vtcm+l->zero);
    for(uint32_t n=0;n<64;n++)zero[n]=n<32?0x404000:0;
    uint8_t *completion=vtcm+l->completion;
    HVX_Vector sentinel=Q6_V_vsplat_R(-1);
    for(uint32_t i=0;i<2048;i+=128)*(HVX_Vector *)(completion+i)=sentinel;
    barrier();
    for(uint32_t m=0;m<l->mb;m++)for(uint32_t n=0;n<l->nb;n++)
        qd_hmx_wide(vtcm+l->a+m*l->kb*2048,vtcm+l->b+n*l->kb*1024,
                    fi+n*64,vtcm+l->c+(m*l->nb+n)*2048,l->kb);
    qd_hmx_finish(zero,completion);
    uint64_t end=cycles()+1000000000u;
    for(;;) {
        typedef uint32_t HmxWords __attribute__((vector_size(128)));
        HmxWords v=*(volatile HmxWords *)completion;
        if(v[0]==0)break;
        if(cycles()>end)return QDC_BARRIER_TIMEOUT;
    }
    barrier();return 0;
}
/* Complete quantization in HVX, emitting four pixel-major rows per vector.
 * Clamping pre-values at 4096 preserves u8 saturation for every supported
 * multiplier, and keeps the signed halfword multiply exact. */
static HVX_Vector hmx_quant_pair(HVX_Vector pre,HVX_Vector mult) {
    pre=Q6_Vuh_vmin_VuhVuh(pre,Q6_V_vsplat_R(4096|(4096<<16)));
    HVX_VectorPair product=Q6_Ww_vmpy_VhVh(pre,mult);
    HVX_Vector a=Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_V_lo_W(product),Q6_V_vsplat_R(8192)),14);
    HVX_Vector b=Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_V_hi_W(product),Q6_V_vsplat_R(8192)),14);
    return Q6_Vh_vpack_VwVw_sat(b,a);
}
static void hmx_quantize_to(const QdConv *c,const QdHmxLayout *l,uint8_t *vtcm,const QdQuant *q,uint8_t *output,uint32_t tid) {
    uint16_t multipliers[64] __attribute__((aligned(128)));
    uint32_t nb=tid,mb=0;
    while(nb>=l->nb){nb-=l->nb;mb++;}
    uint32_t blocked=c->reserved[0]&QDC_CROUTON8,height=(c->reserved[0]>>8)&255,width=c->reserved[0]>>16;
    uint32_t row=0,column=mb*32,stride=blocked?(c->oh==1?width:c->owp):1;
    if(blocked)while(column>=stride){column-=stride;row++;}
    for(uint32_t tile=tid;tile<l->mb*l->nb;tile+=3) {
        uint32_t n0=nb*32;
        for(uint32_t n=0;n<32;n++)multipliers[2*n]=multipliers[2*n+1]=(uint16_t)q[n0+n].multiplier;
        HVX_Vector mult=*(HVX_Vector *)multipliers;
        const HVX_Vector *src=(const HVX_Vector *)(vtcm+l->c+tile*2048);
        HVX_Vector *dst=(HVX_Vector *)(output+tile*1024);
        uint32_t h=row,w=column;
        for(uint32_t i=0;i<8;i++) {
            HVX_Vector a=hmx_quant_pair(src[2*i],mult),b=hmx_quant_pair(src[2*i+1],mult);
            HVX_Vector v=Q6_Vub_vpack_VhVh_sat(b,a);
            if(!blocked)dst[i]=v;
            else {
                if(h<height && w<width) {
                    uint32_t block=((h/8)*(width/8)+w/8)*l->nb+nb;
                    *(HVX_Vector *)(output+block*2048+(h%8)*256+(w%8)*32)=v;
                }
                w+=4;
                if(c->oh==1 && w==width){w=0;h++;}
            }
        }
        uint32_t prev_mb=mb;
        nb+=3;
        while(nb>=l->nb){nb-=l->nb;mb++;}
        if(blocked) {
            column+=(mb-prev_mb)*32;
            while(column>=stride){column-=stride;row++;}
        }
    }
}
#endif
