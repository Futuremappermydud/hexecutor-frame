/* Q4_0 GGUF matrix-vector multiply on Frame HMX.
 * The ARM host supplies quantized rows and one FP16 vector. All dequantization,
 * matrix arithmetic and result conversion happen on the CDSP.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "../../benchmarks/hmx_power.h"

#define ARENA 0xf1800000u
#define ARENA_CACHED 0x88800000u
#define INPUT (ARENA + 0x20000u)
#define WEIGHTS (ARENA + 0xc0000u)
#define MAGIC 0x34514d48u
#define VTCM 0x31000000u
typedef uint32_t V __attribute__((vector_size(128)));
extern void hmx_tune_init(void *scale);
extern void hmx_tune_finish(void *completion);
extern void hmx_tune_gemm(void *a, void *b, void *c, uint32_t k_bytes, uint32_t n_tiles, uint32_t m_tiles);
extern void hmx_tune_deep(void *a, void *b, void *c, uint32_t range, uint32_t flags);

typedef struct { uint32_t pll[16], cfg, branch; } ClockSaved;

static uint32_t clock_set(ClockSaved *save) {
    volatile uint32_t *pll=(volatile uint32_t *)0x32341000u;
    volatile uint32_t *cc=(volatile uint32_t *)0x32348000u;
    for (uint32_t i=0;i<16;i++) save->pll[i]=pll[i];
    save->cfg=cc[0x184/4];save->branch=cc[0x1a0/4];
    if (save->cfg!=0 || (save->pll[0]&5) || save->pll[1]!=0) return 2;
    pll[0]=0;pll[4]=0x44440033u;pll[5]=0;pll[6]=0x00008401u;
    pll[1]=0;pll[0]=0x00800000u;pll[0]=0x00800004u;pll[1]=1;
    uint64_t end=cycles()+200000000u;
    while (!(pll[0]&0x80000000u)) if (cycles()>end) return 3;
    pll[0]=5;
    __asm__ volatile("syncht" ::: "memory");
    cc[0x184/4]=0x200;cc[0x180/4]=1;
    end=cycles()+200000000u;
    while (cc[0x180/4]&1) if (cycles()>end) return 4;
    __asm__ volatile("syncht" ::: "memory");
    return 0;
}

static void clock_restore(const ClockSaved *save) {
    volatile uint32_t *pll=(volatile uint32_t *)0x32341000u;
    volatile uint32_t *cc=(volatile uint32_t *)0x32348000u;
    cc[0x184/4]=save->cfg;cc[0x180/4]=1;
    uint64_t end=cycles()+200000000u;
    while (cc[0x180/4]&1) if (cycles()>end) break;
    cc[0x1a0/4]=save->branch;
    __asm__ volatile("syncht" ::: "memory");
    pll[0]=0;pll[1]=0;
    pll[4]=save->pll[4];pll[5]=save->pll[5];pll[6]=save->pll[6];
    pll[1]=save->pll[1];pll[0]=save->pll[0];
    __asm__ volatile("syncht" ::: "memory");
}

static float half_to_float(uint16_t h) {
    union { uint32_t u; float f; } v;
    uint32_t sign=((uint32_t)h&0x8000u)<<16;
    uint32_t exp=(h>>10)&31u, mant=h&1023u;
    if (!exp) {
        float x=(float)mant * 0x1p-24f;
        return sign ? -x : x;
    }
    v.u=sign|((exp+112u)<<23)|(mant<<13);
    return v.f;
}

static uint16_t float_to_half(float f) {
    union { uint32_t u; float f; } v;
    v.f=f;
    uint32_t sign=(v.u>>16)&0x8000u;
    int32_t exp=(int32_t)((v.u>>23)&255u)-112;
    uint32_t mant=v.u&0x7fffffu;
    if (exp<=0) {
        if (exp < -10) return (uint16_t)sign;
        uint32_t x=(mant|0x800000u) >> (uint32_t)(1-exp);
        uint32_t rem=x&0x1fffu;
        x=(x>>13)+(rem>0x1000u || (rem==0x1000u && ((x>>13)&1u)));
        return (uint16_t)(sign|x);
    }
    if (exp>=31) return (uint16_t)(sign|0x7c00u);
    uint32_t rem=mant&0x1fffu;
    uint32_t body=((uint32_t)exp<<10)|(mant>>13);
    body+=rem>0x1000u || (rem==0x1000u && (body&1u));
    return (uint16_t)(sign|body);
}

/* The Q4_0 scale is FP16 and each code is a small integer. This avoids
 * software FP multiplication for every quantized weight on Hexagon. */
static uint16_t half_mul_q4(uint16_t scale, int32_t q) {
    if (!q || !(scale&0x7fffu)) return 0;
    uint32_t sign=(scale&0x8000u) ^ (q<0 ? 0x8000u : 0u);
    uint32_t mag=(uint32_t)(q<0 ? -q : q);
    uint32_t exponent=(scale>>10)&31u;
    uint32_t mant=scale&1023u;
    if (exponent==31u) return (uint16_t)(sign|0x7c00u);
    uint32_t product=(exponent ? 1024u+mant : mant)*mag;
    uint32_t highest=31u-(uint32_t)__builtin_clz(product);
    if (!exponent) {
        if (product<1024u) return (uint16_t)(sign|product);
        exponent=highest-9u;
    } else {
        exponent+=highest-10u;
    }
    uint32_t shift=highest-10u;
    uint32_t rounded=product>>shift;
    if (shift) {
        uint32_t remainder=product&((1u<<shift)-1u);
        uint32_t midpoint=1u<<(shift-1u);
        rounded+=remainder>midpoint || (remainder==midpoint && (rounded&1u));
    }
    if (rounded==2048u) {rounded=1024u;exponent++;}
    if (exponent>=31u) return (uint16_t)(sign|0x7c00u);
    return (uint16_t)(sign|(exponent<<10)|(rounded&1023u));
}

static void invalidate_cached(uint32_t address, uint32_t bytes) {
    uint32_t start=address&~31u, end=address+bytes;
    for (uint32_t p=start;p<end;p+=32u)
        __asm__ volatile("dcinva(%0)" :: "r"(p) : "memory");
    __asm__ volatile("barrier" ::: "memory");
}

static void pack_activations(volatile uint16_t *a, const float *x,
                             uint32_t full_k, uint32_t start, uint32_t step, uint32_t rows) {
    for (uint32_t kt=0;kt<step/32;kt++) {
        for (uint32_t rp=0;rp<16;rp++) {
            uint16_t row[64] __attribute__((aligned(128)));
            uint32_t r0=rp*2u,r1=r0+1u;
            for (uint32_t ki=0;ki<32;ki++) {
                uint32_t col=start+kt*32u+ki;
                row[ki*2]=r0<rows ? float_to_half(x[r0*full_k+col]) : 0;
                row[ki*2+1]=r1<rows ? float_to_half(x[r1*full_k+col]) : 0;
            }
            ((HVX_Vector *)a)[kt*16+rp]=*(const HVX_Vector *)row;
        }
    }
}

static void dequant_pack_weights(volatile uint16_t *b, const uint8_t *q4,
                                  uint32_t full_k, uint32_t start, uint32_t step, uint32_t n) {
    uint32_t kb=step/32, row_bytes=(full_k/32)*18;
    for (uint32_t nt=0;nt<n/32;nt++) {
        for (uint32_t kt=0;kt<kb;kt++) {
            uint8_t blocks[32][16];
            uint16_t values[32][16];
            for (uint32_t ni=0;ni<32;ni++) {
                const uint8_t *block=q4+(nt*32+ni)*row_bytes+(start/32+kt)*18;
                uint16_t dh=(uint16_t)block[0]|((uint16_t)block[1]<<8);
                for (uint32_t q=0;q<16;q++) values[ni][q]=half_mul_q4(dh,(int32_t)q-8);
                for (uint32_t i=0;i<16;i++) blocks[ni][i]=block[2+i];
            }
            for (uint32_t kp=0;kp<16;kp++) {
                uint16_t row[64] __attribute__((aligned(128)));
                for (uint32_t ni=0;ni<32;ni++) {
                    uint8_t q0=blocks[ni][(kp*2)&15u];
                    uint8_t q1=blocks[ni][(kp*2+1u)&15u];
                    uint32_t v0=kp<8 ? (q0&15u) : (q0>>4);
                    uint32_t v1=kp<8 ? (q1&15u) : (q1>>4);
                    row[ni*2]=values[ni][v0];
                    row[ni*2+1]=values[ni][v1];
                }
                ((HVX_Vector *)b)[(nt*kb+kt)*16+kp]=*(const HVX_Vector *)row;
            }
        }
    }
    __asm__ volatile("syncht" ::: "memory");
}

/* Model conversion has already arranged FP16 weights in HMX B tile order. */
static void copy_packed_weights(volatile uint16_t *b, const uint8_t *packed,
                                uint32_t full_k, uint32_t start, uint32_t step, uint32_t n) {
    uint32_t all_kb=full_k/32u, kb=step/32u;
    for (uint32_t nt=0;nt<n/32u;nt++) {
        for (uint32_t kt=0;kt<kb;kt++) {
            const HVX_Vector *source=(const HVX_Vector *)(packed+
                ((nt*all_kb)+(start/32u)+kt)*16u*128u);
            volatile HVX_Vector *destination=(volatile HVX_Vector *)b+(nt*kb+kt)*16u;
            for (uint32_t pair=0;pair<16u;pair++) destination[pair]=source[pair];
        }
    }
    __asm__ volatile("syncht" ::: "memory");
}

/* 32x32 Q4 tiles, prepared offline in the layout used by llama.cpp's
 * Hexagon backend. Four 128-byte code vectors and 32 FP16 row scales expand
 * into the same 16-vector HMX B tile used by the FP16 path. */
static void dequant_packed_weights(volatile uint16_t *b, const uint8_t *packed,
                                   uint32_t full_k, uint32_t start, uint32_t step, uint32_t n) {
    uint32_t all_kb=full_k/32u,kb=step/32u;
    HVX_Vector mask=Q6_Vb_vsplat_R(15), eight=Q6_Vb_vsplat_R(8);
    for (uint32_t nt=0;nt<n/32u;nt++) for (uint32_t kt=0;kt<kb;kt++) {
        const uint8_t *tile=packed+((nt*all_kb)+(start/32u)+kt)*640u;
        HVX_Vector scales=*(const HVX_Vector *)(tile+512u);
        HVX_Vector scale=Q6_V_lo_W(Q6_W_vshuff_VVR(scales,scales,-2));
        volatile HVX_Vector *dest=(volatile HVX_Vector *)b+(nt*kb+kt)*16u;
        for (uint32_t group=0;group<4u;group++) {
            HVX_Vector codes=*(const HVX_Vector *)(tile+group*128u);
            HVX_Vector low=Q6_Vb_vsub_VbVb(Q6_V_vand_VV(codes,mask),eight);
            HVX_Vector high=Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(codes,4),eight);
            HVX_VectorPair shuffled=Q6_W_vshuff_VVR(high,low,-1);
            HVX_VectorPair unpacked0=Q6_Wh_vunpack_Vb(Q6_V_lo_W(shuffled));
            HVX_VectorPair unpacked1=Q6_Wh_vunpack_Vb(Q6_V_hi_W(shuffled));
            if (nt==0u && kt==0u && group==0u) {
                *(volatile HVX_Vector *)(ARENA+0x3000u)=scale;
                *(volatile HVX_Vector *)(ARENA+0x3080u)=codes;
                *(volatile HVX_Vector *)(ARENA+0x3100u)=Q6_V_lo_W(unpacked0);
                *(volatile HVX_Vector *)(ARENA+0x3180u)=Q6_Vhf_vcvt_Vh(Q6_V_lo_W(unpacked0));
            }
            dest[group*4u+0u]=Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(
                Q6_Vhf_vcvt_Vh(Q6_V_lo_W(unpacked0)),scale));
            dest[group*4u+1u]=Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(
                Q6_Vhf_vcvt_Vh(Q6_V_hi_W(unpacked0)),scale));
            dest[group*4u+2u]=Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(
                Q6_Vhf_vcvt_Vh(Q6_V_lo_W(unpacked1)),scale));
            dest[group*4u+3u]=Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(
                Q6_Vhf_vcvt_Vh(Q6_V_hi_W(unpacked1)),scale));
        }
    }
    __asm__ volatile("syncht" ::: "memory");
}

static float reciprocal_sqrt(float value) {
    union { uint32_t bits; float number; } u;
    u.number=value;
    u.bits=0x5f3759dfu-(u.bits>>1);
    float y=u.number, half=0.5f*value;
    y=y*(1.5f-half*y*y);
    y=y*(1.5f-half*y*y);
    y=y*(1.5f-half*y*y);
    return y;
}

static float exp_approx(float value) {
    if (value < -80.0f) return 0.0f;
    if (value > 80.0f) value=80.0f;
    float scaled=value*1.4426950408889634f;
    int32_t nearest=(int32_t)(scaled+(scaled>=0.0f ? 0.5f : -0.5f));
    float r=scaled-(float)nearest;
    float polynomial=1.0f+r*(0.69314718056f+r*(0.24022650695f+
        r*(0.05550410866f+r*(0.00961812911f+r*(0.00133335581f+
        r*0.00015403530f)))));
    union { uint32_t bits; float number; } scale;
    scale.bits=(uint32_t)(nearest+127)<<23;
    return scale.number*polynomial;
}

static uint32_t run_scalar_op(uint32_t op, volatile uint32_t *cmd) {
    volatile float *output=(volatile float *)INPUT;
    const float *input=(const float *)(ARENA_CACHED+0x20000u);
    const float *weights=(const float *)(ARENA_CACHED+0xc0000u);
    if (op==1u) { /* RMSNorm: width, groups, shared per-group weights */
        uint32_t width=cmd[4],groups=cmd[5];
        if ((width!=1024u && width!=128u) || !groups || groups>512u || width*groups>65536u) return 21;
        invalidate_cached(ARENA_CACHED+0x20000u,width*groups*4u);
        invalidate_cached(ARENA_CACHED+0xc0000u,width*4u);
        for (uint32_t g=0;g<groups;g++) {
            float sum=0.0f;
            for (uint32_t i=0;i<width;i++) {float x=input[g*width+i];sum+=x*x;}
            float factor=reciprocal_sqrt(sum/(float)width+0.000001f);
            for (uint32_t i=0;i<width;i++) output[g*width+i]=input[g*width+i]*factor*weights[i];
        }
    } else if (op==2u) { /* Residual add */
        uint32_t count=cmd[4];
        if (!count || count>98304u) return 22;
        invalidate_cached(ARENA_CACHED+0x20000u,count*4u);
        invalidate_cached(ARENA_CACHED+0xc0000u,count*4u);
        for (uint32_t i=0;i<count;i++) output[i]=input[i]+weights[i];
    } else if (op==3u) { /* SwiGLU */
        uint32_t count=cmd[4];
        if (!count || count>98304u) return 23;
        invalidate_cached(ARENA_CACHED+0x20000u,count*4u);
        invalidate_cached(ARENA_CACHED+0xc0000u,count*4u);
        for (uint32_t i=0;i<count;i++) {
            float x=input[i];
            output[i]=(x/(1.0f+exp_approx(-x)))*weights[i];
        }
    } else if (op==4u) { /* Qwen3 NeoX RoPE: pair the two 64-value head halves */
        uint32_t rows=cmd[4] ? cmd[4] : 1u;
        if (rows>32u) return 24;
        invalidate_cached(ARENA_CACHED+0x20000u,rows*3072u*4u);
        invalidate_cached(ARENA_CACHED+0xc0000u,rows*128u*4u);
        for (uint32_t row=0;row<rows;row++) {
            for (uint32_t head=0;head<24u;head++) {
                uint32_t base=row*3072u+head*128u;
                for (uint32_t i=0;i<64u;i++) {
                    float even=input[base+i], odd=input[base+64u+i];
                    float cosine=weights[row*128u+2u*i],sine=weights[row*128u+2u*i+1u];
                    output[base+i]=even*cosine-odd*sine;
                    output[base+64u+i]=even*sine+odd*cosine;
                }
            }
        }
    } else if (op==5u) { /* GQA attention, prior KV at WEIGHTS as [T,2,8,128] */
        uint32_t prior=cmd[4],rows=cmd[5] ? cmd[5] : 1u;
        if (rows>32u || prior+rows>257u) return 25;
        invalidate_cached(ARENA_CACHED+0x20000u,rows*4096u*4u);
        invalidate_cached(ARENA_CACHED+0xc0000u,prior*2048u*4u);
        float scores[257];
        for (uint32_t row=0;row<rows;row++) for (uint32_t head=0;head<16u;head++) {
            const float *query=input+row*2048u;
            uint32_t kvhead=head/2u;
            float maximum=-1.0e30f;
            for (uint32_t t=0;t<=prior+row;t++) {
                const float *key=t>=prior ? input+rows*2048u+(t-prior)*1024u+kvhead*128u :
                    weights+t*2048u+kvhead*128u;
                float dot=0.0f;
                for (uint32_t i=0;i<128u;i++) dot+=query[head*128u+i]*key[i];
                float score=dot*0.08838834764831845f;
                scores[t]=score;
                if (score>maximum) maximum=score;
            }
            float denominator=0.0f;
            for (uint32_t t=0;t<=prior+row;t++) {
                scores[t]=exp_approx(scores[t]-maximum);
                denominator+=scores[t];
            }
            float inv=1.0f/denominator;
            for (uint32_t i=0;i<128u;i++) {
                float sum=0.0f;
                for (uint32_t t=0;t<=prior+row;t++) {
                    const float *value=t>=prior ? input+rows*3072u+(t-prior)*1024u+kvhead*128u :
                        weights+t*2048u+1024u+kvhead*128u;
                    sum+=scores[t]*value[i];
                }
                output[row*2048u+head*128u+i]=sum*inv;
            }
        }
    } else if (op==6u) { /* Q4_0 token embedding row to FP32 */
        uint32_t width=cmd[4],rows=cmd[5] ? cmd[5] : 1u;
        if (width!=1024u || rows>32u) return 26;
        invalidate_cached(ARENA_CACHED+0xc0000u,rows*(width/32u)*18u);
        const uint8_t *raw=(const uint8_t *)(ARENA_CACHED+0xc0000u);
        for (uint32_t row=0;row<rows;row++) for (uint32_t block=0;block<width/32u;block++) {
            const uint8_t *q=raw+(row*width/32u+block)*18u;
            uint16_t d=(uint16_t)q[0]|((uint16_t)q[1]<<8);
            for (uint32_t i=0;i<16u;i++) {
                uint8_t packed=q[2u+i];
                output[row*width+block*32u+i]=half_to_float(half_mul_q4(d,(int32_t)(packed&15u)-8));
                output[row*width+block*32u+i+16u]=half_to_float(half_mul_q4(d,(int32_t)(packed>>4)-8));
            }
        }
    } else return 20;
    __asm__ volatile("syncht" ::: "memory");
    return 0;
}

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid, ssr=0;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    if (tid>=3) return;
    volatile uint32_t *cmd=(volatile uint32_t *)ARENA;
    volatile uint32_t *st=(volatile uint32_t *)(ARENA+128+16*tid);
    volatile uint32_t *out=(volatile uint32_t *)(ARENA+2048);
    uint32_t last_seq=0;
    for (;;) {
    if (last_seq) {
        while (cmd[2]==last_seq) { }
    }
    uint32_t seq=cmd[2];
    if (last_seq || cmd[1]==2u) {
        if (cmd[9]!=(~seq) ||
            !((cmd[0]==MAGIC && cmd[1]==2u) ||
              (cmd[0]==0x32564451u && cmd[3]==2u)))
            continue;
    }
    last_seq=seq;
    if (cmd[0]==0x32564451u && cmd[3]==2u) {
        st[3]=0x54495845u;st[0]=seq;
        __asm__ volatile("barrier" ::: "memory");
        return;
    }
    uint32_t persistent=cmd[0]==MAGIC && cmd[1]==2u;
    if (tid) {
        st[3]=persistent ? 0u : 0x54495845u;
        st[0]=seq;
        __asm__ volatile("barrier" ::: "memory");
        if (!persistent) return;
        continue;
    }
    out[0]=1;
    out[1]=0;
    out[15]=seq;
    uint64_t deadline=cycles()+2000000000u;
    while (*(volatile uint32_t *)(ARENA+144)!=seq || *(volatile uint32_t *)(ARENA+160)!=seq) {
        if (cycles()>deadline) {out[1]=1;goto done;}
    }
    if (cmd[3]) {
        out[0]=2;
        out[1]=run_scalar_op(cmd[3],cmd);
        out[0]=6;
        goto done;
    }
    uint32_t k=cmd[4],n=cmd[5],rows=cmd[7] ? cmd[7] : 1u;
    uint32_t weight_format=cmd[8];
    uint32_t slice_k=k<1024u ? k : 1024u;
    uint32_t ab=32*slice_k*2,bb=slice_k*n*2,cb=32*n*2;
    uint32_t weight_bytes=weight_format==1u ? k*n*2u :
        weight_format==2u ? (k/32u)*(n/32u)*640u : (k/32u)*n*18u;
    if (cmd[0]!=MAGIC || !k || !n || k>3072 || n>3072 || rows>32u || ((k|n)&31u) ||
        weight_format>2u || ab+bb+cb+4096u+rows*n*4u>0x800000u || weight_bytes>0x740000u) {
        out[1]=2;goto done;
    }
    volatile uint16_t *a=(volatile uint16_t *)VTCM;
    volatile uint16_t *b=(volatile uint16_t *)(VTCM+ab);
    volatile uint16_t *c=(volatile uint16_t *)(VTCM+ab+bb);
    volatile V *scale=(volatile V *)(VTCM+ab+bb+cb);
    volatile V *completion=scale+16;
    volatile float *sums=(volatile float *)(VTCM+ab+bb+cb+4096u);
    out[0]=2;
    invalidate_cached(ARENA_CACHED+0x20000u,rows*k*4u);
    if (weight_format!=1u) invalidate_cached(ARENA_CACHED+0xc0000u,weight_bytes);
    out[13]=out[14]=0;
    V zero, scales;
    for (uint32_t i=0;i<32;i++) {zero[i]=0;scales[i]=0x00003c00u;}
    for (uint32_t i=0;i<cb/128;i++) ((volatile V *)c)[i]=zero;
    scale[0]=scales;scale[1]=zero;
    for (uint32_t i=0;i<16;i++) completion[i]=zero;
    volatile uint32_t *rsc=(volatile uint32_t *)0x323c0000u;
    uint32_t saved[17],trig=rsc[0x460/4],addr=rsc[0x464/4];
    if (rsc[0]!=0x30200u || (rsc[0x404/4]&1u)) {out[1]=3;goto done;}
    uint32_t occupied=0;
    for (uint32_t i=0;i<17;i++) {saved[i]=rsc[0x550/4+i];occupied|=saved[i];}
    if (occupied) {out[1]=4;goto done;}
    for (uint32_t i=0;i<17;i++) rsc[0x550/4+i]=hmx_power_words[i];
    __asm__ volatile("syncht" ::: "memory");
    out[2]=power_trigger(rsc,0);out[3]=rsc[0x408/4];
    if (out[2] || out[3]!=32u) {out[1]=5;goto restore;}
    ClockSaved clocks;
    out[4]=clock_set(&clocks);
    if (out[4]) {out[1]=7;if (out[4]>=3) clock_restore(&clocks);goto power_off;}
    __asm__ volatile("%0 = ssr" : "=r"(ssr));
    __asm__ volatile("ssr = %0\n isync" :: "r"(ssr|(1u<<26)) : "memory");
    hmx_tune_init((void *)scale);
    uint64_t t0=cycles();
    for (uint32_t i=0;i<rows*n;i++) sums[i]=0.0f;
    for (uint32_t start=cmd[6];start<k;start+=1024u) {
        uint32_t step=k-start<1024u ? k-start : 1024u;
        uint64_t phase=cycles();
        pack_activations(a,(const float *)(ARENA_CACHED+0x20000u),k,start,step,rows);
        out[13]+=(uint32_t)(cycles()-phase);
        phase=cycles();
        if (weight_format==1u)
            copy_packed_weights(b,(const uint8_t *)(ARENA+0xc0000u),k,start,step,n);
        else if (weight_format==2u)
            dequant_packed_weights(b,(const uint8_t *)(ARENA_CACHED+0xc0000u),k,start,step,n);
        else
            dequant_pack_weights(b,(const uint8_t *)(ARENA_CACHED+0xc0000u),k,start,step,n);
        if (weight_format==2u) {
            out[20]=((const volatile uint32_t *)b)[0];
            out[21]=((const volatile uint32_t *)b)[1];
            out[22]=((const volatile uint32_t *)b)[2];
            out[23]=((const volatile uint32_t *)b)[3];
            out[24]=((const uint32_t *)(ARENA_CACHED+0xc0000u+512u))[0];
        }
        out[14]+=(uint32_t)(cycles()-phase);
        uint32_t step_bytes=(step/32u)*2048u;
        for (uint32_t nt=0;nt<n/32u;nt++) {
            hmx_tune_deep((void *)a,
                          (void *)(b+nt*(step/32u)*1024u),
                          (void *)(c+nt*1024u),step_bytes-1u,1u);
        }
        hmx_tune_finish((void *)completion);
        for (uint32_t row=0;row<rows;row++) {
            for (uint32_t no=0;no<n;no++) {
                uint32_t index=(no/32u)*1024u+(row/2u)*64u+(no%32u)*2u+(row&1u);
                sums[row*n+no]+=half_to_float(c[index]);
            }
        }
        uint32_t part=start/1024u;
        out[16+part*2]=c[0];
        union { float f; uint32_t u; } sample;
        sample.f=sums[0];
        out[17+part*2]=sample.u;
    }
    uint64_t elapsed=cycles()-t0;
    out[5]=(uint32_t)elapsed;out[6]=(uint32_t)(elapsed>>32);
    out[7]=completion[0][0];
    for (uint32_t i=0;i<rows*n;i++) ((volatile float *)INPUT)[i]=sums[i];
    __asm__ volatile("syncht" ::: "memory");
    __asm__ volatile("ssr = %0\n isync" :: "r"(ssr) : "memory");
    clock_restore(&clocks);
    out[8]=*(volatile uint32_t *)0x32348184u;
    out[9]=*(volatile uint32_t *)0x32341000u;
    out[10]=(out[8]!=clocks.cfg || out[9]!=clocks.pll[0]);
power_off:
    out[11]=power_trigger(rsc,32);out[12]=rsc[0x408/4];
    if (out[11] || out[12]!=68u) out[1]=6;
restore:
    if (!(rsc[0x404/4]&1u)) {
        for (uint32_t i=0;i<17;i++) rsc[0x550/4+i]=saved[i];
        rsc[0x464/4]=addr;rsc[0x460/4]=trig;
        __asm__ volatile("syncht" ::: "memory");
    }
    out[0]=6;
done:
    __asm__ volatile("barrier" ::: "memory");
    st[3]=persistent ? 0u : 0x54495845u;st[0]=seq;
    __asm__ volatile("barrier" ::: "memory");
    if (!persistent) return;
    }
}
