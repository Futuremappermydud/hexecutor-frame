/* Offline capture of the legacy 4-tensor S2 worker from pinned V69. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>

#define PAGE 2048u
#define VTCM_BYTES 0x100000u
#define IN_PAGES 136u
#define OUT_PAGES 36u
#define TABLE_N 256u
#define FIXTURE_MAGIC 0x3253324eu
#ifdef S2_BASIS_FIXTURE
#define FIXTURE_PATH "/src/frame/native_runtime/stride2/s2_basis_fixture.bin"
#else
#define FIXTURE_PATH "/src/native_stride2/s2_fixture_224.bin"
#endif

typedef struct { uint32_t flags, dims[4], max_dims[4]; uint8_t padding[4]; } Shape4;
typedef struct { uint32_t vtable, interface_ptr, shape, data; } TensorObj;
typedef struct { uint32_t reserved, input, weights, fi, output; } LegacyOp;

extern void qd_v69_s2_wrapper(LegacyOp *);
extern void qd_v69_s2_generic_leaf(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
extern void qd_v69_s2_leaf_1xn(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
extern void qd_v69_s2_leaf_nx1(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
extern void qd_hmx_finish(void *,void *);
extern void qd_v69_s2_zero_stub(void), qd_v69_s2_identity_stub(void);
extern void qd_v69_s2_noop_stub(void);
extern void qd_v69_s2_dtype_in(void), qd_v69_s2_dtype_out(void), qd_v69_s2_dtype_qint8(void);

static uint8_t fixture[600000] __attribute__((aligned(128)));
static uint32_t output_vt[16], input_vt[16], weight_vt[16];
static uint32_t output_iface[32], input_iface[32], weight_iface[32], fi_iface[32];
static Shape4 input_shape, output_shape, weight_shape, fi_shape;
static TensorObj input_obj, output_obj, weight_obj, fi_obj;
static LegacyOp op;
volatile uint32_t qd_v69_s2_leaf_capture_args[6];
volatile uint32_t qd_v69_s2_leaf_selected;
volatile uint32_t qd_v69_s2_leaf_request_words[16];
volatile uint32_t qd_v69_s2_leaf_outdesc_words[16];
volatile uint32_t qd_v69_s2_leaf_config_words[16];
static volatile uint32_t *completion;
static uint8_t *out_pages;
static const uint8_t *reference_pages;
static uint32_t active_out_bytes;
static void dump_words(const char *label,uint32_t address,uint32_t words) {
    const volatile uint32_t *p=(const volatile uint32_t *)(uintptr_t)address;
    printf("%s",label);
    for(uint32_t i=0;i<words;i++)printf(" %08x",p[i]);
    putchar('\n');
}
static void dump_capture(const char *label,const volatile uint32_t *p,uint32_t words) {
    printf("%s",label);
    for(uint32_t i=0;i<words;i++)printf(" %08x",p[i]);
    putchar('\n');
}
static void clean(void *p,uint32_t n) {
    uintptr_t a=(uintptr_t)p,end=a+n;
    for(;a<end;a+=32u)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
}

int main(void) {
    FILE *f=fopen(FIXTURE_PATH,"rb");
    if(!f){puts("FAIL fixture open");return 1;}
    uint32_t h[48];
    if(fread(h,sizeof(h),1,f)!=1||h[0]!=FIXTURE_MAGIC){puts("FAIL fixture header");fclose(f);return 2;}
    uint32_t xbytes=h[15]*PAGE, wb=h[35], fb=h[36], ybytes=h[29]*PAGE;
    uint64_t payload=(uint64_t)xbytes+wb+fb+ybytes;
    if(h[1]!=224u||h[15]!=IN_PAGES||h[29]!=OUT_PAGES||payload>sizeof(fixture)||
       fread(fixture,(size_t)payload,1,f)!=1||fgetc(f)!=EOF){puts("FAIL fixture payload/geometry");fclose(f);return 3;}
    fclose(f);
    uint8_t *base=(uint8_t *)(uintptr_t)(__rdcfg(__vtcm_base)<<16);
    for(uint32_t i=0;i<VTCM_BYTES;i+=4096u)add_translation(base+i,base+i,6);
    memset(base,0,VTCM_BYTES);
    uint8_t *x=base,*y=base+0x50000,*w=base+0x62f00,*fi=base+0x68000;
    uint32_t *itab=(uint32_t *)(base+0x6a000),*otab=itab+TABLE_N;
    memcpy(x,fixture,xbytes);memcpy(w,fixture+xbytes,wb);memcpy(fi,fixture+xbytes+wb,fb);
    memset(y,0xa5,OUT_PAGES*PAGE);
    out_pages=y;reference_pages=fixture+xbytes+wb+fb;active_out_bytes=ybytes;completion=(volatile uint32_t *)(base+0x71000);
    memset(base+0x70000,0,0x100u);
    for(uint32_t i=0;i<TABLE_N;i++){itab[i]=(uint32_t)(uintptr_t)x;otab[i]=(uint32_t)(uintptr_t)(y+(OUT_PAGES-1u)*PAGE);}
    for(uint32_t i=0;i<IN_PAGES;i++)itab[i]=(uint32_t)(uintptr_t)(x+i*PAGE);
    for(uint32_t i=0;i<OUT_PAGES;i++)otab[i]=(uint32_t)(uintptr_t)(y+i*PAGE);

    output_vt[9]=(uint32_t)(uintptr_t)qd_v69_s2_dtype_out;
    input_vt[9]=(uint32_t)(uintptr_t)qd_v69_s2_dtype_in;
    weight_vt[9]=(uint32_t)(uintptr_t)qd_v69_s2_dtype_qint8;
    input_shape=(Shape4){0,{1,9,130,128},{1,16,136,128},{0,0,0,0}};
    output_shape=(Shape4){0,{1,8,128,64},{1,8,128,64},{0,0,0,0}};
    weight_shape=(Shape4){0,{3,3,32,64},{3,3,32,64},{0,0,0,0}};
    fi_shape=(Shape4){0,{1,2,1,64},{1,2,1,64},{0,0,0,0}};
    input_obj=(TensorObj){(uint32_t)(uintptr_t)input_vt,(uint32_t)(uintptr_t)input_iface,(uint32_t)(uintptr_t)&input_shape,(uint32_t)(uintptr_t)itab};
    output_obj=(TensorObj){(uint32_t)(uintptr_t)output_vt,(uint32_t)(uintptr_t)output_iface,(uint32_t)(uintptr_t)&output_shape,(uint32_t)(uintptr_t)otab};
    weight_obj=(TensorObj){(uint32_t)(uintptr_t)weight_vt,(uint32_t)(uintptr_t)weight_iface,(uint32_t)(uintptr_t)&weight_shape,(uint32_t)(uintptr_t)w};
    fi_obj=(TensorObj){0,(uint32_t)(uintptr_t)fi_iface,(uint32_t)(uintptr_t)&fi_shape,(uint32_t)(uintptr_t)fi};
    /* The wrapper forwards fields +0x10,+4,+8,+0xc to worker r0-r3. */
    op=(LegacyOp){0,(uint32_t)(uintptr_t)&input_obj,(uint32_t)(uintptr_t)&weight_obj,
                  (uint32_t)(uintptr_t)&fi_obj,(uint32_t)(uintptr_t)&output_obj};
    clean(base,VTCM_BYTES);clean(&input_shape,sizeof(input_shape));clean(&output_shape,sizeof(output_shape));
    clean(&weight_shape,sizeof(weight_shape));clean(&fi_shape,sizeof(fi_shape));
    clean(&input_obj,sizeof(input_obj));clean(&output_obj,sizeof(output_obj));clean(&weight_obj,sizeof(weight_obj));clean(&fi_obj,sizeof(fi_obj));
    clean(output_vt,sizeof(output_vt));clean(input_vt,sizeof(input_vt));clean(weight_vt,sizeof(weight_vt));
    clean(input_iface,sizeof(input_iface));clean(output_iface,sizeof(output_iface));clean(weight_iface,sizeof(weight_iface));clean(fi_iface,sizeof(fi_iface));
    clean(&op,sizeof(op));clean((void *)qd_v69_s2_leaf_capture_args,sizeof(qd_v69_s2_leaf_capture_args));
    clean((void *)qd_v69_s2_leaf_request_words,sizeof(qd_v69_s2_leaf_request_words));
    clean((void *)qd_v69_s2_leaf_outdesc_words,sizeof(qd_v69_s2_leaf_outdesc_words));
    clean((void *)qd_v69_s2_leaf_config_words,sizeof(qd_v69_s2_leaf_config_words));
    clean((void *)&qd_v69_s2_leaf_selected,sizeof(qd_v69_s2_leaf_selected));__asm__ volatile("barrier":::"memory");
    uint32_t old_ssr;__asm__ volatile("%0=ssr":"=r"(old_ssr));
    __asm__ volatile("ssr=%0\n isync"::"r"(old_ssr|(1u<<26)):"memory");
    qd_v69_s2_wrapper(&op);
#ifndef S2_CAPTURE_ONLY
    *completion=0xffffffffu;
    qd_hmx_finish(base+0x70000u,(void *)completion);
    uint32_t spins=0;while(*completion&&spins++<100001u)__asm__ volatile("nop");
#else
    *completion=0u;
#endif
    __asm__ volatile("ssr=%0\n isync"::"r"(old_ssr):"memory");
    printf("S2 legacy capture selected=%u callbacks=%u X=%08x W=%08x FI=%08x Y=%08x\n",
           qd_v69_s2_leaf_selected,qd_v69_s2_leaf_selected!=0,(uint32_t)(uintptr_t)x,(uint32_t)(uintptr_t)w,(uint32_t)(uintptr_t)fi,(uint32_t)(uintptr_t)y);
    printf("leaf args");for(uint32_t a=0;a<6u;a++)printf(" %08x",qd_v69_s2_leaf_capture_args[a]);putchar('\n');
    dump_capture("entry request",qd_v69_s2_leaf_request_words,16u);
    dump_capture("entry output-desc",qd_v69_s2_leaf_outdesc_words,16u);
    dump_capture("entry hmx-config",qd_v69_s2_leaf_config_words,16u);
    dump_words("request words",qd_v69_s2_leaf_capture_args[0],16u);
    dump_words("output-desc words",qd_v69_s2_leaf_capture_args[1],16u);
    dump_words("hmx-config words",qd_v69_s2_leaf_capture_args[4],16u);
    printf("shapes X=1,9,130,128 max=1,16,136,128 Y=1,8,128,64 W=3,3,32,64; raw=%u FI=%u\n",wb,fb);
    uint32_t bad=0,plus=0,minus=0,other=0;
    for(uint32_t i=0;i<active_out_bytes;i++){
      int d=(int)out_pages[i]-(int)reference_pages[i];
      if(d){bad++;if(d==1)plus++;else if(d==-1)minus++;else other++;}
    }
    printf("S2 legacy output bytes=%u mismatches=%u +1=%u -1=%u other=%u completion=%u\n",active_out_bytes,bad,plus,minus,other,*completion);
    printf("first Y bytes");for(uint32_t a=0;a<32u;a++)printf(" %u/%u",out_pages[a],reference_pages[a]);putchar('\n');
    FILE *dump=fopen("/src/frame/native_runtime/stride2/v69_s2_output_224.bin","wb");
    if(dump){fwrite(out_pages,active_out_bytes,1,dump);fclose(dump);}
#ifdef S2_CAPTURE_ONLY
    return qd_v69_s2_leaf_selected?0:4;
#else
    return qd_v69_s2_leaf_selected&&!*completion&&!minus&&!other?0:4;
#endif
}
