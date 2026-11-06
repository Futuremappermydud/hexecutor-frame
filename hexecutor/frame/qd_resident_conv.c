/* Reuse the exact validated convolution implementation. The surrounding graph
 * engine supplies resident inputs and a private staging arena; no host traffic
 * or service wake occurs between graph operations. */
#define payload unused_conv_entry
#define QDC_RESIDENT_BUILD 1
#include "qd_conv.c"
#undef payload
static uint32_t keys[3][3];
#if QDS_NATIVE_1X1
/* Called by the master only while every legacy request is quiescent. Native
 * leaves share HMX configuration with these workers. */
void qdrc_invalidate_native(void) {
    for(uint32_t t=0;t<3;t++)for(uint32_t k=0;k<3;k++)keys[t][k]=0;
    __asm__ volatile("barrier":::"memory");
}
#endif
uint32_t qdrc_execute(const QdConv *shared,uint8_t *arena,uint32_t tid,uint32_t prepared_a) {
    QdConv local;load_command(&local,(volatile QdConv *)shared);
    const QdConv *c=&local;
    Sizes s;uint32_t error=validate(c,&s);
    if(error)return error;
    if(prepared_a>1 || (prepared_a && !(c->reserved[0]&QDC_USE_HMX)))return QDC_BAD_SHAPE;
    /* Private worker stacks occupy L2 +512..704 KB. Parameter packets stay
     * at the top of L2; HMX prepare tables use the first 48 KB. */
    if(!(c->reserved[0]&QDC_USE_HMX) && round_chunk(s.wb)>0x80000u)return QDC_BAD_SHAPE;
    volatile uint32_t *counter=(volatile uint32_t *)(arena+QDC_BARRIER_OFFSET);
    if(c->reserved[0]&QDC_USE_HMX)
        hmx_conv(c,&s,arena,(uint8_t *)QDC_VTCM_BASE,(uint8_t *)QDC_L2TCM_BASE,counter,tid,&error,keys[tid],prepared_a);
    else {
        keys[tid][0]=0;
        conv(c,&s,arena,(uint8_t *)QDC_VTCM_BASE,(uint8_t *)QDC_L2TCM_BASE,counter,tid,&error);
    }
    return error;
}
uint32_t qdrc_power(uint32_t on) {
    uint8_t *a=(uint8_t *)0xf1800000u;
    QdHmxState *h=(QdHmxState *)(a+QDC_HMX_STATE_OFFSET);
    volatile uint32_t *s=(volatile uint32_t *)(a+QDC_CLOCK_OFFSET);
    if(on){uint32_t e=clock_raise(s);return e?e:hmx_start(h);}
    uint32_t e=hmx_shutdown(h),c=clock_restore(s);return e?e:c;
}
