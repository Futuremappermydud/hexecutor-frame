#ifndef QD_HMX_LAYOUT_H
#define QD_HMX_LAYOUT_H
/* Logical 32-pixel tiles use the high-byte lanes of a 64x32 integer HMX tile.
 * A power-of-two wide conversion yields the exact reference pre-shift. */
typedef struct { uint32_t kb,mb,nb,a,b,c,q,fi,zero,completion,bytes; } QdHmxLayout;
static inline QdHmxLayout hmx_layout(const QdConv *c,uint32_t xb) {
    QdHmxLayout l;
    l.kb=(c->kh*c->kw*c->g+7)>>3;l.mb=c->oh*(c->owp>>5);l.nb=c->n>>5;
#ifdef QDC_NATIVE_BUILD
    if(c->version>=QDC_NATIVE_VERSION)l.mb=(l.mb+1)>>1;
#endif
    l.a=(xb+128+2047)&~2047u;
    l.b=l.a+l.mb*l.kb*2048;
    l.c=(l.b+l.nb*l.kb*1024+2047)&~2047u;
    l.q=l.c+l.mb*l.nb*2048;
    l.fi=l.q+l.mb*l.nb*1024;
#ifdef QDC_NATIVE_BUILD
    if(c->version>=QDC_NATIVE_VERSION)l.fi=l.q;
#endif
    l.zero=l.fi+((c->n*8+127)&~127u);
    l.completion=(l.zero+256+2047)&~2047u;l.bytes=l.completion+2048;
    return l;
}
#endif
