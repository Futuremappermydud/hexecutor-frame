/* Focused sim harness for the u16 native HMX conv (hmx_native16): builds
 * synthetic blocked u16 tensors in VTCM, calls hmx_native16 directly and
 * prints, per config, the path taken, a hash of the conv's OUTPUT blocks
 * only, and pcycles/phase cycles. Build once per -DN16_* variant; compare
 * hashes across builds. */
#include "../../native_program/prepared_reference.c"
#include <stdio.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
#include <hexagon_sim_timer.h>
extern void qd_hmx_wide(void *,void *,void *,void *,uint32_t);
extern void qd_hmx_finish(void *,void *);
extern void qd_hmx_wide_n(void *,void *,void *,void *,uint32_t,uint32_t);
extern void qd_hmx_native_n(void *,void *,void *,void *const *,uint32_t,uint32_t);
extern void qd_hmx_native16_n(void *,void *,void *,void *const *,uint32_t,uint32_t);
extern void qd_hmx_slide3_n(const uint32_t *,const void *,const void *,const uint32_t *,uint32_t,uint32_t);
static uint32_t enter(void){uint32_t s;__asm__ volatile("%0=ssr":"=r"(s));uint32_t on=s|(1u<<26);__asm__ volatile("ssr=%0\n isync\n barrier"::"r"(on):"memory");return s;}
static void leave(uint32_t s){__asm__ volatile("ssr=%0\n isync\n barrier"::"r"(s):"memory");}
static uint64_t clk(void){return hexagon_sim_read_pcycles();}
static const Nprg2HmxOps ops={enter,leave,qd_hmx_wide,qd_hmx_finish,clk,qd_hmx_wide_n,qd_hmx_native_n,qd_hmx_native16_n,qd_hmx_slide3_n};
static uint32_t rs=987654321;static uint32_t rnd(void){rs^=rs<<13;rs^=rs>>17;rs^=rs<<5;return rs;}
static uint32_t *mkblocks(uint8_t *pool,uint32_t *at,int nbh,int nbw,int nbd,int fill,int holes){
    int n=nbh*nbw*nbd;uint32_t *b=malloc(n*4);
    for(int i=0;i<n;i++){
        *at=(*at+2047)&~2047u;
        if(holes && (rnd()%29)==0){b[i]=0;continue;}
        b[i]=(*at)|1u;
        if(fill) for(int k=0;k<2048;k+=2)*(uint16_t *)(pool+*at+k)=(uint16_t)(rnd()%3==0?0:rnd());
        else memset(pool+*at,0xee,2048);
        *at+=2048;
    }
    return b;
}
static void run(uint8_t *vt,int cfg,int cbs,int NB,int oh,int ow,int p1,int p2,int packed,int reverse,int holes){
    const int I=cbs*32,co=NB*32,H=3,W=3;
    memset(vt,0,0x100000);
    uint32_t at=0;
    int ih=p1+oh+2,iw=p2+ow+2;
    T xt;memset(&xt,0,sizeof xt);
    xt.flags=F_BLOCKED;xt.esize=2;xt.pool=1;xt.m[0]=1;xt.m[1]=ih;xt.m[2]=iw;xt.m[3]=I;xt.d[0]=1;xt.d[1]=ih;xt.d[2]=iw;xt.d[3]=I;
    xt.p[1]=p1;xt.p[2]=p2;
    xt.blocks=mkblocks(vt,&at,(ih+7)/8,(iw+3)/4,cbs,1,holes);xt.nblocks=((ih+7)/8)*((iw+3)/4)*cbs;
    T o;memset(&o,0,sizeof o);
    o.flags=F_BLOCKED;o.esize=2;o.pool=1;o.m[0]=1;o.m[1]=oh+8;o.m[2]=ow+4;o.m[3]=co;o.d[0]=1;o.d[1]=oh+8;o.d[2]=ow+4;o.d[3]=co;
    o.p[1]=8;o.p[2]=4;
    uint32_t oat=at;
    o.blocks=mkblocks(vt,&at,(oh+8)/8,(ow+4)/4,NB,0,holes);o.nblocks=((oh+8)/8)*((ow+4)/4)*NB;
    uint32_t oend=at;
    at=(at+1023)&~1023u;
    uint32_t woff=at;at+=NB*9*cbs*1024;
    for(uint32_t i=0;i<NB*9*cbs*1024;i++)vt[woff+i]=(uint8_t)rnd();
    T wt;memset(&wt,0,sizeof wt);wt.pool=1;wt.offset=woff;wt.m[0]=H;wt.m[1]=W;wt.m[2]=I;wt.m[3]=co;
    at=(at+127)&~127u;uint32_t foff=at;at+=NB*256;
    for(int k=0;k<co;k++){
        uint32_t m=((8+rnd()%12)<<10)|(rnd()&1023);int32_t b=(int32_t)(rnd()%2000000)-1000000;
        *(uint32_t *)(vt+foff+(k>>5)*256+(k&31)*4)=m;*(int32_t *)(vt+foff+(k>>5)*256+128+(k&31)*4)=b;
    }
    T ft;memset(&ft,0,sizeof ft);ft.pool=1;ft.offset=foff;
    if(at>0xC0000){printf("cfg %d pool overflow %x\n",cfg,at);return;}
    static uint8_t near[0x8000] __attribute__((aligned(128)));
    static uint64_t phase[16];memset(phase,0,sizeof phase);
    Nprg2ReferenceConv conv;memset(&conv,0,sizeof conv);
    conv.workspace=vt+0x100000;conv.workspace_bytes=0x7e000;conv.near=near;conv.near_bytes=sizeof near;conv.phase=phase;conv.clock=clk;conv.hmx=&ops;
    static Nprg2ExecCall call;memset(&call,0,sizeof call);
    G g;memset(&g,0,sizeof g);
    g.call=&call;g.conv=&conv;g.pool[1]=vt;g.pool_bytes[1]=0x100000;g.write_limit[1]=0x100000;g.cm_ok=1;g.chk_state=1;
    int32_t err=0;
    uint64_t t=clk();
    conv_path=0;
    int r=hmx_native16(&g,&xt,&wt,&ft,&o,packed,reverse,co,oh,ow,&err);
    t=clk()-t;
    uint32_t hsh=2166136261u;
    for(uint32_t i=oat;i<oend;i+=4){hsh=(hsh^*(uint32_t *)(vt+i))*16777619u;}
    printf("cfg %d cbs=%d NB=%d %dx%d p=%d,%d pk=%d rev=%d holes=%d: ret=%d err=%d path=%u hash=%08x total=%llu stage=%llu mac=%llu out=%llu setup=%llu\n",
           cfg,cbs,NB,oh,ow,p1,p2,packed,reverse,holes,r,err,conv_path,hsh,(unsigned long long)t,(unsigned long long)phase[4],
           (unsigned long long)phase[5],(unsigned long long)phase[7],(unsigned long long)phase[6]);
#if N16_PT
    {
        static const uint8_t *pt[4096];int nR=oh/8+2,ncol=ow/4+3;
        uint64_t t2=clk();n16_build_ptab(pt,&g,&xt,nR,ncol,cbs,((int64_t)xt.m[2]+3)/4,((int64_t)xt.m[3]+31)/32,1);t2=clk()-t2;
        uint64_t t3=clk();n16_build_ptab(pt,&g,&xt,nR,ncol,cbs,((int64_t)xt.m[2]+3)/4,((int64_t)xt.m[3]+31)/32,0);t3=clk()-t3;
        uint32_t ph=2166136261u;for(int i=0;i<nR*ncol*cbs;i++)ph=(ph^(uint32_t)(uintptr_t)pt[i])*16777619u;
        printf("  ptab vec=%llu scalar=%llu (nR=%d ncol=%d) scalar-hash=%08x\n",(unsigned long long)t2,(unsigned long long)t3,nR,ncol,ph);
        n16_build_ptab(pt,&g,&xt,nR,ncol,cbs,((int64_t)xt.m[2]+3)/4,((int64_t)xt.m[3]+31)/32,1);
        uint32_t vh=2166136261u;for(int i=0;i<nR*ncol*cbs;i++)vh=(vh^(uint32_t)(uintptr_t)pt[i])*16777619u;
        printf("  ptab vec-hash=%08x\n",vh);
    }
#endif
    free((void *)xt.blocks);free((void *)o.blocks);
}
int main(void){
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x400000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *vt=(uint8_t *)base;
    int c=0;
    rs=1;run(vt,c++,1,1,40,64,0,0,0,1,0);   /* target shape: ci32 co32 */
    rs=2;run(vt,c++,1,1,40,64,7,1,0,1,0);
    rs=3;run(vt,c++,1,1,40,64,7,3,1,0,0);
    rs=4;run(vt,c++,1,1,40,64,3,2,0,1,0);
    rs=5;run(vt,c++,1,1,40,64,7,1,0,1,1);   /* absent blocks */
    rs=6;run(vt,c++,2,2,24,32,1,1,1,0,0);   /* cbs=2 NB=2 */
    rs=7;run(vt,c++,2,1,16,16,6,2,0,1,1);
    rs=8;run(vt,c++,1,1,8,4,0,0,0,1,0);     /* single tile */
    return 0;
}
