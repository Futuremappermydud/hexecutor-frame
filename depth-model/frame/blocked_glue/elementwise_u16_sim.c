/* V75 simulator test of blocked_glue/elementwise_u16.c against an independent
 * int64 oracle (the reference formulas of prepared_reference.c, copied) that
 * also reads/writes the CH layout element by element (so the block/row/pair/
 * lane addressing is checked, not assumed).
 *
 * Trials use random geometry: per-tensor meta, origins (same origin[2]%4 and
 * origin[3]%32 across tensors, different block offsets and row origins),
 * partial dims (partial vectors, partial pixel pairs and channel groups),
 * absent blocks in inputs and outputs, slices, batch > 1, and memory outside
 * the written lanes pre-filled with a pattern that must survive.
 * Parameters: the ones that occur in the schedules, plus random draws that
 * the make_* validators accept.
 *
 *   hexagon-sim elementwise_u16_sim.elf -- [trials per kind] */
#include "elementwise_u16.h"
#include <hexagon_standalone.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs=0x243f6a8885a308d3ull;
static uint32_t rnd(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;return (uint32_t)(rs>>16);}
static int64_t clip64(int64_t v,int64_t lo,int64_t hi){return v<lo?lo:v>hi?hi:v;}
static int64_t s32(uint64_t v){return (int64_t)(int32_t)(uint32_t)v;}

typedef struct { int64_t r[10]; int32_t zp; } Raw;
static int64_t ref_op(int kind,int64_t a,int64_t b,const Raw *w) {
    const int64_t *r=w->r;
    switch(kind) {
    case QD_EW16_REQUANT: {int64_t m=r[3]&0xffff,s=15-(r[3]>>24),zin=r[4]&0xffff,zout=r[4]>>16;
        return clip64((((a-zin)*m+((int64_t)1<<(s-1)))>>s)+zout,0,65535);}
    case QD_EW16_ADD1111: {int64_t k2=r[6]+1,k3=2*s32((uint64_t)r[7])+((int64_t)1<<r[6]);return clip64((a*r[4]+b*r[5]+k3)>>k2,0,65535);}
    case QD_EW16_MUL: {int64_t P1=r[4],zb=r[5],za=r[6],K=r[7],P2=r[8];
        int64_t D=s32((uint64_t)(a*b)+(uint64_t)K-(uint64_t)(b*(zb&0xffff)+a*(za&0xffff)));
        int64_t lo=P1&0xffff,hi=s32((uint64_t)P1)>>16,e=(D*lo)>>16;
        int64_t q=clip64((2*e+2*D*hi)>>16,-2147483648LL,2147483647LL);
        int64_t s=clip64(q+s32((uint64_t)P2),-2147483648LL,2147483647LL);
        return clip64((s+4)>>3,0,65535);}
    case QD_EW16_ADD: return clip64((a*r[4]+b*r[5]+((int64_t)1<<r[6]))>>(r[6]+1),0,65535);
    default: {int64_t k2=r[6]+1,k3=((int64_t)w->zp<<k2)+((int64_t)1<<r[6])+1;return clip64((a*r[4]-b*r[5]+k3)>>k2,0,65535);}
    }
}

typedef struct {
    QdBlockedU16 t;
    uint8_t *mem;uint32_t pages;
    uint8_t *pools[128];uint32_t blocks[256];
    int nbh,nbw,nbd;
} Tn;
static uint8_t memA[3][40*2048+128] __attribute__((aligned(2048)));

static int nbl(int m,int w){return (m+w-1)/w;}
/* Random tensor geometry for logical region `dims`, origin residues r2,r3. */
static void make_tensor(Tn *t,uint8_t *mem,const int32_t dims[4],int r2,int r3,int absent_pct,int keep_full_meta) {
    memset(t,0,sizeof *t);t->mem=mem;
    int32_t o[4],m[4];
    o[0]=rnd()%2;m[0]=o[0]+dims[0]+(rnd()%2);
    o[1]=keep_full_meta?0:rnd()%12;m[1]=o[1]+dims[1]+(keep_full_meta?0:rnd()%5);
    o[2]=keep_full_meta?0:4*(rnd()%2)+r2;m[2]=o[2]+dims[2]+(keep_full_meta?0:rnd()%7);
    o[3]=keep_full_meta?0:32*(rnd()%2)+r3;m[3]=o[3]+dims[3]+(keep_full_meta?0:rnd()%40);
    if(keep_full_meta){o[2]=r2;o[3]=r3;m[2]+=r2;m[3]+=r3;}
    t->nbh=nbl(m[1],8);t->nbw=nbl(m[2],4);t->nbd=nbl(m[3],32);
    uint32_t cnt=(uint32_t)m[0]*t->nbh*t->nbw*t->nbd,slice=rnd()%3;
    if(cnt+slice>256){fprintf(stderr,"geometry too big\n");exit(2);}
    t->t.pools=t->pools;t->t.blocks=t->blocks;t->t.block_count=cnt+slice;t->t.slice=slice;
    for(int i=0;i<4;i++){t->t.meta[i]=m[i];t->t.dims[i]=dims[i];t->t.origin[i]=o[i];}
    t->pools[1]=mem;
    uint32_t used=0;
    for(uint32_t i=0;i<cnt+slice;i++){
        if(i>=slice&&(int)(rnd()%100)<absent_pct)t->blocks[i]=0;
        else {if(used>=40){t->blocks[i]=0;continue;}t->blocks[i]=1u|(used*2048u);used++;}
    }
    t->pages=used;
}
/* Element (b,h,w,c) of the logical region -> byte address or NULL. */
static uint16_t *elem(const Tn *t,int b,int h,int w,int c) {
    int B=b+t->t.origin[0],H=h+t->t.origin[1],W=w+t->t.origin[2],C=c+t->t.origin[3];
    int64_t ix=(((int64_t)B*t->nbh+H/8)*t->nbw+W/4)*t->nbd+t->t.slice+C/32;
    uint32_t e=t->blocks[ix];
    if(!(e&127u))return 0;
    return (uint16_t *)(t->mem+(e&~127u)+(H%8)*256+((W%4)/2)*128+2*(2*(C%32)+(W%2)));
}
static uint16_t val(uint32_t mode) {
    static const uint16_t sp[]={0,1,2,255,256,4095,4096,32767,32768,32769,49151,49152,65534,65535};
    if(mode%3==0)return sp[rnd()%(sizeof sp/sizeof sp[0])];
    if(mode%3==1)return (uint16_t)rnd();
    return (uint16_t)(32768+(int)(rnd()%2001)-1000);
}

static int trials(int kind,const Raw *raw,const QdEw16Params *p,int n,int bulk,uint32_t seedmode) {
    int bad=0;
    for(int it=0;it<n;it++) {
        int32_t dims[4];
        int r2=rnd()%4,r3=rnd()%32;
        if(bulk){dims[0]=1;dims[1]=8;dims[2]=64;dims[3]=32;r2=0;r3=0;}
        else {dims[0]=1+(rnd()%5==0);dims[1]=1+rnd()%10;dims[2]=1+rnd()%9;dims[3]=1+rnd()%70;}
        if(dims[0]*dims[1]*dims[2]*dims[3]>3000&&!bulk){dims[3]=1+rnd()%20;}
        Tn a,b,o;
        int ab=bulk?0:rnd()%2?0:20;
        /* make_tensor can need more pages than 24 for big meta; retry smaller */
        make_tensor(&a,memA[0],dims,r2,r3,ab,bulk);
        make_tensor(&b,memA[1],dims,r2,r3,ab,bulk);
        make_tensor(&o,memA[2],dims,r2,r3,bulk?0:rnd()%2?0:20,bulk);
        /* fill memory: inputs random values, output pattern */
        for(uint32_t i=0;i<sizeof memA[0]/2;i++){((uint16_t *)memA[0])[i]=val(seedmode+i/7);((uint16_t *)memA[1])[i]=val(seedmode+it+i/5);}
        memset(memA[2],0xa5,sizeof memA[2]);
        static uint8_t expect[sizeof memA[2]];
        memcpy(expect,memA[2],sizeof expect);
        int bscalar=-1;
        int two=kind==QD_EW16_MUL||kind==QD_EW16_ADD||kind==QD_EW16_ADDSUB||kind==QD_EW16_ADDSUBU;
        int use_scalar=(kind==QD_EW16_ADD1111)||(kind==QD_EW16_MUL&&(rnd()%4==0));
        if(use_scalar)bscalar=val(seedmode+it);
        for(int B=0;B<dims[0];B++)for(int h=0;h<dims[1];h++)for(int w=0;w<dims[2];w++)for(int c=0;c<dims[3];c++){
            uint16_t *ep=elem(&o,B,h,w,c);
            if(!ep)continue;
            uint16_t *ap=elem(&a,B,h,w,c);int64_t av=ap?*ap:0,bv;
            if(use_scalar)bv=bscalar;
            else if(two){uint16_t *bp=elem(&b,B,h,w,c);bv=bp?*bp:0;}
            else bv=0;
            *(uint16_t *)((uint8_t *)ep-memA[2]+expect)=(uint16_t)ref_op(kind==QD_EW16_ADDSUBU?QD_EW16_ADDSUB:kind,av,bv,raw);
        }
        int rc=qd_blocked_ew16_apply(&a.t,(two&&!use_scalar)?&b.t:0,&o.t,p,bscalar);
        if(!rc){fprintf(stderr,"apply rejected kind=%d\n",kind);return 1;}
        if(bulk&&!memcmp(expect,memA[2],sizeof expect)) {
            /* the same region split over 3 parts (round-robin blocks) must give the same bytes */
            memset(memA[2],0xa5,sizeof memA[2]);
            for(uint32_t part=0;part<3;part++)
                if(!qd_blocked_ew16_apply_whole(&a.t,(two&&!use_scalar)?&b.t:0,&o.t,p,bscalar,part,3)){fprintf(stderr,"apply_whole rejected kind=%d\n",kind);return 1;}
        }
        if(memcmp(expect,memA[2],sizeof expect)){
            uint32_t i=0;while(expect[i]==memA[2][i])i++;
            fprintf(stderr,"MISMATCH kind=%d trial=%d byte=%u got=%02x want=%02x dims=%ld,%ld,%ld,%ld o2=%ld o3=%ld\n",kind,it,(unsigned)i,memA[2][i],expect[i],
                    (long)dims[0],(long)dims[1],(long)dims[2],(long)dims[3],(long)o.t.origin[2],(long)o.t.origin[3]);
            if(++bad>3)return 1;
        }
    }
    return bad;
}

int main(int argc,char **argv) {
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    int n=argc>1?atoi(argv[1]):40;
    int fail=0;
    uint32_t total=0;
    /* observed schedule parameters (r[] = call arguments r0..r9) */
    static const Raw req[]={{{0,0,0,16793600,0},0},{{0,0,0,16793600,2147516416},0},{{0,0,0,16793764,0},0},{{0,0,0,32765,0},0},{{0,0,0,32765,9196ll<<16},0},{{0,0,0,16793600,2147516416ll},0}};
    static const Raw a1111[]={{{0,0,0,0,16384,16384,13,3758104576ll},0},{{0,0,0,0,16384,8192,13,4026531840ll},0}};
    static const Raw mul[]={{{0,0,0,0,262127,0,0,2147549184ll,262119},0},{{0,0,0,0,262148,0,0,2147549184ll,262140},0},
                            {{0,0,0,0,262156,0,0,2147549184ll,262148},0},{{0,0,0,0,262281,0,0,2147549184ll,262273},0},
                            {{0,0,0,0,284378,0,0,2147549184ll,284369},0},{{0,0,0,0,262251,0,0,2147549184ll,262243},0},{{0,0,0,0,262152,0,0,2147549184ll,262140},0},{{0,0,0,0,262150,0,0,2147549184ll,262142},0},
                            /* not in the schedules: P2 near the int32 limits so the saturating q+P2 add engages (|q| ~ 3e5) */
                            {{0,0,0,0,262148,0,0,2147549184ll,2147383648ll},0},{{0,0,0,0,262148,0,0,2147549184ll,2147483647ll},0},
                            {{0,0,0,0,262148,0,0,2147549184ll,2147600000ll},0},{{0,0,0,0,262148,0,0,2147549184ll,2147483648ll},0}};
    static const Raw add[]={{{0,0,0,0,13699,26008,14,0},0},{{0,0,0,0,16995,14224,13,0},0}};
    static const Raw asub[]={{{0,0,0,0,14379,18491,14,0},36775},{{0,0,0,0,17144,3455,13,0},10701},{{0,0,0,0,30497,30497,14,0},61378}};
    QdEw16Params p;
    for(unsigned i=0;i<sizeof req/sizeof req[0];i++){if(!qd_ew16_make_requant(req[i].r[3],req[i].r[4],&p)){fprintf(stderr,"requant params\n");return 3;}
        fail|=trials(QD_EW16_REQUANT,&req[i],&p,n,0,i);fail|=trials(QD_EW16_REQUANT,&req[i],&p,3,1,i);total+=n+3;}
    for(unsigned i=0;i<sizeof a1111/sizeof a1111[0];i++){const int64_t *r=a1111[i].r;if(!qd_ew16_make_add1111(r[4],r[5],r[6],r[7],&p)){fprintf(stderr,"a1111 params\n");return 3;}
        fail|=trials(QD_EW16_ADD1111,&a1111[i],&p,n,0,i);fail|=trials(QD_EW16_ADD1111,&a1111[i],&p,3,1,i);total+=n+3;}
    for(unsigned i=0;i<sizeof mul/sizeof mul[0];i++){const int64_t *r=mul[i].r;if(!qd_ew16_make_mul(r[4],r[5],r[6],r[7],r[8],&p)){fprintf(stderr,"mul params\n");return 3;}
        fail|=trials(QD_EW16_MUL,&mul[i],&p,n,0,i);fail|=trials(QD_EW16_MUL,&mul[i],&p,3,1,i);total+=n+3;}
    for(unsigned i=0;i<sizeof add/sizeof add[0];i++){const int64_t *r=add[i].r;if(!qd_ew16_make_add(r[4],r[5],r[6],&p)){fprintf(stderr,"add params\n");return 3;}
        fail|=trials(QD_EW16_ADD,&add[i],&p,n,0,i);fail|=trials(QD_EW16_ADD,&add[i],&p,3,1,i);total+=n+3;}
    for(unsigned i=0;i<sizeof asub/sizeof asub[0];i++){const int64_t *r=asub[i].r;if(!qd_ew16_make_addsub(r[4],r[5],r[6],asub[i].zp,&p)){fprintf(stderr,"addsub params\n");return 3;}
        fail|=trials(p.kind,&asub[i],&p,n,0,i);fail|=trials(p.kind,&asub[i],&p,3,1,i);total+=n+3;}
    /* random accepted parameters, including both ADDSUB domains and negative MUL hi */
    int drawn[6]={0};
    for(int it=0;it<n*6&&fail==0;it++) {
        Raw w;memset(&w,0,sizeof w);int kind=1+it%5;
        uint32_t x=rnd();
        if(kind==QD_EW16_REQUANT){int64_t m=rnd()&0xffff,s=1+rnd()%20,zin=rnd()&0xffff,zout=rnd()%70000;
            if(x&1)zin=(x>>1&1)?0:32768;w.r[3]=m|((15-s<0?0:(15-s))<<24);w.r[4]=zin|(zout<<16);
            if(15-s<0)continue;if(!qd_ew16_make_requant(w.r[3],w.r[4],&p))continue;}
        else if(kind==QD_EW16_ADD1111){w.r[4]=rnd()%40000;w.r[5]=rnd()%40000;w.r[6]=rnd()%20;w.r[7]=(int64_t)(uint32_t)rnd();if(x&1)w.r[7]|=0xf0000000u;
            if(!qd_ew16_make_add1111(w.r[4],w.r[5],w.r[6],w.r[7],&p))continue;}
        else if(kind==QD_EW16_MUL){w.r[4]=(int64_t)(rnd()%(1u<<20));w.r[7]=(int64_t)(uint32_t)rnd();w.r[8]=(int64_t)(uint32_t)rnd();
            if(x&1)w.r[4]=(int64_t)(uint32_t)(-(int32_t)(rnd()%(1u<<20)));
            if(!qd_ew16_make_mul(w.r[4],0,0,w.r[7],w.r[8],&p))continue;}
        else if(kind==QD_EW16_ADD){w.r[4]=rnd()%60000;w.r[5]=rnd()%60000;w.r[6]=rnd()%16;
            if(!qd_ew16_make_add(w.r[4],w.r[5],w.r[6],&p))continue;}
        else {w.r[4]=rnd()%40000;w.r[5]=rnd()%40000;w.r[6]=rnd()%16;w.zp=(int32_t)(rnd()%65536);
            if(!qd_ew16_make_addsub(w.r[4],w.r[5],w.r[6],w.zp,&p))continue;kind=p.kind;}
        drawn[it%5]++;
        fail|=trials(kind,&w,&p,4,0,x);total+=4;
    }
    printf("%s elementwise_u16 sim: %lu trials, random-parameter draws %d/%d/%d/%d/%d\n",fail?"FAIL":"PASS",(unsigned long)total,drawn[0],drawn[1],drawn[2],drawn[3],drawn[4]);
    SIM_RELEASE_HVX;
    return fail;
}
