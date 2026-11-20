/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Host exactness proof for the 32-bit lane forms in blocked_glue/elementwise_u16.h
 * against the int64 reference formulas of prepared_reference.c (copied here
 * verbatim). Parameters that occur in the two schedules are checked over the
 * COMPLETE input space (every (a,b) pair, 2^32 per parameter set; every D for
 * the MUL second stage); randomly drawn parameter sets that the make_*
 * functions accept are checked on all single-operand inputs or on dense
 * samples. Build:  cc -O2 -std=c11 -Iframe/blocked_glue test_ew16_lanes_host.c
 * Run:  test_ew16_lanes_host [quick]   (full run forks one process per job) */
#include "elementwise_u16.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int64_t clip64(int64_t v,int64_t lo,int64_t hi){return v<lo?lo:v>hi?hi:v;}
static int64_t s32(uint64_t v){return (int64_t)(int32_t)(uint32_t)v;}
static int64_t ref_mul(int64_t a,int64_t b,int64_t P1,int64_t zb,int64_t za,int64_t K,int64_t P2) {
    int64_t D=s32((uint64_t)(a*b)+(uint64_t)K-(uint64_t)(b*(zb&0xffff)+a*(za&0xffff)));
    int64_t lo=P1&0xffff,hi=s32((uint64_t)P1)>>16;
    int64_t e=(D*lo)>>16;
    int64_t q=clip64((2*e+2*D*hi)>>16,-2147483648LL,2147483647LL);
    int64_t s=clip64(q+s32((uint64_t)P2),-2147483648LL,2147483647LL);
    return clip64((s+4)>>3,0,65535);
}
/* The reference's D-to-output stage, for the all-D check. */
static int64_t ref_mul_stage2(int64_t D,int64_t P1,int64_t P2) {
    int64_t lo=P1&0xffff,hi=s32((uint64_t)P1)>>16;
    int64_t e=(D*lo)>>16;
    int64_t q=clip64((2*e+2*D*hi)>>16,-2147483648LL,2147483647LL);
    int64_t s=clip64(q+s32((uint64_t)P2),-2147483648LL,2147483647LL);
    return clip64((s+4)>>3,0,65535);
}
static int64_t ref_requant(int64_t v,int64_t r3,int64_t r4) {
    int64_t m=r3&0xffff,s=15-(r3>>24),zin=r4&0xffff,zout=r4>>16;
    return clip64((((v-zin)*m+((int64_t)1<<(s-1)))>>s)+zout,0,65535);
}
static int64_t ref_add1111(int64_t x,int64_t s,int64_t r4,int64_t r5,int64_t r6,int64_t r7) {
    int64_t k2=r6+1,k3=2*s32((uint64_t)r7)+((int64_t)1<<r6);
    return clip64((x*r4+s*r5+k3)>>k2,0,65535);
}
static int64_t ref_add(int64_t a,int64_t b,int64_t r4,int64_t r5,int64_t r6) {
    return clip64((a*r4+b*r5+((int64_t)1<<r6))>>(r6+1),0,65535);
}
static int64_t ref_addsub(int64_t a,int64_t b,int64_t r4,int64_t r5,int64_t r6,int32_t zp) {
    int64_t k2=r6+1,k3=((int64_t)zp<<k2)+((int64_t)1<<r6)+1;
    return clip64((a*r4-b*r5+k3)>>k2,0,65535);
}

static uint64_t rng_state=0x9e3779b97f4a7c15ull;
static uint64_t rnd64(void){rng_state^=rng_state<<13;rng_state^=rng_state>>7;rng_state^=rng_state<<17;return rng_state;}
#define FAILF(...) do{fprintf(stderr,"MISMATCH ");fprintf(stderr,__VA_ARGS__);fprintf(stderr,"\n");exit(1);}while(0)

typedef struct { const char *name; int kind; int64_t r[5]; int32_t zp; } Job;
static Job jobs[256];static int njobs;
static void addjob(const char *n,int kind,int64_t a,int64_t b,int64_t c,int64_t d,int64_t e,int32_t zp) {
    jobs[njobs++]=(Job){n,kind,{a,b,c,d,e},zp};
}

static void run_requant(const Job *j) {
    QdEw16Params p;
    if(!qd_ew16_make_requant(j->r[0],j->r[1],&p))FAILF("requant observed params rejected %"PRId64" %"PRId64,j->r[0],j->r[1]);
    for(int64_t v=0;v<65536;v++)if(qd_ew16_requant_lane(&p,(uint16_t)v)!=ref_requant(v,j->r[0],j->r[1]))FAILF("requant v=%"PRId64,v);
    printf("PASS requant r3=%"PRId64" r4=%"PRId64" all 65536 v\n",j->r[0],j->r[1]);
}
static void run_requant_random(void) {
    int acc=0,rej=0;
    for(int it=0;it<400000;it++) {
        uint64_t x=rnd64();int64_t m=x&0xffff,s=1+(int64_t)((x>>16)%34),zin,zout;
        /* bias toward the accept boundary: large m with small shift, big zero points */
        if(x>>60&1)m|=0x8000;
        zin=(int64_t)((rnd64()&1)?(rnd64()&0xffff):(rnd64()%3)*32768);zout=(int64_t)((rnd64()&1)?(rnd64()&0xffff):(rnd64()%4)*20000);
        int64_t r3,r4=zin|(zout<<16);
        /* s=15-(r3>>24)  =>  r3>>24 = 15-s, only representable when 15-s>=0 */
        if(15-s<0)continue;
        r3=m|((15-s)<<24);
        QdEw16Params p;
        if(!qd_ew16_make_requant(r3,r4,&p)){rej++;continue;}
        acc++;
        for(int64_t v=0;v<65536;v++)if(qd_ew16_requant_lane(&p,(uint16_t)v)!=ref_requant(v,r3,r4))FAILF("requant rnd r3=%"PRId64" r4=%"PRId64" v=%"PRId64,r3,r4,v);
    }
    printf("PASS requant random params: %d accepted (all v each), %d rejected\n",acc,rej);
}
static void run_add1111(const Job *j,int sampled) {
    QdEw16Params p;
    if(!qd_ew16_make_add1111(j->r[0],j->r[1],j->r[2],j->r[3],&p))FAILF("add1111 observed params rejected");
    if(sampled) {
        for(int it=0;it<20000000;it++){uint64_t x=rnd64();uint32_t xv=x&0xffff,s=(x>>16)&0xffff;if(it&1)xv=(it&2)?65535:0;
            if(qd_ew16_add1111_lane(&p,qd_ew16_add1111_const(&p,s),(uint16_t)xv)!=ref_add1111(xv,s,j->r[0],j->r[1],j->r[2],j->r[3]))FAILF("add1111 rnd");}
    } else {
        for(uint32_t s=0;s<65536;s++){int32_t cc=qd_ew16_add1111_const(&p,s);
            for(uint32_t x=0;x<65536;x++)if(qd_ew16_add1111_lane(&p,cc,(uint16_t)x)!=ref_add1111(x,s,j->r[0],j->r[1],j->r[2],j->r[3]))FAILF("add1111 x=%u s=%u",x,s);}
    }
    printf("PASS add1111 k0=%"PRId64" k1=%"PRId64" r6=%"PRId64" r7=%"PRId64" %s\n",j->r[0],j->r[1],j->r[2],j->r[3],sampled?"sampled":"all 2^32 (x,s)");
}
static void run_add_like(const Job *j,int kind) {
    QdEw16Params p;
    int ok=kind==QD_EW16_ADD?qd_ew16_make_add(j->r[0],j->r[1],j->r[2],&p):qd_ew16_make_addsub(j->r[0],j->r[1],j->r[2],j->zp,&p);
    if(!ok)FAILF("%s observed params rejected k0=%"PRId64" k1=%"PRId64" r6=%"PRId64" zp=%d",j->name,j->r[0],j->r[1],j->r[2],j->zp);
    for(uint32_t a=0;a<65536;a++)for(uint32_t b=0;b<65536;b++) {
        uint16_t got=kind==QD_EW16_ADD?qd_ew16_add_lane(&p,(uint16_t)a,(uint16_t)b):qd_ew16_addsub_lane(&p,(uint16_t)a,(uint16_t)b);
        int64_t want=kind==QD_EW16_ADD?ref_add(a,b,j->r[0],j->r[1],j->r[2]):ref_addsub(a,b,j->r[0],j->r[1],j->r[2],j->zp);
        if(got!=want)FAILF("%s a=%u b=%u got=%u want=%"PRId64,j->name,a,b,got,want);
    }
    printf("PASS %s k0=%"PRId64" k1=%"PRId64" r6=%"PRId64" zp=%d mode=%u all 2^32 (a,b)\n",j->name,j->r[0],j->r[1],j->r[2],j->zp,p.kind);
}
static void run_add_random(void) {
    int acc[2]={0,0},rej[2]={0,0},uns=0;
    for(int it=0;it<300000;it++) {
        uint64_t x=rnd64();
        int64_t k0=rnd64()%(1ull<<((x&31)+1)),k1=rnd64()%(1ull<<(((x>>5)&31)+1)),r6=(x>>10)%33;
        int32_t zp=(int32_t)(rnd64()&0xffff);if(x>>40&1)zp=(int32_t)(rnd64()%4096)-2048;
        QdEw16Params p;
        for(int w=0;w<2;w++) {
            int ok=w==0?qd_ew16_make_add(k0,k1,r6,&p):qd_ew16_make_addsub(k0,k1,r6,zp,&p);
            if(!ok){rej[w]++;continue;}
            acc[w]++;if(w==1&&p.kind==QD_EW16_ADDSUBU)uns++;
            for(int s=0;s<3000;s++){
                uint32_t a=rnd64()&0xffff,b=rnd64()&0xffff;
                if(s<4){a=(s&1)?65535:0;b=(s&2)?65535:0;}
                uint16_t got=w==0?qd_ew16_add_lane(&p,(uint16_t)a,(uint16_t)b):qd_ew16_addsub_lane(&p,(uint16_t)a,(uint16_t)b);
                int64_t want=w==0?ref_add(a,b,k0,k1,r6):ref_addsub(a,b,k0,k1,r6,zp);
                if(got!=want)FAILF("add%s random k0=%"PRId64" k1=%"PRId64" r6=%"PRId64" zp=%d a=%u b=%u",w?"sub":"",k0,k1,r6,zp,a,b);
            }
        }
    }
    printf("PASS add random params: add %d acc/%d rej, addsub %d acc (%d unsigned)/%d rej\n",acc[0],rej[0],acc[1],uns,rej[1]);
}
static void run_mul(const Job *j,int all_ab) {
    QdEw16Params p;
    if(!qd_ew16_make_mul(j->r[0],0,0,j->r[1],j->r[2],&p))FAILF("mul observed params rejected");
    if(all_ab) {
        for(uint32_t a=0;a<65536;a++)for(uint32_t b=0;b<65536;b++)
            if(qd_ew16_mul_lane(&p,(uint16_t)a,(uint16_t)b)!=ref_mul(a,b,j->r[0],0,0,j->r[1],j->r[2]))FAILF("mul a=%u b=%u",a,b);
        printf("PASS mul P1=%"PRId64" K=%"PRId64" P2=%"PRId64" all 2^32 (a,b)\n",j->r[0],j->r[1],j->r[2]);
    } else {
        uint32_t D=0;
        do { if(qd_ew16_mul_stage2(&p,D)!=ref_mul_stage2((int32_t)D,j->r[0],j->r[2]))FAILF("mul stage2 D=0x%08x",D); } while(++D);
        printf("PASS mul stage2 P1=%"PRId64" P2=%"PRId64" all 2^32 D\n",j->r[0],j->r[2]);
    }
}
static void run_mul_random(void) {
    int acc=0,rej=0;
    for(int it=0;it<300;it++) {
        int64_t P1=(int64_t)(rnd64()&0xffffffffull),P2=(int64_t)(rnd64()&0xffffffffull),K=(int64_t)(rnd64()&0xffffffffull);
        if(it&1)P1=(int64_t)((rnd64()%40000)|((rnd64()%9)<<16)); /* small hi */
        QdEw16Params p;
        if(!qd_ew16_make_mul(P1,0,0,K,P2,&p)){rej++;continue;}
        acc++;
        /* all D in a 2^26 window around random + extremes and sign boundaries, plus 2^24 random */
        uint32_t starts[4]={0u,0x7fff0000u,0x80000000u-65536u,(uint32_t)rnd64()};
        for(int w=0;w<4;w++)for(uint32_t i=0;i<(1u<<20);i++){uint32_t D=starts[w]+i;
            if(qd_ew16_mul_stage2(&p,D)!=ref_mul_stage2((int32_t)D,P1,P2))FAILF("mul rnd P1=%"PRId64" P2=%"PRId64" D=0x%08x",P1,P2,D);}
        for(int i=0;i<(1<<22);i++){uint32_t D=(uint32_t)rnd64();
            if(qd_ew16_mul_stage2(&p,D)!=ref_mul_stage2((int32_t)D,P1,P2))FAILF("mul rnd P1=%"PRId64" P2=%"PRId64" D=0x%08x",P1,P2,D);}
    }
    printf("PASS mul random params: %d accepted (2^22+4*2^20 D each), %d rejected\n",acc,rej);
}

static void run_job(int i,int quick) {
    const Job *j=&jobs[i];
    switch(j->kind) {
    case 0: run_requant(j);break;
    case 1: run_add1111(j,quick);break;
    case 2: if(quick){QdEw16Params p;if(!qd_ew16_make_add(j->r[0],j->r[1],j->r[2],&p))FAILF("add rejected");
                 for(int s=0;s<20000000;s++){uint32_t a=rnd64()&0xffff,b=rnd64()&0xffff;if(qd_ew16_add_lane(&p,a,b)!=ref_add(a,b,j->r[0],j->r[1],j->r[2]))FAILF("add");}printf("PASS add sampled\n");}
            else run_add_like(j,QD_EW16_ADD);break;
    case 3: if(quick){QdEw16Params p;if(!qd_ew16_make_addsub(j->r[0],j->r[1],j->r[2],j->zp,&p))FAILF("addsub rejected");
                 for(int s=0;s<20000000;s++){uint32_t a=rnd64()&0xffff,b=rnd64()&0xffff;if(qd_ew16_addsub_lane(&p,a,b)!=ref_addsub(a,b,j->r[0],j->r[1],j->r[2],j->zp))FAILF("addsub");}printf("PASS addsub sampled\n");}
            else run_add_like(j,QD_EW16_ADDSUB);break;
    case 4: run_mul(j,quick?0:1);break;
    case 5: run_mul(j,0);break;
    case 6: run_requant_random();run_add_random();run_mul_random();break;
    }
}
int main(int argc,char **argv) {
    int quick=argc>1&&!strcmp(argv[1],"quick");
    /* Observed (r3,r4) requant sets */
    static const int64_t rq[][2]={{16793600,0},{16793600,2147516416},{16793764,0},{32765,0}};
    for(unsigned i=0;i<sizeof rq/sizeof rq[0];i++)addjob("requant",0,rq[i][0],rq[i][1],0,0,0,0);
    addjob("add1111",1,16384,16384,13,3758104576ll,0,0);
    addjob("add1111",1,16384,8192,13,4026531840ll,0,0);
    addjob("add",2,13699,26008,14,0,0,0);
    addjob("add",2,16995,14224,13,0,0,0);
    /* (k0,k1,r6) with the output tensor zero point they occur with */
    static const int64_t as[3][4]={{14379,18491,14,36775},{17144,3455,13,10701},{30497,30497,14,61378}};
    for(int i=0;i<3;i++)addjob("addsub",3,as[i][0],as[i][1],as[i][2],0,0,(int32_t)as[i][3]);
    static const int64_t ms[5][2]={{262127,262119},{262148,262140},{262156,262148},{262281,262273},{284378,284369}};
    for(int i=0;i<5;i++){addjob("mul",4,ms[i][0],2147549184ll,ms[i][1],0,0,0);addjob("mulD",5,ms[i][0],2147549184ll,ms[i][1],0,0,0);}
    /* P2 near the int32 limits (not in the schedules): the saturating add engages */
    static const int64_t p2x[4]={2147383648ll,2147483647ll,2147600000ll,2147483648ll};
    for(int i=0;i<4;i++)addjob("mulD",5,262148,2147549184ll,p2x[i],0,0,0);
    addjob("random",6,0,0,0,0,0,0);
    if(argc>2&&!strcmp(argv[1],"job")){run_job(atoi(argv[2]),0);return 0;}
    int running=0,failed=0;
    for(int i=0;i<njobs;i++) {
        if(running>=12){int st;wait(&st);running--;if(st)failed=1;}
        pid_t pid=fork();
        if(!pid){rng_state^=(uint64_t)i*0x1234567ull+1;run_job(i,quick);fflush(stdout);_exit(0);}
        running++;
    }
    while(running--){int st;wait(&st);if(st)failed=1;}
    printf(failed?"FAILED\n":"ALL PASS (%d jobs%s)\n",njobs,quick?", quick":"");
    return failed;
}
