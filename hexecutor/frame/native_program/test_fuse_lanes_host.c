/* Host exhaustive check of the fused conv-epilogue lane arithmetic (prepared_reference.c, rq_emit_fx and
 * fuse_scalar) against the epilogue commands' own integer formulas (prepared_reference.c k_requant_u8 / k_add_u8
 * generic int64 form):
 *   REQUANT_U8:  clip255((q*m + 2^(s-1)) >> s)                       (m = arg3&0xffff, s = 15-(arg3>>24))
 *   ADD_U8:      clip255((a*k0 + b*k1 + 2^(s-1)) >> s)               (s = arg6+8)
 * The fused form is e = min(lsr32(q*mc + b*mo + rd), 255) in u32 lanes with vmpyie (low unsigned halfword
 * multiplier, wrap mod 2^32). Every u8 input (and every u8 pair for ADD) is checked for
 *   (1) every parameter set of the two real schedules' fusion pairs (read from the bound program), and
 *   (2) random parameter sets accepted by the fuse_decode predicate (full input space each).
 * Run from depth-model: test_fuse_lanes_host <model> */
#include "nprg2_runtime.h"
#include "prepared_fuse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd32(const uint8_t *p){return p[0]|p[1]<<8|p[2]<<16|(uint32_t)p[3]<<24;}
static uint64_t rd64(const uint8_t *p){return rd32(p)|((uint64_t)rd32(p+4)<<32);}
static uint8_t *rf(const char *p,uint32_t *n){FILE *f=fopen(p,"rb");if(!f){perror(p);exit(1);}fseek(f,0,SEEK_END);long z=ftell(f);fseek(f,0,SEEK_SET);uint8_t *b=malloc(z);if(fread(b,1,z,f)!=(size_t)z)exit(1);fclose(f);*n=z;return b;}
static int res(void *c,const Nprg2Program *p,uint32_t i,const Nprg2Command *cm,Nprg2Capability *cap){(void)c;(void)p;(void)i;(void)cm;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;}

/* reference: the commands' int64 formulas */
static int64_t ref_requant(int64_t q,int64_t m,int64_t s){int64_t v=(q*m+((int64_t)1<<(s-1)))>>s;return v>255?255:v;}
static int64_t ref_add(int64_t a,int64_t b,int64_t k0,int64_t k1,int64_t s){int64_t v=(a*k0+b*k1+((int64_t)1<<(s-1)))>>s;return v>255?255:v<0?0:v;}
/* fused lane form */
static uint32_t vmpyie(uint32_t x,uint32_t m){return x*(m&0xffffu);}
static uint32_t lane(uint32_t q,uint32_t b,uint32_t mc,uint32_t mo,uint32_t rd,uint32_t sh,int add){
    uint32_t qc=q>255?255:q;                /* vmin signed on a nonnegative lane */
    uint32_t t=vmpyie(qc,mc)+rd;            /* vmpyieacc(rd, qc, mc) */
    if(add)t+=vmpyie(b,mo);
    t>>=sh;return t>255?255:t;
}
static uint64_t checked,bad;
static int accepts_add(int64_t k0,int64_t k1,int64_t k2){return k0>=0&&k1>=0&&k0<=65535&&k1<=65535&&k2>=1&&k2<=31&&((uint64_t)k0+k1)*255+(1ull<<(k2-1))<=0xffffffffull;}
static void test_requant(int64_t m,int64_t s){
    for(int64_t q=0;q<256;q++){checked++;if(lane((uint32_t)q,0,(uint32_t)m,0,1u<<(s-1),(uint32_t)s,0)!=(uint32_t)ref_requant(q,m,s))bad++;}
}
static void test_add(int64_t k0,int64_t k1,int64_t s){
    for(int64_t a=0;a<256;a++)for(int64_t b=0;b<256;b++){
        checked++;
        if(lane((uint32_t)a,(uint32_t)b,(uint32_t)k0,(uint32_t)k1,1u<<(s-1),(uint32_t)s,1)!=(uint32_t)ref_add(a,b,k0,k1,s))bad++;
        /* conv operand second: the roles swap, the sum is symmetric */
        if(lane((uint32_t)b,(uint32_t)a,(uint32_t)k1,(uint32_t)k0,1u<<(s-1),(uint32_t)s,1)!=(uint32_t)ref_add(a,b,k0,k1,s))bad++;
    }
}
int main(int argc,char **argv){
    const char *model=argc>1?argv[1]:"primary";
    char path[256];uint32_t qn,nn;
    snprintf(path,256,"frame/build/resident-%s.qdrx",model);uint8_t *q=rf(path,&qn);
    snprintf(path,256,"frame/native_program/examples/%s-v2.nprg",model);uint8_t *n=rf(path,&nn);
    Nprg2Pool pools[7];memset(pools,0,sizeof pools);
    for(int i=0;i<7;i++){pools[i].bytes=rd32(n+144+i*4);if(pools[i].bytes)pools[i].base=calloc(1,pools[i].bytes);}
    uint8_t qs[32],ns[32];nprg2_sha256(q,qn,qs);nprg2_sha256(n,nn,ns);
    Nprg2ExpectedHashes e={q+32,q+64,qs,ns};Nprg2Program P;
    if(nprg2_open(&P,n,nn,&e,pools,(Nprg2Allocator){0})){fprintf(stderr,"open\n");return 1;}
    uint8_t *hot=calloc(1,300000);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=res};Nprg2Bound B;
    if(nprg2_bind_mock(&P,pools,&bc,&B)){fprintf(stderr,"bind\n");return 1;}
    Nprg2FusePlan pl;if(nprg2_fuse_plan_build(&pl,&P,&B,(Nprg2Allocator){0})){fprintf(stderr,"plan\n");return 1;}
    uint32_t nreq=0,nadd=0;
    for(uint32_t i=0;i<pl.count;i++){
        Nprg2Command c;nprg2_command_at(&B,pl.pairs[i].epi,&c);
        for(unsigned k=0;k<c.call_count;k++){
            const uint8_t *r=B.calls.data+(size_t)(c.call_first+k)*88u;
            int64_t a[10];for(int j=0;j<10;j++)a[j]=(int64_t)rd64(r+8+8*j);
            if(c.opcode==8&&rd32(r)==QDR_CALL_REQUANTIZE_CROUTON_U8){
                int64_t m=a[3]&0xffff,s=15-(a[3]>>24);
                if(a[4]||a[5]||s<1||s>31){fprintf(stderr,"pair %u: requant params outside the fused set\n",i);return 1;}
                test_requant(m,s);nreq++;
            } else if(c.opcode==10&&rd32(r)==QDR_CALL_ELEMENTWISE_ADD_CROUTON_U8){
                int64_t k0=a[4],k1=a[5],s=a[6]+8;
                if(a[7]||!accepts_add(k0,k1,s)){fprintf(stderr,"pair %u: add params outside the fused set\n",i);return 1;}
                test_add(k0,k1,s);nadd++;
            }
        }
    }
    printf("%s: %u requant and %u add pair parameter sets checked (all inputs each)\n",model,nreq,nadd);
    /* all (m, s) for requant */
    for(int64_t s=1;s<=31;s++)for(int64_t m=0;m<=65535;m++)test_requant(m,s);
    /* random ADD sets over the accepted domain, boundary-biased */
    srand(12345);
    uint64_t sets=0;
    for(int it=0;it<3000;it++){
        int64_t s=1+rand()%31,k0,k1;
        int64_t lim=65535;
        if(it%3==0){k0=rand()%1024;k1=rand()%1024;}
        else if(it%3==1){k0=65535-rand()%16;k1=65535-rand()%16;}
        else{k0=((int64_t)rand()<<8^rand())&65535;k1=((int64_t)rand()<<8^rand())&65535;}
        (void)lim;
        if(!accepts_add(k0,k1,s))continue;
        test_add(k0,k1,s);sets++;
    }
    printf("%s: requant all (m 0..65535, s 1..31) and %llu random add sets; %llu cases, %llu mismatches\n",model,(unsigned long long)sets,
           (unsigned long long)checked,(unsigned long long)bad);
    printf(bad?"FAIL\n":"PASS\n");
    return bad!=0;
}
