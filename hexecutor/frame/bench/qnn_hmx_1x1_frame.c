/* Hardware gate for the already simulator-validated, aligned SDK leaf.
 * Default uses the first 512 KB of verified VTCM scratch. Optional upper
 * placement saves/restores its complete 512 KB range during the exclusive
 * callback, after the upper-page preservation gate. No graph/pool mapping.
 * All other stock benchmark workers leave before this probe touches scratch. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "../qd_protocol.h"
#include "../qd_conv.h"
static uint64_t cycles(void){uint64_t v;__asm__ volatile("%0=upcycle":"=r"(v));return v;}
#include "../qd_axim.h"
#include "../qd_hmx_power.h"
#define ARENA 0xf1800000u
#define MAGIC 0x314e5151u
#define DATA 0x10000u
#define OUTPUT 0x100000u
#define ECHO 0x200000u
#define BACKUP 0x300000u
extern void qd_qnn_hmx_1x1(uint32_t *,uint32_t *,void *,void *,uint32_t *,uint32_t *);
extern void qd_hmx_finish(void *,void *);
extern void qds_on_stack(void (*)(uint32_t),uint32_t,uint32_t);
static void copy(void *dst,const void *src,uint32_t n) {
    for(uint32_t i=0;i<n;i+=128)*(HVX_Vector *)((uint8_t *)dst+i)=*(const HVX_Vector *)((const uint8_t *)src+i);
}
static void fill(void *dst,uint32_t bytes,uint32_t word) {
    HVX_Vector v=Q6_V_vsplat_R(word);
    for(uint32_t i=0;i<bytes;i+=128)*(HVX_Vector *)((uint8_t *)dst+i)=v;
}
static int same(const uint8_t *a,const uint8_t *b,uint32_t bytes) {
    HVX_Vector diff=Q6_V_vzero();
    for(uint32_t i=0;i<bytes;i+=128)diff=Q6_V_vor_VV(diff,Q6_V_vxor_VV(*(const HVX_Vector *)(a+i),*(const HVX_Vector *)(b+i)));
    uint32_t words[32] __attribute__((aligned(128)));*(HVX_Vector *)words=diff;
    uint32_t any=0;for(uint32_t i=0;i<32;i++)any|=words[i];return !any;
}
static int finish(int32_t *zero,uint8_t *done) {
    *(volatile uint32_t *)done=0xffffffffu;qd_hmx_finish(zero,done);
    uint64_t deadline=cycles()+200000000u;
    while(*(volatile uint32_t *)done)if(cycles()>deadline)return 5;
    return 0;
}
static void run_case(uint32_t unused) {
    (void)unused;
    const volatile uint32_t *cmd=(const volatile uint32_t *)ARENA;
    volatile uint32_t *result=(volatile uint32_t *)(ARENA+2048);
    uint32_t h=cmd[3],w=cmd[4],ci=cmd[5],co=cmd[6],reps=cmd[7];
    uint32_t icb=ci/32,ocb=co/32,spatial=(h/8)*(w/8),ib=spatial*icb,ob=spatial*ocb;
    uint32_t *inblocks=(uint32_t *)0x30000000u,*outblocks=(uint32_t *)0x30000400u;
    uint8_t *base=(uint8_t *)(QDC_VTCM_BASE+cmd[9]),*input=base,*output=base+0x20000;
    uint8_t *weights=base+0x40000,*bias=base+0x50000,*done=base+0x52000,*guard=base+0x54000;
    int32_t *zero=(int32_t *)(base+0x51000);
    uint64_t started=cycles();
    fill(base,0x80000,0);fill(output,0x20000,0xa5a5a5a5u);fill(guard,2048,0xa5a5a5a5u);
    const uint8_t *src=(const uint8_t *)(ARENA+DATA);
    for(uint32_t i=0;i<256;i++){inblocks[i]=(uint32_t)(base+0x56000);outblocks[i]=(uint32_t)guard;}
    for(uint32_t i=0;i<ib;i++){inblocks[i]=(uint32_t)(input+(ib-1-i)*2048);copy((void *)inblocks[i],src+i*2048,2048);}
    for(uint32_t i=0;i<ob;i++)outblocks[i]=(uint32_t)(output+(ob-1-i)*2048);
    copy(weights,src+ib*2048,ci*co);copy(bias,src+ib*2048+ci*co,ocb*256);
    uint8_t *expected=(uint8_t *)0x30010000u;
    if(cmd[8])copy(expected,src+ib*2048+ci*co+ocb*256,ob*2048);
    for(uint32_t i=0;i<32;i++)zero[i]=0x404000;
    uint32_t in[]={ (uint32_t)inblocks,icb,ib,spatial,64,ci };
    uint32_t out[]={ (uint32_t)outblocks,ocb,ob,spatial,64,co };
    uint32_t cfg[16]={0},mode[]={1,0};cfg[3]=2047;cfg[6]=1023;
    result[2]=2;result[4]=(uint32_t)(cycles()-started);
    uint64_t t0=cycles(),elapsed=0;
    for(uint32_t r=0;r<reps;r++) {
        if(cmd[8]){fill(output,ob*2048,0xa5a5a5a5u);t0=cycles();}
        __asm__ volatile("mxclracc":::"memory");
        qd_qnn_hmx_1x1(out,in,weights,bias,cfg,mode);
        if(cmd[8]) {
            result[1]=finish(zero,done);elapsed+=cycles()-t0;
            if(result[1])return;
            HVX_Vector diff=Q6_V_vzero();
            for(uint32_t i=0;i<ob;i++)for(uint32_t j=0;j<2048;j+=128)
                diff=Q6_V_vor_VV(diff,Q6_V_vxor_VV(*(const HVX_Vector *)(outblocks[i]+j),*(const HVX_Vector *)(expected+i*2048+j)));
            uint32_t lanes[32] __attribute__((aligned(128)));*(HVX_Vector *)lanes=diff;
            for(uint32_t j=0;j<32;j++)if(lanes[j]){result[1]=10;result[14]=r;return;}
            result[12]++;
        }
    }
    if(!cmd[8]){result[1]=finish(zero,done);elapsed=cycles()-t0;if(result[1])return;}
    result[5]=(uint32_t)elapsed;result[6]=elapsed>>32;result[13]=cmd[8];
    result[2]=3;started=cycles();
    for(uint32_t i=0;i<ob;i++)copy((void *)(ARENA+OUTPUT+i*2048),(void *)outblocks[i],2048);
    /* Verify the leaf did not alter either operand, including scattered input. */
    for(uint32_t i=0;i<ib;i++)copy((void *)(ARENA+ECHO+i*2048),(void *)inblocks[i],2048);
    copy((void *)(ARENA+ECHO+ib*2048),weights,ci*co);
    copy((void *)(ARENA+ECHO+ib*2048+ci*co),bias,ocb*256);
    for(uint32_t i=0;i<2048;i++)if(guard[i]!=0xa5){result[1]=6;break;}
    for(uint32_t i=ob*2048;i<0x20000;i++)if(output[i]!=0xa5){result[1]=7;break;}
    result[7]=(uint32_t)(cycles()-started);result[8]=ob*2048;result[9]=ib*2048+ci*co+ocb*256;
    result[2]=4;
}
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;__asm__ volatile("%0=htid":"=r"(tid));if(tid>=3)return;
    volatile uint32_t *cmd=(volatile uint32_t *)ARENA,*result=(volatile uint32_t *)(ARENA+2048);
    uint32_t seq=cmd[2];
    if(!tid) {
        for(uint32_t i=0;i<32;i++)result[i]=0;
        result[0]=MAGIC;result[2]=1;
        uint64_t deadline=cycles()+2000000000u;
        while(*(volatile uint32_t *)(ARENA+144)!=seq || *(volatile uint32_t *)(ARENA+160)!=seq)
            if(cycles()>deadline){result[1]=1;goto done;}
        if(cmd[0]!=MAGIC || cmd[1]!=1 || (cmd[3]!=8 && cmd[3]!=16) || (cmd[4]!=8 && cmd[4]!=24) ||
           cmd[5]<32 || cmd[5]>128 || cmd[6]<32 || cmd[6]>128 || ((cmd[5]|cmd[6])&31) ||
           !cmd[7] || cmd[7]>10000 || cmd[8]>1 ||
           (cmd[9] && cmd[9]!=0x300000u && cmd[9]!=0x700000u)){result[1]=2;goto done;}
        if(cmd[9]) {
            uint64_t e8,e9;
            __asm__ volatile("%0=tlbr(%1)":"=r"(e8):"r"(8));
            __asm__ volatile("%0=tlbr(%1)":"=r"(e9):"r"(9));
            if(e8!=0xc0031000f7062020ull || e9!=0xc0031400f7062820ull){result[1]=12;goto done;}
        }
        volatile uint32_t *clocks=(volatile uint32_t *)(ARENA+QDC_CLOCK_OFFSET);
        QdHmxState *hmx=(QdHmxState *)(ARENA+QDC_HMX_STATE_OFFSET);
        result[1]=clock_raise(clocks);
        if(!result[1])result[1]=hmx_start(hmx);
        if(!result[1]) {
            if(cmd[9]) {
                uint64_t started=cycles();copy((void *)(ARENA+BACKUP),(void *)(QDC_VTCM_BASE+cmd[9]),0x80000);
                __asm__ volatile("syncht":::"memory");result[15]=(uint32_t)(cycles()-started);result[18]=1;
            }
            volatile uint32_t *guard=(volatile uint32_t *)0x30080000u;
            for(uint32_t i=0;i<32;i++)guard[i]=0x7154c9a3u;
            qds_on_stack(run_case,0,0x30090000u);
            for(uint32_t i=0;i<32;i++)if(guard[i]!=0x7154c9a3u)result[1]=8;
        }
        result[10]=hmx_shutdown(hmx);
        if(result[18] && !result[10]) {
            uint64_t started=cycles();copy((void *)(QDC_VTCM_BASE+cmd[9]),(void *)(ARENA+BACKUP),0x80000);
            __asm__ volatile("syncht":::"memory");result[16]=(uint32_t)(cycles()-started);
            result[17]=same((const uint8_t *)(QDC_VTCM_BASE+cmd[9]),(const uint8_t *)(ARENA+BACKUP),0x80000);
            if(!result[17])result[1]=11;
        }
        result[11]=clock_restore(clocks);
        if(result[10] || result[11])result[1]=9;
    }
done:
    __asm__ volatile("barrier":::"memory");
    volatile QdStatus *st=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET)+tid;
    st->error=tid?0:result[1];st->rows=0;st->reserved=QDC_EXITED;
    __asm__ volatile("barrier":::"memory");st->done_sequence=seq;
}
