/* Randomised check of qd_hmx_slide3p_n and qd_hmx_nativep_n plus the HVX merge
 * against the simulator-rounding reference clip((q*M+8192)>>14). */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
#include <hexagon_sim_timer.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
extern void qd_hmx_finish(void *,void *);
extern void qd_hmx_slide3p_n(const uint32_t *rec,const void *b,const void *fi,const uint32_t *st,uint32_t cbs,uint32_t ncol);
extern void qd_hmx_nativep_n(void *,void *,void *,void *const *,uint32_t,uint32_t);
static uint8_t *done;static int32_t *zero;
static void fin(void){*(volatile uint32_t *)done=0xffffffff;qd_hmx_finish(zero,done);uint32_t t=0;while(*(volatile uint32_t *)done&&t++<1000000)__asm__ volatile("nop");if(*(volatile uint32_t*)done)puts("NO COMPLETION");}
static uint32_t rs=12345;static uint32_t rnd(void){rs^=rs<<13;rs^=rs>>17;rs^=rs<<5;return rs;}
static uint8_t *heap;static uint32_t at;static uint8_t *ALLOC(uint32_t n){at=(at+2047)&~2047u;uint8_t *r=heap+at;at+=n;return r;}
static void merges(uint8_t *o,const uint8_t *lo,const uint8_t *hi,const uint8_t *lo2,const uint8_t *hi2,HVX_Vector mv){
    const HVX_Vector c4096=Q6_Vh_vsplat_R(4096),c8192=Q6_Vh_vsplat_R(8192);
    for(int y=0;y<8;y++){
        const HVX_Vector *L=(const HVX_Vector *)(lo+y*256),*H=(const HVX_Vector *)(hi+y*256);
        HVX_Vector l0=L[0],l1=L[1],l2=*(const HVX_Vector *)(lo2+y*256),h0=H[0],h1=H[1],h2=*(const HVX_Vector *)(hi2+y*256);
        HVX_Vector ls[2]={Q6_V_valign_VVR(l1,l0,32),Q6_V_valign_VVR(l2,l1,32)},hs[2]={Q6_V_valign_VVR(h1,h0,32),Q6_V_valign_VVR(h2,h1,32)};
        for(int i=0;i<2;i++){
            HVX_VectorPair w=Q6_W_vshuff_VVR(hs[i],ls[i],-1);
            HVX_Vector off=Q6_Vh_vsplat_R(239<<8);
            HVX_Vector a=Q6_Vuh_vsub_VuhVuh_sat(Q6_V_lo_W(w),off),b=Q6_Vuh_vsub_VuhVuh_sat(Q6_V_hi_W(w),off);
            a=Q6_Vh_vmpy_VhVh_s1_rnd_sat(a,mv);b=Q6_Vh_vmpy_VhVh_s1_rnd_sat(b,mv);
            ((HVX_Vector *)(o+y*256))[i]=Q6_Vub_vpack_VhVh_sat(b,a);
        }
    }
}
static void merge(uint8_t *o,const uint8_t *lo,const uint8_t *hi,HVX_Vector mv){
    const HVX_Vector c4096=Q6_Vh_vsplat_R(4096),c8192=Q6_Vh_vsplat_R(8192);
    for(int i=0;i<16;i++){
        HVX_VectorPair w=Q6_W_vshuff_VVR(((const HVX_Vector *)hi)[i],((const HVX_Vector *)lo)[i],-1);
        HVX_Vector off=Q6_Vh_vsplat_R(239<<8);
        HVX_Vector a=Q6_Vuh_vsub_VuhVuh_sat(Q6_V_lo_W(w),off),b=Q6_Vuh_vsub_VuhVuh_sat(Q6_V_hi_W(w),off);
        a=Q6_Vh_vmpy_VhVh_s1_rnd_sat(a,mv);b=Q6_Vh_vmpy_VhVh_s1_rnd_sat(b,mv);
        ((HVX_Vector *)o)[i]=Q6_Vub_vpack_VhVh_sat(b,a);
    }
}
int main(int argc,char **argv) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x400000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    int cbs=atoi(argv[1]),nbw=atoi(argv[2]),NB=atoi(argv[3]),ncolin=nbw+2;
    heap=(uint8_t *)base;at=0;
    int nblk=3*ncolin*cbs;uint8_t **ib=malloc(nblk*sizeof *ib);
    uint8_t *pool=ALLOC(nblk*2048+2048);
    for(int i=0;i<nblk;i++){ib[i]=pool+i*2048;for(int k=0;k<2048;k++)ib[i][k]=(uint8_t)(rnd()>>(rnd()%8));}
#define IB(r,c,cb) ib[((r)*ncolin+(c))*cbs+(cb)]
    int KB=9*cbs;int8_t *B=(int8_t *)ALLOC(NB*KB*1024);for(int i=0;i<NB*KB*1024;i++)B[i]=(int8_t)((int32_t)rnd()>>(24+rnd()%6));
    uint8_t *C=ALLOC(NB*ncolin*4096),*ob=ALLOC(NB*nbw*2048);
    int32_t *fi=(int32_t *)ALLOC(NB*512);zero=(int32_t *)ALLOC(256);done=ALLOC(2048);
    uint32_t *rec=(uint32_t *)ALLOC(ncolin*cbs*16),*st=(uint32_t *)ALLOC(ncolin*8);
    HVX_Vector *mh=(HVX_Vector *)ALLOC(NB*128);
    int32_t pre[1024],M[1024],corr[1024];
    for(int n=0;n<32*NB;n++){pre[n]=2+rnd()%6;M[n]=1024+rnd()%1024;
        /* centre x/2^pre in the range that matters so every branch of the merge is hit */
        corr[n]=(int32_t)(rnd()%(4096u<<pre[n]))-(int32_t)(rnd()%(600000u));
        fi[(n>>5)*128+(n&31)]=(int32_t)((uint32_t)(24-pre[n])<<10);fi[(n>>5)*128+32+(n&31)]=corr[n];
        fi[(n>>5)*128+64+(n&31)]=(int32_t)((uint32_t)(16-pre[n])<<10);fi[(n>>5)*128+96+(n&31)]=corr[n]+(239<<(8+pre[n]));
        ((uint16_t *)mh)[(n>>5)*64+(n&31)]=(uint16_t)(2*M[n]);((uint16_t *)mh)[(n>>5)*64+32+(n&31)]=(uint16_t)(2*M[n]);}
    for(int k=0;k<64;k++)zero[k]=k<32?0x404000:0;
    for(int c=0;c<ncolin;c++)for(int cb=0;cb<cbs;cb++){uint32_t *r=rec+(c*cbs+cb)*4;
        uint32_t a=(uint32_t)IB(0,c,cb),b=(uint32_t)IB(1,c,cb),d=(uint32_t)IB(2,c,cb);
        r[0]=a|0x740;r[1]=(b-a)|0x71f;r[2]=b|0x140;r[3]=(d-b)|0x71f;}
    memset(ob,0xee,NB*nbw*2048);
    uint64_t t0=hexagon_sim_read_pcycles();
    for(int j=0;j<NB;j++){
        for(int c=0;c<ncolin;c++){
            if(c<1){st[2*c]=0;st[2*c+1]=0;continue;}
            uint8_t *d=C+((size_t)j*ncolin+c)*4096;
            st[2*c]=(uint32_t)d;st[2*c+1]=(uint32_t)(d+2048);}
        qd_hmx_slide3p_n(rec,B+(size_t)j*KB*1024,fi+j*128,st,cbs,ncolin);
    }
    fin();
    uint64_t t1=hexagon_sim_read_pcycles();
    for(int j=0;j<NB;j++)for(int bx=0;bx<nbw;bx++){uint8_t *c=C+((size_t)j*ncolin+bx+1)*4096;merges(ob+((size_t)j*nbw+bx)*2048,c,c+2048,c+4096,c+4096+2048,mh[j]);}
    uint64_t t2=hexagon_sim_read_pcycles();
    long bad=0,tot=0,cn=0,cz=0,cs=0;
    for(int j=0;j<NB;j++)for(int bx=0;bx<nbw;bx++)for(int y=0;y<8;y++)for(int x=0;x<8;x++)for(int n=0;n<32;n++){
        int64_t acc=0;
        for(int cb=0;cb<cbs;cb++)for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++){
            int row=7+y+ky,col=7+8*bx+x+kx;const uint8_t *p=IB(row>>3,col>>3,cb)+((row&7)*8+(col&7))*32;
            const int8_t *w=B+((size_t)j*KB+cb*9+ky*3+(2-kx))*1024;
            for(int ci=0;ci<32;ci++)acc+=(int64_t)p[ci]*w[((ci/4)*32+n)*4+ci%4];
        }
        int ch=32*j+n;int64_t q=(acc+corr[ch])>>pre[ch];int64_t v=(q*M[ch]+8192)>>14;uint8_t e=v<0?0:v>255?255:v;
        cn+=q<0;cs+=q>=4096;cz+=q>=0&&q<4096;
        uint8_t g=ob[((size_t)j*nbw+bx)*2048+(y*8+x)*32+n];tot++;
        if(g!=e){if(bad<6){uint8_t *cc=C+((size_t)j*ncolin+bx+1)*4096+(y*8+x)*32+n;printf("bad j%d bx%d y%d x%d n%d q=%lld M=%d got %u want %u lo %u hi %u wantlo %u wanthi %lld\n",j,bx,y,x,n,(long long)q,M[ch],g,e,cc[0],cc[2048],(unsigned)(q&255),(long long)((q>>8)+16));}bad++;}
    }
    printf("slide cbs=%d nbw=%d NB=%d: %ld/%ld bad (q<0 %ld, in range %ld, q>=4096 %ld), hmx %llu merge %llu pcycles\n",cbs,nbw,NB,bad,tot,cn,cz,cs,(unsigned long long)(t1-t0),(unsigned long long)(t2-t1));
    /* native (A staged by hand: one tile per tap of block row 1, no shift taps): 1x1 of K=cbs */
    uint8_t *A=ALLOC(cbs*2048);for(int cb=0;cb<cbs;cb++)memcpy(A+cb*2048,IB(1,2,cb),2048);
    void *outs[4];uint8_t *Cn=ALLOC(NB*4096);
    int8_t *B2=(int8_t *)ALLOC(NB*cbs*1024);for(int i=0;i<NB*cbs*1024;i++)B2[i]=B[i];
    long bad2=0;
    for(int j=0;j<NB;j++){outs[0]=Cn+j*4096;qd_hmx_nativep_n(A,B2+j*cbs*1024,fi+j*128,(void *const *)outs,cbs,1);}
    fin();
    for(int j=0;j<NB;j++){merge(ob,Cn+j*4096,Cn+j*4096+2048,mh[j]);
        for(int p=0;p<64;p++)for(int n=0;n<32;n++){int64_t acc=0;
            for(int cb=0;cb<cbs;cb++){const uint8_t *a=A+cb*2048+p*32;const int8_t *w=B2+((size_t)j*cbs+cb)*1024;
                for(int ci=0;ci<32;ci++)acc+=(int64_t)a[ci]*w[((ci/4)*32+n)*4+ci%4];}
            int ch=32*j+n;int64_t q=(acc+corr[ch])>>pre[ch];int64_t v=(q*M[ch]+8192)>>14;uint8_t e=v<0?0:v>255?255:v;
            if(ob[p*32+n]!=e){if(!bad2)printf("native bad p%d n%d got %u want %u\n",p,n,ob[p*32+n],e);bad2++;}
        }}
    printf("native: %ld bad\n",bad2);
    return 0;
}
