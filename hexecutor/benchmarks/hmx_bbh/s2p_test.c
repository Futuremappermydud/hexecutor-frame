/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Randomised check of qd_hmx_slide2p_n and the realigning merge against a scalar
 * model using the engine's stride-2 tap table (S2_TAPS, s2_ktab). */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
extern void qd_hmx_finish(void *,void *);
extern void qd_hmx_slide2p_n(const uint32_t *rec,const void *b,const void *fi,const uint32_t *st,uint32_t groups,uint32_t ncol);
static const int S2_TAPS[9][2]={{0,2},{0,0},{2,2},{2,0},{0,1},{2,1},{1,2},{1,0},{1,1}};
static uint8_t *done;static int32_t *zero;
static void fin(void){*(volatile uint32_t *)done=0xffffffff;qd_hmx_finish(zero,done);uint32_t t=0;while(*(volatile uint32_t *)done&&t++<1000000)__asm__ volatile("nop");if(*(volatile uint32_t*)done)puts("NO COMPLETION");}
static uint32_t rs=777;static uint32_t rnd(void){rs^=rs<<13;rs^=rs>>17;rs^=rs<<5;return rs;}
static uint8_t *heap;static uint32_t at;static uint8_t *ALLOC(uint32_t n){at=(at+2047)&~2047u;uint8_t *r=heap+at;at+=n;return r;}
static void merges(uint8_t *o,const uint8_t *lo,const uint8_t *hi,const uint8_t *lo2,const uint8_t *hi2,HVX_Vector mv){
    const HVX_Vector c4096=Q6_Vh_vsplat_R(4096),c8192=Q6_Vh_vsplat_R(8192);
    for(int y=0;y<8;y++){
        const HVX_Vector *L=(const HVX_Vector *)(lo+y*256),*H=(const HVX_Vector *)(hi+y*256);
        HVX_Vector l0=L[0],l1=L[1],l2=*(const HVX_Vector *)(lo2+y*256),h0=H[0],h1=H[1],h2=*(const HVX_Vector *)(hi2+y*256);
        HVX_Vector ls[2]={Q6_V_valign_VVR(l1,l0,32),Q6_V_valign_VVR(l2,l1,32)},hs[2]={Q6_V_valign_VVR(h1,h0,32),Q6_V_valign_VVR(h2,h1,32)};
        for(int i=0;i<2;i++){
            HVX_VectorPair w=Q6_W_vshuff_VVR(hs[i],ls[i],-1);
            HVX_Vector a=Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vmax_VuhVuh(Q6_V_lo_W(w),c4096),c8192),b=Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vmax_VuhVuh(Q6_V_hi_W(w),c4096),c8192);
            a=Q6_Vh_vsub_VhVh(a,c4096);b=Q6_Vh_vsub_VhVh(b,c4096);
            a=Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_Vh_vadd_VhVh(a,a),mv);b=Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_Vh_vadd_VhVh(b,b),mv);
            ((HVX_Vector *)(o+y*256))[i]=Q6_Vub_vpack_VhVh_sat(b,a);
        }
    }
}
int main(int argc,char **argv) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x400000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    int cbO=atoi(argv[1]),nbw=atoi(argv[2]),NB=atoi(argv[3]),ncol=nbw+1,cbs=4*cbO;
    heap=(uint8_t *)base;at=0;
    /* input blocks [2 rows][ncol][cbs], scattered order */
    int nblk=2*ncol*cbs;uint8_t **ib=malloc(nblk*sizeof *ib);uint8_t *pool=ALLOC(nblk*2048+2048);
    int *perm=malloc(nblk*sizeof(int));for(int i=0;i<nblk;i++)perm[i]=i;for(int i=nblk-1;i>0;i--){int j=rnd()%(i+1),t=perm[i];perm[i]=perm[j];perm[j]=t;}
    for(int i=0;i<nblk;i++){ib[i]=pool+perm[i]*2048;for(int k=0;k<2048;k++)ib[i][k]=(uint8_t)(rnd()>>(rnd()%8));}
#define IB(r,c,cb) ib[((r)*ncol+(c))*cbs+(cb)]
    int KB=9*cbO;int8_t *B=(int8_t *)ALLOC(NB*KB*1024);for(int i=0;i<NB*KB*1024;i++)B[i]=(int8_t)((int32_t)rnd()>>(24+rnd()%6));
    uint8_t *C=ALLOC(NB*ncol*4096),*ob=ALLOC(NB*nbw*2048);
    int32_t *fi=(int32_t *)ALLOC(NB*512);zero=(int32_t *)ALLOC(256);done=ALLOC(2048);
    uint32_t *rec=(uint32_t *)ALLOC(ncol*cbO*32),*st=(uint32_t *)ALLOC(ncol*8);HVX_Vector *mh=(HVX_Vector *)ALLOC(NB*128);
    int32_t pre[1024],M[1024],corr[1024];
    for(int n=0;n<32*NB;n++){pre[n]=2+rnd()%6;M[n]=1024+rnd()%1024;corr[n]=(int32_t)(rnd()%(4096u<<pre[n]))-(int32_t)(rnd()%(300000u));
        fi[(n>>5)*128+(n&31)]=(int32_t)((uint32_t)(24-pre[n])<<10);fi[(n>>5)*128+32+(n&31)]=corr[n];
        fi[(n>>5)*128+64+(n&31)]=(int32_t)((uint32_t)(16-pre[n])<<10);fi[(n>>5)*128+96+(n&31)]=corr[n]+(1<<(12+pre[n]));
        ((uint16_t *)mh)[(n>>5)*64+(n&31)]=(uint16_t)M[n];((uint16_t *)mh)[(n>>5)*64+32+(n&31)]=(uint16_t)M[n];}
    for(int k=0;k<64;k++)zero[k]=k<32?0x404000:0;
    static const uint32_t rl[4]={0x120,0x100,0x020,0x000};
    for(int c=0;c<ncol;c++)for(int g=0;g<cbO;g++){uint32_t *r=rec+(c*cbO+g)*8;
        for(int q=0;q<4;q++){uint32_t a=(uint32_t)IB(0,c,g*4+q),b=(uint32_t)IB(1,c,g*4+q);r[2*q]=a|rl[q];r[2*q+1]=(b-a)|0x71fu;}}
    for(int j=0;j<NB;j++){
        for(int c=0;c<ncol;c++){uint8_t *d=C+((size_t)j*ncol+c)*4096;st[2*c]=(uint32_t)d;st[2*c+1]=(uint32_t)(d+2048);}
        qd_hmx_slide2p_n(rec,B+(size_t)j*KB*1024,fi+j*128,st,cbO,ncol);
    }
    fin();
    for(int j=0;j<NB;j++)for(int bx=0;bx<nbw;bx++){uint8_t *c=C+((size_t)j*ncol+bx)*4096;merges(ob+((size_t)j*nbw+bx)*2048,c,c+2048,c+4096,c+4096+2048,mh[j]);}
    long bad=0,tot=0,cn=0,cz=0,cs=0;
    for(int j=0;j<NB;j++)for(int bx=0;bx<nbw;bx++)for(int y=0;y<8;y++)for(int x=0;x<8;x++)for(int n=0;n<32;n++){
        int64_t acc=0;
        for(int g=0;g<cbO;g++)for(int r=0;r<9;r++){
            int y3=S2_TAPS[r][0],x3=S2_TAPS[r][1],ky=y3>>1,kx=(x3+1)>>1,q=(y3&1)*2+(x3&1);
            int row=y+ky,col=8*bx+x+kx;const uint8_t *p=IB(row>>3,col>>3,g*4+q)+((row&7)*8+(col&7))*32;
            const int8_t *w=B+((size_t)j*KB+g*9+r)*1024;
            for(int ci=0;ci<32;ci++)acc+=(int64_t)p[ci]*w[((ci/4)*32+n)*4+ci%4];
        }
        int ch=32*j+n;int64_t q=(acc+corr[ch])>>pre[ch];int64_t v=(q*M[ch]+8192)>>14;uint8_t e=v<0?0:v>255?255:v;
        cn+=q<0;cs+=q>=4096;cz+=q>=0&&q<4096;
        uint8_t g2=ob[((size_t)j*nbw+bx)*2048+(y*8+x)*32+n];tot++;
        if(g2!=e){if(bad<6)printf("bad j%d bx%d y%d x%d n%d q=%lld got %u want %u\n",j,bx,y,x,n,(long long)q,g2,e);bad++;}
    }
    printf("s2 cbO=%d nbw=%d NB=%d: %ld/%ld bad (q<0 %ld, in range %ld, q>=4096 %ld)\n",cbO,nbw,NB,bad,tot,cn,cz,cs);
    return 0;
}
