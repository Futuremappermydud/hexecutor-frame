/* Randomised check of qd_hmx_slide3_n vs a scalar model (silicon cvt.ub). */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
#include <hexagon_sim_timer.h>
extern void qd_hmx_finish(void *,void *);
extern void qd_hmx_slide3_n(const uint32_t *rec,const void *b,const void *fi,const uint32_t *st,uint32_t cbs,uint32_t ncol);
static uint8_t *done;static int32_t *zero;
static void fin(void){*(volatile uint32_t *)done=0xffffffff;qd_hmx_finish(zero,done);uint32_t t=0;while(*(volatile uint32_t *)done&&t++<1000000)__asm__ volatile("nop");if(*(volatile uint32_t*)done)puts("NO COMPLETION");}
static uint32_t rs=12345;static uint32_t rnd(void){rs^=rs<<13;rs^=rs>>17;rs^=rs<<5;return rs;}
/* */
static uint8_t *heap;static uint32_t at;static uint8_t *ALLOC(uint32_t n){at=(at+2047)&~2047u;uint8_t *r=heap+at;at+=n;return r;}
int main(int argc,char **argv) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x400000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    int cbs=atoi(argv[1]),nbw=atoi(argv[2]),NB=atoi(argv[3]),ncolin=nbw+2;
    heap=(uint8_t *)base;at=0;
    /* input blocks [3][ncolin][cbs], shuffled placement */
    int nblk=3*ncolin*cbs;uint8_t **ib=malloc(nblk*sizeof *ib);
    uint8_t *pool=ALLOC(nblk*2048+2048);
    int *perm=malloc(nblk*sizeof(int));for(int i=0;i<nblk;i++)perm[i]=i;for(int i=nblk-1;i>0;i--){int j=rnd()%(i+1),t=perm[i];perm[i]=perm[j];perm[j]=t;}
    for(int i=0;i<nblk;i++){ib[i]=pool+perm[i]*2048;for(int k=0;k<2048;k++)ib[i][k]=rnd();}
#define IB(r,c,cb) ib[((r)*ncolin+(c))*cbs+(cb)]
    int KB=9*cbs;int8_t *B=(int8_t *)ALLOC(NB*KB*1024);for(int i=0;i<NB*KB*1024;i++)B[i]=(int8_t)rnd();
    uint8_t *ob=ALLOC(NB*nbw*2048),*dummy=ALLOC(4096);
    int32_t *fi=(int32_t *)ALLOC(NB*256);zero=(int32_t *)ALLOC(256);done=ALLOC(2048);
    uint32_t *rec=(uint32_t *)ALLOC(ncolin*cbs*16),*st=(uint32_t *)ALLOC(ncolin*8);
    int32_t pre[1024],M[1024],corr[1024];
    for(int n=0;n<32*NB;n++){pre[n]=8+rnd()%9;M[n]=1024+rnd()%1024;corr[n]=(int32_t)(rnd()%200000)-100000;
        fi[(n>>5)*64+(n&31)]=(int32_t)(0x400000u|((uint32_t)(20-pre[n])<<10)|(uint32_t)(M[n]-1024));fi[(n>>5)*64+32+(n&31)]=corr[n];}
    for(int k=0;k<64;k++)zero[k]=k<32?0x404000:0;
    for(int c=0;c<ncolin;c++)for(int cb=0;cb<cbs;cb++){uint32_t *r=rec+(c*cbs+cb)*4;
        uint32_t a=(uint32_t)IB(0,c,cb),b=(uint32_t)IB(1,c,cb),d=(uint32_t)IB(2,c,cb);
        r[0]=a|0x740;r[1]=(b-a)|0x71f;r[2]=b|0x140;r[3]=(d-b)|0x71f;}
    memset(ob,0xee,NB*nbw*2048);
    uint64_t t0=hexagon_sim_read_pcycles();
    for(int j=0;j<NB;j++){
        /* store for column c (virtual tile c, c=0..ncolin-1) after loading input col c:
         * tile T=c: pixel 7 of out block T-2, pixels 0..6 of block T-1 */
        for(int c=0;c<ncolin;c++){
            uint8_t *d=(c-2>=0&&c-2<nbw)?ob+((size_t)j*nbw+c-2)*2048:dummy;
            st[2*c]=(uint32_t)d|0xe0;st[2*c+1]=0x700;}
        qd_hmx_slide3_n(rec,B+(size_t)j*KB*1024,fi+j*64,st,cbs,ncolin);
    }
    fin();
    uint64_t t1=hexagon_sim_read_pcycles();
    long bad=0,tot=0;
    for(int j=0;j<NB;j++)for(int bx=0;bx<nbw;bx++)for(int y=0;y<8;y++)for(int x=0;x<8;x++)for(int n=0;n<32;n++){
        int64_t acc=0;
        for(int cb=0;cb<cbs;cb++)for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++){
            int row=7+y+ky,col=7+8*bx+x+kx;const uint8_t *p=IB(row>>3,col>>3,cb)+((row&7)*8+(col&7))*32;
            const int8_t *w=B+((size_t)j*KB+cb*9+ky*3+(2-kx))*1024;
            for(int ci=0;ci<32;ci++)acc+=(int64_t)p[ci]*w[((ci/4)*32+n)*4+ci%4];
        }
        int ch=32*j+n;int64_t q=(acc+corr[ch])>>pre[ch];int64_t v=((2*q+1)*M[ch]+(1<<14))>>15;uint8_t e=v<0?0:v>255?255:v;
        uint8_t g=ob[((size_t)j*nbw+bx)*2048+(y*8+x)*32+n];tot++;
        if(g!=e){if(bad<5)printf("bad j%d bx%d y%d x%d n%d got %u want %u\n",j,bx,y,x,n,g,e);bad++;}
    }
    printf("cbs=%d nbw=%d NB=%d: %ld/%ld bad, %llu pcycles\n",cbs,nbw,NB,bad,tot,(unsigned long long)(t1-t0));
    return 0;
}
