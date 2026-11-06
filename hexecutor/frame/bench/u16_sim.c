/* ISA-simulator check of the payload's u16 HVX convolution and fused epilogue
 * against the exact int64 reference requantization (htp_emu conv_requant). */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "../qd_conv.c"
static uint8_t xbuf[1u<<20] __attribute__((aligned(128)));
static int32_t wbuf[1u<<16] __attribute__((aligned(128)));
static uint8_t obuf[1u<<20] __attribute__((aligned(128)));
static QdQuant16 qbuf[256] __attribute__((aligned(128)));
static uint16_t xs[1u<<18];
static int8_t ws[1u<<16];
static uint32_t seed=7;
static uint32_t rnd(void){seed=seed*1103515245u+12345u;return seed>>8;}
int main(void) {
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    /* kh, kw, ci, n, oh, owp, zp */
    static const uint32_t cases[][7]={{3,3,32,32,8,64,0},{1,1,12,32,1,256,32768},{3,3,8,32,4,32,65432},{1,1,64,64,1,128,1234},{3,3,32,32,8,96,65535}};
    uint32_t failures=0;
    for(uint32_t t=0;t<sizeof cases/sizeof cases[0];t++) {
        QdConv c;memset(&c,0,sizeof c);
        uint32_t ci=cases[t][2],zp=cases[t][6];
        c.kh=cases[t][0];c.kw=cases[t][1];c.n=cases[t][3];c.oh=cases[t][4];c.owp=cases[t][5];
        c.g=(ci+3)/4;c.h=c.oh+c.kh-1;c.wp=c.owp+c.kw-1;c.reserved[0]=QDC_U16;
        uint32_t W=c.wp,H=c.h,N=c.n,plane=H*c.g*W*4;
        for(uint32_t i=0;i<H*W*ci;i++)xs[i]=(uint16_t)(t==4?65535-(rnd()&7):rnd());
        for(uint32_t i=0;i<c.kh*c.kw*ci*N;i++)ws[i]=(int8_t)rnd();
        memset(xbuf,0,2*plane+256);
        for(uint32_t h=0;h<H;h++)for(uint32_t w=0;w<W;w++)for(uint32_t ch=0;ch<ci;ch++) {
            uint16_t v=xs[(h*W+w)*ci+ch];size_t at=((size_t)h*c.g+ch/4)*W*4+w*4+ch%4;
            xbuf[at]=(uint8_t)v;xbuf[plane+at]=(uint8_t)(v>>8);
        }
        memset(wbuf,0,sizeof wbuf);
        for(uint32_t k=0;k<c.kh*c.kw;k++)for(uint32_t g=0;g<c.g;g++)for(uint32_t n=0;n<N;n++)for(uint32_t q=0;q<4;q++)
            ((int8_t *)wbuf)[(((size_t)k*c.g+g)*N+n)*4+q]=g*4+q<ci?ws[((size_t)k*ci+g*4+q)*N+n]:0;
        int64_t M[256],post[256],bias[256],wsum[256];
        for(uint32_t n=0;n<N;n++) {
            M[n]=1024+(rnd()&1023);post[n]=1+rnd()%15;bias[n]=(int64_t)(int32_t)rnd()>>(rnd()%24);
            wsum[n]=0;for(uint32_t i=0;i<c.kh*c.kw*ci;i++)wsum[n]+=ws[(size_t)i*N+n];
            int64_t corr=256*bias[n]-(int64_t)zp*wsum[n],ch=corr>>8,rd=1ll<<(post[n]-1);
            int64_t top=((int64_t)65535<<post[n])-rd,pmax=(top+M[n]-1)/M[n];
            qbuf[n]=(QdQuant16){(int32_t)ch,(int32_t)(corr-256*ch),(int32_t)pmax,(uint32_t)M[n],(uint32_t)post[n],(uint32_t)rd,{0,0}};
        }
        uint32_t units=0;
        for(uint32_t tid=0;tid<3;tid++)units+=compute(&c,xbuf,wbuf,(int32_t *)obuf,tid,(const QdQuant *)qbuf);
        uint32_t bad=0,checked=0;
        for(uint32_t oy=0;oy<c.oh;oy++)for(uint32_t ox=0;ox<c.owp;ox++)for(uint32_t n=0;n<N;n++) {
            int64_t acc=0;
            for(uint32_t ky=0;ky<c.kh;ky++)for(uint32_t kx=0;kx<c.kw;kx++)for(uint32_t ch=0;ch<ci;ch++)
                acc+=(int64_t)xs[((oy+ky)*W+ox+kx)*ci+ch]*ws[((size_t)(ky*c.kw+kx)*ci+ch)*N+n];
            int64_t a=acc-(int64_t)zp*wsum[n]+bias[n]*256;
            int64_t q=(((a>>8)*M[n])+((int64_t)1<<(post[n]-1)))>>post[n];
            q=q<0?0:q>65535?65535:q;
            uint16_t got=((uint16_t *)obuf)[(((size_t)oy*(N/2)+n/2)*(c.owp/32)+ox/32)*64+(n%2)*32+ox%32];
            bad+=got!=(uint16_t)q;checked++;
        }
        printf("u16 case %u kh=%u ci=%u n=%u oh=%u owp=%u zp=%u: %s (%u/%u bad, %u units)\n",(unsigned)t,(unsigned)c.kh,(unsigned)ci,
               (unsigned)N,(unsigned)c.oh,(unsigned)c.owp,(unsigned)zp,bad?"FAIL":"PASS",(unsigned)bad,(unsigned)checked,(unsigned)units);
        failures+=bad!=0;
    }
    SIM_RELEASE_HVX;
    printf(failures?"FAILURES\n":"ALL PASS\n");
    return failures?1:0;
}
