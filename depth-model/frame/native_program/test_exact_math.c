/* Host check: prepared_exact_math.h against the reference double formulas.
 *   cc -O2 -ffp-contract=off -fno-fast-math test_exact_math.c -lm -o t && ./t */
#include "prepared_exact_math.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
static uint32_t seed=12345u;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
static uint32_t fb(float f){uint32_t b;memcpy(&b,&f,4);return b;}
static float bf(uint32_t b){float f;memcpy(&f,&b,4);return f;}
static uint32_t dq_ref(int32_t v,int32_t zp,double scale){
    double sc=(double)(float)scale;
    double exact=((double)v-zp)*sc;
    float rounded=(float)exact;
    float away=nextafterf(rounded,exact<0?-INFINITY:INFINITY);
    return fb(exact==((double)rounded+(double)away)*0.5?away:rounded);
}
static uint16_t q_ref(float s,int32_t zp,double scale){
    double inv=(double)(float)(1.0/scale);
    double v=floor((double)s*inv+0.5)+zp;
    return (uint16_t)(v<0?0:v>65535?65535:v);
}
int main(void){
    long n=0,bad=0,fb_=0;
    /* DEQUANTIZE: every u16 value for many scales and zero points. */
    for(int t=0;t<400;t++){
        double scale=t<5?(double)(1<<t)/65536.0:ldexp(1.0+(rnd()%100000)/100000.0,(int)(rnd()%40)-30);
        if(t%7==0) scale=(double)bf(0x3f800000u|(rnd()&0x7fffffu))*ldexp(1,(int)(rnd()%20)-10);
        int32_t zp=t%3==0?32768:t%3==1?(int32_t)(rnd()%65536):0;
        uint32_t m24;int32_t e;
        if(!dq_scale(fb((float)scale),&m24,&e)) continue;
        for(int32_t v=0;v<65536;v++){
            uint32_t got,want=dq_ref(v,zp,scale);n++;
            if(!dq_exact(v-zp,m24,e,&got)){fb_++;continue;}
            if(got!=want){if(bad++<10)printf("DQ v%d zp%d scale%.9g got%08x want%08x\n",v,zp,scale,got,want);}
        }
    }
    printf("dequantize: %ld values, %ld mismatches, %ld fallbacks\n",n,bad,fb_);
    long n2=0,bad2=0,fb2=0;
    /* QUANTIZE: random float bit patterns, values near n*scale and (n+1/2)*scale. */
    for(int t=0;t<300;t++){
        double scale=ldexp(1.0+(rnd()%1000000)/1000000.0,(int)(rnd()%30)-25);
        if(t%5==0) scale=ldexp(1.0,(int)(rnd()%30)-25);
        int32_t zp=t%2?32768:(int32_t)(rnd()%65536);
        float invf=(float)(1.0/scale);uint32_t mi;int32_t ei;
        if(!dq_scale(fb(invf),&mi,&ei)) continue;
        for(int i=0;i<200000;i++){
            float s;
            switch(i%4){
            case 0: s=bf(rnd());break;
            case 1: s=(float)(((int32_t)(rnd()%140000)-70000)*scale);break;
            case 2: s=(float)((((int32_t)(rnd()%140000)-70000)+0.5)*scale);break;
            default: {float b=(float)((((int32_t)(rnd()%140000)-70000)+0.5)/invf);uint32_t u=fb(b)+(rnd()%5)-2;s=bf(u);}
            }
            if(isnan(s)) continue;
            uint16_t got,want=q_ref(s,zp,scale);n2++;
            if(!q_exact(fb(s),mi,ei,zp,&got)){fb2++;continue;}
            if(got!=want){if(bad2++<10)printf("Q s%.9g(%08x) zp%d scale%.9g got%u want%u\n",s,fb(s),zp,scale,got,want);}
        }
    }
    printf("quantize: %ld values, %ld mismatches, %ld fallbacks\n",n2,bad2,fb2);
    return bad||bad2;
}
