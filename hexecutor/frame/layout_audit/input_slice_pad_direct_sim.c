#include "input_slice_pad_direct.h"
#include <stdio.h>
#include <stdlib.h>

static size_t count4(const int32_t d[4]) {
    return (size_t)d[0] * d[1] * d[2] * d[3];
}

static void reference(uint8_t *dst, const uint8_t *src,
        const int32_t in[4], const int32_t out[4],
        const int32_t start[4], const int32_t pad[4], uint8_t fill) {
    int32_t lo[4], s0[4], s1[4];
    for (int i = 0; i < 4; ++i) {
        lo[i] = start[i] - pad[i];
        s0[i] = lo[i] > 0 ? lo[i] : 0;
        s1[i] = lo[i] + out[i] < in[i] ? lo[i] + out[i] : in[i];
        if (s1[i] < s0[i]) s1[i] = s0[i];
    }
    memset(dst, fill, count4(out));
    for (int32_t b=s0[0]; b<s1[0]; ++b)
      for (int32_t h=s0[1]; h<s1[1]; ++h)
       for (int32_t w=s0[2]; w<s1[2]; ++w)
        for (int32_t c=s0[3]; c<s1[3]; ++c) {
            size_t si=(((size_t)b*in[1]+h)*in[2]+w)*in[3]+c;
            size_t di=(((size_t)(b-lo[0])*out[1]+(h-lo[1]))*out[2]
                       +(w-lo[2]))*out[3]+(c-lo[3]);
            dst[di]=src[si];
        }
}

static int check_case(const int32_t in[4], const int32_t out[4],
        const int32_t start[4], const int32_t pad[4]) {
    size_t ni=count4(in), no=count4(out);
    uint8_t *src=malloc(ni ? ni : 1), *got=malloc(no ? no : 1),
            *want=malloc(no ? no : 1);
    if (!src || !got || !want) return 1;
    for (size_t i=0; i<ni; ++i) src[i]=(uint8_t)(i*37u+11u);
    memset(got,0xa5,no); reference(want,src,in,out,start,pad,0x6d);
    int handled=qd_layout_input_slice_pad_direct_u8(got,no,src,ni,in,out,start,pad,0x6d);
    int bad=!handled || memcmp(got,want,no)!=0;
    free(src);free(got);free(want);return bad;
}

int main(void) {
    unsigned cases=0;
    /* Exhaust crop/pad placements on small NHWC shapes, including no overlap,
     * channel padding, batch padding, and empty dimensions. */
    for (int32_t b=1;b<=2;b++) for (int32_t h=1;h<=2;h++)
      for (int32_t w=1;w<=2;w++) for (int32_t c=1;c<=2;c++) {
        int32_t in[4]={b,h,w,c};
        for (int32_t ob=1;ob<=3;ob++) for (int32_t oh=1;oh<=3;oh++)
          for (int32_t ow=1;ow<=3;ow++) for (int32_t oc=1;oc<=3;oc++) {
            int32_t out[4]={ob,oh,ow,oc};
            for (int32_t sb=-1;sb<=2;sb++)
             for (int32_t sh=-1;sh<=2;sh++)
              for (int32_t sw=-1;sw<=2;sw++)
               for (int32_t sc=-1;sc<=2;sc++) {
                int32_t start[4]={sb,sh,sw,sc};
                for (int32_t pb=0;pb<=1;pb++) for (int32_t ph=0;ph<=1;ph++)
                 for (int32_t pw=0;pw<=1;pw++) for (int32_t pc=0;pc<=1;pc++) {
                    int32_t pad[4]={pb,ph,pw,pc};
                    if (check_case(in,out,start,pad)) {
                        fprintf(stderr,"mismatch case %u in=%d,%d,%d,%d out=%d,%d,%d,%d start=%d,%d,%d,%d pad=%d,%d,%d,%d\n",cases,in[0],in[1],in[2],in[3],out[0],out[1],out[2],out[3],start[0],start[1],start[2],start[3],pad[0],pad[1],pad[2],pad[3]);return 1;
                    }
                    ++cases;
                 }
               }
          }
      }

    /* The alias guard and byte-capacity guard must reject without mutation. */
    {
        uint8_t buf[128], before[128];
        int32_t d[4]={1,1,8,1}, start[4]={0,0,0,0}, pad[4]={0,0,0,0};
        for (size_t i=0;i<sizeof buf;i++)buf[i]=(uint8_t)i;
        memcpy(before,buf,sizeof buf);
        if (qd_layout_input_slice_pad_direct_u8(buf+4,8,buf,8,d,d,start,pad,0)) return 2;
        if (memcmp(buf,before,sizeof buf)) return 3;
        if (qd_layout_input_slice_pad_direct_u8(buf+32,7,buf,8,d,d,start,pad,0)) return 4;
        if (memcmp(buf,before,sizeof buf)) return 5;
    }
    printf("PASS exhaustive crop/pad cases=%u; alias/capacity guards\n",cases);
    return 0;
}
