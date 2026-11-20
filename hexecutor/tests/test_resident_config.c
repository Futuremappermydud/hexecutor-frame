/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdio.h>
#include <stdlib.h>
#include "../native/qd_graph_dsp.h"
static void check(int ok){if(!ok)abort();}
int main(void) {
    QdsConfig c={0};c.magic=QDS_MAGIC;c.version=1;c.code_bytes=1024;
    c.image=0x100000;c.image_bytes=256;c.pool[5]=0x200000;c.pool_bytes[5]=4096;
    c.pool[1]=0x1000000;c.pool_bytes[1]=4096;
    for(int i=0;i<4;i++){c.input[i]=0x1100000+i*4096;c.input_bytes[i]=4096;}
    for(int i=0;i<2;i++){c.output[i]=0x1200000+i*4096;c.output_bytes[i]=4096;}
    c.heap=0x2000000;c.heap_bytes=0x2000000-4096;check(qds_config_valid(&c));
    unsigned count=1;QdsConfig bad;
#define BAD(expr) do {bad=c;expr;check(!qds_config_valid(&bad));count++;}while(0)
    BAD(bad.magic=0);BAD(bad.code_bytes=0x100001);BAD(bad.image_bytes=255);
    BAD(bad.image=0);BAD(bad.image_bytes=0xffffffff);BAD(bad.pool[1]=bad.image);
    BAD(bad.input[0]=bad.output[0]);BAD(bad.input[0]=0x1ffff80);
    BAD(bad.output[0]=0x4000000);BAD(bad.output_bytes[0]=0xffffffff);
    BAD(bad.heap=bad.pool[1]);BAD(bad.heap_bytes+=4096);BAD(bad.input[0]++);
    BAD(bad.input_bytes[0]=0);BAD(bad.pool[5]=0x1000000);
    printf("PASS %u valid/negative allocation contracts\n",count);return 0;
}
