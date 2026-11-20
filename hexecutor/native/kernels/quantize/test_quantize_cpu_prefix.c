/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "quantize_u16.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while (0)
static int read_exact(const char *path, void *p, size_t n) {
    FILE *f=fopen(path,"rb"); if(!f)return 0;
    int ok=fread(p,1,n,f)==n && fgetc(f)==EOF; fclose(f); return ok;
}
int main(int argc,char **argv) {
    CHECK(argc==3);
    char p[1024]; const char *model=argv[2];
    CHECK(!strcmp(model,"primary")||!strcmp(model,"alternate"));
    uint32_t src_off=!strcmp(model,"primary")?1089536u:3997696u;
    uint32_t dst_off=!strcmp(model,"primary")?7589888u:2775040u;
    float src[32]; uint16_t expected[32], guarded[40];
    snprintf(p,sizeof p,"%s/%s-smoke.f32",argv[1],model); CHECK(read_exact(p,src,sizeof src));
    snprintf(p,sizeof p,"%s/%s-smoke.u16",argv[1],model); CHECK(read_exact(p,expected,sizeof expected));
    uint8_t *pool=malloc(8u*1024u*1024u),*before=malloc(8u*1024u*1024u); CHECK(pool&&before);
    snprintf(p,sizeof p,"%s/%s-smoke.pool1",argv[1],model); CHECK(read_exact(p,pool,8u*1024u*1024u));
    memcpy(before,pool,8u*1024u*1024u);
    CHECK(!memcmp(pool+src_off,src,sizeof src));
    CHECK(nq_quantize_f32_to_u16((float *)(void *)(pool+src_off),128,(uint16_t *)(void *)(pool+dst_off),64,32,32768u,0.0003051804378628731f));
    CHECK(!memcmp(pool+dst_off,expected,64));
    CHECK(!memcmp(pool,before,dst_off));
    CHECK(!memcmp(pool+dst_off+64u,before+dst_off+64u,8u*1024u*1024u-dst_off-64u));
    for(unsigned i=0;i<40;i++)guarded[i]=0xa55au;
    CHECK(nq_quantize_f32_to_u16(src,sizeof src,&guarded[4],64,32,32768u,0.0003051804378628731f));
    CHECK(!memcmp(&guarded[4],expected,64));
    for(unsigned i=0;i<4;i++)CHECK(guarded[i]==0xa55au&&guarded[36+i]==0xa55au);
    CHECK(!nq_quantize_f32_to_u16(src,sizeof src,&guarded[4],62,32,32768u,0.0003051804378628731f));
    CHECK(!nq_quantize_f32_to_u16(src,sizeof src,(uint16_t *)(void *)src,64,32,32768u,0.0003051804378628731f));
    float saved=src[3]; src[3]=0.0f/0.0f;
    for(unsigned i=0;i<40;i++)guarded[i]=0xa55au;
    CHECK(!nq_quantize_f32_to_u16(src,sizeof src,&guarded[4],64,32,32768u,0.0003051804378628731f));
    for(unsigned i=0;i<40;i++)CHECK(guarded[i]==0xa55au);
    src[3]=saved;
    free(before); free(pool);
    puts("PASS Quantize@FH.Ff CPU-prefix: exact 32-value output, pool1 outside-target bytes, output guards, bad size/alias/nonfinite rejection");
    return 0;
}
