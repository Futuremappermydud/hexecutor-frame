/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Compare direct arena collection with the established dense -> blocked write,
 * including canaries, holes, shuffled pools, slices, partial rows and flattening.
 * cc -O1 -g -fsanitize=address,undefined -ffp-contract=off -fno-strict-aliasing
 *    frame/tests/test_native_collect.c frame/qd_resident.c -lm -pthread -o TEST
 */
#include "../native/qd_graph.c"
#include <stdio.h>

static void one(int height, int width, int channels, int flatten, int partial, int helpers, int blockout, int aliases) {
    QdgGraph g={0},ref={0};
    g.block_output=1;
    ConvState cs={0};g.conv=&cs;
    T t={0};
    C c={0};uint32_t output=0;c.out=&output;
    g.t=&t;g.c=&c;ref.t=&t;ref.c=&c;
    t.flags=QDR_BLOCKED;t.esize=1;t.dtype=DT_U8;t.crouton=1;
    t.m[0]=2;t.m[1]=height+13;t.m[2]=width+19;t.m[3]=channels+64;t.slice=3;
    t.nblocks=3+2*((t.m[1]+7)/8)*((t.m[2]+7)/8)*((t.m[3]+31)/32);
    uint32_t *blocks=calloc(t.nblocks,sizeof *blocks);t.blocks=blocks;
    size_t bytes=(size_t)t.nblocks*2048+128;
    for(int pool=1;pool<=3;pool+=2) {
        g.pool[pool]=malloc(bytes);ref.pool[pool]=malloc(bytes);
        memset(g.pool[pool],0xa5,bytes);memset(ref.pool[pool],0xa5,bytes);
    }
    for(uint32_t i=0;i<t.nblocks;i++)
        blocks[i]=i%11==0?0:((t.nblocks-1-i)*2048)|(i%2?1:3);
    if(aliases)for(uint32_t i=4;i<t.nblocks;i+=5)blocks[i]=blocks[i-3];
    Job j={0};j.oh=height;j.ow=width;j.co=channels;j.pending=j.busy=1;
    j.origin[0]=1;j.origin[1]=blockout?8:5;j.origin[2]=partial?3:8;j.origin[3]=32;
    Req q={0};q.active=1;q.job=&j;
    Layout *l=&q.L;l->hmx=1;l->n=channels;l->oh=flatten?1:height;
    l->ow=flatten?height*width:width;l->owp=(l->ow+31)/32*32;
    l->co=128;l->cb=l->oh*l->owp*channels;
    if(blockout){l->out_h=height;l->out_w=width;}
    g.arena=malloc(l->co+l->cb+128);memset(g.arena,0xcc,l->co+l->cb+128);
    uint8_t *dense=malloc((size_t)height*width*channels);
    for(int h=0;h<height;h++)for(int w=0;w<width;w++)for(int n=0;n<channels;n++) {
        size_t i=((size_t)h*width+w)*channels+n;
        uint8_t value=(uint8_t)(i*73+i/47+19);dense[i]=value;
        uint32_t p=flatten?h*width+w:h*l->owp+w;
        size_t offset=blockout?((((size_t)(h/8)*(width/8)+w/8)*(channels/32)+n/32)*2048+
                               (h%8)*256+(w%8)*32+n%32):
                              (((size_t)(p/32)*(channels/32)+n/32)*32+p%32)*32+n%32;
        g.arena[l->co+offset]=value;
    }
    A a={{1,height,width,channels},DT_U8,0,dense};
    if(t_write_region(&ref,&t,&a,j.origin))abort();
    if(helpers){int cpus[3]={-1,-1,-1};if(qdg_set_helpers(&g,3,cpus)!=3)abort();helpers_active(&g,1);}
    if(collect(&g,&q) || q.active || j.busy || j.pending)abort();
    for(int pool=1;pool<=3;pool+=2)if(memcmp(g.pool[pool],ref.pool[pool],bytes)) {
        fprintf(stderr,"FAIL h=%d w=%d c=%d flat=%d partial=%d helpers=%d pool=%d\n",
                height,width,channels,flatten,partial,helpers,pool);abort();
    }
    /* Every unsupported layout must keep the existing collection path. */
    j.pending=1;t.flags|=QDR_DIRECT;
    if(hmx_collect_direct(&g,&q))abort();
    t.flags=QDR_BLOCKED;j.origin[3]=1;
    if(hmx_collect_direct(&g,&q))abort();
    stop_helpers(&g);
    for(int pool=1;pool<=3;pool+=2){free(g.pool[pool]);free(ref.pool[pool]);}
    free(g.arena);free(dense);free(blocks);
}
int main(void) {
    const int shapes[][3]={{1,1,32},{3,7,64},{8,8,32},{9,35,96},{16,64,128},{17,67,32}};
    unsigned count=0;
    for(unsigned s=0;s<sizeof shapes/sizeof shapes[0];s++)
        for(int f=0;f<2;f++)for(int p=0;p<2;p++)for(int helpers=0;helpers<2;helpers++) {
            one(shapes[s][0],shapes[s][1],shapes[s][2],f,p,helpers,0,0);count++;
        }
    for(int f=0;f<2;f++)for(int helpers=0;helpers<2;helpers++) {
        one(8,8,32,f,0,helpers,1,0);one(16,64,128,f,0,helpers,1,0);
        one(24,24,96,f,0,helpers,1,0);count+=3;
        one(16,64,128,f,0,helpers,1,1);one(17,67,96,f,1,helpers,0,1);count+=2;
    }
    printf("PASS: %u direct HMX collection cases\n",count);return 0;
}
