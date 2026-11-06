#include "nprg2_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t u16(const uint8_t *p){return (uint16_t)p[0]|(uint16_t)p[1]<<8;}
static uint32_t u24(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16;}
static uint32_t u32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static void w16(uint8_t *p,uint16_t v){p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8);}
static void w32(uint8_t *p,uint32_t v){p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8);p[2]=(uint8_t)(v>>16);p[3]=(uint8_t)(v>>24);}
#define CHECK(c) do { if(!(c)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1);} } while(0)

static uint8_t *read_file(const char *path,uint32_t *n){FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long z=ftell(f);CHECK(z>0&&z<=NPRG2_MAX_IMAGE_BYTES*8L);CHECK(!fseek(f,0,SEEK_SET));uint8_t *p=malloc((size_t)z);CHECK(p);CHECK(fread(p,1,(size_t)z,f)==(size_t)z);fclose(f);*n=(uint32_t)z;return p;}
static void hashes_for_qdrx(const uint8_t *q,uint32_t n,Nprg2ExpectedHashes *h,uint8_t source[32]){CHECK(n>=96);nprg2_sha256(q,n,source);h->context_sha256=q+32;h->constants_sha256=q+64;h->source_sha256=source;}
static void alloc_pools(const uint8_t *nprg,Nprg2Pool p[7]){for(int i=0;i<7;i++){uint32_t n=u32(nprg+144+i*4);p[i].bytes=n;p[i].base=n?calloc(1,n):NULL;CHECK(!n||p[i].base);}}
static void free_pools(Nprg2Pool p[7]){for(int i=0;i<7;i++)free(p[i].base);}
static void reset_crc(uint8_t *p,uint32_t n){w32(p+12,nprg2_crc32(p,n,12,4));}
static int opens_as(uint8_t *candidate,uint32_t n,const Nprg2ExpectedHashes *h,Nprg2Pool pools[7],int expected){Nprg2Program p;Nprg2ExpectedHashes local=*h;uint8_t prepared[32];nprg2_sha256(candidate,n,prepared);local.prepared_sha256=prepared;int e=nprg2_open(&p,candidate,n,&local,pools,(Nprg2Allocator){0});if(e==NPRG2_OK)nprg2_close(&p);return e==expected;}

static int resolve_none(void *ctx,const Nprg2Program *p,uint32_t i,const Nprg2Command *c,Nprg2Capability *cap){(void)ctx;(void)p;(void)i;(void)c;(void)cap;return 0;}
static int resolve_fixture(void *ctx,const Nprg2Program *p,uint32_t i,const Nprg2Command *c,Nprg2Capability *cap){
    (void)ctx;(void)p;(void)i;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;
}
static int resolve_scratch_over(void *ctx,const Nprg2Program *p,uint32_t i,const Nprg2Command *c,Nprg2Capability *cap){
    (void)resolve_fixture(ctx,p,i,c,cap);cap->scratch_bytes=900000;return 1;
}

static void compare_qdrx(const uint8_t *q,uint32_t qbytes,const Nprg2Program *p){
    CHECK(qbytes>=256);uint32_t qto=u32(q+148),qbc=u32(q+156),qco=u32(q+164),qro=u32(q+172),qca=u32(q+188);
    CHECK(u32(q+152)==p->tensor_count);CHECK(u32(q+168)==p->command_count);
    for(int i=0;i<7;i++)CHECK(u32(q+96+4*i)==p->pool_bytes[i]);
    CHECK(u32(q+160)==p->expanded_block_count);
    for(uint32_t i=0;i<p->tensor_count;i++){
        const uint8_t *t=q+qto+(size_t)i*128,*g=p->geometry.data+(size_t)p->tensors[i].geometry*44;
        uint32_t flags=u32(t+12);CHECK(u32(t+4)==g[0]);CHECK(u32(t+8)==u32(g+2));
        CHECK((flags&(1u|64u|8u|16u|32u))==g[1]);
        if(flags&1u)for(int j=0;j<12;j++)CHECK((int32_t)u32(t+16+j*4)==(int16_t)((uint16_t)g[6+j*2]|(uint16_t)g[7+j*2]<<8));
        if(flags&8u)CHECK(!memcmp(t+104,g+6,16));
        if(flags&16u)CHECK(!memcmp(t+100,g+6,4));
        CHECK((int32_t)u32(t+88)==(int32_t)u32(g+30));CHECK(memcmp(t+92,g+34,8)==0);
        const Nprg2Tensor *nt=&p->tensors[i];
        if(flags&4u){CHECK(nt->storage_kind==1);CHECK(nt->pool==u32(t+64));CHECK(nt->offset==u32(t+68));}
        else if(flags&2u){CHECK(nt->storage_kind==2);CHECK(nt->block_count==u32(t+76));CHECK(nt->slice_origin==u32(t+80));uint32_t first=u32(t+72);for(uint32_t j=0;j<nt->block_count;j++)CHECK(p->expanded_blocks[nt->block_first+j]==u32(q+qbc+(size_t)(first+j)*4));}
        else CHECK(nt->storage_kind==0);
    }
    for(uint32_t i=0;i<p->command_count;i++){
        const uint8_t *qcmd=q+qco+(size_t)i*96,*pcmd=p->command.data+(size_t)i*20;
        CHECK(u32(qcmd)==u16(pcmd));CHECK(u32(qcmd+4)==pcmd[2]);CHECK(u32(qcmd+8)==pcmd[3]);
        CHECK((uint16_t)u32(qcmd+20)==u16(pcmd+4));
        CHECK(u32(qcmd+24)==u16(pcmd+6));CHECK(u32(qcmd+28)==pcmd[14]);
        CHECK(u32(qcmd+32)==u16(pcmd+8));CHECK(u32(qcmd+36)==pcmd[15]);
        CHECK(u32(qcmd+60)==pcmd[16]);CHECK(u32(qcmd+68)==pcmd[17]);
        CHECK(!!u32(qcmd+12)==!!pcmd[18]);
        for(uint32_t j=0;j<pcmd[14];j++)CHECK(u32(q+qro+(size_t)(u32(qcmd+24)+j)*4)==u16(p->refs.data+(u16(pcmd+6)+j)*2));
        for(uint32_t j=0;j<pcmd[15];j++)CHECK(u32(q+qro+(size_t)(u32(qcmd+32)+j)*4)==u16(p->refs.data+(u16(pcmd+8)+j)*2));
        uint32_t qwo=u32(q+180),qcoff=u32(qcmd+56),qcall=u32(qcmd+64);
        for(uint32_t j=0;j<pcmd[16];j++)CHECK(u32(q+qwo+(size_t)(qcoff+j)*4)==u32(p->vectors.data+(u16(pcmd+10)+j)*4));
        for(uint32_t j=0;j<pcmd[17];j++)CHECK(!memcmp(q+qca+(size_t)(qcall+j)*88,p->calls.data+(size_t)(u16(pcmd+12)+j)*88,88));
        (void)qca;
    }
    CHECK(p->graph_input_first==u32(q+132)&&p->graph_input_count==u32(q+136));
    CHECK(p->graph_output_first==u32(q+140)&&p->graph_output_count==u32(q+144));
    for(uint32_t i=0;i<p->graph_input_count;i++)CHECK(u32(q+qro+(size_t)(u32(q+132)+i)*4)==u16(p->refs.data+(p->graph_input_first+i)*2));
    for(uint32_t i=0;i<p->graph_output_count;i++)CHECK(u32(q+qro+(size_t)(u32(q+140)+i)*4)==u16(p->refs.data+(p->graph_output_first+i)*2));
}

static void test_malformed(const uint8_t *raw,uint32_t n,const Nprg2ExpectedHashes *h,Nprg2Pool pools[7]){
    uint8_t *m=malloc(n);CHECK(m);
    memcpy(m,raw,n);m[n-1]^=1;CHECK(opens_as(m,n,h,pools,NPRG2_BAD_CRC));
    memcpy(m,raw,n);m[180]=0xff;m[181]=0xff;m[182]=0xff;m[183]=0x7f;reset_crc(m,n);CHECK(opens_as(m,n,h,pools,NPRG2_BAD_SECTION));
    uint8_t badhash[32];memcpy(badhash,h->source_sha256,32);badhash[0]^=1;Nprg2ExpectedHashes wrong=*h;wrong.source_sha256=badhash;CHECK(opens_as((uint8_t *)raw,n,&wrong,pools,NPRG2_BAD_HASH));
    uint8_t badprepared[32];memcpy(badprepared,h->prepared_sha256,32);badprepared[0]^=1;wrong=*h;wrong.prepared_sha256=badprepared;Nprg2Program rejected;CHECK(nprg2_open(&rejected,raw,n,&wrong,pools,(Nprg2Allocator){0})==NPRG2_BAD_HASH);

    /* Make a referenced block delta a non-terminating five-byte varint. */
    memcpy(m,raw,n);uint32_t to=u32(m+188),bo=u32(m+196);int found=0;
    for(uint32_t i=0;i<u32(m+24);i++){uint8_t *t=m+to+(size_t)i*12;if(t[2]==2){uint32_t off=u24(t+3),count=u16(t+6);if(count>1){CHECK(off+8<n);for(int j=0;j<5;j++)m[bo+off+3+j]=0x80;found=1;break;}}}
    CHECK(found);reset_crc(m,n);CHECK(opens_as(m,n,h,pools,NPRG2_BAD_BLOCK));

    /* Slice origin must remain inside its tensor's own block span. */
    memcpy(m,raw,n);to=u32(m+188);found=0;
    for(uint32_t i=0;i<u32(m+24);i++){uint8_t *t=m+to+(size_t)i*12;if(t[2]==2){w32(t+8,u16(t+6));found=1;break;}}
    CHECK(found);reset_crc(m,n);CHECK(opens_as(m,n,h,pools,NPRG2_BAD_BLOCK));

    /* A command ref outside the tensor table is rejected before binding. */
    memcpy(m,raw,n);uint32_t ro=u32(m+212);w16(m+ro,0xffff);reset_crc(m,n);CHECK(opens_as(m,n,h,pools,NPRG2_BAD_COMMAND));
    /* Direct tensor address beyond the declared pool is rejected. */
    memcpy(m,raw,n);to=u32(m+188);found=0;
    for(uint32_t i=0;i<u32(m+24);i++){uint8_t *t=m+to+(size_t)i*12;if(t[2]==1){t[3]=1;w32(t+4,0xffffffffu);found=1;break;}}
    CHECK(found);reset_crc(m,n);CHECK(opens_as(m,n,h,pools,NPRG2_BAD_POOL));
    free(m);
}

static void run_model(const char *qpath,const char *npath,uint32_t expected_tensors,uint32_t expected_commands,uint32_t expected_geometries,uint32_t expected_blocks){
    uint32_t qn,nn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),source_sha[32],prepared_sha[32];Nprg2ExpectedHashes hashes;hashes_for_qdrx(q,qn,&hashes,source_sha);nprg2_sha256(raw,nn,prepared_sha);hashes.prepared_sha256=prepared_sha;
    Nprg2Pool pools[7]={0};alloc_pools(raw,pools);Nprg2Program p;int e=nprg2_open(&p,raw,nn,&hashes,pools,(Nprg2Allocator){0});if(e!=NPRG2_OK)fprintf(stderr,"nprg2_open(%s)=%d\n",npath,e);CHECK(e==NPRG2_OK);
    CHECK(p.tensor_count==expected_tensors&&p.command_count==expected_commands&&p.geometry_count==expected_geometries&&p.expanded_block_count==expected_blocks);
    Nprg2Pool badp[7];memcpy(badp,pools,sizeof badp);CHECK(badp[1].bytes>1);badp[1].bytes--;CHECK(opens_as(raw,nn,&hashes,badp,NPRG2_BAD_POOL));
    memcpy(badp,pools,sizeof badp);for(int i=0;i<7;i++)if(i!=1&&badp[i].bytes){badp[i].base=badp[1].base;break;}CHECK(opens_as(raw,nn,&hashes,badp,NPRG2_BAD_POOL));
    compare_qdrx(q,qn,&p);test_malformed(raw,nn,&hashes,pools);

    uint8_t *hot=malloc(300000);CHECK(hot);memset(hot,0xa5,300000);uint8_t before[32];memcpy(before,hot,sizeof before);
    Nprg2BindConfig cfg={.hot_memory=hot,.hot_capacity=300000,.scratch_budget=491968,.l2_available_bytes=786432,.pool1_is_verified_vtcm=0,.resolve=resolve_fixture};Nprg2Bound bound;
    CHECK(nprg2_bind_mock(&p,pools,&cfg,&bound)==NPRG2_BAD_BIND);CHECK(!memcmp(before,hot,sizeof before));
    cfg.pool1_is_verified_vtcm=1;cfg.resolve=NULL;CHECK(nprg2_bind_mock(&p,pools,&cfg,&bound)==NPRG2_UNSUPPORTED);CHECK(!memcmp(before,hot,sizeof before));
    cfg.resolve=resolve_none;CHECK(nprg2_bind_mock(&p,pools,&cfg,&bound)==NPRG2_UNSUPPORTED);CHECK(!memcmp(before,hot,sizeof before));
    cfg.resolve=resolve_scratch_over;CHECK(nprg2_bind_mock(&p,pools,&cfg,&bound)==NPRG2_SCRATCH);CHECK(!memcmp(before,hot,sizeof before));
    cfg.resolve=resolve_fixture;cfg.scratch_budget=491968;
    Nprg2BindConfig overlap=cfg;overlap.hot_memory=raw;overlap.hot_capacity=nn;
    CHECK(nprg2_bind_mock(&p,pools,&overlap,&bound)==NPRG2_BAD_BIND);
    uint8_t still_prepared[32];nprg2_sha256(raw,nn,still_prepared);CHECK(!memcmp(still_prepared,prepared_sha,32));
    overlap.hot_memory=(uint8_t *)p.tensors;overlap.hot_capacity=300000;
    CHECK(nprg2_bind_mock(&p,pools,&overlap,&bound)==NPRG2_BAD_BIND);
    overlap.hot_memory=(uint8_t *)p.expanded_blocks;
    CHECK(nprg2_bind_mock(&p,pools,&overlap,&bound)==NPRG2_BAD_BIND);
    CHECK(nprg2_bind_mock(&p,pools,&cfg,&bound)==NPRG2_OK);CHECK(bound.ready&&bound.hot_bytes<250000);
    Nprg2Command c;CHECK(nprg2_command_at(&bound,p.command_count-1,&c)==NPRG2_OK);CHECK(c.opcode>0&&c.variant>0);
    /* Command-time access must not consult the setup-only DDR tensor table. */
    Nprg2Tensor *setup=p.tensors;p.tensors=NULL;int got_blocked=0;
    for(uint32_t i=0;i<p.tensor_count;i++) {
        Nprg2TensorView tv;CHECK(nprg2_bound_tensor_at(&p,&bound,i,&tv)==NPRG2_OK);
        CHECK(tv.geometry_record==bound.geometry.data+(size_t)setup[i].geometry*NPRG2_GEOMETRY_BYTES);
        CHECK(tv.geometry==setup[i].geometry&&tv.storage_kind==setup[i].storage_kind);
        if(setup[i].storage_kind==2) {
            CHECK(tv.blocks==p.expanded_blocks+setup[i].block_first);
            CHECK(tv.block_count==setup[i].block_count&&tv.slice_origin==setup[i].slice_origin);got_blocked=1;
        } else if(setup[i].storage_kind==1)CHECK(tv.pool==setup[i].pool&&tv.offset==setup[i].offset);
    }
    p.tensors=setup;CHECK(got_blocked);
    for(int i=0;i<7;i++)if(pools[i].bytes)CHECK(pools[i].base[0]==0&&pools[i].base[pools[i].bytes-1]==0);
    free(hot);nprg2_close(&p);free_pools(pools);free(q);free(raw);
    printf("PASS %s\n",qpath);
}

int main(int argc,char **argv){CHECK(argc==5);uint8_t digest[32];nprg2_sha256("abc",3,digest);static const uint8_t abc[32]={0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};CHECK(!memcmp(digest,abc,32));
    run_model(argv[1],argv[2],5336,5068,671,120330);run_model(argv[3],argv[4],5343,5040,667,120592);return 0;}
