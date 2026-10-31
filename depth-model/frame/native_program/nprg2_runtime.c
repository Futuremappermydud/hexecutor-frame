#include "nprg2_runtime.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

enum { HEADER_CRC=12, HASH_CONTEXT=48, HASH_CONSTANTS=80, HASH_SOURCE=112,
       POOL_TABLE=144, IO_TABLE=172, SECTION_TABLE=180, SECTION_COUNT=7 };
enum { F_SHAPE=1, F_CROUTON=64, F_SHAPE_VALUE=8, F_SCALAR=16, F_QUANT=32 };
enum { ST_NONE=0, ST_DIRECT=1, ST_BLOCKED=2 };
enum { MAX_EXPANDED_BLOCKS=NPRG2_MAX_IMAGE_BYTES/4 };

typedef struct {
    uint32_t key, count, first, end;
    uint8_t used, decoded, pad[2];
} BlockMemo;

static uint16_t rd16(const uint8_t *p) { return (uint16_t)p[0] | (uint16_t)p[1]<<8; }
static uint32_t rd24(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16; }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24; }
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | (uint64_t)rd32(p+4)<<32; }
static int16_t rdi16(const uint8_t *p) { return (int16_t)rd16(p); }
static uint32_t align8(uint32_t n) { return (n+7u)&~7u; }
static int known_pool(uint32_t p) { return p==1 || p==3 || p==4 || p==5; }

uint32_t nprg2_crc32(const void *data, size_t bytes, size_t zero_offset, size_t zero_bytes) {
    const uint8_t *p=(const uint8_t *)data;
    uint32_t crc=0xffffffffu;
    for(size_t i=0;i<bytes;i++) {
        uint8_t b=(i>=zero_offset && i-zero_offset<zero_bytes)?0:p[i];
        crc^=b;
        for(unsigned j=0;j<8;j++)crc=(crc>>1)^(0xedb88320u&((uint32_t)-(int32_t)(crc&1u)));
    }
    return ~crc;
}

/* Small self-contained SHA256 so host tests and the future DSP caller use the
 * same package-hash implementation without a platform crypto dependency. */
typedef struct { uint32_t h[8]; uint64_t bits; uint8_t block[64]; uint32_t used; } Sha256;
static const uint32_t K256[64]={
  0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
  0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
  0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
  0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
  0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
  0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
  0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
  0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};
static uint32_t rr(uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));}
static void sha_block(Sha256 *s,const uint8_t *p) {
    uint32_t w[64];for(int i=0;i<16;i++)w[i]=(uint32_t)p[i*4]<<24|(uint32_t)p[i*4+1]<<16|(uint32_t)p[i*4+2]<<8|p[i*4+3];
    for(int i=16;i<64;i++){uint32_t a=rr(w[i-15],7)^rr(w[i-15],18)^(w[i-15]>>3),b=rr(w[i-2],17)^rr(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+a+w[i-7]+b;}
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3],e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
    for(int i=0;i<64;i++){uint32_t s1=rr(e,6)^rr(e,11)^rr(e,25),ch=(e&f)^(~e&g),t1=h+s1+ch+K256[i]+w[i];uint32_t s0=rr(a,2)^rr(a,13)^rr(a,22),maj=(a&b)^(a&c)^(b&c),t2=s0+maj;h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}
static void sha_init(Sha256 *s){static const uint32_t h[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};memcpy(s->h,h,sizeof h);s->bits=0;s->used=0;}
static void sha_update(Sha256 *s,const uint8_t *p,size_t n){s->bits+=(uint64_t)n*8;while(n){size_t k=64-s->used;if(k>n)k=n;memcpy(s->block+s->used,p,k);s->used+=(uint32_t)k;p+=k;n-=k;if(s->used==64){sha_block(s,s->block);s->used=0;}}}
static void sha_final(Sha256 *s,uint8_t out[32]){uint64_t bits=s->bits;s->block[s->used++]=0x80;if(s->used>56){while(s->used<64)s->block[s->used++]=0;sha_block(s,s->block);s->used=0;}while(s->used<56)s->block[s->used++]=0;for(int i=7;i>=0;i--)s->block[s->used++]=(uint8_t)(bits>>(i*8));sha_block(s,s->block);for(int i=0;i<8;i++){out[i*4]=(uint8_t)(s->h[i]>>24);out[i*4+1]=(uint8_t)(s->h[i]>>16);out[i*4+2]=(uint8_t)(s->h[i]>>8);out[i*4+3]=(uint8_t)s->h[i];}}
void nprg2_sha256(const void *data,size_t bytes,uint8_t out[32]){Sha256 s;sha_init(&s);sha_update(&s,(const uint8_t *)data,bytes);sha_final(&s,out);}

static void *default_alloc(void *ctx,size_t n){(void)ctx;return malloc(n?n:1);}
static void default_free(void *ctx,void *p){(void)ctx;free(p);}
static void *a_alloc(const Nprg2Allocator *a,size_t n){return a->alloc(a->ctx,n);}
static void a_free(const Nprg2Allocator *a,void *p){if(p)a->free(a->ctx,p);}
static int ptr_end(const void *p,uint32_t bytes,uintptr_t *start,uintptr_t *end){uintptr_t a=(uintptr_t)p;if(!p || a>UINTPTR_MAX-bytes)return 0;*start=a;*end=a+bytes;return 1;}
static int pools_valid(const Nprg2Pool pools[7],const uint32_t required[7]){
    uintptr_t lo[7],hi[7];
    for(unsigned i=0;i<7;i++){
        if(pools[i].bytes<required[i] || (required[i]&&!pools[i].base))return 0;
        if(pools[i].bytes){if(!ptr_end(pools[i].base,pools[i].bytes,&lo[i],&hi[i]))return 0;}
        else lo[i]=hi[i]=0;
    }
    for(unsigned i=0;i<7;i++)if(pools[i].bytes)for(unsigned j=0;j<i;j++)if(pools[j].bytes&&lo[i]<hi[j]&&lo[j]<hi[i])return 0;
    return 1;
}
static uint32_t dtype_size(uint8_t d){static const uint8_t z[8]={0,1,1,2,2,4,4,4};return d<8?z[d]:0;}
static const uint8_t *geo_at(const Nprg2Program *p,uint32_t i){return p->geometry.data+(size_t)i*NPRG2_GEOMETRY_BYTES;}
static int geometry_valid(const uint8_t *g){
    uint8_t dtype=g[0],flags=g[1];
    if(dtype>=8 || flags&~(F_SHAPE|F_CROUTON|F_SHAPE_VALUE|F_SCALAR|F_QUANT) || g[42]||g[43])return 0;
    unsigned form=!!(flags&F_SHAPE)+!!(flags&F_SHAPE_VALUE)+!!(flags&F_SCALAR);
    if(form!=1)return 0;
    if(flags&F_SHAPE)for(int d=0;d<4;d++){
        int32_t actual=rdi16(g+6+d*2),maximum=rdi16(g+14+d*2);
        if(actual<0 || maximum<0 || actual>maximum)return 0;
    }
    if(flags&F_SHAPE_VALUE)for(int i=22;i<30;i++)if(g[i])return 0;
    if(flags&F_SCALAR)for(int i=10;i<30;i++)if(g[i])return 0;
    if(!(flags&F_QUANT))for(int i=30;i<42;i++)if(g[i])return 0;
    return 1;
}
static uint32_t hash_key(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;}
static BlockMemo *memo_find(BlockMemo *m,uint32_t cap,uint32_t first){
    uint32_t key=first+1u,at=hash_key(key)&(cap-1u);
    for(uint32_t n=0;n<cap;n++,at=(at+1)&(cap-1u)){if(!m[at].used)return &m[at];if(m[at].key==key)return &m[at];}
    return NULL;
}
static uint32_t zigzag_decode(uint32_t v){return (v&1u)?(uint32_t)(-(int32_t)(v>>1)-1):(v>>1);}
static int read_uleb(const uint8_t *s,uint32_t n,uint32_t *at,uint32_t *out){
    uint32_t v=0,shift=0;
    for(unsigned i=0;i<5;i++){
        if(*at>=n)return 0;uint8_t b=s[(*at)++];
        if(shift==28 && (b&0xf0u))return 0;
        v|=(uint32_t)(b&127u)<<shift;
        if(!(b&128u)){if(i && !(b&127u))return 0;*out=v;return 1;}shift+=7;
    }
    return 0;
}
static int decode_block_list(const Nprg2Program *p,const Nprg2Pool pools[7],uint32_t first,uint32_t count,
                             uint32_t output_first,uint32_t *out,uint32_t *end){
    const uint8_t *s=p->block_stream.data;uint32_t n=p->block_stream.bytes;
    if(!count || first>n || n-first<3)return 0;
    uint32_t code=rd24(s+first),pool=code>>21,off=code&0x1fffffu;
    if(!known_pool(pool))return 0;
    for(uint32_t i=0;i<count;i++){
        if(i){uint32_t v;if(!read_uleb(s,n,end,&v))return 0;uint32_t dz=zigzag_decode(v);
            int64_t next=(int64_t)off+(int32_t)dz;if(next<0 || next>=0x200000) return 0;off=(uint32_t)next;}
        uint64_t byteoff=(uint64_t)off*128u;
        if(byteoff+2048u>p->pool_bytes[pool] || byteoff+2048u>pools[pool].bytes)return 0;
        out[output_first+i]=(pool&127u)|(uint32_t)byteoff;
    }
    return 1;
}
static int command_decode_section(const Nprg2Section *sec,uint32_t count,uint32_t index,Nprg2Command *c){
    if(index>=count)return NPRG2_BAD_COMMAND;const uint8_t *r=sec->data+(size_t)index*NPRG2_COMMAND_BYTES;
    c->sequence=rd16(r);c->opcode=r[2];c->variant=r[3];c->io_index=rd16(r+4);
    c->input_first=rd16(r+6);c->output_first=rd16(r+8);c->vector_first=rd16(r+10);c->call_first=rd16(r+12);
    c->input_count=r[14];c->output_count=r[15];c->vector_count=r[16];c->call_count=r[17];c->duplicate_pad=r[18];
    if(!c->opcode || c->opcode>QDR_OPCODES || !c->variant || c->variant>QDR_VARIANTS || c->duplicate_pad>1 || r[19])return NPRG2_BAD_COMMAND;
    return NPRG2_OK;
}

static int validate_first_pass(Nprg2Program *p,const Nprg2Pool pools[7],BlockMemo *memo,uint32_t memo_cap,uint32_t *unique_blocks){
    uint32_t total=0;
    for(uint32_t i=0;i<p->geometry_count;i++)if(!geometry_valid(geo_at(p,i)))return NPRG2_BAD_TENSOR;
    for(uint32_t i=0;i<p->tensor_count;i++){
        const uint8_t *r=p->tensor.data+(size_t)i*NPRG2_TENSOR_BYTES;
        uint32_t gi=rd16(r),kind=r[2];const uint8_t *pay=r+3;
        if(gi>=p->geometry_count || kind>ST_BLOCKED)return NPRG2_BAD_TENSOR;
        const uint8_t *g=geo_at(p,gi);uint8_t flags=g[1];
        int croton=(kind==ST_BLOCKED && (g[2]=='C'||g[2]=='c'));
        if(!!(flags&F_CROUTON)!=croton)return NPRG2_BAD_TENSOR;
        if(kind==ST_NONE){for(int j=0;j<9;j++)if(pay[j])return NPRG2_BAD_TENSOR;}
        else if(kind==ST_DIRECT){
            uint32_t pool=pay[0],off=rd32(pay+1);for(int j=5;j<9;j++)if(pay[j])return NPRG2_BAD_TENSOR;
            if(!known_pool(pool))return NPRG2_BAD_POOL;
            uint64_t bytes=0;
            if(flags&F_SHAPE){uint64_t elems=1;for(int d=0;d<4;d++){int32_t dim=rdi16(g+14+d*2);if(dim<0)return NPRG2_BAD_TENSOR;elems*=((uint32_t)dim);if(elems>NPRG2_MAX_IMAGE_BYTES)return NPRG2_BAD_TENSOR;}bytes=elems*dtype_size(g[0]);}
            else bytes=dtype_size(g[0]);
            if((uint64_t)off+bytes>p->pool_bytes[pool] || (uint64_t)off+bytes>pools[pool].bytes)return NPRG2_BAD_POOL;
        } else {
            uint32_t first=rd24(pay),n=rd16(pay+3),slice=rd32(pay+5);
            if(!n || first>p->block_stream.bytes || p->block_stream.bytes-first<3)return NPRG2_BAD_BLOCK;
            if(slice>=n)return NPRG2_BAD_BLOCK;
            BlockMemo *m=memo_find(memo,memo_cap,first);if(!m)return NPRG2_NO_MEMORY;
            if(!m->used){m->used=1;m->key=first+1u;m->count=n;m->first=total;if(total>MAX_EXPANDED_BLOCKS-n)return NPRG2_BAD_BLOCK;total+=n;}
            else if(m->count!=n)return NPRG2_BAD_BLOCK;
        }
    }
    *unique_blocks=total;return NPRG2_OK;
}

int nprg2_open(Nprg2Program *out,const void *raw,uint32_t raw_bytes,const Nprg2ExpectedHashes *expected,
               const Nprg2Pool pools[7],Nprg2Allocator allocator){
    if(!out || !raw || !expected || !expected->context_sha256 || !expected->constants_sha256 || !expected->source_sha256 || !expected->prepared_sha256 || !pools)return NPRG2_BAD_ARGUMENT;
    memset(out,0,sizeof *out);
    if((allocator.alloc==NULL)!=(allocator.free==NULL))return NPRG2_BAD_ARGUMENT;
    if(!allocator.alloc)allocator.alloc=default_alloc;if(!allocator.free)allocator.free=default_free;
    if(raw_bytes<NPRG2_HEADER_BYTES || raw_bytes>NPRG2_MAX_IMAGE_BYTES)return NPRG2_BAD_HEADER;
    const uint8_t *b=(const uint8_t *)raw;
    if(rd32(b)!=NPRG2_MAGIC || rd16(b+4)!=NPRG2_VERSION || rd16(b+6)!=NPRG2_HEADER_BYTES || rd32(b+8)!=raw_bytes || rd32(b+16)!=QDR_ABI)return NPRG2_BAD_HEADER;
    if(nprg2_crc32(raw,raw_bytes,HEADER_CRC,4)!=rd32(b+HEADER_CRC))return NPRG2_BAD_CRC;
    uint8_t image_hash[32];nprg2_sha256(raw,raw_bytes,image_hash);
    if(memcmp(image_hash,expected->prepared_sha256,32))return NPRG2_BAD_HASH;
    if(memcmp(b+HASH_CONTEXT,expected->context_sha256,32)||memcmp(b+HASH_CONSTANTS,expected->constants_sha256,32)||memcmp(b+HASH_SOURCE,expected->source_sha256,32))return NPRG2_BAD_HASH;
    for(uint32_t i=236;i<NPRG2_HEADER_BYTES;i++)if(b[i])return NPRG2_BAD_HEADER;
    out->raw=b;out->raw_bytes=raw_bytes;out->geometry_count=rd32(b+20);out->tensor_count=rd32(b+24);out->block_stream_bytes=rd32(b+28);out->command_count=rd32(b+32);out->ref_count=rd32(b+36);out->vector_count=rd32(b+40);out->call_count=rd32(b+44);
    out->graph_input_first=rd16(b+172);out->graph_input_count=rd16(b+174);out->graph_output_first=rd16(b+176);out->graph_output_count=rd16(b+178);
    for(unsigned i=0;i<7;i++)out->pool_bytes[i]=rd32(b+POOL_TABLE+i*4);
    if(!out->geometry_count || out->geometry_count>65535u || !out->tensor_count || out->tensor_count>65535u || !out->command_count || out->command_count>65535u || out->ref_count>65535u || out->vector_count>65535u || out->call_count>65535u)return NPRG2_BAD_HEADER;
    if(!pools_valid(pools,out->pool_bytes))return NPRG2_BAD_POOL;
    Nprg2Section *ss[7]={&out->geometry,&out->tensor,&out->block_stream,&out->command,&out->refs,&out->vectors,&out->calls};
    uint32_t expected_bytes[7]={out->geometry_count*NPRG2_GEOMETRY_BYTES,out->tensor_count*NPRG2_TENSOR_BYTES,out->block_stream_bytes,out->command_count*NPRG2_COMMAND_BYTES,out->ref_count*2u,out->vector_count*4u,out->call_count*88u};
    uint32_t end=NPRG2_HEADER_BYTES;
    for(unsigned i=0;i<7;i++){
        uint32_t off=rd32(b+SECTION_TABLE+i*8),size=rd32(b+SECTION_TABLE+i*8+4),aligned=align8(end);
        if(aligned<end || off!=aligned || size!=expected_bytes[i] || off>raw_bytes || size>raw_bytes-off)return NPRG2_BAD_SECTION;
        for(uint32_t z=end;z<off;z++)if(b[z])return NPRG2_BAD_SECTION;
        ss[i]->offset=off;ss[i]->bytes=size;ss[i]->data=b+off;end=off+size;
    }
    if(end!=raw_bytes || out->graph_input_first+out->graph_input_count>out->ref_count || out->graph_output_first+out->graph_output_count>out->ref_count)return NPRG2_BAD_SECTION;
    for(unsigned i=0;i<7;i++)out->session_pools[i]=pools[i];
    memcpy(out->context_sha256,b+HASH_CONTEXT,32);memcpy(out->constants_sha256,b+HASH_CONSTANTS,32);memcpy(out->source_sha256,b+HASH_SOURCE,32);out->allocator=allocator;
    uint32_t memo_cap=1;while(memo_cap<out->tensor_count*2u){if(memo_cap>0x40000000u)return NPRG2_NO_MEMORY;memo_cap<<=1;}
    BlockMemo *memo=(BlockMemo *)a_alloc(&allocator,(size_t)memo_cap*sizeof *memo);if(!memo)return NPRG2_NO_MEMORY;memset(memo,0,(size_t)memo_cap*sizeof *memo);
    uint32_t unique_blocks=0;int e=validate_first_pass(out,pools,memo,memo_cap,&unique_blocks);
    if(e){a_free(&allocator,memo);nprg2_close(out);return e;}
    out->tensors=(Nprg2Tensor *)a_alloc(&allocator,(size_t)out->tensor_count*sizeof *out->tensors);
    out->expanded_blocks=(uint32_t *)a_alloc(&allocator,(size_t)unique_blocks*sizeof(uint32_t));
    if(!out->tensors || (unique_blocks && !out->expanded_blocks)){a_free(&allocator,memo);nprg2_close(out);return NPRG2_NO_MEMORY;}
    memset(out->tensors,0,(size_t)out->tensor_count*sizeof *out->tensors);
    out->expanded_block_count=unique_blocks;
    uint32_t block_cursor=0,blocks_out=0;
    for(uint32_t i=0;i<out->tensor_count;i++){
        const uint8_t *r=out->tensor.data+(size_t)i*NPRG2_TENSOR_BYTES,*pay=r+3;Nprg2Tensor *t=&out->tensors[i];
        t->geometry=rd16(r);t->storage_kind=r[2];
        if(t->storage_kind==ST_DIRECT){t->pool=pay[0];t->offset=rd32(pay+1);}
        else if(t->storage_kind==ST_BLOCKED){uint32_t first=rd24(pay),count=rd16(pay+3);t->block_count=count;t->slice_origin=rd32(pay+5);
            if(t->slice_origin>=count){e=NPRG2_BAD_BLOCK;break;}
            BlockMemo *m=memo_find(memo,memo_cap,first);if(!m || !m->used || m->count!=count){e=NPRG2_BAD_BLOCK;break;}
            if(!m->decoded){if(first!=block_cursor){e=NPRG2_BAD_BLOCK;break;}uint32_t at=first+3,endoff=at;
                if(!decode_block_list(out,pools,first,count,blocks_out,out->expanded_blocks,&endoff)){e=NPRG2_BAD_BLOCK;break;}
                m->first=blocks_out;m->end=endoff;m->decoded=1;blocks_out+=count;block_cursor=endoff;
            } else if(m->end<first+3){e=NPRG2_BAD_BLOCK;break;}
            t->block_first=m->first;
        }
    }
    if(!e && (blocks_out!=unique_blocks || block_cursor!=out->block_stream_bytes))e=NPRG2_BAD_BLOCK;
    if(!e)for(uint32_t i=0;i<out->ref_count;i++)if(rd16(out->refs.data+i*2)>=out->tensor_count){e=NPRG2_BAD_COMMAND;break;}
    if(!e)for(uint32_t i=0;i<out->call_count;i++){const uint8_t *r=out->calls.data+i*88;uint32_t id=rd32(r),mask=rd32(r+4);if(!id || id>QDR_CALLS || mask>=1024){e=NPRG2_BAD_COMMAND;break;}for(unsigned j=0;j<10;j++)if((mask&(1u<<j))&&rd64(r+8+j*8)){e=NPRG2_BAD_COMMAND;break;}}
    if(!e)for(uint32_t i=0;i<out->command_count;i++){Nprg2Command c;if(command_decode_section(&out->command,out->command_count,i,&c)){e=NPRG2_BAD_COMMAND;break;}
        if(c.input_first+c.input_count>out->ref_count || c.output_first+c.output_count>out->ref_count || c.vector_first+c.vector_count>out->vector_count || c.call_first+c.call_count>out->call_count){e=NPRG2_BAD_COMMAND;break;}}
    if(!e)for(uint32_t i=0;i<out->graph_input_count;i++)if(rd16(out->refs.data+(out->graph_input_first+i)*2)>=out->tensor_count)e=NPRG2_BAD_COMMAND;
    if(!e)for(uint32_t i=0;i<out->graph_output_count;i++)if(rd16(out->refs.data+(out->graph_output_first+i)*2)>=out->tensor_count)e=NPRG2_BAD_COMMAND;
    a_free(&allocator,memo);
    if(e){nprg2_close(out);return e;}
    return NPRG2_OK;
}

void nprg2_close(Nprg2Program *p){if(!p)return;Nprg2Allocator a=p->allocator;if(!a.free)a.free=default_free;a_free(&a,p->expanded_blocks);a_free(&a,p->tensors);memset(p,0,sizeof *p);}

int nprg2_command_at(const Nprg2Bound *b,uint32_t index,Nprg2Command *out){if(!b||!b->ready||!out)return NPRG2_BAD_BIND;return command_decode_section(&b->command,b->command.bytes/NPRG2_COMMAND_BYTES,index,out);}
int nprg2_tensor_at(const Nprg2Program *p,uint32_t index,const Nprg2Tensor **t,const uint8_t **geometry){if(!p||index>=p->tensor_count||!t||!geometry)return NPRG2_BAD_ARGUMENT;*t=&p->tensors[index];*geometry=p->geometry.data+(size_t)(*t)->geometry*NPRG2_GEOMETRY_BYTES;return NPRG2_OK;}
int nprg2_bound_tensor_at(const Nprg2Program *p,const Nprg2Bound *b,uint32_t index,Nprg2TensorView *o){
    if(!p||!b||!b->ready||!o||index>=p->tensor_count)return NPRG2_BAD_ARGUMENT;
    const uint8_t *r=b->tensor.data+(size_t)index*NPRG2_TENSOR_BYTES,*payload=r+3;
    o->geometry=rd16(r);o->storage_kind=r[2];o->pool=0;o->offset=0;o->blocks=NULL;o->block_count=0;o->slice_origin=0;
    if(o->geometry>=p->geometry_count || o->storage_kind>ST_BLOCKED)return NPRG2_BAD_TENSOR;
    o->geometry_record=b->geometry.data+(size_t)o->geometry*NPRG2_GEOMETRY_BYTES;
    if(o->storage_kind==ST_DIRECT){o->pool=payload[0];o->offset=rd32(payload+1);}
    else if(o->storage_kind==ST_BLOCKED){uint32_t first=rd24(payload),count=rd16(payload+3);o->slice_origin=rd32(payload+5);
        if(!count || o->slice_origin>=count || first>p->expanded_block_count || count>p->expanded_block_count-first)return NPRG2_BAD_TENSOR;
        o->block_count=count;o->blocks=p->expanded_blocks+first;
    }
    return NPRG2_OK;
}

static int map_same(const Nprg2Program *p,const Nprg2Pool pools[7]){for(unsigned i=0;i<7;i++)if(pools[i].base!=p->session_pools[i].base||pools[i].bytes!=p->session_pools[i].bytes)return 0;return 1;}
static int hot_push(uint8_t *base,uint32_t cap,uint32_t *cursor,const Nprg2Section *src,Nprg2Section *dst){uint32_t at=align8(*cursor);if(at<*cursor || at>cap || src->bytes>cap-at)return 0;memset(base+*cursor,0,at-*cursor);memcpy(base+at,src->data,src->bytes);dst->offset=at;dst->bytes=src->bytes;dst->data=base+at;*cursor=at+src->bytes;return 1;}
static int memory_overlaps(const void *a,uint32_t an,const void *b,uint32_t bn){uintptr_t al,ah,bl,bh;if(!an||!bn)return 0;if(!ptr_end(a,an,&al,&ah)||!ptr_end(b,bn,&bl,&bh))return 1;return al<bh&&bl<ah;}
int nprg2_bind_mock(const Nprg2Program *p,const Nprg2Pool pools[7],const Nprg2BindConfig *cfg,Nprg2Bound *bound){
    if(!p||!pools||!cfg||!bound)return NPRG2_BAD_ARGUMENT;memset(bound,0,sizeof *bound);
    if(!cfg->pool1_is_verified_vtcm || !map_same(p,pools) || !cfg->hot_memory)return NPRG2_BAD_BIND;
    uint32_t hot_need=0;const Nprg2Section *srcs[6]={&p->geometry,&p->tensor,&p->command,&p->refs,&p->vectors,&p->calls};
    for(unsigned i=0;i<6;i++){hot_need=align8(hot_need);if(hot_need>UINT32_MAX-srcs[i]->bytes)return NPRG2_BAD_BIND;hot_need+=srcs[i]->bytes;}
    if(hot_need>cfg->hot_capacity || hot_need>cfg->l2_available_bytes || cfg->scratch_budget>cfg->l2_available_bytes-hot_need)return NPRG2_BAD_BIND;
    for(unsigned i=0;i<7;i++)if(memory_overlaps(cfg->hot_memory,cfg->hot_capacity,pools[i].base,pools[i].bytes))return NPRG2_BAD_BIND;
    if(memory_overlaps(cfg->hot_memory,cfg->hot_capacity,p->raw,p->raw_bytes) ||
       memory_overlaps(cfg->hot_memory,cfg->hot_capacity,p->tensors,p->tensor_count*(uint32_t)sizeof *p->tensors) ||
       memory_overlaps(cfg->hot_memory,cfg->hot_capacity,p->expanded_blocks,p->expanded_block_count*4u))return NPRG2_BAD_BIND;
    /* Bound tensor rows carry an expanded-block index in their existing
     * 24-bit field. Validate every rewrite before writing the hot buffer. */
    for(uint32_t i=0;i<p->tensor_count;i++)if(p->tensors[i].storage_kind==ST_BLOCKED) {
        uint32_t first=p->tensors[i].block_first,n=p->tensors[i].block_count;
        if(first>0xffffffu || first>p->expanded_block_count || n>p->expanded_block_count-first)return NPRG2_BAD_BIND;
    }
    /* Full schedule coverage and scratch checks precede every hot-memory or
     * graph-pool write. There is intentionally no legacy fallback route. */
    if(!cfg->resolve)return NPRG2_UNSUPPORTED;
    uint32_t max_scratch=0;
    for(uint32_t i=0;i<p->command_count;i++){
        Nprg2Command c;Nprg2Capability cap={0};if(command_decode_section(&p->command,p->command_count,i,&c))return NPRG2_BAD_COMMAND;
        if(!cfg->resolve(cfg->resolver_ctx,p,i,&c,&cap) || !cap.supported || cap.may_fallback)return NPRG2_UNSUPPORTED;
        uint64_t n=(uint64_t)cap.scratch_bytes+cap.concurrent_worker_scratch_bytes;if(n>cfg->scratch_budget)return NPRG2_SCRATCH;if(n>max_scratch)max_scratch=(uint32_t)n;
    }
    uint32_t cursor=0;
    if(!hot_push(cfg->hot_memory,cfg->hot_capacity,&cursor,&p->geometry,&bound->geometry) ||
       !hot_push(cfg->hot_memory,cfg->hot_capacity,&cursor,&p->tensor,&bound->tensor) ||
       !hot_push(cfg->hot_memory,cfg->hot_capacity,&cursor,&p->command,&bound->command) ||
       !hot_push(cfg->hot_memory,cfg->hot_capacity,&cursor,&p->refs,&bound->refs) ||
       !hot_push(cfg->hot_memory,cfg->hot_capacity,&cursor,&p->vectors,&bound->vectors) ||
       !hot_push(cfg->hot_memory,cfg->hot_capacity,&cursor,&p->calls,&bound->calls))return NPRG2_BAD_BIND;
    for(uint32_t i=0;i<p->tensor_count;i++)if(p->tensors[i].storage_kind==ST_BLOCKED) {
        uint8_t *r=(uint8_t *)bound->tensor.data+(size_t)i*NPRG2_TENSOR_BYTES+3;
        uint32_t first=p->tensors[i].block_first;
        r[0]=(uint8_t)first;r[1]=(uint8_t)(first>>8);r[2]=(uint8_t)(first>>16);
    }
    bound->hot_bytes=cursor;bound->maximum_command_scratch=max_scratch;bound->ready=1;return NPRG2_OK;
}

int nprg2_bind_metadata(const Nprg2Program *p,const Nprg2Pool pools[7],const Nprg2BindConfig *cfg,Nprg2Bound *bound){
    if(!p||!pools||!cfg||!bound)return NPRG2_BAD_ARGUMENT;memset(bound,0,sizeof *bound);
    if(!cfg->pool1_is_verified_vtcm||!map_same(p,pools)||!cfg->hot_memory)return NPRG2_BAD_BIND;
    uint32_t hot_need=0;const Nprg2Section *srcs[6]={&p->geometry,&p->tensor,&p->command,&p->refs,&p->vectors,&p->calls};
    for(unsigned i=0;i<6;i++){hot_need=align8(hot_need);if(hot_need>UINT32_MAX-srcs[i]->bytes)return NPRG2_BAD_BIND;hot_need+=srcs[i]->bytes;}
    if(hot_need>cfg->hot_capacity||hot_need>cfg->l2_available_bytes)return NPRG2_BAD_BIND;
    for(unsigned i=0;i<7;i++)if(memory_overlaps(cfg->hot_memory,cfg->hot_capacity,pools[i].base,pools[i].bytes))return NPRG2_BAD_BIND;
    if(memory_overlaps(cfg->hot_memory,cfg->hot_capacity,p->raw,p->raw_bytes)||
       memory_overlaps(cfg->hot_memory,cfg->hot_capacity,p->tensors,p->tensor_count*(uint32_t)sizeof *p->tensors)||
       memory_overlaps(cfg->hot_memory,cfg->hot_capacity,p->expanded_blocks,p->expanded_block_count*4u))return NPRG2_BAD_BIND;
    for(uint32_t i=0;i<p->tensor_count;i++)if(p->tensors[i].storage_kind==ST_BLOCKED){
        uint32_t first=p->tensors[i].block_first,n=p->tensors[i].block_count;
        if(first>0xffffffu||first>p->expanded_block_count||n>p->expanded_block_count-first)return NPRG2_BAD_BIND;
    }
    uint32_t cursor=0;
    if(!hot_push(cfg->hot_memory,cfg->hot_capacity,&cursor,&p->geometry,&bound->geometry)||
       !hot_push(cfg->hot_memory,cfg->hot_capacity,&cursor,&p->tensor,&bound->tensor)||
       !hot_push(cfg->hot_memory,cfg->hot_capacity,&cursor,&p->command,&bound->command)||
       !hot_push(cfg->hot_memory,cfg->hot_capacity,&cursor,&p->refs,&bound->refs)||
       !hot_push(cfg->hot_memory,cfg->hot_capacity,&cursor,&p->vectors,&bound->vectors)||
       !hot_push(cfg->hot_memory,cfg->hot_capacity,&cursor,&p->calls,&bound->calls))return NPRG2_BAD_BIND;
    for(uint32_t i=0;i<p->tensor_count;i++)if(p->tensors[i].storage_kind==ST_BLOCKED){
        uint8_t *r=(uint8_t*)bound->tensor.data+(size_t)i*NPRG2_TENSOR_BYTES+3;uint32_t first=p->tensors[i].block_first;
        r[0]=(uint8_t)first;r[1]=(uint8_t)(first>>8);r[2]=(uint8_t)(first>>16);
    }
    bound->hot_bytes=cursor;bound->maximum_command_scratch=0;bound->ready=1;return NPRG2_OK;
}
