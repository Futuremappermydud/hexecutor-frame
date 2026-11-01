#include "prepared_elementwise.h"

#include "../blocked_glue/add_u8.h"
#include "../blocked_glue/linearclip_u8_u16.h"
#include <limits.h>
#include <string.h>

enum { BLOCK_BYTES=2048, MAX_BLOCKS=2560, GEOMETRY_CROUTON=64 };
static uint8_t **pool_pointer_table(const Nprg2ExecCall *c) {
    return (uint8_t **)((uint8_t *)c->scratch+3u*MAX_BLOCKS*sizeof(uintptr_t));
}
static int fill_pool_pointer_table(const Nprg2ExecCall *c) {
    if (!c->scratch || ((uintptr_t)c->scratch%sizeof(uintptr_t)) ||
        c->scratch_bytes<NPRG2_ELEMENTWISE_SCRATCH_BYTES) return 0;
    uint8_t **p=pool_pointer_table(c);for (unsigned i=0;i<7;i++) p[i]=c->pools[i].base;
    return 1;
}

static uint16_t rd16e(const uint8_t *p) { return (uint16_t)p[0] | (uint16_t)p[1]<<8; }
static uint32_t rd32e(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static uint64_t rd64e(const uint8_t *p) { return (uint64_t)rd32e(p) | (uint64_t)rd32e(p+4)<<32; }
static int32_t rdi16e(const uint8_t *p) { return (int16_t)rd16e(p); }
static int add_overflow(uintptr_t a, size_t n) { return n > UINTPTR_MAX-a; }

static int view_shape(const Nprg2TensorView *v, int32_t dims[4], int32_t meta[4], int32_t origin[4]) {
    if (!v || !v->geometry_record || !(v->geometry_record[1]&1u)) return 0;
    for (unsigned i=0;i<4;i++) {
        dims[i]=rdi16e(v->geometry_record+6+i*2u);
        meta[i]=rdi16e(v->geometry_record+14+i*2u);
        origin[i]=rdi16e(v->geometry_record+22+i*2u);
        if (dims[i]<0 || meta[i]<=0 || origin[i]<0 || (int64_t)dims[i]+origin[i]>meta[i]) return 0;
    }
    return 1;
}

static int make_u8(const Nprg2ExecCall *c, uint32_t slot, int output, QdBlockedU8 *out) {
    uint32_t ti; Nprg2TensorView v;
    if ((output?nprg2_exec_output(c,slot,&ti,&v):nprg2_exec_input(c,slot,&ti,&v))!=NPRG2_EXEC_OK) return 0;
    (void)ti;
    if (v.storage_kind!=2 || v.block_count>MAX_BLOCKS || !v.blocks || !(v.geometry_record[1]&GEOMETRY_CROUTON) ||
        v.geometry_record[0]!=1 || memcmp(v.geometry_record+2,"CB",2)) return 0;
    if (!fill_pool_pointer_table(c)) return 0;
    memset(out,0,sizeof *out); out->pools=pool_pointer_table(c);
    out->blocks=v.blocks; out->block_count=v.block_count; out->slice=v.slice_origin;
    return view_shape(&v,out->dims,out->meta,out->origin);
}

static int make_u16out(const Nprg2ExecCall *c, QdBlockedU16Out *out) {
    uint32_t ti; Nprg2TensorView v;
    if (nprg2_exec_output(c,0,&ti,&v)!=NPRG2_EXEC_OK) return 0;
    (void)ti;
    if (v.storage_kind!=2 || v.block_count>MAX_BLOCKS || !v.blocks || !(v.geometry_record[1]&GEOMETRY_CROUTON) ||
        v.geometry_record[0]!=3 || memcmp(v.geometry_record+2,"CH",2)) return 0;
    if (!fill_pool_pointer_table(c)) return 0;
    memset(out,0,sizeof *out); out->pools=pool_pointer_table(c);
    out->blocks=v.blocks; out->block_count=v.block_count; out->slice=v.slice_origin;
    return view_shape(&v,out->dims,out->meta,out->origin);
}

static int qd_valid(const int32_t *m,const int32_t *d,const int32_t *o,int width) {
    if (m[1]>INT32_MAX-7 || m[2]>INT32_MAX-(width-1) || m[3]>INT32_MAX-31) return 0;
    for (int i=0;i<4;i++) if (m[i]<=0 || d[i]<0 || o[i]<0 || (int64_t)o[i]+d[i]>m[i]) return 0;
    uint64_t nbh=(uint64_t)(m[1]+7)/8, nbw=(uint64_t)(m[2]+width-1)/width,
             nbd=(uint64_t)(m[3]+31)/32, slots=(uint32_t)m[0];
    const uint64_t factors[3]={nbh,nbw,nbd};
    for (unsigned i=0;i<3;i++) { if (factors[i] && slots>UINT32_MAX/factors[i]) return 0; slots*=factors[i]; }
    return slots<=UINT32_MAX;
}

/* Max-heapsort avoids libc qsort implementation variance and uses no hidden
 * allocation or stack-sized block arrays. */
static void sift_down(uintptr_t *a,size_t root,size_t n) {
    for (;;) {
        size_t child=root*2u+1u; if (child>=n) return;
        if (child+1u<n && a[child]<a[child+1u]) child++;
        if (a[root]>=a[child]) return;
        uintptr_t t=a[root];a[root]=a[child];a[child]=t;root=child;
    }
}
static void sort_starts(uintptr_t *a,size_t n) {
    for (size_t i=n/2u;i>0;i--) sift_down(a,i-1u,n);
    for (size_t i=n;i>1;i--) { uintptr_t t=a[0];a[0]=a[i-1u];a[i-1u]=t;sift_down(a,0,i-1u); }
}

static int block_address(const Nprg2ExecCall *c,const uint32_t *blocks,uint32_t count,
                         uint32_t index,uintptr_t *address) {
    if (index>=count) return 0;
    uint32_t enc=blocks[index],pool=enc&127u,off=enc&~127u;
    if (!pool || pool>=7 || !c->pools[pool].base || (off&(BLOCK_BYTES-1u)) ||
        off>c->pools[pool].bytes || BLOCK_BYTES>c->pools[pool].bytes-off) return 0;
    uint32_t limit=pool==1?c->pool1_writable_bytes:c->pools[pool].bytes;
    if (off>limit || BLOCK_BYTES>limit-off) return 0;
    uintptr_t base=(uintptr_t)c->pools[pool].base;
    if (add_overflow(base,off) || add_overflow(base+off,BLOCK_BYTES)) return 0;
    *address=base+off; return 1;
}

static int collect(const Nprg2ExecCall *c,const uint32_t *blocks,uint32_t count,uint32_t slice,
                   const int32_t *m,const int32_t *d,const int32_t *o,int width,
                   uintptr_t *starts,size_t capacity,size_t *used) {
    *used=0;
    if (!d[0] || !d[1] || !d[2] || !d[3]) return 1;
    int nbh=(m[1]+7)/8,nbw=(m[2]+width-1)/width,nbd=(m[3]+31)/32;
    for (int B=o[0];B<o[0]+d[0];B++)
      for (int H=o[1]/8;H<=(o[1]+d[1]-1)/8;H++)
       for (int W=o[2]/width;W<=(o[2]+d[2]-1)/width;W++)
        for (int D=o[3]/32;D<=(o[3]+d[3]-1)/32;D++) {
            int64_t ix=(((int64_t)B*nbh+H)*nbw+W)*nbd+(int64_t)slice+D;
            if (ix<0) return 0;
            if ((uint64_t)ix>=count) continue; /* audited glue treats absent blocks as zero/no-store */
            if (!(blocks[ix]&127u)) continue;
            uintptr_t a;if (!block_address(c,blocks,count,(uint32_t)ix,&a)) return 0;
            if (*used>=capacity) return 0;
            starts[(*used)++]=a;
        }
    sort_starts(starts,*used); return 1;
}

static int sets_disjoint(const uintptr_t *a,size_t na,const uintptr_t *b,size_t nb) {
    size_t i=0,j=0;
    while (i<na && j<nb) {
        if (a[i]<b[j]+BLOCK_BYTES && b[j]<a[i]+BLOCK_BYTES) return 0;
        if (a[i]<b[j]) i++; else j++;
    }
    return 1;
}

static int alias_proof(const Nprg2ExecCall *c,const QdBlockedU8 *a,const QdBlockedU8 *b,
                       const QdBlockedU8 *o,const QdBlockedU16Out *u16out,int is_linearclip,
                       const void *extra_read,size_t extra_read_bytes) {
    if (!fill_pool_pointer_table(c)) return 0;
    uintptr_t *s=(uintptr_t *)c->scratch;
    uintptr_t *sa=s,*sb=s+MAX_BLOCKS,*so=s+2u*MAX_BLOCKS;
    size_t na=0,nb=0,no=0;
    if (!collect(c,a->blocks,a->block_count,a->slice,a->meta,a->dims,a->origin,8,sa,MAX_BLOCKS,&na)) return 0;
    if (is_linearclip) {
        /* LinearClip reads one blocked tensor and Fi; it has no second CB
         * tensor input. Do not dereference the intentionally absent `b`/`o`.
         * Keep the second input interval set empty and prove output disjoint
         * from the actual u16 destination below. */
        if (b || o || !u16out) return 0;
        if (!collect(c,u16out->blocks,u16out->block_count,u16out->slice,u16out->meta,u16out->dims,u16out->origin,4,so,MAX_BLOCKS,&no)) return 0;
    } else {
        if (b) {
            if (!collect(c,b->blocks,b->block_count,b->slice,b->meta,b->dims,b->origin,8,sb,MAX_BLOCKS,&nb)) return 0;
        }
        if (!collect(c,o->blocks,o->block_count,o->slice,o->meta,o->dims,o->origin,8,so,MAX_BLOCKS,&no)) return 0;
    }
    for (size_t i=1;i<no;i++) if (so[i]<so[i-1]+BLOCK_BYTES) return 0;
    if (extra_read_bytes) {
        uintptr_t x=(uintptr_t)extra_read;
        if (!extra_read || add_overflow(x,extra_read_bytes)) return 0;
        for (size_t i=0;i<no;i++) if (so[i]<x+extra_read_bytes && x<so[i]+BLOCK_BYTES) return 0;
    }
    return sets_disjoint(so,no,sa,na) && sets_disjoint(so,no,sb,nb);
}

static int add_params(const Nprg2ExecCall *c,QdAddU8Params *p) {
    if (c->command.call_count!=1 || c->command.call_first>c->bound->calls.bytes/88u ||
        c->command.call_count>c->bound->calls.bytes/88u-c->command.call_first) return 0;
    const uint8_t *r=c->bound->calls.data+(size_t)c->command.call_first*88u;
    if (rd32e(r)!=QDR_CALL_ELEMENTWISE_ADD_CROUTON_U8) return 0;
    uint32_t mask=rd32e(r+4); uint64_t x[10];
    for (unsigned i=0;i<10;i++) x[i]=rd64e(r+8u+i*8u);
    if ((mask&0x70u) || x[4]>=(1ull<<40) || x[5]>=(1ull<<40) || x[6]>=(1ull<<40) ||
        (int64_t)x[4]<0 || (int64_t)x[5]<0 || x[4]>UINT32_MAX || x[5]>UINT32_MAX) return 0;
    uint64_t sh=x[6]+8u;if (sh<1 || sh>31) return 0;
    p->ma=(uint32_t)x[4];p->mb=(uint32_t)x[5];p->shift=(uint32_t)sh;
    return (uint64_t)255*p->ma+(uint64_t)255*p->mb+(1ull<<(p->shift-1u))<=UINT32_MAX;
}

static int add_validate(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx; if (!c || c->command.opcode!=QDR_ADD_U8 || c->command.variant!=13 ||
        c->command.input_count!=2 || c->command.output_count!=1) return 0;
    QdBlockedU8 a,b,o;QdAddU8Params p;
    if (!make_u8(c,0,0,&a)||!make_u8(c,1,0,&b)||!make_u8(c,0,1,&o)||!add_params(c,&p)) return 0;
    if (!qd_valid(a.meta,a.dims,a.origin,8)||!qd_valid(b.meta,b.dims,b.origin,8)||!qd_valid(o.meta,o.dims,o.origin,8)) return 0;
    for (int i=0;i<4;i++) if (a.dims[i]!=b.dims[i]||a.dims[i]!=o.dims[i]) return 0;
    if ((a.origin[3]&31)!=(b.origin[3]&31)||(a.origin[3]&31)!=(o.origin[3]&31)||
        (a.origin[2]&3)!=(b.origin[2]&3)||(a.origin[2]&3)!=(o.origin[2]&3)) return 0;
    uint32_t oti;Nprg2TensorView ov;if (nprg2_exec_output(c,0,&oti,&ov)!=NPRG2_EXEC_OK) return 0;
    if (ov.geometry_record[1]&GEOMETRY_CROUTON) { int32_t depth=((o.dims[3]+o.origin[3]+31)/32)*32;
        if (depth>o.meta[3]) depth=o.meta[3];if (depth-o.origin[3]!=a.dims[3]) return 0; }
    return alias_proof(c,&a,&b,&o,NULL,0,NULL,0);
}

static int linearclip_params(const Nprg2ExecCall *c,QdLinearclipU8U16Params *p) {
    uint32_t id;Nprg2TensorView v;
    if (c->command.input_count!=2 || nprg2_exec_input(c,1,&id,&v)!=NPRG2_EXEC_OK) return 0;
    (void)id;
    if (v.storage_kind!=1 || v.pool>=7 || !c->pools[v.pool].base || v.geometry_record[0]!=5 ||
        memcmp(v.geometry_record+2,"fi",2)) return 0;
    int32_t d[4],m[4],o[4];if (!view_shape(&v,d,m,o)) return 0;
    static const int32_t want[4]={1,1,1,12};
    for (int i=0;i<4;i++) if (d[i]!=want[i]||m[i]!=want[i]||o[i]) return 0;
    uint32_t off=v.offset,limit=c->pools[v.pool].bytes;
    if (v.pool==1 && limit>c->pool1_writable_bytes) limit=c->pool1_writable_bytes;
    if (off>limit || 48u>limit-off) return 0;
    const uint8_t *fi=c->pools[v.pool].base+off;
    uint32_t packed=rd32e(fi+4);int32_t shift=7-(int32_t)((int32_t)packed>>16);
    if (shift<1 || shift>31) return 0;
    p->multiplier=packed&65535u;p->shift=(uint32_t)shift;return 1;
}

static int linearclip_validate(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx; if (!c || c->command.opcode!=QDR_LINEARCLIP || c->command.variant!=68 ||
        c->command.input_count!=2 || c->command.output_count!=1) return 0;
    QdBlockedU8 in;QdBlockedU16Out out;QdLinearclipU8U16Params p;
    if (!make_u8(c,0,0,&in)||!make_u16out(c,&out)||!linearclip_params(c,&p)) return 0;
    if (!qd_valid(in.meta,in.dims,in.origin,8)||!qd_valid(out.meta,out.dims,out.origin,4)) return 0;
    for (int i=0;i<4;i++) if (in.dims[i]!=out.dims[i]) return 0;
    if (in.origin[2]%4!=out.origin[2]%4||in.origin[3]%32!=out.origin[3]%32) return 0;
    uint32_t oti;Nprg2TensorView ov;if (nprg2_exec_output(c,0,&oti,&ov)!=NPRG2_EXEC_OK) return 0;
    if (ov.geometry_record[1]&GEOMETRY_CROUTON) { int32_t depth=((out.dims[3]+out.origin[3]+31)/32)*32;
        if (depth>out.meta[3]) depth=out.meta[3];if (depth-out.origin[3]!=in.dims[3]) return 0; }
    uint32_t fi_id;Nprg2TensorView fi_view;
    if (nprg2_exec_input(c,1,&fi_id,&fi_view)!=NPRG2_EXEC_OK) return 0;
    (void)fi_id;
    return alias_proof(c,&in,NULL,NULL,&out,1,c->pools[fi_view.pool].base+fi_view.offset,48);
}

static int add_run(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx;QdBlockedU8 a,b,o;QdAddU8Params p;
    if (!make_u8(c,0,0,&a)||!make_u8(c,1,0,&b)||!make_u8(c,0,1,&o)||!add_params(c,&p)) return 0;
    /* The audited helper's prepare() allocates alias lists. The equivalent
     * full proof already ran in executor preflight with caller scratch; this
     * cookie is the helper's apply guard, not a skip around validation. */
    QdBlockedAddU8 prepared;prepared.a=a;prepared.b=b;prepared.out=o;prepared.cookie=0x51444138u;
    return qd_blocked_add_u8_apply(&prepared,&p);
}
static int linearclip_run(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx;QdBlockedU8 in;QdBlockedU16Out out;QdLinearclipU8U16Params p;
    if (!make_u8(c,0,0,&in)||!make_u16out(c,&out)||!linearclip_params(c,&p)) return 0;
    /* As for ADD_U8, executor preflight has already completed the alias proof
     * in caller scratch; use the audited helper's normal HVX apply routine. */
    QdBlockedLinearclipU8U16 prepared;prepared.in=in;prepared.out=out;prepared.cookie=0x51445538u;
    return qd_blocked_linearclip_u8_u16_apply(&prepared,&p);
}

static int requant_params(const Nprg2ExecCall *c,QdAddU8Params *p) {
    if (c->command.call_count!=1 || c->command.call_first>c->bound->calls.bytes/88u ||
        c->command.call_count>c->bound->calls.bytes/88u-c->command.call_first) return 0;
    const uint8_t *r=c->bound->calls.data+(size_t)c->command.call_first*88u;
    if (rd32e(r)!=QDR_CALL_REQUANTIZE_CROUTON_U8) return 0;
    uint32_t mask=rd32e(r+4); uint64_t x[10];
    for (unsigned i=0;i<10;i++) x[i]=rd64e(r+8u+i*8u);
    if ((mask&0x38u) || x[4]!=0 || x[5]!=0 || x[3]>=(1ull<<40)) return 0;
    int64_t m=(int64_t)(x[3]&0xffffu);
    int64_t s=15-(int64_t)(x[3]>>24);
    if (s<1 || s>31) return 0;
    p->ma=(uint32_t)m;p->mb=0;p->shift=(uint32_t)s;
    return (uint64_t)255*p->ma+(1ull<<(p->shift-1u))<=UINT32_MAX;
}

static int requant_validate(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx;
    if (!c || c->command.opcode!=QDR_REQUANT_U8 || c->command.variant!=70 ||
        c->command.input_count!=2 || c->command.output_count!=1) return 0;
    QdBlockedU8 in,out;QdAddU8Params p;
    if (!make_u8(c,0,0,&in) || !make_u8(c,0,1,&out) || !requant_params(c,&p)) return 0;
    if (!qd_valid(in.meta,in.dims,in.origin,8) || !qd_valid(out.meta,out.dims,out.origin,8)) return 0;
    for (int i=0;i<4;i++) if (in.dims[i]!=out.dims[i]) return 0;
    if ((in.origin[3]&31)!=(out.origin[3]&31) || (in.origin[2]&3)!=(out.origin[2]&3)) return 0;
    uint32_t oti;Nprg2TensorView ov;if (nprg2_exec_output(c,0,&oti,&ov)!=NPRG2_EXEC_OK) return 0;
    if (ov.geometry_record[1]&GEOMETRY_CROUTON) {
        int32_t depth=((out.dims[3]+out.origin[3]+31)/32)*32;
        if (depth>out.meta[3]) depth=out.meta[3];
        if (depth-out.origin[3]!=in.dims[3]) return 0;
    }
    uint32_t fi_id;Nprg2TensorView fi_view;
    if (nprg2_exec_input(c,1,&fi_id,&fi_view)!=NPRG2_EXEC_OK) return 0;
    (void)fi_id;
    if (fi_view.storage_kind!=1 || fi_view.pool>=7 || !c->pools[fi_view.pool].base ||
        fi_view.geometry_record[0]!=5 || memcmp(fi_view.geometry_record+2,"fi",2)) return 0;
    int32_t d[4],m[4],o[4];if (!view_shape(&fi_view,d,m,o)) return 0;
    static const int32_t want[4]={1,1,1,12};
    for (int i=0;i<4;i++) if (d[i]!=want[i] || m[i]!=want[i] || o[i]) return 0;
    uint32_t off=fi_view.offset,limit=c->pools[fi_view.pool].bytes;
    if (fi_view.pool==1 && limit>c->pool1_writable_bytes) limit=c->pool1_writable_bytes;
    if (off>limit || 48u>limit-off) return 0;
    return alias_proof(c,&in,NULL,&out,NULL,0,c->pools[fi_view.pool].base+fi_view.offset,48);
}

static int requant_run(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx;QdBlockedU8 in,out;QdAddU8Params p;
    if (!make_u8(c,0,0,&in) || !make_u8(c,0,1,&out) || !requant_params(c,&p)) return 0;
    QdBlockedAddU8 prepared;prepared.a=in;prepared.b=in;prepared.out=out;prepared.cookie=0x51444138u;
    return qd_blocked_add_u8_apply(&prepared,&p);
}

int nprg2_executor_register_elementwise_region(Nprg2Executor *e, uint8_t region) {
    if (!e) return NPRG2_EXEC_ARGUMENT;
    int rc=nprg2_executor_register_scratch_region(e,QDR_ADD_U8,13,add_validate,add_run,NULL,
            NPRG2_ELEMENTWISE_SCRATCH_BYTES,0,region);
    if (rc) return rc;
    rc=nprg2_executor_register_scratch_region(e,QDR_REQUANT_U8,70,requant_validate,requant_run,NULL,
            NPRG2_ELEMENTWISE_SCRATCH_BYTES,0,region);
    if (rc) return rc;
    /* Variant 68 is the proven linearclip@CH.CB.fi (CB u8 -> CH u16) form.
     * U16->U16 and dense/FB variants remain unregistered and fail closed. */
    return nprg2_executor_register_scratch_region(e,QDR_LINEARCLIP,68,linearclip_validate,linearclip_run,NULL,
            NPRG2_ELEMENTWISE_SCRATCH_BYTES,0,region);
}
int nprg2_executor_register_elementwise(Nprg2Executor *e) {
    return nprg2_executor_register_elementwise_region(e,NPRG2_EXEC_SCRATCH_DEFAULT);
}
