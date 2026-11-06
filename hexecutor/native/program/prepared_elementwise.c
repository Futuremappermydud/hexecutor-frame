#include "prepared_elementwise.h"

#include "../kernels/blocked/add_u8.h"
#include "../kernels/blocked/elementwise_u16.h"
#include "../kernels/blocked/linearclip_u8_u16.h"
#include "../kernels/blocked/linearclip_flat_u8.h"
/* blocked_glue/linearclip_u16.h and sign_blocked.h declare their own QdBlockedU16, which
 * elementwise_u16.h already typedefs (to the layout-identical QdBlockedU16Out);
 * give the glue's the name QdBlockedU16Glue and convert by copy. */
#define QdBlockedU16 QdBlockedU16Glue
#include "../kernels/blocked/sign_blocked.h"
#undef QdBlockedU16
#include <limits.h>
#include <stdlib.h>
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


/* ---- threads: the handlers split the rows (dims[1]) of the region. A part
 * covers pixel rows [H*part/parts, H*(part+1)/parts) of every tensor; output
 * pixel rows are distinct 256-byte rows of the blocks and the stores are
 * byte-masked, so parts write disjoint bytes and read only inputs. */
#ifndef PAR_EW_MIN_ELEMS
#define PAR_EW_MIN_ELEMS 8192u /* Frame sweep 2K-32K: flat below 8K (t4c) */
#endif
#define EW_SPLIT(T,t,h0,h1) do { (t).dims[1]=(h1)-(h0); (t).origin[1]+=(h0); } while(0)
static int ew_range(int32_t H,uint32_t part,uint32_t parts,int32_t *h0,int32_t *h1) {
    *h0=(int32_t)((int64_t)H*part/parts);*h1=(int32_t)((int64_t)H*(part+1u)/parts);
    return *h1>*h0;
}
static uint32_t ew_parts(const Nprg2ExecCall *c,const int32_t *dims) {
    uint64_t n=(uint64_t)dims[0]*(uint64_t)dims[1]*(uint64_t)dims[2]*(uint64_t)dims[3];
    if (n<PAR_EW_MIN_ELEMS || dims[1]<2) return 1;
    return nprg2_parallel_parts(c->parallel,(uint32_t)dims[1]);
}
typedef struct { QdBlockedU8 a,b,out; QdAddU8Params p; } AddJob;
static void add_part(void *ctx,uint32_t part,uint32_t parts) {
    AddJob j=*(const AddJob *)ctx;int32_t h0,h1;
    if (!ew_range(j.a.dims[1],part,parts,&h0,&h1)) return;
    EW_SPLIT(0,j.a,h0,h1);EW_SPLIT(0,j.b,h0,h1);EW_SPLIT(0,j.out,h0,h1);
    QdBlockedAddU8 prepared;prepared.a=j.a;prepared.b=j.b;prepared.out=j.out;prepared.cookie=0x51444138u;
    (void)qd_blocked_add_u8_apply(&prepared,&j.p);
}
typedef struct { QdBlockedAddU8 pr; QdAddU8Params p; } AddWholeJob;
static void add_whole_part(void *ctx,uint32_t part,uint32_t parts) {
    const AddWholeJob *j=(const AddWholeJob *)ctx;
    (void)qd_blocked_add_u8_apply_whole(&j->pr,&j->p,part,parts);
}
static int add_apply_parts(const Nprg2ExecCall *c,const QdBlockedU8 *a,const QdBlockedU8 *b,const QdBlockedU8 *o,const QdAddU8Params *p) {
    {
        AddWholeJob j;j.pr.a=*a;j.pr.b=*b;j.pr.out=*o;j.pr.cookie=0x51444138u;j.p=*p;
        if (qd_blocked_add_u8_whole(&j.pr)) {
            uint32_t parts=ew_parts(c,a->dims);
            if (parts>1) nprg2_parallel_for(c->parallel,NPRG2_PAR_EW,parts,add_whole_part,&j);
            else return qd_blocked_add_u8_apply_whole(&j.pr,&j.p,0,1);
            return 1;
        }
    }
    uint32_t parts=ew_parts(c,a->dims);
    if (parts>1) {
        AddJob j={*a,*b,*o,*p};
        nprg2_parallel_for(c->parallel,NPRG2_PAR_EW,parts,add_part,&j);
        return 1;
    }
    QdBlockedAddU8 prepared;prepared.a=*a;prepared.b=*b;prepared.out=*o;prepared.cookie=0x51444138u;
    return qd_blocked_add_u8_apply(&prepared,p);
}
typedef struct { QdBlockedLinearclipU8U16 pr; QdLinearclipU8U16Params p; } LcWholeJob;
static void lc_whole_part(void *ctx,uint32_t part,uint32_t parts) {
    const LcWholeJob *j=(const LcWholeJob *)ctx;
    (void)qd_blocked_linearclip_u8_u16_apply_whole(&j->pr,&j->p,part,parts);
}
typedef struct { QdBlockedU8 in; QdBlockedU16Out out; QdLinearclipU8U16Params p; } LcJob;
static void lc_part(void *ctx,uint32_t part,uint32_t parts) {
    LcJob j=*(const LcJob *)ctx;int32_t h0,h1;
    if (!ew_range(j.in.dims[1],part,parts,&h0,&h1)) return;
    EW_SPLIT(0,j.in,h0,h1);EW_SPLIT(0,j.out,h0,h1);
    QdBlockedLinearclipU8U16 prepared;prepared.in=j.in;prepared.out=j.out;prepared.cookie=0x51445538u;
    (void)qd_blocked_linearclip_u8_u16_apply(&prepared,&j.p);
}
typedef struct { QdBlockedU16Out a,b,o; int has_b; QdEw16Params p; int32_t bs; } W16Job;
static void w16_part(void *ctx,uint32_t part,uint32_t parts) {
    W16Job j=*(const W16Job *)ctx;int32_t h0,h1;
    if (!ew_range(j.a.dims[1],part,parts,&h0,&h1)) return;
    EW_SPLIT(0,j.a,h0,h1);EW_SPLIT(0,j.b,h0,h1);EW_SPLIT(0,j.o,h0,h1);
    (void)qd_blocked_ew16_apply(&j.a,j.has_b?&j.b:NULL,&j.o,&j.p,j.bs);
}
/* Whole 8x4x32 blocks: parts take blocks round-robin (no row split). */
static void w16_whole_part(void *ctx,uint32_t part,uint32_t parts) {
    const W16Job *j=(const W16Job *)ctx;
    (void)qd_blocked_ew16_apply_whole(&j->a,j->has_b?&j->b:NULL,&j->o,&j->p,j->bs,part,parts);
}

#ifdef EW_MEMO
/* Worker lane, preflighted commands: descriptors and parameters come only from the
 * bound program (call records, geometry, block tables), so they are parsed once per
 * command. u16 scalars are operand values and are still read on every run. */
typedef struct {
    uint32_t valid;
    union {
        struct { QdBlockedU8 a,b,o; QdAddU8Params p; } add;
        struct { QdBlockedU16Out a,b,o; QdEw16Params p; const uint8_t *scalar; } w16;
    } u;
} EwMemo;
enum { EW_MEMO_COMMANDS=8192, EW_MEMO_LANE_SLOTS=512, EW_MEMO_SLOTS=3*EW_MEMO_LANE_SLOTS };
static EwMemo *ew_slab;
static uint16_t *ew_ix;     /* by command index: slab index + 1 */
/* Lists run concurrently: each allocates from its own part of the slab. */
static uint32_t ew_used[3];
static const void *ew_program;
static uint8_t *ew_pool_tab[7];
void nprg2_exec_elementwise_memo_begin(const Nprg2Program *program,const Nprg2Pool *pools) {
    if (!ew_slab) ew_slab=malloc(sizeof(EwMemo)*EW_MEMO_SLOTS);
    if (!ew_ix) ew_ix=malloc(sizeof(uint16_t)*EW_MEMO_COMMANDS);
    if (!ew_slab || !ew_ix) return;
    /* Memos stay valid for the same program and pool contents. */
    int same=ew_program==(const void *)program;
    for (unsigned i=0;same && i<7;i++) same=ew_pool_tab[i]==pools[i].base;
    if (same) return;
    memset(ew_ix,0,sizeof(uint16_t)*EW_MEMO_COMMANDS);ew_used[0]=ew_used[1]=ew_used[2]=0;
    ew_program=program;
    for (unsigned i=0;i<7;i++) ew_pool_tab[i]=pools[i].base;
}
static inline EwMemo *ew_hit(const Nprg2ExecCall *c) {
    if (!c->preflighted || c->execution_lane>2u || !ew_ix || c->command_index>=EW_MEMO_COMMANDS ||
        c->program!=ew_program) return NULL;
    uint32_t ix=ew_ix[c->command_index];
    return ix?&ew_slab[ix-1u]:NULL;
}
static inline EwMemo *ew_new(const Nprg2ExecCall *c) {
    if (!c->preflighted || c->execution_lane>2u || !ew_ix || c->command_index>=EW_MEMO_COMMANDS ||
        c->program!=ew_program || ew_used[c->execution_lane]>=EW_MEMO_LANE_SLOTS) return NULL;
    return &ew_slab[c->execution_lane*EW_MEMO_LANE_SLOTS+ew_used[c->execution_lane]];
}
static inline void ew_commit(const Nprg2ExecCall *c) {
    uint32_t l=c->execution_lane;ew_used[l]++;
    ew_ix[c->command_index]=(uint16_t)(l*EW_MEMO_LANE_SLOTS+ew_used[l]);
}
#endif
static int add_run(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx;QdBlockedU8 a,b,o;QdAddU8Params p;
#ifdef EW_MEMO
    {EwMemo *m=ew_hit(c);if (m) return add_apply_parts(c,&m->u.add.a,&m->u.add.b,&m->u.add.o,&m->u.add.p);}
#endif
    if (!make_u8(c,0,0,&a)||!make_u8(c,1,0,&b)||!make_u8(c,0,1,&o)||!add_params(c,&p)) return 0;
#ifdef EW_MEMO
    {EwMemo *m=ew_new(c);if (m){m->u.add.a=a;m->u.add.b=b;m->u.add.o=o;m->u.add.p=p;
        m->u.add.a.pools=m->u.add.b.pools=m->u.add.o.pools=ew_pool_tab;ew_commit(c);}}
#endif
    /* The audited helper's prepare() allocates alias lists. The equivalent
     * full proof already ran in executor preflight with caller scratch; this
     * cookie is the helper's apply guard, not a skip around validation. */
    return add_apply_parts(c,&a,&b,&o,&p);
}
static int linearclip_run(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx;QdBlockedU8 in;QdBlockedU16Out out;QdLinearclipU8U16Params p;
    if (!make_u8(c,0,0,&in)||!make_u16out(c,&out)||!linearclip_params(c,&p)) return 0;
    /* As for ADD_U8, executor preflight has already completed the alias proof
     * in caller scratch; use the audited helper's normal HVX apply routine. */
    {
        LcWholeJob j;j.pr.in=in;j.pr.out=out;j.pr.cookie=0x51445538u;j.p=p;
        if (qd_blocked_linearclip_u8_u16_whole(&j.pr)) {
            uint32_t wparts=ew_parts(c,in.dims);
            if (wparts>1) { nprg2_parallel_for(c->parallel,NPRG2_PAR_EW,wparts,lc_whole_part,&j);return 1; }
            return qd_blocked_linearclip_u8_u16_apply_whole(&j.pr,&j.p,0,1);
        }
    }
    uint32_t parts=ew_parts(c,in.dims);
    if (parts>1) {
        LcJob j={in,out,p};
        nprg2_parallel_for(c->parallel,NPRG2_PAR_EW,parts,lc_part,&j);
        return 1;
    }
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
#ifdef EW_MEMO
    {EwMemo *m=ew_hit(c);if (m) return add_apply_parts(c,&m->u.add.a,&m->u.add.a,&m->u.add.o,&m->u.add.p);}
#endif
    if (!make_u8(c,0,0,&in) || !make_u8(c,0,1,&out) || !requant_params(c,&p)) return 0;
#ifdef EW_MEMO
    {EwMemo *m=ew_new(c);if (m){m->u.add.a=in;m->u.add.b=in;m->u.add.o=out;m->u.add.p=p;
        m->u.add.a.pools=m->u.add.b.pools=m->u.add.o.pools=ew_pool_tab;ew_commit(c);}}
#endif
    return add_apply_parts(c,&in,&in,&out,&p);
}


/* ------------------------------------------------------------------------
 * u16 crouton ("CH") elementwise: REQUANT_U16/71, ADD1111_U16/59,
 * MUL_U16/74, ADD_U16/14, ADDSUB_U16/60, MUL1111_U16/73. Exact 32-bit lane
 * forms are derived and checked in blocked_glue/elementwise_u16.h; this
 * section only maps the reference kernels' arguments onto them and rejects
 * (-> reference kernel) everything outside the proven ranges. Same scratch as
 * the u8 handlers (alias proof), so the region demand is unchanged.
 * ------------------------------------------------------------------------ */
enum { W16_REQ, W16_ADD1111, W16_MUL, W16_ADD, W16_ADDSUB, W16_MUL1111, W16_MULINL, W16_COUNT };
typedef struct { uint8_t opcode,variant,nin; uint32_t fn,used; int two_tensors,scalar; } W16Spec;
static const W16Spec w16_spec[W16_COUNT]={
    {QDR_REQUANT_U16,71,2,QDR_CALL_REQUANTIZE_CROUTON_U16,0x18,0,0},
    {QDR_ADD1111_U16,59,3,QDR_CALL_ELEMENTWISE_ADD1111_CROUTON_U16,0xf0,0,1},
    {QDR_MUL_U16,74,3,QDR_CALL_ELEMENTWISE_MUL_CROUTON_U16,0x1f0,1,0},
    {QDR_ADD_U16,14,2,QDR_CALL_ELEMENTWISE_ADD_CROUTON_U16,0x1f0,1,0},
    {QDR_ADDSUB_U16,60,3,QDR_CALL_ELEMENTWISE_ADDSUB_CROUTON_U16,0x70,1,0},
    {QDR_MUL1111_U16,73,3,QDR_CALL_ELEMENTWISE_MUL1111_CROUTON_U16,0x1f0,0,1},
    /* MUL_INLINE/72 (mul_op@CH*2.cH.fi): no call record; the scalar is a 1-pixel
     * "cH" crouton tensor and the six multiplier words an i32 tensor in pool 5. */
    {QDR_MUL_INLINE,72,3,0,0,0,1},
};

static int make_u16(const Nprg2ExecCall *c,uint32_t slot,int output,QdBlockedU16Out *out) {
    uint32_t ti;Nprg2TensorView v;
    if ((output?nprg2_exec_output(c,slot,&ti,&v):nprg2_exec_input(c,slot,&ti,&v))!=NPRG2_EXEC_OK) return 0;
    (void)ti;
    if (v.storage_kind!=2 || v.block_count>MAX_BLOCKS || !v.blocks || !(v.geometry_record[1]&GEOMETRY_CROUTON) ||
        v.geometry_record[0]!=3 || memcmp(v.geometry_record+2,"CH",2)) return 0;
    if (!fill_pool_pointer_table(c)) return 0;
    memset(out,0,sizeof *out); out->pools=pool_pointer_table(c);
    out->blocks=v.blocks; out->block_count=v.block_count; out->slice=v.slice_origin;
    return view_shape(&v,out->dims,out->meta,out->origin);
}

/* The 1x1x1x1 direct u16 scalar operand of ADD1111/MUL1111: address of its
 * two bytes, or NULL. The value is read at run time (it may be a pool 3
 * activation written by an earlier command). */
static const uint8_t *w16_scalar_addr(const Nprg2ExecCall *c,uint32_t slot) {
    uint32_t id;Nprg2TensorView v;
    if (nprg2_exec_input(c,slot,&id,&v)!=NPRG2_EXEC_OK) return NULL;
    (void)id;
    if (v.storage_kind!=1 || v.pool>=7 || !c->pools[v.pool].base || v.geometry_record[0]!=3) return NULL;
    int32_t d[4],m[4],o[4];if (!view_shape(&v,d,m,o)) return NULL;
    for (int i=0;i<4;i++) if (d[i]!=1||m[i]!=1||o[i]) return NULL;
    uint32_t limit=c->pools[v.pool].bytes;
    if (v.pool==1 && limit>c->pool1_writable_bytes) limit=c->pool1_writable_bytes;
    if (v.offset>limit || 2u>limit-v.offset) return NULL;
    return c->pools[v.pool].base+v.offset;
}

/* MUL_INLINE's scalar operand: element (0,0,0,0) of a 1x1x1x1 u16 "cH" crouton
 * tensor, i.e. the first halfword of its first block. NULL unless that block is
 * present and inside its pool. Read at run time like the direct scalar. */
static const uint8_t *mulinl_scalar_addr(const Nprg2ExecCall *c,uint32_t slot) {
    uint32_t id;Nprg2TensorView v;
    if (nprg2_exec_input(c,slot,&id,&v)!=NPRG2_EXEC_OK) return NULL;
    (void)id;
    if (v.storage_kind!=2 || !v.blocks || v.geometry_record[0]!=3 || !(v.geometry_record[1]&GEOMETRY_CROUTON) ||
        memcmp(v.geometry_record+2,"cH",2)) return NULL;
    int32_t d[4],m[4],o[4];if (!view_shape(&v,d,m,o)) return NULL;
    for (int i=0;i<4;i++) if (d[i]!=1||o[i]) return NULL;
    if (m[0]!=1||m[1]>8||m[2]>4||m[3]>32) return NULL;
    if (v.slice_origin>=v.block_count) return NULL;
    uint32_t e=v.blocks[v.slice_origin],pool=e&127u;
    if (!pool || pool>=7 || !c->pools[pool].base) return NULL;
    uint32_t limit=c->pools[pool].bytes;
    if ((e&~127u)>limit || BLOCK_BYTES>limit-(e&~127u)) return NULL;
    return c->pools[pool].base+(e&~127u);
}
/* MUL_INLINE multipliers: six i32 words (pool 5, immutable), the reference's
 * k[0]=p[2], zb=p[3]&0xffff, za=p[3]>>16, k[3]=p[4], k[4]=p[5] (needs >= 6 words). */
static int mulinl_params(const Nprg2ExecCall *c,QdEw16Params *p) {
    uint32_t id;Nprg2TensorView v;
    if (nprg2_exec_input(c,2,&id,&v)!=NPRG2_EXEC_OK) return 0;
    (void)id;
    if (v.storage_kind!=1 || v.pool!=5 || !c->pools[5].base || v.geometry_record[0]!=5) return 0;
    int32_t d[4],m[4],o[4];if (!view_shape(&v,d,m,o)) return 0;
    if (d[0]!=1||d[1]!=1||d[2]!=1||d[3]<6||m[0]!=1||m[1]!=1||m[2]!=1||m[3]!=d[3]||o[0]|o[1]|o[2]|o[3]) return 0;
    uint32_t limit=c->pools[5].bytes;
    if (v.offset>limit || 24u>limit-v.offset) return 0;
    const uint8_t *w=c->pools[5].base+v.offset;
    uint32_t z=rd32e(w+12);
    return qd_ew16_make_mul((int32_t)rd32e(w+8),(int64_t)(z&65535u),(int64_t)(z>>16),(int32_t)rd32e(w+16),(int32_t)rd32e(w+20),p);
}

/* The reference's call_args: exactly one matching call record, the "used"
 * arguments unmasked. Fills r[10]. */
static int w16_call(const Nprg2ExecCall *c,const W16Spec *sp,int64_t r[10]) {
    if (c->command.call_count!=1 || c->command.call_first>c->bound->calls.bytes/88u ||
        c->command.call_count>c->bound->calls.bytes/88u-c->command.call_first) return 0;
    const uint8_t *rec=c->bound->calls.data+(size_t)c->command.call_first*88u;
    if (rd32e(rec)!=sp->fn) return 0;
    uint32_t mask=rd32e(rec+4);
    if (sp->used&mask) return 0;
    for (unsigned i=0;i<10;i++) r[i]=(int64_t)rd64e(rec+8u+i*8u);
    return 1;
}
static int w16_params(const Nprg2ExecCall *c,int which,QdEw16Params *p) {
    const W16Spec *sp=&w16_spec[which];int64_t r[10];
    if (which==W16_MULINL) return mulinl_params(c,p);
    if (!w16_call(c,sp,r)) return 0;
    switch (which) {
    case W16_REQ: return qd_ew16_make_requant(r[3],r[4],p);
    case W16_ADD1111: return qd_ew16_make_add1111(r[4],r[5],r[6],r[7],p);
    case W16_MUL: case W16_MUL1111: return qd_ew16_make_mul(r[4],r[5],r[6],r[7],r[8],p);
    case W16_ADD: return r[7]==0 && qd_ew16_make_add(r[4],r[5],r[6],p);
    default: {
        uint32_t oti;Nprg2TensorView ov;
        if (nprg2_exec_output(c,0,&oti,&ov)!=NPRG2_EXEC_OK) return 0;
        return qd_ew16_make_addsub(r[4],r[5],r[6],(int32_t)rd32e(ov.geometry_record+30),p);
    }
    }
}

static int alias_proof16(const Nprg2ExecCall *c,const QdBlockedU16Out *a,const QdBlockedU16Out *b,
                         const QdBlockedU16Out *o,const void *extra_read,size_t extra_read_bytes) {
    if (!fill_pool_pointer_table(c)) return 0;
    uintptr_t *s=(uintptr_t *)c->scratch;
    uintptr_t *sa=s,*sb=s+MAX_BLOCKS,*so=s+2u*MAX_BLOCKS;
    size_t na=0,nb=0,no=0;
    if (!collect(c,a->blocks,a->block_count,a->slice,a->meta,a->dims,a->origin,4,sa,MAX_BLOCKS,&na)) return 0;
    if (b && !collect(c,b->blocks,b->block_count,b->slice,b->meta,b->dims,b->origin,4,sb,MAX_BLOCKS,&nb)) return 0;
    if (!collect(c,o->blocks,o->block_count,o->slice,o->meta,o->dims,o->origin,4,so,MAX_BLOCKS,&no)) return 0;
    for (size_t i=1;i<no;i++) if (so[i]<so[i-1]+BLOCK_BYTES) return 0;
    if (extra_read_bytes) {
        uintptr_t x=(uintptr_t)extra_read;
        if (!extra_read || add_overflow(x,extra_read_bytes)) return 0;
        for (size_t i=0;i<no;i++) if (so[i]<x+extra_read_bytes && x<so[i]+BLOCK_BYTES) return 0;
    }
    return sets_disjoint(so,no,sa,na) && sets_disjoint(so,no,sb,nb);
}

static int w16_validate(const Nprg2ExecCall *c,int which) {
    const W16Spec *sp=&w16_spec[which];
    if (!c || c->command.opcode!=sp->opcode || c->command.variant!=sp->variant ||
        c->command.input_count!=sp->nin || c->command.output_count!=1) return 0;
    QdEw16Params p;QdBlockedU16Out a,b,o;
    if (!w16_params(c,which,&p) || !make_u16(c,0,0,&a) || !make_u16(c,0,1,&o)) return 0;
    if (sp->two_tensors && !make_u16(c,1,0,&b)) return 0;
    const uint8_t *scalar=NULL;
    if (sp->scalar && !(scalar=which==W16_MULINL?mulinl_scalar_addr(c,1):w16_scalar_addr(c,1))) return 0;
    if (!qd_valid(a.meta,a.dims,a.origin,4) || !qd_valid(o.meta,o.dims,o.origin,4)) return 0;
    if (sp->two_tensors && !qd_valid(b.meta,b.dims,b.origin,4)) return 0;
    for (int i=0;i<4;i++) if (a.dims[i]!=o.dims[i] || (sp->two_tensors && a.dims[i]!=b.dims[i])) return 0;
    if ((a.origin[3]&31)!=(o.origin[3]&31) || (a.origin[2]&3)!=(o.origin[2]&3)) return 0;
    if (sp->two_tensors && ((a.origin[3]&31)!=(b.origin[3]&31) || (a.origin[2]&3)!=(b.origin[2]&3))) return 0;
    uint32_t oti;Nprg2TensorView ov;if (nprg2_exec_output(c,0,&oti,&ov)!=NPRG2_EXEC_OK) return 0;
    if (ov.geometry_record[1]&GEOMETRY_CROUTON) {
        /* Memory.write pads a narrower value to the 32-lane boundary with zeros;
         * only the unpadded form is the elementwise result. */
        int32_t depth=((o.dims[3]+o.origin[3]+31)/32)*32;
        if (depth>o.meta[3]) depth=o.meta[3];
        if (depth-o.origin[3]!=a.dims[3]) return 0;
    }
    return alias_proof16(c,&a,sp->two_tensors?&b:NULL,&o,scalar,scalar?2u:0u);
}
static int w16_run(const Nprg2ExecCall *c,int which) {
    const W16Spec *sp=&w16_spec[which];
    QdEw16Params p;QdBlockedU16Out a,b,o;int32_t bs=-1;
#ifdef EW_MEMO
    {
        EwMemo *m=ew_hit(c);
        if (m) {
            a=m->u.w16.a;o=m->u.w16.o;p=m->u.w16.p;
            if (sp->two_tensors) b=m->u.w16.b;
            if (sp->scalar) bs=(int32_t)rd16e(m->u.w16.scalar);
            goto w16_apply;
        }
    }
#endif
    if (!w16_params(c,which,&p) || !make_u16(c,0,0,&a) || !make_u16(c,0,1,&o)) return 0;
    if (sp->two_tensors && !make_u16(c,1,0,&b)) return 0;
    {
        const uint8_t *sa=NULL;
        if (sp->scalar) {
            sa=which==W16_MULINL?mulinl_scalar_addr(c,1):w16_scalar_addr(c,1);
            if (!sa) return 0;
            bs=(int32_t)rd16e(sa);
        }
#ifdef EW_MEMO
        EwMemo *m=ew_new(c);
        if (m) {
            m->u.w16.a=a;m->u.w16.o=o;m->u.w16.p=p;m->u.w16.scalar=sa;
            if (sp->two_tensors) m->u.w16.b=b;
            m->u.w16.a.pools=m->u.w16.o.pools=ew_pool_tab;
            if (sp->two_tensors) m->u.w16.b.pools=ew_pool_tab;
            ew_commit(c);
        }
#endif
    }
#ifdef EW_MEMO
w16_apply:;
#endif
    uint32_t parts=ew_parts(c,a.dims);
    if (parts>1) {
        W16Job j={a,b,o,sp->two_tensors,p,bs};
        if (!sp->two_tensors) memset(&j.b,0,sizeof j.b);
        if (qd_blocked_ew16_whole(&a,sp->two_tensors?&b:NULL,&o)) {
            nprg2_parallel_for(c->parallel,NPRG2_PAR_EW,parts,w16_whole_part,&j);
            return 1;
        }
        nprg2_parallel_for(c->parallel,NPRG2_PAR_EW,parts,w16_part,&j);
        return 1;
    }
    return qd_blocked_ew16_apply(&a,sp->two_tensors?&b:NULL,&o,&p,bs);
}
#define W16_HANDLERS(N,W) \
    static int N##_validate(const Nprg2ExecCall *c,void *ctx){(void)ctx;return w16_validate(c,W);} \
    static int N##_run(const Nprg2ExecCall *c,void *ctx){(void)ctx;return w16_run(c,W);}
W16_HANDLERS(req16,W16_REQ)
W16_HANDLERS(add1111,W16_ADD1111)
W16_HANDLERS(mul16,W16_MUL)
W16_HANDLERS(add16,W16_ADD)
W16_HANDLERS(addsub16,W16_ADDSUB)
W16_HANDLERS(mul1111,W16_MUL1111)
W16_HANDLERS(mulinl,W16_MULINL)


/* ------------------------------------------------------------------------
 * SIGN/39 (CH u16 -> CH u16) and LINEARCLIP/67 (CH u16 -> CH u16), the
 * blocked_glue/sign_blocked.c and linearclip_u16.c kernels, and LINEARCLIP/69
 * (flat u8 -> flat u8). Shapes, alias proof and the derived constants are the
 * ones the reference kernels use (k_sign, k_linearclip); anything outside them
 * is rejected so the chained reference kernel runs it. Constants are derived
 * in validation and again at run time from the same immutable sources (the
 * geometry records, and the pool 5 `fi` tensor), so a run never relies on a
 * stale validation.
 * ------------------------------------------------------------------------ */
_Static_assert(sizeof(QdBlockedU16Out)==sizeof(QdBlockedU16Glue),"glue descriptor layout");
static void to_glue(const QdBlockedU16Out *s,QdBlockedU16Glue *d) { memcpy(d,s,sizeof *d); }

/* Common shape/alias proof for one u16 CH input and one u16 CH output of equal
 * logical shape (what w16_validate proves for a single-input op). */
static int u16_pair_ok(const Nprg2ExecCall *c,QdBlockedU16Out *a,QdBlockedU16Out *o) {
    if (!make_u16(c,0,0,a) || !make_u16(c,0,1,o)) return 0;
    if (!qd_valid(a->meta,a->dims,a->origin,4) || !qd_valid(o->meta,o->dims,o->origin,4)) return 0;
    for (int i=0;i<4;i++) if (a->dims[i]!=o->dims[i]) return 0;
    if ((a->origin[3]&31)!=(o->origin[3]&31) || (a->origin[2]&3)!=(o->origin[2]&3)) return 0;
    uint32_t oti;Nprg2TensorView ov;if (nprg2_exec_output(c,0,&oti,&ov)!=NPRG2_EXEC_OK) return 0;
    if (ov.geometry_record[1]&GEOMETRY_CROUTON) {
        int32_t depth=((o->dims[3]+o->origin[3]+31)/32)*32;
        if (depth>o->meta[3]) depth=o->meta[3];
        if (depth-o->origin[3]!=a->dims[3]) return 0;
    }
    return alias_proof16(c,a,NULL,o,NULL,0);
}

static int sign_levels_of(const Nprg2ExecCall *c,int32_t *input_zero,QdSignLevels *lv) {
    uint32_t ti;Nprg2TensorView iv,ov;
    if (nprg2_exec_input(c,0,&ti,&iv)!=NPRG2_EXEC_OK || nprg2_exec_output(c,0,&ti,&ov)!=NPRG2_EXEC_OK) return 0;
    const uint8_t *ig=iv.geometry_record,*og=ov.geometry_record;
    if (!ig || !og || !(ig[1]&32u) || !(og[1]&32u)) return 0;
    *input_zero=(int32_t)rd32e(ig+30);
    if (*input_zero<0 || *input_zero>65535) return 0;
    double scale;memcpy(&scale,og+34,8);
    return qd_sign_prepare_levels(scale,(int32_t)rd32e(og+30),lv);
}
static int sign_validate(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx;
    if (!c || c->command.opcode!=QDR_SIGN || c->command.variant!=39 ||
        c->command.input_count!=1 || c->command.output_count!=1) return 0;
    QdBlockedU16Out a,o;int32_t iz;QdSignLevels lv;
    return u16_pair_ok(c,&a,&o) && sign_levels_of(c,&iz,&lv);
}
static int sign_run(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx;
    QdBlockedU16Out a,o;int32_t iz;QdSignLevels lv;
    if (!make_u16(c,0,0,&a) || !make_u16(c,0,1,&o) || !sign_levels_of(c,&iz,&lv)) return 0;
    /* The alias proof ran in preflight (see add_run); this is the audited
     * helper's apply guard. */
    QdBlockedSignU16 prepared;to_glue(&a,&prepared.in);to_glue(&o,&prepared.out);prepared.cookie=0x51445347u;
    return qd_blocked_sign_u16_apply(&prepared,iz,&lv);
}

/* The 12-word `fi` operand of LinearClip: a direct constant (pool 5, immutable)
 * 1x1x1x12 i32 tensor. */
static const uint8_t *lc_fi(const Nprg2ExecCall *c) {
    uint32_t id;Nprg2TensorView v;
    if (c->command.input_count!=2 || nprg2_exec_input(c,1,&id,&v)!=NPRG2_EXEC_OK) return NULL;
    if (v.storage_kind!=1 || v.pool!=5 || !c->pools[5].base || v.geometry_record[0]!=5 ||
        memcmp(v.geometry_record+2,"fi",2)) return NULL;
    int32_t d[4],m[4],o[4];if (!view_shape(&v,d,m,o)) return NULL;
    static const int32_t want[4]={1,1,1,12};
    for (int i=0;i<4;i++) if (d[i]!=want[i]||m[i]!=want[i]||o[i]) return NULL;
    if (v.offset>c->pools[5].bytes || 48u>c->pools[5].bytes-v.offset) return NULL;
    return c->pools[5].base+v.offset;
}
/* The reference's linearclip constants for a u16-input form:
 * P=fi[10] (low half multiplier, high byte shift), zp=fi[11], negated
 * multiplier when fi[6]<0. */
static void lc_consts(const uint8_t *fi,int64_t *m,int64_t *sh,uint32_t *zp) {
    uint32_t P=rd32e(fi+40);*zp=rd32e(fi+44);
    *m=(int64_t)(P&0xffffu);*sh=15-(int64_t)(P>>24);
    if ((int32_t)rd32e(fi+24)<0) *m=-*m;
}
/* linearclip_u16.c's affine_fits, on the parameters the kernel will see. */
static int lc16_fits(const QdLinearclipParams *p) {
    if (p->shift<1 || p->shift>31 || p->input_zero<0 || p->input_zero>65535 ||
        p->output_zero<0 || p->output_zero>65535) return 0;
    int64_t r=1ll<<(p->shift-1),a=-(int64_t)p->input_zero*p->multiplier+r;
    int64_t b=(65535-(int64_t)p->input_zero)*p->multiplier+r;
    if (a<INT32_MIN||a>INT32_MAX||b<INT32_MIN||b>INT32_MAX) return 0;
    a=(a>>p->shift)+p->output_zero;b=(b>>p->shift)+p->output_zero;
    return a>=INT32_MIN&&a<=INT32_MAX&&b>=INT32_MIN&&b<=INT32_MAX;
}
static int lc16_params(const Nprg2ExecCall *c,QdLinearclipParams *p) {
    const uint8_t *fi=lc_fi(c);int64_t m,sh;uint32_t zp;
    if (!fi) return 0;
    lc_consts(fi,&m,&sh,&zp);
    if (m<INT32_MIN||m>INT32_MAX||sh<1||sh>31) return 0;
    p->multiplier=(int32_t)m;p->input_zero=(int32_t)(zp&0xffffu);p->output_zero=(int32_t)(zp>>16);
    p->shift=(uint32_t)sh;p->reserved=0;
    return lc16_fits(p);
}
static int lc16_validate(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx;
    if (!c || c->command.opcode!=QDR_LINEARCLIP || c->command.variant!=67 ||
        c->command.input_count!=2 || c->command.output_count!=1) return 0;
    QdBlockedU16Out a,o;QdLinearclipParams p;
    return lc16_params(c,&p) && u16_pair_ok(c,&a,&o);
}
static int lc16_run(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx;
    QdBlockedU16Out a,o;QdLinearclipParams p;
    if (!make_u16(c,0,0,&a) || !make_u16(c,0,1,&o) || !lc16_params(c,&p)) return 0;
    QdBlockedU16Linearclip prepared;to_glue(&a,&prepared.in);to_glue(&o,&prepared.out);prepared.params=p;
    prepared.cookie=0x51444c43u;
    return qd_blocked_linearclip_u16_apply_params(&prepared,&p);
}

/* LINEARCLIP/69: flat u8 direct tensors, dense (dims == metadata, no origin). */
typedef struct { uint8_t *src,*dst;uint32_t n,m; } LcFlatPlan;
static int flat_u8(const Nprg2ExecCall *c,uint32_t slot,int output,uint8_t **ptr,uint32_t *n) {
    uint32_t id;Nprg2TensorView v;
    if ((output?nprg2_exec_output(c,slot,&id,&v):nprg2_exec_input(c,slot,&id,&v))!=NPRG2_EXEC_OK) return 0;
    if (v.storage_kind!=1 || v.pool>=7 || !c->pools[v.pool].base || v.geometry_record[0]!=1) return 0;
    if (output && v.pool==5) return 0;
    int32_t d[4],m[4],o[4];if (!view_shape(&v,d,m,o)) return 0;
    uint64_t count=1;
    for (int i=0;i<4;i++) { if (d[i]!=m[i]||o[i]) return 0;count*=(uint32_t)d[i];if (count>UINT32_MAX) return 0; }
    uint32_t limit=c->pools[v.pool].bytes;
    if (output && v.pool==1 && limit>c->pool1_writable_bytes) limit=c->pool1_writable_bytes;
    if (v.offset>limit || count>limit-v.offset) return 0;
    *ptr=c->pools[v.pool].base+v.offset;*n=(uint32_t)count;return 1;
}
static int lcflat_plan(const Nprg2ExecCall *c,LcFlatPlan *p) {
    if (!c || c->command.opcode!=QDR_LINEARCLIP || c->command.variant!=69 ||
        c->command.input_count!=2 || c->command.output_count!=1) return 0;
    uint32_t nin,nout;
    if (!flat_u8(c,0,0,&p->src,&nin) || !flat_u8(c,0,1,&p->dst,&nout) || nin!=nout) return 0;
    p->n=nin;
    uintptr_t s=(uintptr_t)p->src,d=(uintptr_t)p->dst;
    if (s!=d && s<d+nin && d<s+nin) return 0; /* only identical or disjoint */
    const uint8_t *fi=lc_fi(c);int64_t m,sh;uint32_t zp;
    if (!fi) return 0;
    lc_consts(fi,&m,&sh,&zp);
    /* k_linearclip's u8 input form requires exactly this. */
    if (sh!=14 || zp!=0 || m<0) return 0;
    p->m=(uint32_t)m;return 1;
}
static int lcflat_validate(const Nprg2ExecCall *c,void *ctx) { (void)ctx;LcFlatPlan p;return lcflat_plan(c,&p); }
static int lcflat_run(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx;LcFlatPlan p;
    if (!lcflat_plan(c,&p)) return 0;
    qd_linearclip_flat_u8(p.src,p.dst,p.n,p.m);return 1;
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
    rc=nprg2_executor_register_scratch_region(e,QDR_LINEARCLIP,68,linearclip_validate,linearclip_run,NULL,
            NPRG2_ELEMENTWISE_SCRATCH_BYTES,0,region);
    if (rc) return rc;
    rc=nprg2_executor_register_scratch_region(e,QDR_SIGN,39,sign_validate,sign_run,NULL,
            NPRG2_ELEMENTWISE_SCRATCH_BYTES,0,region);
    if (rc) return rc;
    rc=nprg2_executor_register_scratch_region(e,QDR_LINEARCLIP,67,lc16_validate,lc16_run,NULL,
            NPRG2_ELEMENTWISE_SCRATCH_BYTES,0,region);
    if (rc) return rc;
    rc=nprg2_executor_register_scratch_region(e,QDR_LINEARCLIP,69,lcflat_validate,lcflat_run,NULL,
            NPRG2_ELEMENTWISE_SCRATCH_BYTES,0,region);
    if (rc) return rc;
    /* u16 crouton elementwise (blocked_glue/elementwise_u16.h). */
    static const struct { int w; Nprg2ExecValidate v; Nprg2ExecRun r; } h16[W16_COUNT]={
        {W16_REQ,req16_validate,req16_run},{W16_ADD1111,add1111_validate,add1111_run},
        {W16_MUL,mul16_validate,mul16_run},{W16_ADD,add16_validate,add16_run},
        {W16_ADDSUB,addsub16_validate,addsub16_run},{W16_MUL1111,mul1111_validate,mul1111_run},
        {W16_MULINL,mulinl_validate,mulinl_run}};
    for (unsigned i=0;i<W16_COUNT;i++) {
        rc=nprg2_executor_register_scratch_region(e,w16_spec[h16[i].w].opcode,w16_spec[h16[i].w].variant,
                h16[i].v,h16[i].r,NULL,NPRG2_ELEMENTWISE_SCRATCH_BYTES,0,region);
        if (rc) return rc;
    }
    return NPRG2_EXEC_OK;
}
int nprg2_executor_register_elementwise(Nprg2Executor *e) {
    return nprg2_executor_register_elementwise_region(e,NPRG2_EXEC_SCRATCH_DEFAULT);
}
