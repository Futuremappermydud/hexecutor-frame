/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_fuse.h"

#include <stdlib.h>
#include <string.h>

/* Block-level (2048-byte) liveness analysis of pool 1 over the bound
 * schedule. See prepared_fuse.h for the contract and EXECUTOR.md
 * (2026-11-01, conv epilogue fusion) for the argument. */
enum { NB_MAX=4096, WORDS=NB_MAX/32, BLOCK=2048, OP_CONV=6, OP_REQUANT_U8=8, OP_ADD_U8=10,
       VAR_REQUANT_U8=70, VAR_ADD_U8=13, OP_SPILL_FILL=26, OP_BLOCK_ZAP=44 };

static uint32_t rdu32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static int32_t rdi16f(const uint8_t *p){return (int16_t)(uint16_t)(p[0]|p[1]<<8);}
static void *fz_alloc(const Nprg2Allocator *a,size_t n){return a->alloc?a->alloc(a->ctx,n):malloc(n);}
static void fz_free(const Nprg2Allocator *a,void *p){if(!p)return;if(a->free)a->free(a->ctx,p);else if(!a->alloc)free(p);}

typedef struct {
    uint32_t w[3][WORDS];     /* 0 read, 1 write (touch), 2 cover */
    uint16_t l[3][NB_MAX];
    uint32_t n[3];
} Foot;
static void foot_clear(Foot *f){
    for(int k=0;k<3;k++){for(uint32_t i=0;i<f->n[k];i++){uint16_t b=f->l[k][i];f->w[k][b>>5]=0;}f->n[k]=0;}
}
static void foot_add(Foot *f,int k,uint32_t b){
    if(f->w[k][b>>5]&(1u<<(b&31)))return;
    f->w[k][b>>5]|=1u<<(b&31);f->l[k][f->n[k]++]=(uint16_t)b;
}
static int has(const uint32_t *bm,uint32_t b){return (bm[b>>5]>>(b&31))&1u;}

/* View facts shared by the footprint and the candidate tests. */
typedef struct {
    Nprg2TensorView v;
    const uint8_t *g;
    uint8_t dt,flags;
    int32_t d[4],m[4],p[4];
    uint32_t esize;
} View;
static uint32_t esize_of(uint8_t dt){static const uint8_t s[8]={0,1,1,2,2,4,4,4};return dt<8?s[dt]:0;}
static int load_view(const Nprg2Program *P,const Nprg2Bound *B,const Nprg2Command *c,int out,uint32_t slot,View *v){
    uint32_t first=out?c->output_first:c->input_first;
    if(first>B->refs.bytes/2u||slot>=(uint32_t)(out?c->output_count:c->input_count)||slot>B->refs.bytes/2u-first)return 0;
    uint32_t ix=rdu32((uint8_t[]){B->refs.data[(first+slot)*2u],B->refs.data[(first+slot)*2u+1u],0,0});
    if(nprg2_bound_tensor_at(P,B,ix,&v->v))return 0;
    v->g=v->v.geometry_record;if(!v->g)return 0;
    v->dt=v->g[0];v->flags=v->g[1];v->esize=esize_of(v->dt);
    for(int i=0;i<4;i++){
        v->d[i]=rdi16f(v->g+6+2*i);v->m[i]=rdi16f(v->g+14+2*i);v->p[i]=rdi16f(v->g+22+2*i);
        if(!(v->flags&1u)){v->d[i]=v->m[i]=v->p[i]=0;}
    }
    return 1;
}
static int blocked_dims(const View *v,int32_t *nbh,int32_t *nbw,int32_t *nbd,int32_t *cw){
    if(!(v->flags&1u)||!v->esize||v->v.storage_kind!=2)return 0;
    *cw=v->esize==1?8:v->esize==2?4:2;
    for(int i=0;i<4;i++)if(v->m[i]<=0&&v->d[i]>0)return 0;
    *nbh=(v->m[1]+7)/8;*nbw=(v->m[2]+*cw-1)/ *cw;*nbd=(v->m[3]+31)/32;
    return 1;
}
/* Adds a byte range of pool 1 (cover: blocks it fully contains). 0 on an
 * unmodelled range. */
static int add_range(Foot *f,int k,uint64_t off,uint64_t bytes,int cover,uint64_t pool_bytes){
    if(!bytes)return 1;
    if(off+bytes>pool_bytes||(off+bytes-1)/BLOCK>=NB_MAX)return 0;
    for(uint64_t b=off/BLOCK;b<=(off+bytes-1)/BLOCK;b++){
        foot_add(f,k,(uint32_t)b);
        if(cover&&b*BLOCK>=off&&(b+1)*BLOCK<=off+bytes)foot_add(f,2,(uint32_t)b);
    }
    return 1;
}
/* Adds the pool-1 footprint of one tensor view. write: touch set goes to
 * the write list (and the exactly covered blocks to the cover list). */
static int add_view(Foot *f,const View *v,int write,uint64_t pool_bytes){
    int k=write?1:0;
    if(v->v.storage_kind==1){
        if(v->v.pool!=1)return 1;
        if(!(v->flags&1u)||!v->esize)return 0;
        uint64_t n=1;for(int i=0;i<4;i++){if(v->m[i]<=0)return 0;n*=(uint64_t)v->m[i];}
        int full=1;for(int i=0;i<4;i++)if(v->p[i]!=0||v->d[i]!=v->m[i])full=0;
        return add_range(f,k,v->v.offset,n*v->esize,write&&full,pool_bytes);
    }
    if(v->v.storage_kind!=2)return 1;
    const uint32_t *t=v->v.blocks;uint32_t cnt=v->v.block_count;
    int32_t nbh,nbw,nbd,cw;
    uint64_t lo,hi;
    if(blocked_dims(v,&nbh,&nbw,&nbd,&cw)){
        uint64_t n=(uint64_t)(v->m[0]>0?v->m[0]:0)*nbh*nbw*nbd;
        lo=v->v.slice_origin;hi=lo+n;if(hi>cnt)hi=cnt;
    } else if(!(v->flags&1u)){lo=0;hi=cnt;}   /* no shape: whole table */
    else return 0;
    for(uint64_t i=lo;i<hi;i++){
        uint32_t e=t[i];if(!(e&127u))continue;
        if((e&127u)!=1u)continue;
        uint32_t off=e&~127u;if(off&(BLOCK-1u)||off/BLOCK>=NB_MAX||off+BLOCK>pool_bytes)return 0;
        foot_add(f,k,off/BLOCK);
    }
    if(write&&blocked_dims(v,&nbh,&nbw,&nbd,&cw)&&v->d[0]>0&&v->d[1]>0&&v->d[2]>0&&v->d[3]>0){
        int32_t h0=(v->p[1]+7)/8,h1=(v->p[1]+v->d[1])/8,w0=(v->p[2]+cw-1)/cw,w1=(v->p[2]+v->d[2])/cw;
        int32_t z0=(v->p[3]+31)/32,z1=(v->p[3]+v->d[3])/32;
        for(int32_t b=v->p[0];b<v->p[0]+v->d[0];b++)for(int32_t h=h0;h<h1;h++)for(int32_t w=w0;w<w1;w++)for(int32_t z=z0;z<z1;z++){
            uint64_t ix=(((uint64_t)b*nbh+h)*nbw+w)*nbd+v->v.slice_origin+z;
            if(ix>=cnt)continue;uint32_t e=t[ix];
            if((e&127u)==1u)foot_add(f,2,(e&~127u)/BLOCK);
        }
    }
    return 1;
}
static int command_foot(const Nprg2Program *P,const Nprg2Bound *B,const Nprg2Command *c,Foot *f,uint64_t pool_bytes){
    foot_clear(f);
    View v;
    for(uint32_t k=0;k<c->input_count;k++){if(!load_view(P,B,c,0,k,&v)||!add_view(f,&v,0,pool_bytes))return 0;}
    for(uint32_t k=0;k<c->output_count;k++){if(!load_view(P,B,c,1,k,&v)||!add_view(f,&v,1,pool_bytes))return 0;}
    if(c->vector_count){
        if(!B->vectors.data||c->vector_first>B->vectors.bytes/4u||c->vector_count>B->vectors.bytes/4u-c->vector_first)return 0;
        const uint8_t *fv=B->vectors.data+(size_t)c->vector_first*4u;
        if(c->opcode==OP_SPILL_FILL&&c->vector_count>=4u){
            for(uint32_t i=4;i+1<c->vector_count;i+=2){
                uint32_t p=rdu32(fv+4u*i),n=rdu32(fv+4u*(i+1));
                if((p&0x7ffu)!=1u)continue;
                if(c->variant==9){if(!add_range(f,0,p&~0x7ffu,n,0,pool_bytes))return 0;}
                else if(c->variant==8){if(!add_range(f,1,p&~0x7ffu,n,1,pool_bytes))return 0;}
                else return 0;
            }
        } else if(c->opcode==OP_BLOCK_ZAP&&c->vector_count>=2u){
            uint32_t header=rdu32(fv+4),count=header&0x7fffu;
            for(uint32_t j=0;j<count;j++){
                if(2u+2u*j+1u>=c->vector_count)return 0;
                if(!add_range(f,0,rdu32(fv+4u*(2u+2u*j)),BLOCK,0,pool_bytes)||
                   !add_range(f,1,rdu32(fv+4u*(3u+2u*j)),BLOCK,1,pool_bytes))return 0;
            }
        } else return 0;   /* unmodelled framework vectors */
    }
    return 1;
}

/* ------------------------------------------------------------ candidates */
typedef struct {
    uint32_t conv, epi; uint8_t operand, inplace;
    uint16_t *rd, *wr; uint32_t nrd, nwr;   /* the epilogue's footprint */
    uint16_t *ti; uint32_t nti;             /* conv output blocks */
    uint8_t gap_read, live_after, accepted;
} Cand;

/* u8 CH tensor the fused path can reproduce exactly. */
static int fuse_geom(const View *v){
    int32_t nbh,nbw,nbd,cw;
    if(v->dt!=1||!blocked_dims(v,&nbh,&nbw,&nbd,&cw))return 0;
    for(int i=0;i<4;i++)if(v->p[i]!=0||v->d[i]!=v->m[i])return 0;
    return v->m[0]==1&&v->m[1]>0&&v->m[1]%8==0&&v->m[2]>0&&v->m[2]%8==0&&v->m[3]>0&&v->m[3]%32==0;
}
static int same_shape(const View *a,const View *b){return a->dt==b->dt&&!memcmp(a->m,b->m,sizeof a->m);}
/* Entries [slice, slice+n) must all be present, aligned pool-1 blocks, pairwise distinct;
 * fills ids. */
static int entries(const View *v,uint16_t *ids,uint32_t *n_out,uint32_t *stamp,uint32_t *stamp_gen,uint64_t pool_bytes){
    int32_t nbh,nbw,nbd,cw;if(!blocked_dims(v,&nbh,&nbw,&nbd,&cw))return 0;
    uint64_t n=(uint64_t)nbh*nbw*nbd;
    if((uint64_t)v->v.slice_origin+n>v->v.block_count||n>NB_MAX)return 0;
    (*stamp_gen)++;
    for(uint64_t i=0;i<n;i++){
        uint32_t e=v->v.blocks[v->v.slice_origin+i];
        if((e&127u)!=1u)return 0;
        uint32_t off=e&~127u;if((off&(BLOCK-1u))||off/BLOCK>=NB_MAX||off+BLOCK>pool_bytes)return 0;
        uint32_t b=off/BLOCK;if(stamp[b]==*stamp_gen)return 0;stamp[b]=*stamp_gen;ids[i]=(uint16_t)b;
    }
    *n_out=(uint32_t)n;return 1;
}

int nprg2_fuse_plan_build(Nprg2FusePlan *plan,const Nprg2Program *P,const Nprg2Bound *B,Nprg2Allocator al){
    if(!plan||!P||!B||!B->ready)return NPRG2_FUSE_ARGUMENT;
    memset(plan,0,sizeof *plan);plan->allocator=al;plan->commands=P->command_count;
    uint64_t pool_bytes=P->pool_bytes[1];
    if(!pool_bytes||pool_bytes/BLOCK>NB_MAX)return NPRG2_FUSE_UNSUPPORTED;
    uint32_t N=P->command_count;
    int rc=NPRG2_FUSE_MEMORY;
    Foot *f=fz_alloc(&al,sizeof *f);
    uint32_t *lw=fz_alloc(&al,NB_MAX*4u),*lr=fz_alloc(&al,NB_MAX*4u),*stamp=fz_alloc(&al,NB_MAX*4u),*L=fz_alloc(&al,WORDS*4u);
    uint16_t *ida=fz_alloc(&al,NB_MAX*2u),*idb=fz_alloc(&al,NB_MAX*2u),*idc=fz_alloc(&al,NB_MAX*2u),*idd=fz_alloc(&al,NB_MAX*2u);
    uint16_t *role=fz_alloc(&al,(size_t)(N?N:1)*2u);
    Cand *cand=NULL;uint32_t ncand=0,capc=0;
    Nprg2FusePair *pairs=NULL;
    if(!f||!lw||!lr||!stamp||!L||!ida||!idb||!idc||!idd||!role)goto done;
    memset(f,0,sizeof *f);memset(lw,0,NB_MAX*4u);memset(lr,0,NB_MAX*4u);memset(stamp,0,NB_MAX*4u);memset(L,0,WORDS*4u);
    memset(role,0,(size_t)(N?N:1)*2u);
    uint32_t gen=0;
    rc=NPRG2_FUSE_UNSUPPORTED;
    /* ---- forward pass: candidates */
    for(uint32_t y=0;y<N;y++){
        Nprg2Command c;if(nprg2_command_at(B,y,&c))goto done;
        if((c.opcode==OP_REQUANT_U8&&c.variant==VAR_REQUANT_U8)||(c.opcode==OP_ADD_U8&&c.variant==VAR_ADD_U8)){
            int add=c.opcode==OP_ADD_U8;
            if(c.input_count>=1&&c.input_count<=2&&c.output_count==1&&(!add||c.input_count==2)){
                View eo;
                for(uint32_t k=0;k<(add?2u:1u);k++){
                    View ei,ov;uint32_t nti,nto,nov;
                    if(!load_view(P,B,&c,0,k,&ei)||!fuse_geom(&ei)||!entries(&ei,ida,&nti,stamp,&gen,pool_bytes))continue;
                    uint32_t w=lw[ida[0]];
                    if(!w)continue;
                    int uniq=1;for(uint32_t i=0;i<nti;i++)if(lw[ida[i]]!=w){uniq=0;break;}
                    if(!uniq)continue;
                    w--;
                    Nprg2Command cc;if(nprg2_command_at(B,w,&cc)||cc.opcode!=OP_CONV||cc.output_count!=1)continue;
                    View co;
                    if(!load_view(P,B,&cc,1,0,&co)||!fuse_geom(&co)||!same_shape(&co,&ei))continue;
                    uint32_t ntc;if(!entries(&co,idb,&ntc,stamp,&gen,pool_bytes)||ntc!=nti)continue;
                    int eq=1;for(uint32_t i=0;i<nti;i++)if(idb[i]!=ida[i]){eq=0;break;}
                    if(!eq)continue;
                    /* other operand (ADD) */
                    uint32_t nother=0;
                    if(add){
                        if(!load_view(P,B,&c,0,1-k,&ov)||!fuse_geom(&ov)||!same_shape(&ov,&ei)||!entries(&ov,idc,&nov,stamp,&gen,pool_bytes))continue;
                        int bad=0;
                        for(uint32_t i=0;i<nov&&!bad;i++){
                            uint32_t b=idc[i];
                            if(lw[b]>w)bad=1;                              /* written at/after the conv */
                            for(uint32_t j=0;j<nti&&!bad;j++)if(ida[j]==b)bad=1;
                        }
                        if(bad)continue;
                        nother=nov;
                    }
                    if(!load_view(P,B,&c,1,0,&eo)||!fuse_geom(&eo)||!same_shape(&eo,&ei)||!entries(&eo,idd,&nto,stamp,&gen,pool_bytes))continue;
                    uint16_t *opnd=idc;
                    int inplace=0;
                    {   /* output vs conv output */
                        uint32_t common=0;
                        for(uint32_t i=0;i<nto;i++)for(uint32_t j=0;j<nti;j++)if(idd[i]==ida[j])common++;
                        if(common){
                            if(common==nto&&nto==nti){
                                inplace=1;
                                for(uint32_t i=0;i<nto;i++)if(idd[i]!=ida[i])inplace=0;
                            }
                            if(!inplace){continue;}
                        }
                    }
                    int bad=0;
                    for(uint32_t i=0;i<nto&&!bad;i++){
                        uint32_t b=idd[i];
                        if(lr[b]>w)bad=1;                                    /* read by the conv or later */
                        if(!inplace&&lw[b]>w)bad=1;                          /* written in between */
                        if(inplace&&lw[b]!=w+1)bad=1;
                        for(uint32_t j=0;j<nother&&!bad;j++)if(opnd[j]==b)bad=1;
                    }
                    if(bad){continue;}
                    /* gap readers of the raw conv output */
                    int gap_read=0;
                    for(uint32_t i=0;i<nti;i++)if(lr[ida[i]]>=w+2u)gap_read=1;
                    if(inplace&&gap_read){continue;}
                    if(ncand==capc){
                        uint32_t nc=capc?capc*2u:64u;Cand *t=fz_alloc(&al,nc*sizeof *t);
                        if(!t){rc=NPRG2_FUSE_MEMORY;goto done;}
                        if(cand){memcpy(t,cand,ncand*sizeof *t);fz_free(&al,cand);}
                        cand=t;capc=nc;
                    }
                    Cand *q=&cand[ncand++];memset(q,0,sizeof *q);
                    q->conv=w;q->epi=y;q->operand=(uint8_t)k;q->inplace=(uint8_t)inplace;q->gap_read=(uint8_t)gap_read;
                    /* epilogue footprint: reads = input operands (conv input, other, and for REQUANT its params
                     * are tiny pool-5 tensors, but record all input tensors), writes = output */
                    if(!command_foot(P,B,&c,f,pool_bytes)){rc=NPRG2_FUSE_UNSUPPORTED;goto done;}
                    q->nrd=f->n[0];q->nwr=f->n[1];q->nti=nti;
                    q->rd=fz_alloc(&al,(size_t)(q->nrd+q->nwr+q->nti+1u)*2u);
                    if(!q->rd){rc=NPRG2_FUSE_MEMORY;goto done;}
                    q->wr=q->rd+q->nrd;q->ti=q->wr+q->nwr;
                    memcpy(q->rd,f->l[0],q->nrd*2u);memcpy(q->wr,f->l[1],q->nwr*2u);memcpy(q->ti,ida,nti*2u);
                    
                }
            }
        }
        if(!command_foot(P,B,&c,f,pool_bytes))goto done;
        for(uint32_t i=0;i<f->n[0];i++)lr[f->l[0][i]]=y+1u;
        for(uint32_t i=0;i<f->n[1];i++)lw[f->l[1][i]]=y+1u;
    }
    /* ---- reverse pass: is the raw output live after the epilogue? */
    {
        uint32_t ci=ncand;
        for(uint32_t y=N;y-->0;){
            Nprg2Command c;if(nprg2_command_at(B,y,&c))goto done;
            while(ci>0&&cand[ci-1].epi>y)ci--;     /* candidates are ordered by epi */
            if(!command_foot(P,B,&c,f,pool_bytes))goto done;
            for(uint32_t k=ci;k>0&&cand[k-1].epi==y;k--){
                Cand *q=&cand[k-1];
                for(uint32_t i=0;i<q->nti;i++){
                    uint32_t b=q->ti[i];
                    if(has(L,b)&&!has(f->w[2],b))q->live_after=1;
                }
            }
            for(uint32_t i=0;i<f->n[2];i++)L[f->l[2][i]>>5]&=~(1u<<(f->l[2][i]&31));
            for(uint32_t i=0;i<f->n[0];i++)L[f->l[0][i]>>5]|=1u<<(f->l[0][i]&31);
        }
    }
    /* ---- greedy selection */
    pairs=fz_alloc(&al,(size_t)(ncand?ncand:1)*sizeof *pairs);
    if(!pairs){rc=NPRG2_FUSE_MEMORY;goto done;}
    uint32_t np=0;
    for(uint32_t i=0;i<ncand;i++){
        Cand *q=&cand[i];
        int dead=!q->gap_read&&!q->live_after;
        if(q->inplace&&!dead)continue;
        int ok=1;
        for(uint32_t j=0;j<np&&ok;j++){
            const Nprg2FusePair *p=&pairs[j];const Cand *o=NULL;
            for(uint32_t t=0;t<ncand;t++)if(cand[t].accepted&&cand[t].conv==p->conv){o=&cand[t];break;}
            if(!o)continue;
            if(p->conv==q->conv||p->epi==q->epi){ok=0;break;}
            if(!(q->conv<o->epi&&o->conv<q->epi))continue;      /* disjoint intervals */
            /* writes(a) vs reads/writes(b) and writes(b) vs reads(a) */
            for(uint32_t a=0;a<q->nwr&&ok;a++){
                for(uint32_t b=0;b<o->nwr&&ok;b++)if(q->wr[a]==o->wr[b])ok=0;
                for(uint32_t b=0;b<o->nrd&&ok;b++)if(q->wr[a]==o->rd[b])ok=0;
            }
            for(uint32_t a=0;a<o->nwr&&ok;a++)for(uint32_t b=0;b<q->nrd&&ok;b++)if(o->wr[a]==q->rd[b])ok=0;
        }
        if(!ok)continue;
        q->accepted=1;
        pairs[np]=(Nprg2FusePair){q->conv,q->epi,q->operand,(uint8_t)dead,0,0};
        np++;
    }
    if(np>=NPRG2_FUSE_EPI-1u)goto done;
    for(uint32_t i=0;i<np;i++){role[pairs[i].conv]=(uint16_t)(i+1u);role[pairs[i].epi]=(uint16_t)(NPRG2_FUSE_EPI|(i+1u));}
    plan->pairs=pairs;plan->count=np;plan->role=role;pairs=NULL;role=NULL;rc=NPRG2_FUSE_OK;
done:
    for(uint32_t i=0;i<ncand;i++)fz_free(&al,cand[i].rd);
    fz_free(&al,cand);fz_free(&al,pairs);fz_free(&al,role);
    fz_free(&al,f);fz_free(&al,lw);fz_free(&al,lr);fz_free(&al,stamp);fz_free(&al,L);fz_free(&al,ida);fz_free(&al,idb);fz_free(&al,idc);fz_free(&al,idd);
    if(rc!=NPRG2_FUSE_OK){Nprg2Allocator keep=plan->allocator;memset(plan,0,sizeof *plan);plan->allocator=keep;}
    return rc;
}
void nprg2_fuse_plan_free(Nprg2FusePlan *plan){
    if(!plan)return;
    fz_free(&plan->allocator,plan->pairs);fz_free(&plan->allocator,plan->role);
    memset(plan,0,sizeof *plan);
}
