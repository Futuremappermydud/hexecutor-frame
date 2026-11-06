/* Bounded, allocation-free QDRX reader. No libc, host pointers in the wire
 * format, physical addresses, power changes, or memory-pool writes. */
#include "qd_resident.h"

static const uint32_t strides[QDR_TABLES] = {128,4,96,4,4,88};
uint32_t qdr_u32(const void *data) {
    const uint8_t *p=(const uint8_t *)data;
    return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
}
uint64_t qdr_u64(const void *data) {
    const uint8_t *p=(const uint8_t *)data;
    return (uint64_t)qdr_u32(p)|((uint64_t)qdr_u32(p+4)<<32);
}
static uint32_t word(const uint8_t *p, uint32_t i) {return qdr_u32(p+4*i);}
static uint32_t fail(QdrStatus *s, uint32_t code, uint32_t detail) {
    s->code=code;s->detail=detail;return code;
}
static void reset(QdrStatus *s) {
    s->code=QDR_OK;s->command=QDR_NONE;s->tensor=QDR_NONE;s->detail=0;
}
static uint32_t zero(const uint8_t *p, uint32_t bytes) {
    for(uint32_t i=0;i<bytes;i++)if(p[i])return 0;
    return 1;
}
static uint32_t span(uint32_t first, uint32_t n, uint32_t count) {
    return first<=count && n<=count-first && (n || first==0);
}
static uint32_t crc32(const uint8_t *p,uint32_t bytes) {
    uint32_t crc=0xffffffffu;
    for(uint32_t i=0;i<bytes;i++) {
        crc^=(i>=16 && i<20)?0:p[i];
        for(uint32_t j=0;j<8;j++)crc=(crc>>1)^(0xedb88320u & (0u-(crc&1)));
    }
    return ~crc;
}
const uint8_t *qdr_record(const QdrImage *v, uint32_t table, uint32_t index) {
    if(table>=QDR_TABLES || index>=v->count[table])return (const uint8_t *)0;
    return v->data+v->offset[table]+index*strides[table];
}
static uint32_t table_word(const QdrImage *v, uint32_t table, uint32_t index) {
    return qdr_u32(qdr_record(v,table,index));
}
static uint32_t pool_span(const QdrImage *v, uint32_t pool, uint32_t off, uint32_t bytes) {
    return pool<7 && pool!=0 && pool!=2 && off<=v->pool_bytes[pool] && bytes<=v->pool_bytes[pool]-off;
}
static uint32_t class_valid(uint32_t c) {
    for(uint32_t i=0;i<sizeof(qdr_classes)/sizeof(qdr_classes[0]);i++)if(c==qdr_classes[i])return 1;
    return 0;
}
static uint32_t tensors(const QdrImage *v,QdrStatus *s,uint32_t *ddr3,uint32_t *ddr4) {
    uint32_t previous=0;
    for(uint32_t i=0;i<v->count[QDR_TENSORS];i++) {
        const uint8_t *t=qdr_record(v,QDR_TENSORS,i);
        uint32_t id=word(t,0),dtype=word(t,1),cls=word(t,2),f=word(t,3);
        s->tensor=i;
        if(id<=previous || dtype>7 || !class_valid(cls) || f&~127u ||
           !zero(t+120,8) || !!(f&QDR_HAS_SHAPE)+!!(f&QDR_SHAPE_VALUE)+!!(f&QDR_SCALAR)!=1)
            return fail(s,QDR_BAD_TENSOR,0);
        previous=id;
        if((f&(QDR_BLOCKED|QDR_DIRECT|QDR_CROUTON)) && !(f&QDR_HAS_SHAPE))return fail(s,QDR_BAD_TENSOR,1);
        if((f&QDR_BLOCKED) && (f&QDR_DIRECT))return fail(s,QDR_BAD_TENSOR,2);
        if(!!(f&QDR_CROUTON)!=!!((f&QDR_BLOCKED) && ((cls&255)=='C' || (cls&255)=='c')))
            return fail(s,QDR_BAD_TENSOR,3);
        if((f&QDR_SHAPE_VALUE)?(dtype!=0):(!dtype))return fail(s,QDR_BAD_TENSOR,4);
        if(!(f&QDR_SHAPE_VALUE) && !zero(t+104,16))return fail(s,QDR_BAD_TENSOR,5);
        if(!(f&QDR_SCALAR) && word(t,25))return fail(s,QDR_BAD_TENSOR,6);
        if((f&QDR_SCALAR) && dtype!=5 && dtype!=6)return fail(s,QDR_BAD_TENSOR,7);
        if(!(f&QDR_QUANT) && !zero(t+88,12))return fail(s,QDR_BAD_TENSOR,8);
        if((f&QDR_QUANT) && ((word(t,24)&0x7ff00000u)==0x7ff00000u || word(t,24)&0x80000000u ||
            !(word(t,23)|(word(t,24)&0x7fffffffu))))return fail(s,QDR_BAD_TENSOR,9);
        uint32_t size=dtype==1 || dtype==2?1:dtype==3 || dtype==4?2:4;
        uint64_t total=size;
        if(f&QDR_HAS_SHAPE) {
            for(uint32_t j=0;j<4;j++) {
                uint32_t d=word(t,4+j),m=word(t,8+j),p=word(t,12+j);
                if(!d || !m || m>0x7fffffffu || p>m || d>m-p)
                    return fail(s,QDR_BAD_TENSOR,10+j);
                total*=m;
                if(total>0xffffffffu)return fail(s,QDR_BAD_TENSOR,10+j);
            }
            if(word(t,21)&~1u)return fail(s,QDR_BAD_TENSOR,14);
        } else if(!zero(t+16,48) || word(t,21))return fail(s,QDR_BAD_TENSOR,15);
        if(f&QDR_DIRECT) {
            uint32_t pool=word(t,16),off=word(t,17);
            if(!pool_span(v,pool,off,(uint32_t)total) || pool==6)return fail(s,QDR_BAD_POOL,pool);
            if(pool==3 && off+total>*ddr3)*ddr3=off+(uint32_t)total;
            if(pool==4 && off+total>*ddr4)*ddr4=off+(uint32_t)total;
        } else if(word(t,16) || word(t,17))return fail(s,QDR_BAD_TENSOR,16);
        if(f&QDR_BLOCKED) {
            uint32_t start=word(t,18),n=word(t,19);
            if(!n || !span(start,n,v->count[QDR_BLOCKS]) || word(t,20)>0x7fffffffu)
                return fail(s,QDR_BAD_TENSOR,18);
        } else if(!zero(t+72,12))return fail(s,QDR_BAD_TENSOR,19);
    }
    s->tensor=QDR_NONE;
    return QDR_OK;
}
static uint32_t required_call(uint32_t opcode) {
    switch(opcode) {
    case QDR_REQUANT_U8:return QDR_CALL_REQUANTIZE_CROUTON_U8;
    case QDR_REQUANT_U16:return QDR_CALL_REQUANTIZE_CROUTON_U16;
    case QDR_ADD_U8:return QDR_CALL_ELEMENTWISE_ADD_CROUTON_U8;
    case QDR_ADD_U16:return QDR_CALL_ELEMENTWISE_ADD_CROUTON_U16;
    case QDR_ADDSUB_U16:return QDR_CALL_ELEMENTWISE_ADDSUB_CROUTON_U16;
    case QDR_ADD1111_U16:return QDR_CALL_ELEMENTWISE_ADD1111_CROUTON_U16;
    case QDR_MUL_U16:return QDR_CALL_ELEMENTWISE_MUL_CROUTON_U16;
    case QDR_MUL1111_U16:return QDR_CALL_ELEMENTWISE_MUL1111_CROUTON_U16;
    default:return 0;
    }
}
static uint32_t commands(const QdrImage *v,QdrStatus *s) {
    /* Shared spans may be referenced repeatedly. Bound aggregate work as well
     * as image bytes, so hostile reuse cannot turn a small image into billions
     * of validator iterations. This is in addition to the dispatch budget. */
    uint32_t work=QDR_MAX_BYTES;
    for(uint32_t i=0;i<v->count[QDR_COMMANDS];i++) {
        const uint8_t *c=qdr_record(v,QDR_COMMANDS,i);
        uint32_t op=word(c,1),variant=word(c,2),flags=word(c,3),io=word(c,5);
        s->command=i;
        if(!variant || variant>QDR_VARIANTS || flags>1 || word(c,4)!=(i?i-1:QDR_NONE) || !zero(c+72,24))
            return fail(s,QDR_BAD_COMMAND,0);
        if(op!=(flags?QDR_NOP:qdr_variant_ops[variant]) || (flags && qdr_variant_ops[variant]!=QDR_INPUT_SLICE_PAD))
            return fail(s,QDR_BAD_COMMAND,1);
        for(uint32_t j=6;j<=16;j+=2) {
            uint32_t table=j<10?QDR_REFS:j<16?QDR_WORDS:QDR_CALL_TABLE;
            if(!span(word(c,j),word(c,j+1),v->count[table]))return fail(s,QDR_BAD_COMMAND,j);
        }
        uint32_t orig=qdr_variant_ops[variant];
        if(orig==QDR_INPUT_SLICE || orig==QDR_INPUT_SLICE_PAD) {
            if(io>=word(v->data,34))return fail(s,QDR_BAD_COMMAND,5);
        } else if(orig==QDR_OUTPUT_SLICE) {
            if(io>=word(v->data,36))return fail(s,QDR_BAD_COMMAND,5);
        } else if(io!=QDR_NONE)return fail(s,QDR_BAD_COMMAND,5);
        uint32_t required=required_call(op),ncall=word(c,17),first=word(c,16),matches=0;
        uint32_t nw=word(c,15);
        if(ncall>work || nw>work-ncall)return fail(s,QDR_BUDGET,i);
        work-=ncall+nw;
        for(uint32_t j=0;j<ncall;j++)if(word(qdr_record(v,QDR_CALL_TABLE,first+j),0)==required)matches++;
        if(required && matches!=1)return fail(s,QDR_BAD_CALL,required);
        if(op==QDR_ADDSUB_FLAT) {
            uint32_t name=ncall?word(qdr_record(v,QDR_CALL_TABLE,first),0):0;
            if(ncall!=1 || (name!=QDR_CALL_ELEMENTWISE_ADD_FLAT_U16 && name!=QDR_CALL_ELEMENTWISE_SUB_FLAT_U16))
                return fail(s,QDR_BAD_CALL,name);
        }
        if(op==QDR_SPILL_FILL) {
            uint32_t firstw=word(c,14),n=word(c,15);
            if(n<4 || n&1)return fail(s,QDR_BAD_COMMAND,14);
            uint32_t off=table_word(v,QDR_WORDS,firstw+3);
            for(uint32_t j=4;j<n;j+=2) {
                uint32_t addr=table_word(v,QDR_WORDS,firstw+j),bytes=table_word(v,QDR_WORDS,firstw+j+1);
                uint32_t pool=addr&0x7ff;
                if(pool==6 || !pool_span(v,pool,addr&~0x7ffu,bytes) || !pool_span(v,6,off,bytes))
                    return fail(s,QDR_BAD_POOL,pool);
                off+=bytes;
            }
        }
        if(op==QDR_BLOCK_ZAP) {
            uint32_t firstw=word(c,14),n=word(c,15);
            if(n<2)return fail(s,QDR_BAD_COMMAND,14);
            uint32_t h=table_word(v,QDR_WORDS,firstw+1),count=h&0x7fff;
            if(!(h&0x8000) || (h&0xc00000)!=0xc00000 || count>(n-2)/2)return fail(s,QDR_BAD_COMMAND,15);
            for(uint32_t j=0;j<count*2;j++)if(!pool_span(v,1,table_word(v,QDR_WORDS,firstw+2+j),2048))
                return fail(s,QDR_BAD_POOL,1);
            for(uint32_t j=2+count*2;j<n;j++)if(table_word(v,QDR_WORDS,firstw+j))return fail(s,QDR_BAD_COMMAND,16);
        }
    }
    s->command=QDR_NONE;
    return QDR_OK;
}

uint32_t qdr_open(const void *data,uint32_t bytes,QdrImage *out,QdrStatus *s) {
    reset(s);
    /* Do not publish even a partial view on failure. */
    out->data=(const uint8_t *)0;out->bytes=0;
    if(!data || bytes<QDR_HEADER_BYTES || bytes>QDR_MAX_BYTES)return fail(s,QDR_BAD_HEADER,0);
    const uint8_t *p=(const uint8_t *)data;
    if(word(p,0)!=0x58524451u || word(p,1)!=1 || word(p,2)!=QDR_HEADER_BYTES || word(p,3)!=bytes ||
       word(p,5)!=QDR_ABI || word(p,6) || word(p,7) || !zero(p+196,60))return fail(s,QDR_BAD_HEADER,1);
    if(crc32(p,bytes)!=word(p,4))return fail(s,QDR_BAD_CRC,0);
    QdrImage v;
    v.data=p;v.bytes=bytes;
    uint32_t end=QDR_HEADER_BYTES;
    for(uint32_t i=0;i<QDR_TABLES;i++) {
        uint32_t off=word(p,37+i*2),n=word(p,38+i*2),aligned=(end+7)&~7u;
        if(off!=aligned || off>bytes || !zero(p+end,off-end) || (uint64_t)n*strides[i]>bytes-off)
            return fail(s,QDR_BAD_SECTION,i);
        v.offset[i]=off;v.count[i]=n;end=off+n*strides[i];
    }
    if(end!=bytes || !v.count[QDR_TENSORS] || !v.count[QDR_COMMANDS] ||
       v.count[QDR_TENSORS]>65536 || v.count[QDR_COMMANDS]>65536)return fail(s,QDR_BAD_SECTION,6);
    for(uint32_t i=0;i<7;i++)v.pool_bytes[i]=word(p,24+i);
    if(v.pool_bytes[0] || v.pool_bytes[2] || v.pool_bytes[1]!=0x800000 ||
       !v.pool_bytes[5] || v.pool_bytes[5]>0x800000 || v.pool_bytes[6]!=1769472 ||
       word(p,31)!=0x2d0000 || word(p,32)!=0x800000)return fail(s,QDR_BAD_POOL,0);
    uint32_t ddr3=0,ddr4=0;
    for(uint32_t i=0;i<v.count[QDR_BLOCKS];i++) {
        uint32_t b=table_word(&v,QDR_BLOCKS,i),pool=b&127,off=b&~127u;
        if(pool==6 || !pool_span(&v,pool,off,2048) || (pool!=5 && (off&2047)))return fail(s,QDR_BAD_BLOCK,i);
        if(pool==3 && off+2048>ddr3)ddr3=off+2048;
        if(pool==4 && off+2048>ddr4)ddr4=off+2048;
    }
    if(tensors(&v,s,&ddr3,&ddr4))return s->code;
    if(ddr3!=v.pool_bytes[3] || ddr4!=v.pool_bytes[4])return fail(s,QDR_BAD_POOL,3);
    for(uint32_t i=0;i<v.count[QDR_REFS];i++)if(table_word(&v,QDR_REFS,i)>=v.count[QDR_TENSORS])return fail(s,QDR_BAD_REF,i);
    if(!span(word(p,33),word(p,34),v.count[QDR_REFS]) || !word(p,34) ||
       !span(word(p,35),word(p,36),v.count[QDR_REFS]) || !word(p,36))return fail(s,QDR_BAD_REF,QDR_NONE);
    for(uint32_t i=0;i<v.count[QDR_CALL_TABLE];i++) {
        const uint8_t *c=qdr_record(&v,QDR_CALL_TABLE,i);
        uint32_t name=word(c,0),mask=word(c,1);
        if(!name || name>QDR_CALLS || mask&~1023u)return fail(s,QDR_BAD_CALL,i);
        for(uint32_t j=0;j<10;j++)if((mask&(1u<<j)) && qdr_u64(c+8+j*8))return fail(s,QDR_BAD_CALL,i);
    }
    if(commands(&v,s))return s->code;
    out->data=v.data;out->bytes=v.bytes;
    for(uint32_t i=0;i<QDR_TABLES;i++){out->offset[i]=v.offset[i];out->count[i]=v.count[i];}
    for(uint32_t i=0;i<7;i++)out->pool_bytes[i]=v.pool_bytes[i];
    return QDR_OK;
}

uint32_t qdr_dispatch(const void *data,uint32_t bytes,const QdrKernel *variants,uint32_t variant_count,
                     uint32_t budget,QdrCancel cancel,void *context,QdrStatus *s) {
    QdrImage v;
    if(qdr_open(data,bytes,&v,s))return s->code;
    if(budget<v.count[QDR_COMMANDS])return fail(s,QDR_BUDGET,v.count[QDR_COMMANDS]);
    for(uint32_t i=0;i<v.count[QDR_COMMANDS];i++) {
        const uint8_t *c=qdr_record(&v,QDR_COMMANDS,i);
        uint32_t variant=word(c,2);
        if(word(c,1)!=QDR_NOP && (!variants || variant>=variant_count || !variants[variant])) {
            s->command=i;return fail(s,QDR_UNSUPPORTED,variant);
        }
    }
    for(uint32_t i=0;i<v.count[QDR_COMMANDS];i++) {
        s->command=i;
        if(cancel && cancel(context))return fail(s,QDR_CANCELLED,0);
        const uint8_t *c=qdr_record(&v,QDR_COMMANDS,i);
        if(word(c,1)!=QDR_NOP) {
            uint32_t code=variants[word(c,2)](&v,i,context);
            if(code)return fail(s,QDR_KERNEL_ERROR,code);
        }
    }
    s->command=QDR_NONE;
    return QDR_OK;
}
