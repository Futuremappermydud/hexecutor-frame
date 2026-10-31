#include "qd_native_1x1_adapter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE "frame/native_adapter/preflight-primary.bin"
#define MAGIC 0x31505151u
#define WS_BYTES QD_N1X1_WORKSPACE_MAX

typedef struct {
    uint32_t dims[4],max_dims[4],padding[4],slice_offset,block_count;
} TensorMeta;
typedef struct {
    uint32_t sequence;
    TensorMeta input,output;
    uint32_t ci,co,kh,kw,input_dtype,output_dtype,weight_dtype,fi_dtype,weight_format;
    uint32_t weight_pool,weight_offset,weight_bytes,fi_pool,fi_offset,fi_bytes;
} Record;
_Static_assert(sizeof(Record)==44u*4u,"preflight record ABI");

int main(void) {
    FILE *f=fopen(FIXTURE,"rb");if(!f){puts("FAIL fixture open");return 1;}
    uint32_t h[9];if(fread(h,sizeof h,1,f)!=1 || h[0]!=MAGIC){puts("FAIL fixture header");return 2;}
    uint8_t *pool[7]={0};uint32_t pool_bytes[7];
    for(uint32_t i=0;i<7;i++) {
        pool_bytes[i]=h[2+i];
        if(pool_bytes[i]) {
            size_t n=(pool_bytes[i]+127u)&~127u;
            pool[i]=aligned_alloc(128,n);
            if(!pool[i]){puts("FAIL pool alloc");return 3;}
            memset(pool[i],0,n);
        }
    }
    static uint8_t workspace[WS_BYTES] __attribute__((aligned(128)));
    static uint32_t tables[512] __attribute__((aligned(128)));
    static uint32_t abi[30] __attribute__((aligned(128)));
    uint32_t passed=0;
    for(uint32_t n=0;n<h[1];n++) {
        Record rec;uint32_t *xb=NULL,*yb=NULL;
        if(fread(&rec,sizeof rec,1,f)!=1){puts("FAIL record read");return 4;}
        xb=malloc((size_t)rec.input.block_count*4u);yb=malloc((size_t)rec.output.block_count*4u);
        if(!xb || !yb || fread(xb,4,rec.input.block_count,f)!=rec.input.block_count ||
           fread(yb,4,rec.output.block_count,f)!=rec.output.block_count){puts("FAIL block refs");return 5;}
        if(rec.weight_pool>=7 || rec.fi_pool>=7 || !pool[rec.weight_pool] || !pool[rec.fi_pool] ||
           rec.weight_offset>pool_bytes[rec.weight_pool] || rec.weight_bytes>pool_bytes[rec.weight_pool]-rec.weight_offset ||
           rec.fi_offset>pool_bytes[rec.fi_pool] || rec.fi_bytes>pool_bytes[rec.fi_pool]-rec.fi_offset) {
            printf("FAIL source range at sequence %u\n",rec.sequence);return 6;
        }
        QdN1x1Request r;memset(&r,0,sizeof r);r.pool_count=7;r.pool_bytes=pool_bytes;
        for(uint32_t i=0;i<7;i++)r.pool[i]=pool[i];
        memcpy(r.input.dims,rec.input.dims,16);memcpy(r.input.max_dims,rec.input.max_dims,16);
        memcpy(r.input.padding,rec.input.padding,16);r.input.slice_offset=rec.input.slice_offset;
        r.input.block_count=rec.input.block_count;r.input.blocks=xb;
        memcpy(r.output.dims,rec.output.dims,16);memcpy(r.output.max_dims,rec.output.max_dims,16);
        memcpy(r.output.padding,rec.output.padding,16);r.output.slice_offset=rec.output.slice_offset;
        r.output.block_count=rec.output.block_count;r.output.blocks=yb;
        r.weight_ci=rec.ci;r.weight_co=rec.co;r.weight_kernel_h=rec.kh;r.weight_kernel_w=rec.kw;
        r.input_dtype=rec.input_dtype;r.output_dtype=rec.output_dtype;
        r.weight_dtype=rec.weight_dtype;r.fi_dtype=rec.fi_dtype;r.weight_format=rec.weight_format;
        r.weights=pool[rec.weight_pool]+rec.weight_offset;r.weights_bytes=rec.weight_bytes;
        r.fi=(const int32_t *)(pool[rec.fi_pool]+rec.fi_offset);r.fi_bytes=rec.fi_bytes;
        r.workspace=workspace;r.workspace_bytes=WS_BYTES;
        r.table_storage=tables;r.table_storage_bytes=sizeof tables;
        r.abi_storage=abi;r.abi_storage_bytes=sizeof abi;
        QdN1x1Prepared p;int e=qd_native_1x1_prepare(&r,&p);
        if(e) {printf("FAIL prepare sequence=%u status=%d\n",rec.sequence,e);return 7;}
        if(p.workspace_used>WS_BYTES || !p.effective_h || !p.effective_w) {
            printf("FAIL prepared bounds sequence=%u\n",rec.sequence);return 8;
        }
        for(uint32_t i=0;i<p.input_blocks;i++)
            if(p.input_table[i]!=(uint32_t)(uintptr_t)(p.input_data+(p.input_blocks-1u-i)*QD_N1X1_BLOCK_BYTES)) {
                printf("FAIL input pointer table sequence=%u\n",rec.sequence);return 9;
            }
        for(uint32_t i=0;i<p.output_blocks;i++)
            if(p.output_table[i]!=(uint32_t)(uintptr_t)(p.output_data+(p.output_blocks-1u-i)*QD_N1X1_BLOCK_BYTES)) {
                printf("FAIL output pointer table sequence=%u\n",rec.sequence);return 10;
            }
        for(uint32_t i=p.input_blocks;i<QD_N1X1_TABLE_ENTRIES;i++)
            if(p.input_table[i]!=(uint32_t)(uintptr_t)p.input_guard_block) {
                printf("FAIL input table guard sequence=%u\n",rec.sequence);return 11;
            }
        for(uint32_t i=p.output_blocks;i<QD_N1X1_TABLE_ENTRIES;i++)
            if(p.output_table[i]!=(uint32_t)(uintptr_t)p.guard_block) {
                printf("FAIL output table guard sequence=%u\n",rec.sequence);return 12;
            }
        passed++;free(xb);free(yb);
    }
    if(fgetc(f)!=EOF || passed!=h[1]){puts("FAIL fixture trailing data/count");return 10;}
    fclose(f);printf("PASS C adapter preflight+guard/scatter for %u actual primary QDRX commands; max workspace %u B\n",passed,0x56800u);
    return 0;
}
