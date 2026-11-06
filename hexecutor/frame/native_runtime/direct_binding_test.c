#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "qd_native_direct.h"

static uint8_t vtcm[QD_DIRECT_VTCM_BYTES] __attribute__((aligned(128)));
static uint32_t tables[512] __attribute__((aligned(128)));
static uint32_t abi[30] __attribute__((aligned(16)));
/* Extra backing makes accidental result writes beyond the seven-word array
 * visible without corrupting adjacent test globals. */
static uint32_t pool_bytes[16];
void qd_set_hmx_params_stride1(uint32_t*c,uint32_t a,uint32_t b,uint32_t d,
                               uint32_t e,uint32_t f,uint32_t g,uint32_t h){
    c[0]=a;c[1]=b;c[2]=d;c[3]=e;c[4]=f;c[5]=g;c[6]=h;c[12]=8;
}
static int fail(const char*s){printf("FAIL %s\n",s);return 1;}
static uint32_t p32(const void*p){return (uint32_t)(uintptr_t)p;}

static int test_1x1(void){
    uint32_t xb[1]={0x1000u|1u},yb[1]={0x2000u|1u};
    pool_bytes[1]=QD_DIRECT_ACTIVE_BYTES;
    QdN1x1Request r;memset(&r,0,sizeof r);r.pool_count=7;r.pool[1]=vtcm;r.pool_bytes=pool_bytes;
    r.input=(QdN1x1Blocked){.dims={1,8,8,32},.max_dims={1,8,8,32},.block_count=1,.blocks=xb};
    r.output=(QdN1x1Blocked){.dims={1,8,8,32},.max_dims={1,8,8,32},.block_count=1,.blocks=yb};
    r.weights=vtcm+0x4000;r.weights_bytes=1024;r.weight_ci=32;r.weight_co=32;
    r.weight_kernel_h=1;r.weight_kernel_w=1;r.input_dtype=1;r.output_dtype=1;r.weight_dtype=2;r.fi_dtype=5;
    r.weight_format=QD_N1X1_WEIGHT_XPB;r.fi=(const int32_t*)(vtcm+0x5000);r.fi_bytes=256;
    memset(tables,0x39,sizeof tables);memset(abi,0x47,sizeof abi);memset(vtcm+QD_DIRECT_TAIL_OFFSET,0x61,QD_DIRECT_TAIL_BYTES);
    uint32_t tcopy[512],acopy[30];memcpy(tcopy,tables,sizeof tables);memcpy(acopy,abi,sizeof abi);
    uint8_t tailcopy[0x2000];memcpy(tailcopy,vtcm+QD_DIRECT_TAIL_OFFSET,sizeof tailcopy);
    QdDirectBinding result;memset(&result,0x73,sizeof result);QdDirectBinding before=result;
    int e=qd_direct_validate_1x1(&r,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,sizeof tables,abi,sizeof abi,
                                vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,&result);
    if(e||memcmp(tables,tcopy,sizeof tables)||memcmp(abi,acopy,sizeof abi)||
       memcmp(tailcopy,vtcm+QD_DIRECT_TAIL_OFFSET,sizeof tailcopy)||memcmp(&result,&before,sizeof result))
        return fail("1x1 validation must be side-effect free");
    e=qd_direct_bind_1x1(&r,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,sizeof tables,abi,sizeof abi,
                         vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,&result);
    if(e||tables[0]!=p32(vtcm+0x1000)||tables[QD_N1X1_TABLE_ENTRIES]!=p32(vtcm+0x2000)||
       result.weights!=r.weights||result.fi!=(void*)r.fi||result.in_pages!=1||result.out_pages!=1)
        return fail("1x1 direct page pointers/leaf args");
    if(result.done==0||*result.done!=0||((uint32_t*)result.zero_fi)[0]!=0x404000u||
       ((uint32_t*)result.zero_fi)[32]!=0)return fail("1x1 reserved-tail initialization");
    {
        uint8_t tail_before[0x2000];uint32_t pool_before[16],t_before[512],a_before[30];
        QdDirectBinding result_before;
        memcpy(tail_before,vtcm+QD_DIRECT_TAIL_OFFSET,sizeof tail_before);
        memcpy(pool_before,pool_bytes,sizeof pool_before);memcpy(t_before,tables,sizeof t_before);
        memcpy(a_before,abi,sizeof a_before);memset(&result,0x6b,sizeof result);result_before=result;
        e=qd_direct_bind_1x1(&r,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,sizeof tables,abi,sizeof abi,
            vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES+128u,&result);
        if(e!=QD_DIRECT_BAD_ARGUMENT||memcmp(tail_before,vtcm+QD_DIRECT_TAIL_OFFSET,sizeof tail_before)||
           memcmp(pool_before,pool_bytes,sizeof pool_before)||memcmp(t_before,tables,sizeof t_before)||
           memcmp(a_before,abi,sizeof a_before)||memcmp(&result_before,&result,sizeof result))
            return fail("1x1 excess tail capacity must reject without writes");
        e=qd_direct_bind_1x1(&r,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,sizeof tables,abi,sizeof abi,
            vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,(QdDirectBinding*)pool_bytes);
        if(e!=QD_DIRECT_ALIAS||memcmp(tail_before,vtcm+QD_DIRECT_TAIL_OFFSET,sizeof tail_before)||
           memcmp(pool_before,pool_bytes,sizeof pool_before)||memcmp(t_before,tables,sizeof t_before)||
           memcmp(a_before,abi,sizeof a_before))
            return fail("1x1 result storage must not alias pool_bytes or write on rejection");
    }
    uint32_t old=yb[0];yb[0]=xb[0];memset(tables,0x39,sizeof tables);memset(abi,0x47,sizeof abi);
    e=qd_direct_validate_1x1(&r,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,sizeof tables,abi,sizeof abi,
                            vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,0);
    yb[0]=old;if(e!=QD_DIRECT_ALIAS||((uint8_t*)tables)[0]!=0x39||((uint8_t*)abi)[0]!=0x47)
        return fail("1x1 aliased pages must reject before writes");
    xb[0]=QD_DIRECT_ACTIVE_BYTES|1u;
    e=qd_direct_validate_1x1(&r,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,sizeof tables,abi,sizeof abi,
                            vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,0);
    if(e!=QD_DIRECT_RANGE)return fail("1x1 reserved-tail page must reject");
    xb[0]=0x1000u|1u;
    uint32_t alt_tables[512] __attribute__((aligned(128)));
    uint32_t alt_abi[30] __attribute__((aligned(16)));
    e=qd_direct_validate_1x1(&r,vtcm+128,QD_DIRECT_ACTIVE_BYTES,alt_tables,sizeof alt_tables,alt_abi,sizeof alt_abi,
                            vtcm+128+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,0);
    if(e!=QD_DIRECT_RANGE)return fail("pool pointer must equal leased VTCM base");
    uint32_t savedw=r.input.dims[2];r.input.dims[2]--;
    e=qd_direct_validate_1x1(&r,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,sizeof tables,abi,sizeof abi,
                            vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,0);r.input.dims[2]=savedw;
    if(e!=QD_DIRECT_UNSUPPORTED)return fail("partial physical tile must reject");
    uint32_t savedpool=xb[0];xb[0]=0x1000u;
    e=qd_direct_validate_1x1(&r,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,sizeof tables,abi,sizeof abi,
                            vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,0);xb[0]=savedpool;
    if(e!=QD_DIRECT_RANGE)return fail("non-pool1 block ref must reject");
    return 0;
}

static int test_3x3(void){
    uint32_t xb[9],yb[1]={0x20000u|1u};pool_bytes[1]=QD_DIRECT_ACTIVE_BYTES;
    for(uint32_t i=0;i<9;i++)xb[i]=0x10000u+i*QD_DIRECT_PAGE_BYTES|1u;
    QdN3x3Request r;memset(&r,0,sizeof r);r.pool_count=7;r.pool[1]=vtcm;r.pool_bytes=pool_bytes;
    r.input=(QdN3x3Blocked){.dims={1,10,10,32},.max_dims={1,24,24,32},.padding={0,7,7,0},.block_count=9,.blocks=xb};
    r.output=(QdN3x3Blocked){.dims={1,8,8,32},.max_dims={1,8,8,32},.block_count=1,.blocks=yb};
    r.weights=vtcm+0x30000;r.weights_bytes=9u*32u*32u;r.weight_ci=32;r.weight_co=32;
    r.weight_kernel_h=3;r.weight_kernel_w=3;r.input_dtype=1;r.output_dtype=1;r.weight_dtype=2;r.fi_dtype=5;
    r.weight_format=QD_N3X3_WEIGHT_XWB;r.fi=(const int32_t*)(vtcm+0x33000);r.fi_bytes=256;
    memset(tables,0x29,sizeof tables);memset(abi,0x57,sizeof abi);
    uint32_t tcopy[512],acopy[30];memcpy(tcopy,tables,sizeof tables);memcpy(acopy,abi,sizeof abi);
    int e=qd_direct_validate_3x3(&r,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,sizeof tables,abi,sizeof abi,
                                vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,0);
    if(e||memcmp(tables,tcopy,sizeof tables)||memcmp(abi,acopy,sizeof abi))return fail("3x3 validation side effects");
    QdDirectBinding b;e=qd_direct_bind_3x3(&r,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,sizeof tables,abi,sizeof abi,
                                          vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,&b);
    if(e||b.in_pages!=9||b.out_pages!=1||tables[0]!=p32(vtcm+0x10000)||
       tables[QD_N3X3_TABLE_ENTRIES]!=p32(vtcm+0x20000)||b.weights!=r.weights||b.fi!=(void*)r.fi)
        return fail("3x3 direct page pointers/leaf args");
    {
        uint8_t tail_before[0x2000];uint32_t pool_before[16],t_before[512],a_before[30];
        QdDirectBinding result;memset(&result,0x6b,sizeof result);QdDirectBinding result_before=result;
        memcpy(tail_before,vtcm+QD_DIRECT_TAIL_OFFSET,sizeof tail_before);
        memcpy(pool_before,pool_bytes,sizeof pool_before);memcpy(t_before,tables,sizeof t_before);
        memcpy(a_before,abi,sizeof a_before);
        e=qd_direct_bind_3x3(&r,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,sizeof tables,abi,sizeof abi,
            vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES+128u,&result);
        if(e!=QD_DIRECT_BAD_ARGUMENT||memcmp(tail_before,vtcm+QD_DIRECT_TAIL_OFFSET,sizeof tail_before)||
           memcmp(pool_before,pool_bytes,sizeof pool_before)||memcmp(t_before,tables,sizeof t_before)||
           memcmp(a_before,abi,sizeof a_before)||memcmp(&result_before,&result,sizeof result))
            return fail("3x3 excess tail capacity must reject without writes");
        e=qd_direct_bind_3x3(&r,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,sizeof tables,abi,sizeof abi,
            vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,(QdDirectBinding*)pool_bytes);
        if(e!=QD_DIRECT_ALIAS||memcmp(tail_before,vtcm+QD_DIRECT_TAIL_OFFSET,sizeof tail_before)||
           memcmp(pool_before,pool_bytes,sizeof pool_before)||memcmp(t_before,tables,sizeof t_before)||
           memcmp(a_before,abi,sizeof a_before))
            return fail("3x3 result storage must not alias pool_bytes or write on rejection");
    }
    return 0;
}

int main(void){if(test_1x1()||test_3x3())return 1;puts("PASS direct binding validate/bind, active-range/alias checks, and direct pointers");return 0;}
