/* Run with ASan/UBSan against generated QDRX images. Mutations have a fresh
 * checksum so they exercise structural bounds rather than only CRC rejection.
 * This is a host-only fuzz driver, never a DSP payload. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../native/qd_resident.h"

static uint32_t rng=0x1234abcd;
static uint32_t random32(void) {rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void put(uint8_t *p,uint32_t v) {for(uint32_t i=0;i<4;i++)p[i]=(uint8_t)(v>>(i*8));}
static void checksum(uint8_t *p,uint32_t bytes) {
    uint32_t crc=0xffffffff;
    put(p+16,0);
    for(uint32_t i=0;i<bytes;i++) {
        crc^=p[i];
        for(uint32_t j=0;j<8;j++)crc=(crc>>1)^(0xedb88320u & (0u-(crc&1)));
    }
    put(p+16,~crc);
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    FILE *f=fopen(argv[1],"rb");if(!f)return 2;
    if(fseek(f,0,SEEK_END)){fclose(f);return 2;}
    long length=ftell(f);
    if(length<QDR_HEADER_BYTES || length>QDR_MAX_BYTES){fclose(f);return 2;}
    uint32_t bytes=(uint32_t)length;
    uint8_t *original=malloc(bytes),*allocated=malloc(bytes+1);
    if(!original || !allocated){fclose(f);free(original);free(allocated);return 2;}
    rewind(f);
    if(fread(original,1,bytes,f)!=bytes){fclose(f);free(original);free(allocated);return 2;}
    fclose(f);
    uint8_t *data=allocated+1; /* Deliberately unaligned. */
    QdrImage view;QdrStatus status;
    if(qdr_open(original,bytes,&view,&status)){free(original);free(allocated);return 3;}
    for(uint32_t iteration=0;iteration<512;iteration++) {
        memcpy(data,original,bytes);
        uint32_t table=random32()%6,off=qdr_u32(original+148+table*8);
        uint32_t where=iteration<64?iteration*4:iteration<256?off+(random32()%16)*4:random32()%(bytes-3);
        if(where>bytes-4)where=bytes-4;
        uint32_t value=random32();
        if(iteration%5==0)value=0xffffffff;
        if(iteration%7==0)value=0;
        put(data+where,value);checksum(data,bytes);
        uint32_t size=iteration%13==0?random32()%bytes:bytes;
        uint32_t code=qdr_open(data,size,&view,&status);
        if(code>QDR_KERNEL_ERROR || code!=status.code || (code && (view.data || view.bytes)))abort();
        code=qdr_dispatch(data,size,NULL,0,65536,NULL,NULL,&status);
        if(code>QDR_KERNEL_ERROR || code!=status.code)abort();
    }
    free(original);free(allocated);
    puts("512 checksummed mutations and dispatch attempts passed");
    return 0;
}
