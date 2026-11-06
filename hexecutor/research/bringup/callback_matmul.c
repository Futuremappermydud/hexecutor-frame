#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
typedef unsigned u32;

/* C[32,32] = A[32,32] * B[32,32], exact signed int32 output.
 * Values in A are small signed int16, packed for HVX's word*halfword MAC.
 * Each hardware worker owns every third output row.
 */
void callback(void) {
    u32 tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    const int *a = (const int *)0x87608000;
    const HVX_Vector *b = (const HVX_Vector *)0x87609000;
    HVX_Vector *c = (HVX_Vector *)0x8760a000;
    volatile u32 *status = (volatile u32 *)0xf0b12100;
    for (int i=0;i<8192;i+=32)
        __asm__ volatile("dcinva(%0)" :: "r"((u32)a+i) : "memory");
    for (int row=tid;row<32;row+=3) {
        HVX_Vector acc = Q6_V_vzero();
        for (int k=0;k<32;k++) {
            u32 packed = (u32)(unsigned short)a[row*32+k] * 0x10001u;
            acc = Q6_Vw_vmpyiacc_VwVwRh(acc,b[k],packed);
        }
        c[row] = acc;
        for (int i=0;i<128;i+=32)
            __asm__ volatile("dccleana(%0)" :: "r"((u32)&c[row]+i) : "memory");
    }
    __asm__ volatile("barrier" ::: "memory");
    status[tid*4] = 0xc0de0000 | tid;
    status[tid*4+1] = 63+tid;
    status[tid*4+3] = 32;
    __asm__ volatile("barrier" ::: "memory");
    status[tid*4+2] = 1;
}
