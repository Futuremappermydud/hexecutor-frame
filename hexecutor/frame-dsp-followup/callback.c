/* A freestanding callback for Valve's existing CDSP worker dispatcher.
 * Called by hardware threads 0..2 after their built-in FLOPS benchmark.
 * Only writes the dedicated 0xb12100 scratch region through its uncached alias.
 */
typedef unsigned u32;
void callback(void) {
    u32 tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    volatile u32 *out = (volatile u32 *)0xf0b12100;
    out[tid * 4 + 0] = 0xc0de0000 | tid;
    out[tid * 4 + 1] = 7 * 9 + tid;
    __asm__ volatile("barrier" ::: "memory");
    out[tid * 4 + 2] = 1;
}
