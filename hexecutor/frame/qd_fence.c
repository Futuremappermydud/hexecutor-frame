/* ARM-side ordering for the persistent DSP protocol: the arena mapping is
 * not ordered against the DSP without explicit barriers. */
void qd_fence(void) {
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}
