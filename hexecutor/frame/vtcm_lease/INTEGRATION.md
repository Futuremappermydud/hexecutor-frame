# Full VTCM lease integration contract

`qd_vtcm_lease.h` is a no-libc helper for a single exclusive callback session.
The offline C harness can be run with:

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror -o /tmp/test_vtcm_lease frame/vtcm_lease/test_vtcm_lease.c
/tmp/test_vtcm_lease
```

## Bootstrap ordering

1. Keep the existing FrameTransport lock and paused eye client for the entire
   borrow. Do not call `stop_persistent` or release the callback before restore.
2. In bootstrap thread 0, call `qdo_map`, then initialize a `QdVtcmLease` in
   private cdsp_mem metadata and call `qdvt_save(lease, (volatile uint8_t *)
   QDVT_BASE, QDVT_BASE, QDVT_BYTES, QDVT_BACKUP_BASE,
   (uint8_t *)QDVT_BACKUP_BASE)`. This must finish before publishing the
   barrier that releases threads 1 and 2 into the graph engine.
3. Abort graph execution if save fails. Do not mark the lease saved, release
   the callback, or resume eye tracking on an error.
4. After every graph worker has returned and the engine has quiesced HMX and
   DMA, thread 0 calls `qdvt_restore_verify`. Only after it succeeds may the
   normal owned DDR mapping be removed and callback workers exit.
5. The host must require both a zero bootstrap error and the lease's verified
   phase before allowing `FrameTransport` to close. A failed verification must
   raise from the `collect` callback passed to `FrameTransport.invoke` or
   `submit`, before those methods clear `job_active`; `_close` then sees an
   active job and keeps the callback/eye paused. Setting `_stuck=True` alone
   after a completed call is insufficient because `_close` starts with
   `safe = not job_active`. The backup must remain intact for recovery; never
   clear it on either path. Any timeout or failed restore likewise must not
   release the callback or restart eye tracking.

The currently proven architectural mappings are TLB 8 and 9 with values
`0xc0031000f7062020` and `0xc0031400f7062820`. The helper rechecks these before
and after each transition. The 8 MiB snapshot belongs at `0xa6000000` in the
private 128 MiB allocation; place guards immediately before and after it and
confirm them on the host. On Hexagon, the helper requires actual VTCM and
backup pointers to equal `0x31000000` and `0xa6000000`. The offline mock build
accepts ordinary aligned, nonoverlapping host buffers while retaining fixed
architectural range arguments. This follows the current resident layout: fixed graph data below 32 MiB,
heap 32–64 MiB, scratch 64–80 MiB, staging 80–88 MiB, and otherwise free space
above 88 MiB. Do not move the backup into graph heap/scratch/staging.

## Graph integration constraints

Current config validation describes every pool as a range inside the first
32 MiB of DDR. Pool 1 is presently allocated there by `resident_dsp.py`; simply
changing that host offset to `0x31000000` is invalid. A native VTCM pool needs
an explicit DSP-side pool binding or pointer translation, and graph tensor
lifetimes must be checked against every VTCM consumer.

The current graph/HMX path uses the low `0x2d0000` bytes for HMX packet,
activation, accumulator, quantization, and fallback scratch; the upper
`0x530000` bytes were the portion probed. It also has small 2 KiB tile copies
and HMX packet aliases. An 8 MiB pool 1 occupies the entire VTCM, so it cannot
coexist with these staging and scratch uses. The native graph path must remove
the old HMX staging users (including fallback paths) while a leased native
pool is active, or introduce a bounded allocator whose tensor lifetimes never
overlap required scratch. The requested full-size native pool therefore
requires graph tiling/lifetime changes; it is not safe to enable the existing
graph and staging modes together.

The snapshot uses an XOR reduction plus a word sum weighted by each 128-byte
vector's index. This makes the checksum position-dependent, so a reordered
backup chunk is rejected before any restore write. Restore then performs a
full vector XOR/OR comparison against the intact snapshot. The offline harness
includes a two-chunk swap regression check.

The lease only proves bounded borrowing between firmware clients. The live
baseline changes between eye sessions and differs from the firmware image;
always save and restore the current bytes. This is not permanent ownership and
does not permit concurrent eye use.
