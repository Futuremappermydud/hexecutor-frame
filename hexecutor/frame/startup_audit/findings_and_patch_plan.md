# Retained DSP startup race audit

Offline source review only. No Frame, DSP, device, or retained-memory access was performed. The observed state supplied by root is: callback/control active (`CTRL=[2,0,8]`), QDS command header all zero (sequence 0), QDO map active, lease phase 1/error 0 with a host-verified backup fingerprint, and no host job flag. The cause of the dispatch is unknown. This note does not attribute it to a particular service or opcode.

## Source path and concrete failure

1. `FrameTransport._open()` writes callback/payload and sets `WORK+24=CODE_BASE` before returning (`frame/frame_transport.py:233-237`). `ResidentDSP.__enter__()` then still has to write input/pool/heap guards, clear the arena, and write `QDS_CONFIG` (`frame/resident_dsp.py:75-84`). The callback is therefore visible during Python setup.
2. The preserving bootstrap's master path only checks the small config header, then calls `qdo_map()` and saves VTCM before dispatching the resident engine (`frame/qd_graph_boot.c:50-78`). It does not validate the QDS RUN command header or sequence before either operation.
3. The payload thread barrier allows an all-zero sequence to pass immediately: the master initializes `sync[0]=0`, and followers wait while `sync[0] != a[2]`. If both are zero, followers enter the engine path without waiting for successful initialization (`qd_graph_boot.c:51-78`).
4. Cleanup can mistake an unowned active callback for a safe idle session. `FrameTransport._close()` starts with `safe = not self.job_active and not self._retain_reason` (`frame/frame_transport.py:434`). It only tries to quiesce when `job_active` is true. It can skip restoring code because CTRL is not 1 and still reach the final eye-restart branch because `safe` remains true (`:456-535`).
5. `ResidentDSP.close()` normally catches a still-active QDO map at arena offset 3072 and calls `retain_session()` (`resident_dsp.py:162-190`), but this protection depends on that state still being visible and the ResidentDSP wrapper reaching `close()`. FrameTransport itself needs a conservative guard for the unowned-active case.

## Required invariants

* Before `qdo_map()` or `qdvt_save()`, the preserving bootstrap has validated a complete QDS RUN header: expected command magic, graph RUN op, protocol version, permitted flags, nonempty command interval, and a nonzero sequence that is unchanged across a barrier-protected snapshot. It must also validate `QdsConfig` with `qds_config_valid()` before mapping.
* Followers wait for a nonzero initialization publication, never for equality with a potentially zero command sequence. Invalid startup publishes an error and releases the barrier without running the resident engine or waiting for followers that may not have been dispatched.
* A host context with no outstanding job may restore pages or restart the eye only after observing the transport's verified idle tuple `(CTRL, CTRL+4, CTRL+8) == (1,0,1)`. If control is active while the host has no job flag, retain callback/backup state and keep the eye stopped. Do not infer the competing dispatch source from control value 8.
* Register the callback only after a command has been prepared. Loading callback bytes is distinct from publishing its pointer. This closes the long setup window in `ResidentDSP.__enter__()`.

## Patch sketch

This is a proposal only; shared production files are unchanged.

### `qd_graph_boot.c`

At the top of `payload()`, take a stable snapshot of the command and config before any QDO mapping or VTCM lease work:

```c
static uint32_t qds_start_header_valid(volatile uint32_t *a,
                                       const QdsConfig *cfg,
                                       uint32_t *sequence) {
    uint32_t s0=a[2];
    uint32_t magic=a[0], version=a[1], op=a[3];
    uint32_t first=a[4], last=a[5], flags=a[6];
    __asm__ volatile("barrier" ::: "memory");
    uint32_t s1=a[2];
    if (!s0 || s0!=s1 || magic!=0x32564451u || version!=6u || op!=3u)
        return QDS_BAD_START;
    if (first>last || (flags&~1u) || !qds_config_valid(cfg))
        return QDS_BAD_START;
    *sequence=s0;
    return 0;
}
```

The master should publish a nonzero init result only after validation/map/save has completed:

```c
if (tid==0) {
    sync[0]=0; sync[1]=sync[2]=sync[3]=0; sync[4]=0;
    uint32_t start_seq=0;
    uint32_t err=qds_start_header_valid(a,cfg,&start_seq);
    if (!err) err=qdo_map(map);
    if (!err) err=save_vtcm_and_publish_lease(...);
    sync[4]=err;
    __asm__ volatile("barrier" ::: "memory");
    sync[0]=err ? 0xffffffffu : start_seq; /* always nonzero after init */
} else {
    while (sync[0]==0) { /* bounded wait; zero sequence cannot pass */ }
}
```

After this barrier, each worker checks `sync[4]`. On rejection, it writes its own exited status and returns without entering the QDO engine. The master must not spin for both followers on this rejection path. `ResidentDSP.prepare()` should clear the init area (`arena[3584:3604]`) before the first callback; callback registration is deferred until after prepare, so the follower wait starts from zero. Keep the existing restore/unmap path only for successful initialization. Define `QDS_BAD_START` as a local nonzero bootstrap error value that cannot be mistaken for a `QDG_ERR_*` success code.

**Stronger option: prove all three worker entries before mapping or saving.** Have the host clear the five init words before callback registration. Each valid worker entry publishes its own `sync[tid+1]=start_seq` and waits for `sync[0] != 0`. Tid 0 first validates the stable header/config, then waits a short bounded interval for both follower arrivals to equal that same sequence. It calls `qdo_map()` and the VTCM save only after all three arrivals; then publishes `sync[4]=err`, a barrier, and `sync[0]=start_seq` (or a nonzero rejection sentinel on any error). If only one foreign thread was invoked despite a syntactically valid header, it times out and rejects without mapping or saving. Do not zero the arrival slots inside tid 0: a follower may have arrived first, and that would erase its proof. Invalid sequence zero is rejected immediately and the master publishes a nonzero rejection sentinel so any followers can leave their wait. Followers that time out also return an error status without touching QDO/VTCM. A short startup bound is appropriate because the host only registers immediately before dispatch; choose and verify a cycle bound for the target clock during normal integration.

The first RUN header is built by `ResidentDSP.prepare()` as `[0x32564451, version=6, sequence-1, op=3, first, last, direct]`, then the sequence at word 2 is published last. For the first request the placeholder is zero, so the new nonzero check rejects exactly the observed sequence-zero startup. If the header writer changes, update the guard and this contract together. A follow-up hardening can set word 2 to zero for every prepare before publishing the new sequence, but it must be checked against the resident loop's sequence semantics and all callback clients first. For a three-worker arrival proof, clear the init words in the first `ResidentDSP.prepare()` before callback registration; the current one-time arena clear in `ResidentDSP.__enter__()` is not sufficient evidence if any callback can arrive during setup.

### `frame_transport.py`

Add separate `code_loaded` and `installed`/`registered` state. `_open()` saves the old work/page bytes and writes the callback and payload pages, but does **not** publish `WORK+24`. Add `_register_callback()` that verifies the pointer is still zero, publishes `CODE_BASE`, and marks registered. In `submit()` and `invoke()`, call it after `prepare(sequence)` and before the first dispatch-control write/socket send. Later persistent jobs keep the existing registration.

On close, before computing `safe`, inspect CTRL only when no host job is active. If the tuple is not `(1,0,1)`, call `retain_session()` with a reason identifying “active control without a host job” and do not clear callback/arena, restore code/payload, or restart the eye. Compute `safe` as false when `_stuck` or any retain reason exists. Restore loaded pages even when the callback was never registered, but only when CTRL is verified idle and the work pointer is either zero or the pointer this context published. If an unexpected pointer/control state is present, retain.

This preserves direct `FrameTransport` callers: both `submit()` and `invoke()` must register lazily, not only ResidentDSP. The expected idle tuple is already used by `_open()` (`frame_transport.py:174-175`, `:198-200`); the proposal does not assign meaning to any non-idle opcode.

## Offline guard model

`test_startup_guard.py` exercises the command-header predicate with valid and malformed snapshots: zero sequence, changed sequence, wrong magic/op/version, empty/reversed interval, and unsupported flags. It is a host model only; it does not validate Hexagon barriers, worker launch behavior, mapping APIs, or transport control registers.

## Validation required after root integration

* Unit-test the pure header predicate, plus target compilation of the preserving bootstrap.
* Static/source check that `qdo_map` and `qdvt_save` are dominated by successful config and command validation.
* Offline mock of the transport close path: `job_active=False` with non-idle CTRL must retain and never call the eye restart path; idle CTRL with callback unregistered should restore the saved pages normally.
* Hardware validation is intentionally not proposed here. Root owns retained-state recovery and any device experiment after explicit reboot approval.
