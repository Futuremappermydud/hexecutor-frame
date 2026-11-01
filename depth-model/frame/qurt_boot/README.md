# Booting stock QuRT (cdsp.mbn) on the Steam Frame via /dev/cdsp — 2026-10-31 notes

Status: NOT booting yet. QuRT now runs ~800 basic blocks of its cold boot without the
instant SoC reset, but later resets remain and probes became non-repeatable.

## Findings
1. **Instant reset #1 = the driver's own BOOT ioctl.** `hexagon-baremetal.c` ends BOOT with
   `readl(QDSP6SS_BOOT_STATUS)` from the ARM CPU right after BOOT_CMD. If QuRT's first
   instructions make QDSP6SS unreadable at that moment, the CPU hangs -> SoC reset.
   Fix: patch the ELF entry vector (`fe000000`) to `jump` into a delay loop in the padding
   at `fe000e10` (`entry_delay.S`, ~0.5 s at 0x40000 iterations; the DSP runs only ~0.4 M packets/s),
   which then jumps back to the original reset path `fe00001c`.
2. **Reset #2 = QuRT's BSS memset.** `fe194e6c..fe194e78` loops `dczeroa` over fe01b9c0..fe17dec0
   (1.4 MB). NOP the `dczeroa` word (0xa0c0c000 at fe194e74): the dmabuf is already zero.
3. Closing /dev/cdsp (or killing the process) after QuRT ran resets the SoC (shutdown touches QDSP6SS).
4. QEMU (`qemu-system-hexagon -M virt -cpu v73`) runs the same cold path; it needs `ciad`
   (0x6400c060|rs<<16) and DMA insns implemented. `tb_first.txt` is its ordered list of block starts.
5. Observability trick: hook any block start with `jump` to `dump_stub.S` (placed at fe000e40) that stores
   regs to VA fe01a000 (MMU on) / PA 0x9dd00000 (MMU off) and spins; read via the fsynced watch in
   `qurt_boot_test.py` (needs DMA_BUF_IOCTL_SYNC). `bis4.py` bisects with it (reset vs reached by boot_id).
6. Probes past ~index 790 (around the cfg lookups at fe17f6ac/b8) are NOT repeatable (793/794 flip).

## Caveats / state
- Needs dsp_service + steamvr.service masked and a reboot; each reset reboots the headset (~70 s).
- Files on the Frame: ~/frame-dsp/{p_base.mbn,p_delay2.mbn,qurt_boot_test.py,steps.log}.

## Update (later 2026-10-31)
- Probes below ~index 750 are deterministic and monotone (bis5). From ~index 753 on (`ciad` at fe17f8c4
  executes for real on hardware) results are chaotic: with the DSP parked in a spin, interrupts are still
  enabled, so QuRT's vector/handler code runs at random times. The reset now follows QuRT start by <0.25 s
  (fsynced ticks stop at 0.50 s; the entry delay ends at ~0.5-0.7 s) even when parked at index 793 in
  hold mode (no close). So the next suspect is an *interrupt/timer path* (QTimer/L2VIC/IPCC), not a single
  straight-line instruction.
- Next: make the hook cave clear SSR.IE (or mask all L2VIC/imask) before spinning so probes after 753
  become deterministic; then bisect again. A driver-side module is only needed to avoid the reset at close.
- Note: `dsp_service` must really be masked: hold4.py runs are vacuous if it is active
  (loader prints "dsp_service is running").
