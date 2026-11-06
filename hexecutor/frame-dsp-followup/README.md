# Frame CDSP follow-up experiments

**Later Oct 25 follow-up:** standalone FP16 HMX matrix multiplication now
works through this callback path. See
[HMX bring-up and validation](../depth-model/frame/HMX.md). The experiments
below record the earlier scalar/HVX work.

Tested on Oct 25, 2026 against Valve CDSP firmware SHA-256
`4f9fd567eb582c24f1f960a6525fafb0bd75733223ffa7ce8d6541cfd2a62b1c`.

The working copy and compiled ELFs are already on `frame` in
`/home/steamos/frame-dsp/followup/`. Build and run there:

```sh
make all
python3 live_cdsp_probe.py --callback callback.elf
python3 live_cdsp_probe.py --callback callback_hvx.elf
python3 live_cdsp_probe.py --callback callback_vector.elf --vector
python3 live_cdsp_probe.py --callback callback_matmul.elf --matmul
```

These use the existing daemon's DMA-buffer transport and firmware callback;
they do not load replacement firmware. They briefly terminate the eye-tracking
client because the stock benchmark reuses gaze buffers, then restart CDSP eye
tracking in `finally`. Keep `dsp_service` running. Run one probe at a time.

The matrix test verifies all 1,024 signed integer outputs against a scalar Python
reference. The roughly 4 ms reported completion includes the stock FLOPS benchmark
used to wake the workers; it is not a standalone matrix-kernel benchmark.

`hexdis.py` disassembles Hexagon ELFs through the Frame's installed libLLVM.
`test_eye_backends.py` checks GPU/CPU/CPUEV2 using shipped weights and restores CDSP.
CPU initialization failures are in Steam's `eyetracking.txt`, despite exit code 0.

Each different callback uses a fresh executable page. If you change a payload,
choose a new verified-empty page and update the linker address, or implement DSP
instruction-cache invalidation. Do not assume future firmware shares this layout.
This is a firmware-specific proof of scalar/HVX computation, not a QNN runtime,
HMX demonstration, or tested concurrent custom-inference/gaze scheduler.

See the parent investigation document for protocol offsets and remaining options.

## Full stereo inference backend

`../depth-model/frame/` now builds on this callback ABI to run the complete
recovered stereo graph. All convolution dot products run on HVX; the ARM CPU
handles graph operations and exact requantization. It packages both models
with a freestanding callback ELF and a Python/NumPy runner. See its README for
validation reports, run commands and the long-session eye-client pause/restore
procedure. It still does not load QNN or use HMX.
