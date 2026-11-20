---
name: frame-dsp-state
description: Read-only check of the Frame CDSP before and after any DSP run (hexecutor_qnn16_run, nirhands_replay, depth worker, broker) - control state, callback, loader/payload/arena regions, eye client, cdsp_mem module, dsp_service journal - and what to do when it is not clean. Use before/after every DSP session, when a run hangs, crashes or times out, or when the user asks "is the DSP OK".
---

# Frame DSP state check (read-only)

Source of truth: `docs/hexecutor/operations.md` (Session lifecycle, Failure handling, `(2,0,8)` sections).

## Check

```sh
ssh frame 'python3 ~/nirhands-skel/hexecutor/benchmarks/depth_state.py'
```
`hexecutor/benchmarks/depth_state.py` (tracked, stdlib only, read-only via the `dsp_service` socket) prints JSON:
`control` (3 words), `callback`, `regions.{loader,payload,arena}.zero`, `eye_clients`. Any copy of the tree on the
Frame works (older handoffs used `~/frame-dsp/quest-depth/wave1-20261101/frame/bench/depth_state.py`). Pass a path
argument to also save the report.

Also useful, all read-only:
```sh
ssh frame 'lsmod | grep cdsp_mem; cat /sys/module/cdsp_mem/parameters/size_mb /sys/module/cdsp_mem/parameters/iova'
ssh frame 'journalctl -b | grep dsp_service | tail -40'
ssh frame 'uptime'          # a recent boot means cdsp_mem must be reloaded (user, sudo)
```
Expected for the depth/hands engines: `size_mb` 128, `iova` 2684354560.

## Interpreting

| State | Meaning | Action |
| --- | --- | --- |
| control `[1,0,1]`, callback 0/cleared, regions zero | idle, clean | OK to run |
| `[1,1,1]`, callback set, regions non-zero | a host died without closing its session | stop; tell the user; recovery has been reboot |
| `[2,0,8]` | firmware wake stuck (long wake / XRService FeatureNN contention) | stop; DSP work is blocked until reboot; new sessions fail after 5 s with `CDSP firmware is stuck at control (2,0,8)` |
| no `/dev/cdsp_mem` | rebooted | user runs `sudo insmod` (or `depth-runtime/scripts/setup_frame.sh --qnn`, which uses sudo; ask first) |

The journal line `Timeout reached while waiting for the CPU to read the results` appears in clean sessions too; it
is not the fault. Each wake logs `Received CDSP FLOPS benchmark function`; a stuck DSP logs no further wakes.

## Rules

- **Never read QDSP6SS registers through `/dev/mem`**: it resets the SoC. Use `depth_state.py`.
- Don't reboot, insmod, edit XRService settings or kill services without the user's go-ahead.
- If a capture harness marked the boot as an NPU-cleanup fault, don't clear the marker to force another run.
- Long holds are risky unless XRService controller FeatureNN is off the CDSP
  (`controllerTrackingOptions.configGlobal` `FeatureNN.enableForLogic=false` + `GripNet.enable=true`); keep sessions
  short regardless.
- Record the before/after state in the topic guide or private run notes for every DSP run ("control `(1,0,1)` afterwards").
