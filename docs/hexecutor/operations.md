# Building and operating Hexecutor

## Toolchains and recipes

The basic Makefile accepts `HEX_CC`, `HEX_LD`, and `HOST_CC`. A host compiler builds scalar substitutes; Hexagon builds need the supported target flags/intrinsics and linker. Some prepared/HMX paths use the local SDK toolchain even when basic callback targets can use an open LLVM build.

From the monorepo root:

```sh
make -C hexecutor/native HEX_CC=clang-14 HEX_LD=ld.lld-14
```

This builds basic callback targets. For the current measured primary recipe:

```sh
sh depth-model/deployment/lanes/build_best.sh NAME
```

That script changes to the monorepo root, uses the prepared engine recipe, and expects separately provisioned program data, measured-cost inputs, and generated lane headers. Its recorded generator options include two worker lanes and `--active block`. Regenerate model-specific headers when the program or schedule policy changes; the last generated header is not a universal schedule.

Record source revision/dirty patch, compiler, flags, generated-header identity, package hashes, engine/bootstrap hashes, and device report together. Files in `build/` are private local artifacts and are not shipped by Git. Do not infer that a renamed ELF is the one a launcher pins.

## Kernel memory module

The depth engine expects `/dev/cdsp_mem`, 128 MB, and the verified IOVA. `runtime/scripts/setup_frame.sh --qnn` checks the configured device package, loads the module when needed, and checks memory chunks. A kernel module must match the running kernel and its ABI; building it against an arbitrary source tree is insufficient.

The language-model client has a different memory-size requirement. Loading its larger allocation is not interchangeable with satisfying the depth engine's exact module-size check.

## Session lifecycle

The host acquires exclusive DSP access, checks service/eye state, verifies memory ownership and package identity, installs loader/payload state, then enters the guarded session. During a run, work publication and completion are explicit. Normal shutdown drains active work, restores borrowed state, clears callback registration, and resumes the eye client only when restoration is safe.

Changing the payload requires instruction-cache maintenance. The fixed loader implements that for its supported range. Replacing a live loader or treating apparently zero firmware memory as a general executable allocator is outside the validated contract.

Keep `dsp_service` alive. One custom inference client runs at a time. Ordinary sessions pause eye tracking; only the broker's Gazenet adapter may keep the eye client running while serving or explicitly yielding its stock requests. A depth worker and the chat client cannot claim the same DSP resources concurrently.

## Failure handling

`hexecutor/benchmarks/depth_state.py` inspects service/worker state. `hexecutor/benchmarks/chunk_map.py` checks owned memory. Preserve the reports when a worker times out or cleanup fails. A zero process exit code, an old JSON file, or restored output buffers alone does not prove the DSP session exited cleanly.

Recorded long firmware wakes have ended at control state `(2,0,8)`: the firmware's request thread never returns, XRService's own DSP work fails, and prior incidents required reboot. Holds up to 2.45 s exited cleanly (about 80 hand-viewer sessions of 2–2.4 s on 2026-11-08 alone), while 17.9 s (2026-11-04) and 46.5 s (hand viewer, 2026-11-08 18:33) stuck. The transport therefore attempts to re-wake every `Artifacts::max_wake_seconds` (1 s). This correlation does not establish an elapsed-time threshold or prove that wake duration alone causes the failure.

`dsp_service`'s log is in the journal (`journalctl -b | grep dsp_service`), including the DSP's own `[cDSP]` lines. Each wake logs `Received CDSP FLOPS benchmark function`. A persistent wake also logs `Timeout reached while waiting for the CPU to read the results` 55–100 ms later; that warning appears in clean sessions too and is not the fault. After a stuck session, later clients connect but log no further wakes.

A session that opens on a stuck DSP now fails with `CDSP firmware is stuck at control (2,0,8)…` after a 5 s wait, without installing anything or pausing eye tracking. If the capture harness marks the current boot as an NPU-cleanup fault, do not clear the marker to force another scan; recover the device and re-establish prerequisites.

### Firmware dispatch finding (2026-11-09)

Read-only disassembly of the pinned `cdsp_sm8650.mbn` (SHA-256 `4f9fd567eb582c24f1f960a6525fafb0bd75733223ffa7ce8d6541cfd2a62b1c`) maps the control words to DSP addresses `0xf0a91000`, `+4`, and `+8`. The request handler writes state 2 at `0x87000438`. It samples the third control word at `0x870005a0`: value 4 calls the FLOPS worker routine at `0x87003850`, while value 8 takes another worker-dispatch path at `0x87000600`. Both paths must pass a heavyweight transaction synchronization at `0x870006a0` (implementation `0x87003040`) before state 3 is written at `0x870006a8`; state 1 follows at `0x87000754`. The value 8 in a later snapshot does not establish which branch was taken when the handler sampled it. No elapsed-time exit was found in this completion path. The 50-million-unit warning loop at `0x87000450` is the separate wait for the CPU to read results; that warning also occurs on clean wakes.

A read-only snapshot on this date found `(2,0,8)` with exit sequence 132, all four worker `EXITED` markers at that sequence and zero worker errors, VTCM lease phase 2, cleared JTLB and clock markers, and the callback still registered. Thus the custom workers had published their exit before the firmware published state 3. The journal shows wakes at 03:27:26.947, 03:27:27.225, 03:27:27.331, and 03:27:28.300 before the stuck state; the final two are 0.969 s apart, consistent with wake rotation, although the exact binary and runtime wake setting for that run have not been established. The snapshot does not identify the exact firmware instruction where it stopped; a DSP PC trace would be required to distinguish the completion-count wait, cache maintenance, and transaction synchronization.

For long host sessions, keep the host memory and package resident but end each firmware wake with the normal `EXIT` packet, wait for the firmware completion state, acknowledge it, verify worker/JTLB/VTCM/clock restoration, and issue a new wake before further commands. The one-second rotation in the transport implements this pattern for idle gaps and between commands. It cannot interrupt a command or host operation while the session mutex is held, so it is not a hard one-second limit. Do not schedule a command whose measured worst-case duration can consume the remaining wake budget. The 2026-11-09 incident means this rotation is not yet a demonstrated reliable solution: identify the deployed binary and capture per-wake exit duration, control transitions, and firmware PC before changing the wake limit or relying on long sessions.

### Re-wake traffic

Each preserving tensor wake currently copies the full 8 MiB VTCM to private DDR on entry. On exit, the DSP reads that backup to fingerprint it, reads it again to restore VTCM, and reads it a third time to compare the restored bytes. The host then reads the 8 MiB backup to verify its fingerprint before another wake can overwrite it. This is approximately 8 MiB of DDR writes and 32 MiB of DDR reads per wake, before model staging. The compacted landmark graph has 3,878,912 bytes of resident packed weights and 665,600 bytes of auxiliary records; palm has 2,074,624 and 496,000 bytes respectively. A fresh wake stages those records from constants again, with a separate cross-worker job for each resident convolution weight copy. These are code-derived traffic counts, not a measured breakdown of the reported 30+ ms re-wake latency.

Two candidates retain the existing firmware restoration contract. First, cache each graph's fully prepared, contiguous resident-record region in private DDR once, then copy it into VTCM with one bulk job after each new lease. This replaces repeated auxiliary conversion and many small worker dispatches; it does not remove the necessary DDR read. Second, fuse the backup fingerprint calculation with the DDR-to-VTCM restore copy, preserving the subsequent exact VTCM comparison and host check; this removes one 8 MiB DDR read. Measure each phase before selecting either change. Keeping model records in VTCM across an acknowledged wake would skip more traffic, but the stock firmware also uses VTCM and this session does not own its contents between wakes.

### XRService controller FeatureNN and continuous wakes

The installed XRService binary contains both `dispatchStereoFeatureNNOnCDSP` and `dispatchStereoFeatureNNOnCPU`, a CPU-model fallback path, and an embedded configuration example setting `stmsCreationOptions.preprocessorConfig.disableCDSPUsage` to `true`. The binary also contains a failure path for the absence of a CPU model compatible with the selected CDSP model. Disabling the whole XRService CDSP path can affect other functions, so controller, HMD, and eye behavior need validation. The separate `featureComputationMethod: CPU` example belongs to the SLAM frontend and does not by itself select the controller FeatureNN backend.

On the initial 2026-11-09 stuck boot, the last logged custom FLOPS wake began at 03:27:28.300; XRService logged `Dispatching Stereo FeatureNN on cDSP timed out` at 03:27:28.919 and subsequent CDSP ownership failures. This showed the stuck session impaired XRService's controller work, without by itself establishing that FeatureNN caused the stuck firmware completion. The controlled comparison below provides stronger evidence for contention. The exact firmware instruction responsible for `(2,0,8)` remains unknown.

#### Controlled CPU-versus-CDSP test (2026-11-09)

The XRService setting was tested on the headset with the deployed 2026-11-06 Hexecutor host, the same palm graph/input, and a temporary runner that performs four inferences, holds the one workload wake idle for a specified time, then calls `Session::close()`. The two logged firmware wakes per run were the package/preflight wake and the workload wake; there was no periodic wake rotation in that deployed host. With `disableCDSPUsage=true`, idle holds of 2, 5, 20, and 60 seconds completed. A 20-second hold and a 20,000-inference run (about 53 seconds) also completed while the right controller was connected. The user directly observed controller tracking working in this configuration. Each successful close returned control to `(1,0,1)`, cleared the callback, and left the loader, payload, and arena regions zero; XRService logged no Stereo FeatureNN CDSP timeout during those holds.

For the comparison, the original XRService settings were restored and the headset rebooted, with the right controller still connected. The same 20-second idle-hold test began its workload wake at 04:08:24.465. XRService logged its first `Dispatching Stereo FeatureNN on cDSP timed out` at 04:08:30.958, then repeated ownership failures. After the requested exit, the DSP remained at `(2,0,8)` with the callback registered; the runner could not close and was killed before reboot. The CPU setting was reapplied and a final reboot returned the DSP to idle `(1,0,1)` with no callback. The settings file's original copy remains beside it as `XRServiceSettings.json.pre-cpu-cdsp-test-20261109`.

`disableCDSPUsage` alone did not keep FeatureNN off the CDSP in the following boot. XRService started at 04:10:40 with that setting loaded, stayed quiet for ten minutes, then dispatched Stereo FeatureNN on the CDSP at 04:21:16, 0.5 s after a controller was picked up and SLAM relocalized; a hand-viewer session held at that moment timed out and left `(2,0,8)`. That setting belongs to the SLAM preprocessor. The controller tracker has its own switch: `xrSessionSettings.controllerTrackingOptions.configGlobal` with `"FeatureNN": {"enableForLogic": false}` and `"GripNet": {"enable": true}` (GripNet is XRService's CPU ONNX controller tracker; enabling both makes XRService log `Should not enable both GripNet and FeatureNN` every frame). XRService names unknown keys in its log (`invalid setting: key 'X' does not exist`); a top-level `controllerTrackingOptions`, `featureExtractorNN`, and `FeatureNN.enable` were all rejected. With the accepted setting and a tracked controller, XRService made no CDSP calls after startup. Controller tracking quality under GripNet has not been evaluated.

This A/B result strongly implicates XRService CDSP contention in this failure, and demonstrates a way to avoid re-wakes for the tested durations. It does not establish an unlimited hold, prove the CPU path's tracking quality under all conditions, or identify the exact stuck firmware instruction. XRService's five-minute controller metric reported `UNINITIALIZED` for 38.2 seconds during the CPU trial even though the user observed tracking working; treat that metric as unresolved, not as a pass/fail proxy. A production long-wake mode should check the CPU setting, validate controller/HMD/eye behavior, and retain an exit/recovery path before raising or removing the current one-second rotation limit.

No device clock, firmware, kernel module, or service is changed merely by reading these guides. Device operations should be performed from a known package and recorded configuration.

Source: [build script](../../hexecutor/build-scripts/build_resident_dsp.sh), [session](../../hexecutor/host/cpp/executor.cpp), [transport](../../hexecutor/host/cpp/frame_transport.cpp), and [module build](../../hexecutor/device/kmod/build.sh).

The native host is built with the root CMake project (`make build-host`). `HEXECUTOR_SDK_ROOT` supplies the Hexagon SDK path for DSP build scripts, and `HEXECUTOR_TOOLCHAIN_IMAGE` selects an already installed Docker toolchain image. Build wrappers mount the source tree explicitly; they no longer rely on a synthetic model/resource directory. Offline Python tools use `make run SCRIPT=...` or an editable development install.
