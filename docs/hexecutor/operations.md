# Building and operating Hexecutor

## Toolchains and recipes

The basic Makefile accepts `HEX_CC`, `HEX_LD`, and `HOST_CC`. A host compiler builds scalar substitutes; Hexagon builds need the supported target flags/intrinsics and linker. Some prepared/HMX paths use the local SDK toolchain even when basic callback targets can use an open LLVM build.

From the monorepo root:

```sh
make -C hexecutor/frame HEX_CC=clang-14 HEX_LD=ld.lld-14
```

This builds basic callback targets. For the current measured primary recipe:

```sh
sh hexecutor/frame/bench/lanes/build_best.sh NAME
```

That script changes to the resource root, uses the prepared engine recipe, and expects separately provisioned program data, measured-cost inputs, and generated lane headers. Its recorded generator options include two worker lanes and `--active block`. Regenerate model-specific headers when the program or schedule policy changes; the last generated header is not a universal schedule.

Record source revision/dirty patch, compiler, flags, generated-header identity, package hashes, engine/bootstrap hashes, and device report together. Files in `build/` are private local artifacts and are not shipped by Git. Do not infer that a renamed ELF is the one a launcher pins.

## Kernel memory module

The depth engine expects `/dev/cdsp_mem`, 128 MB, and the verified IOVA. `runtime/scripts/setup_frame.sh --qnn` checks the configured device package, loads the module when needed, and checks memory chunks. A kernel module must match the running kernel and its ABI; building it against an arbitrary source tree is insufficient.

The language-model client has a different memory-size requirement. Loading its larger allocation is not interchangeable with satisfying the depth engine's exact module-size check.

## Session lifecycle

The host acquires exclusive DSP access, checks service/eye state, verifies memory ownership and package identity, installs loader/payload state, then enters the guarded session. During a run, work publication and completion are explicit. Normal shutdown drains active work, restores borrowed state, clears callback registration, and resumes the eye client only when restoration is safe.

Changing the payload requires instruction-cache maintenance. The fixed loader implements that for its supported range. Replacing a live loader or treating apparently zero firmware memory as a general executable allocator is outside the validated contract.

Keep `dsp_service` alive. One custom inference client runs at a time. Eye tracking can be paused for the session; a depth worker and the chat client cannot claim the same DSP resources concurrently.

## Failure handling

`bench/depth_state.py` inspects service/worker state. `bench/chunk_map.py` checks owned memory. Preserve the reports when a worker times out or cleanup fails. A zero process exit code, an old JSON file, or restored output buffers alone does not prove the DSP session exited cleanly.

The recorded idle-exit failure left control state `(2,0,8)` and required a reboot. The bridge closes idle sessions early to avoid that path. If the capture harness marks the current boot as an NPU-cleanup fault, do not clear the marker to force another scan; recover the device and re-establish prerequisites.

No device clock, firmware, kernel module, or service is changed merely by reading these guides. Device operations should be performed from a known package and recorded configuration.

Source: [build script](../../hexecutor/frame/build_resident_dsp.sh), [session](../../hexecutor/frame/resident_dsp.py), [transport](../../hexecutor/frame/frame_transport.py), and [module build](../../hexecutor/frame/kmod/build.sh).
