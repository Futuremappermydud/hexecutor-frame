# XRService broker integration

Hexecutor has a shared C ABI shim and a stock DMA-mailbox adapter for Gazenet. Both submit to the tensor [broker](broker.md). XRService's FeatureNN call takes CDSP ownership before publishing job 8, so the mailbox adapter cannot serve that job while the broker owns the device. The FeatureNN path uses a version-pinned in-process hook before that ownership step. Worn-device correctness still requires verification.

## Integration paths

- The in-daemon `xrservice::Adapter` connects to `dsp_service`, maps its shared DMA buffer, and serves supported Gazenet work. FeatureNN mailbox mode now refuses startup because XRService does not publish its job until after acquiring CDSP ownership. [Mailbox adapter](../../hexecutor/host/cpp/xrservice_featurenn.cpp)
- The shared shim exposes `broker_shim_dispatch_stereofeaturenn`, `broker_shim_dispatch_gazenet` and `broker_shim_dispatch_gazenet_vector`. It also exports a FeatureNN symbol for interposition where a caller uses that symbol. An installed library, successful `LD_PRELOAD` or portable test alone does not establish that live XRService dispatch passes through it. Check the actual loaded service and stock traffic before claiming that firmware is bypassed. [C header](../../hexecutor/include/hexecutor/xrservice_broker_shim.h), [shim implementation](../../hexecutor/host/cpp/xrservice_broker_shim.cpp)

## StereoFeatureNN

The XRService binary's FeatureNN dispatch takes CDSP ownership and then posts job 8. Its shared block starts at DMA offset `0x1522000`, with request header at `0x80`, packed u8 eye planes at `0x100`, and results at `0x40080`. Each eye has a 0x450-byte result record. The in-process hook intercepts the call before ownership, sends the crops to the broker, and applies the final uncertainty exponential before returning to XRService. It is enabled only with `HEXECUTOR_XRSERVICE_INLINE_HOOK=1` when `libxrservice_broker_shim.so` is loaded into XRService; it checks the executable path, function prologue, mailbox code and embedded function name against the 2026-11-03 binary before patching memory. [Hook source](../../hexecutor/host/cpp/xrservice_inline_hook.cpp)

Enable XRService's normal FeatureNN path (not the GripNet/CPU test setting) and run, using the actual lowered prefix:

```sh
FRAME_STEREOFEATURENN=/home/steamos/hexecutor-broker/featurenn \
    hexecutor/packaging/run_hexecutord.sh --long-wake 30
```

`FRAME_FEATURENN` is also accepted as an environment variable alias. That launcher registers the graph alone, for use by the in-process shim; it does not start a FeatureNN mailbox adapter. Set `HEXECUTORD_SOCKET` to the same socket in both processes, or use their shared default. Add `--graph landmark=...` explicitly if desired. Check total constant capacity for any added graph. The CPU ONNX model that ships with XRService is the numerical reference. Our lowered graph differs from its u8 result by up to about 0.07 on the raw head.

## Gazenet

The stock `CDSPGazenet` library is used by the separate `eyetracking` process. Its control block starts at DMA offset `0x445000`. Job 2 loads weights and the first job 4 acquires the stock I/O buffer; both remain on stock firmware. Once that buffer exists, `--xrservice-gazenet GRAPH` serves job-4 evaluation. The graph contract is one 128×128 three-channel float NHWC input and exactly 64 output floats. The adapter converts the two planar stock float images at cached offset `0x100`, pads channel two with zero, and writes the stock result at `0x20100`.

`--xrservice-gazenet-vector GRAPH` separately adapts the in-place float-vector request. Its input is exactly 4,800 floats at cached offset `0x440`; the graph may return one or several outputs containing 1–4,800 finite floats in total. The adapter writes them back to the same buffer, updates the float count, then clears the ready word at control offset `0x0c`. Other vector sizes are left to stock firmware.

**VTCM ownership gate.** Firmware hardware thread 3 runs the stock eye-tracking handler on its own, separately from our wake. It serves jobs 2/4 posted to the DMA block at `0x1522000`, and it uses VTCM `0x31700000–~0x317d4600`. Pool 1 owns that range while a session is open ([memory layout](memory-layout.md#stock-eye-tracking-thread-and-vtcm)). Once the eye client keeps running, any job 2/4 that reaches the firmware's thread 3 while the broker holds a wake corrupts both sides. The adapter's yield releases the session only for requests it sees in its own blocks. The firmware's handler matches Gazenet's layout but sits at the block this adapter calls FeatureNN, so the offsets above need on-device confirmation before the Gazenet adapter can be called safe. Two fixes would remove the hazard: keep pool 1 out of `0x31700000–0x31800000` in adapter mode, or prove that no job 2/4 can reach `0x1522000` during a wake.

Either Gazenet option sets `Artifacts::keep_eye_running`, so opening the broker session does not pause the eye client. Use this only with the adapter: an ordinary custom session still pauses and restores eye tracking. The launcher equivalents are `FRAME_GAZENET` and `FRAME_GAZENET_VECTOR`, each containing an HXFG prefix. Multiple adapter environment variables may be set together, subject to the session's graph and constant-capacity limits.

`--long-wake` does not enable the CPU-setting tripwire in adapter mode, because stock CDSP activity is expected. `hexecutor/tests/test_xrservice_featurenn.cpp` checks the FeatureNN and Gazenet conversion/result ABIs without a device. No Gazenet model lowering, output equivalence, live camera evaluation, or firmware handoff has been validated yet. Controller/gaze tracking quality on real inputs and long worn sessions remain release gates, as does absence of `(2,0,8)`.

## Packaging

`cmake --install build --prefix <dir>` installs these files from a root or standalone Hexecutor build.

  - `<prefix>/bin/hexecutord`
  - `<prefix>/lib/libxrservice_broker_shim.so` (or `.dylib`)
  - `<prefix>/include/xrservice_broker_shim.h`

The manual installer also copies the launcher to `<prefix>/bin/run_hexecutord.sh` and generates `<prefix>/systemd/10-xrservice-broker.conf`. It reads the build directory from `BUILD_DIR`, which defaults to the repo's `build/`. [install_hexecutord.sh](../../hexecutor/packaging/install_hexecutord.sh)

## Verification

The portable mailbox and shim tests validate conversion, result layout and broker calls. Validate model equivalence on captured inputs, then confirm live dispatch and firmware cleanup under the exact target service configuration. For the historical Frame coverage see the validation guide. [Mailbox ABI tests](../../hexecutor/tests/test_xrservice_featurenn.cpp), [shim tests](../../hexecutor/tests/test_xrservice_broker_shim.cpp), [device evidence](validation.md#recorded-frame-results)

**2026-11-20 Frame trial.** A FeatureNN-only adapter initially refused startup because it treated nonzero data at the Gazenet offset as an active Gazenet request. The FeatureNN-only path now ignores that block, and a portable regression test passes. The updated broker opened a device session on the Frame with the lowered FeatureNN graph. After switching XRService from GripNet to stock FeatureNN, no FeatureNN job reached the broker; XRService logged repeated CDSP ownership failures and Stereo FeatureNN timeouts. Broker shutdown reported worker restoration failure, and the firmware control remained `(2,0,8)` with a registered callback. The original GripNet setting was restored and the Frame rebooted; control returned to `(1,0,1)` with callback `0x0`, and XRService restarted. This trial does not establish a working tracking loop. Confirm the live request block and firmware ownership before another adapter attempt.

**2026-11-20 hook trial.** Static inspection of the same XRService binary showed that its dispatch acquires CDSP ownership before writing the job-8 mailbox, explaining the deadlock above. With stock FeatureNN enabled and the broker running without the mailbox adapter, the version-pinned hook was injected into the live XRService process. A synthetic 128×128 stereo crop passed through the hooked entry point and broker with return code 0 and finite output. Real controller traffic also reached the broker: its final statistics reported four FeatureNN runs total, including the synthetic run, with 1.31 ms mean DSP time and 3.53 s mean end-to-end run time. The sparse requests repeatedly opened cold sessions; XRService logged one 4.43 s FeatureNN call. Each observed session closed cleanly, and firmware control returned to `(1,0,1)` with callback `0x0`. This proves dispatch and return through the broker but does not establish usable latency, tracking quality, or sustained operation. The trial broker was stopped, the original GripNet setting restored, and XRService restarted.
