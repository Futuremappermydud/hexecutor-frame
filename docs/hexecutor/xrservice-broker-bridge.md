# XRService broker integration

Hexecutor has a stock DMA-mailbox adapter in `hexecutord` and a separate shared C ABI shim. Both submit to the tensor [broker](broker.md). Use the portable ABI tests to validate packing and dispatch. Worn-device correctness and firmware ownership still require verification for each stock workload.

## Integration paths

- The in-daemon `xrservice::Adapter` connects to `dsp_service`, maps its shared DMA buffer, and passes supported work to the broker with a tracking deadline. One adapter coordinates subscribed stock jobs, stages the result before publishing completion, and yields to stock firmware for model setup or unsupported requests. [Mailbox adapter](../../hexecutor/host/cpp/xrservice_featurenn.cpp)
- The shared shim exposes `broker_shim_dispatch_stereofeaturenn`, `broker_shim_dispatch_gazenet` and `broker_shim_dispatch_gazenet_vector`. It also exports a FeatureNN symbol for interposition where a caller uses that symbol. An installed library, successful `LD_PRELOAD` or portable test alone does not establish that live XRService dispatch passes through it. Check the actual loaded service and stock traffic before claiming that firmware is bypassed. [C header](../../hexecutor/include/hexecutor/xrservice_broker_shim.h), [shim implementation](../../hexecutor/host/cpp/xrservice_broker_shim.cpp)

## StereoFeatureNN

`hexecutord --xrservice-featurenn GRAPH` (or `--xrservice-stereofeaturenn GRAPH`) serves job 8 from the unmodified built-in XRService. Its shared block starts at DMA offset `0x1522000`. Its words are state, active, and taken/job id at offsets 0, 4, and 8. The request header is at cached offset `0x80`, packed u8 eye planes at `0x100`, and results at `0x40080`. Each eye has a 0x450-byte `cdsp_featurenn_output_data`: four contiguous 69-float arrays (center, uncertainty, plus-x, plus-z). The adapter converts u8 planes to normalized three-channel float NHWC and writes uncertainty as `2·min(raw,10)`; XRService applies the final exponential.

Enable XRService's normal FeatureNN path (not the GripNet/CPU test setting) and run, using the actual lowered prefix:

```sh
FRAME_STEREOFEATURENN=/home/steamos/hexecutor-broker/featurenn \
    hexecutor/packaging/run_hexecutord.sh --long-wake 30
```

`FRAME_FEATURENN` is also accepted as an environment variable alias. That launcher registers stereoFeatureNN alone; add `--graph landmark=...` explicitly if desired. Check total constant capacity for any added graph. The CPU ONNX model that ships with XRService is the numerical reference. Our lowered graph differs from its u8 result by up to about 0.07 on the raw head.

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
