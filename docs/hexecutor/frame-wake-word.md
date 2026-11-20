# Wake word activation on Steam Frame: investigation handoff

Status: investigated on 2026-11-18; no wake word detector has been deployed. The target is detection **while the headset is on**. Waking the headset from system suspend is outside this scope.

This page records the verified mic and notification paths, the low power implementation route, and what remains unproved. It is not a claim that Qualcomm's stock SVA service is running on Valve's firmware.

## Findings at a glance

| Question | Finding | Evidence |
| --- | --- | --- |
| Can the active headset capture microphone PCM? | Yes. `VA Capture va_macro_tx1-1` captured 16 kHz, mono, signed 16-bit PCM while `dsp_service` ran. | Two-second `arecord -D hw:0,1 -f S16_LE -r 16000 -c 1` run exited 0 on 2026-11-18. |
| Is the stock Qualcomm SVA firmware active? | No. Valve's `adsp_sm8650.mbn` is the active aDSP image. The installed Qualcomm `adsp.mbn` contains SVA/ACD strings but is not what `dsp_service` boots. | Live service process and image hash; service loader disassembly. |
| Is the CDSP callback in the TEE? | No. CDSP, aDSP, and the Qualcomm TEE are separate execution environments. The existing Hexecutor callback is for CDSP and does not provide an aDSP code hook. | Hexecutor transport and Valve's separate `/dev/cdsp` and `/dev/adsp` driver paths. |
| Can a client receive an aDSP notification? | Yes. A client obtained the service's shared aDSP DMA-BUF, subscribed to bit `0x40`, and received `service_notification\0` after a **host-simulated** event. | Live end-to-end service test described below. This did not execute new DSP code. |
| Is there a low power microphone memory pool? | Yes. The live device tree attaches a 12 KiB pool at physical `0x06d40000` to the VA LPAIF. The current soundcard chains VA through core LPAIF; its source comment describes a direct LPASS memory mode. | Live device tree and Valve kernel source. |

## Active firmware and ownership

`/opt/steamvr/drivers/cv/bin/linuxarm64/dsp_service` owns `/dev/adsp` exclusively and boots `/opt/steamvr/drivers/cv/bin/linuxarm64/adsp_sm8650.mbn`. The observed image SHA-256 was `b2183a36716ad602871b5c94242bc94faac7e2b3e7f09eb4117c18b00cc61328`. The image is a roughly 1.4 MiB Hexagon ELF with entry/code base `0x89000000`. Service allocates a 16 MiB shared DMA-BUF, copies the image into it, maps it for the aDSP, and boots the aDSP. Its `load_mbn` path copies ELF segments; no signature verification was identified in that host loader. That observation does not rule out other firmware checks.

The installed `/usr/lib/firmware/qcom/sm8650/adsp.mbn` is a separate, roughly 23 MiB Qualcomm image. Strings in that file include `capi_voicewakeup_v2` and `PARAM_ID_REGISTER_MULTI_SOUND_MODEL`. They are evidence that **the installed stock image** includes AudioReach SVA code, not that Valve's active image exposes it. No usable PAL/AudioReach VoiceUI graph service or SVA sound model was found installed. Replacing Valve's image wholesale would also replace its audio filter and other headset tasks; it is not a validated integration route.

Valve's PipeWire LADSPA `audiofilter.so` sends mic PCM to a WebRTC AEC/NS task in Valve's aDSP image. That existing filter is a host-fed stream, not evidence of autonomous aDSP microphone capture.

The [Hexagon bare-metal driver](https://github.com/evlaV/deckard-linux-no-oopsies/blob/sm8650-7.2/drivers/soc/qcom/hexagon-baremetal.c) provides separate aDSP/CDSP devices. `/dev/adsp` is single-owner; opening another instance while the service runs is not a workable extension point. The driver's `DSP_IOCTL_MAP_DIRECT` maps a physical range into the aDSP IOMMU, and `DSP_IOCTL_MAP_DMABUF` maps a DMA-BUF. Its `DSP_IOCTL_HANDOVER` drops initial aDSP power-domain and interconnect votes after boot. The service's shared aDSP memory is accessible to its socket clients without claiming `/dev/adsp`.

## Verified service callback path

The service listens on the abstract Unix socket `\0dsp_service`. On connection it sends an `ADSP` header and two file descriptors: a 16 MiB aDSP DMA-BUF followed by a 32 MiB cDSP DMA-BUF. A client sends a 32-bit request mask. The first bytes of the mapped aDSP DMA-BUF matched the active firmware image at its ELF load offset, confirming this is the live image memory.

Service disassembly shows `process_dsp_requests` consuming a 32-bit aDSP request word at buffer offset `0x450060`, intersecting it with each client's request mask, and sending `service_notification\0` to matching clients. The aDSP image contains a helper at `0x89002c40` that atomically ORs its argument into a word addressed as `0xf0450060` and writes an IPCC notification register. The host offset and firmware address appear to refer to the same shared control word; that address relationship has not been exercised by newly injected DSP code.

On 2026-11-18, a test client subscribed to unused bit `0x40`, wrote that bit to the host view at offset `0x450060` while the word was zero, and received `service_notification\0`; the service cleared the word. This proves client registration, service consumption, and notification delivery. It **does not** prove the aDSP can raise bit `0x40` from a new task. Any implementation must confirm that the bit remains free in the deployed firmware version. The service also checks requests after its 100 ms epoll timeout, so IPCC is the immediate notification route rather than the only delivery mechanism.

## Microphone DMA and buffer size

The live card is `SM8250 LPASS`, with capture PCM `VA Capture va_macro_tx1-1` at `hw:0,1`. At 16 kHz mono S16, a live ALSA read showed `period_size: 1024` frames and `buffer_size: 4096` frames: 64 ms per period and 256 ms of raw audio in the 8 KiB ring. In a three-second capture, the active LPAIF interrupt count rose from 102181 to 102230, consistent with about 16 period interrupts per second. That ordinary ALSA path requires regular host activity.

The [LPAIF driver](https://github.com/evlaV/deckard-linux-no-oopsies/blob/sm8650-7.2/sound/soc/qcom/lpaif.c) sets its allocation and `buffer_bytes_max` to `0x2000` (8 KiB), then programs the DMA length from the actual ALSA buffer size. It already writes `~0u` to the period-length register for its internal bridge to suppress bridge period interrupts. A dedicated armed-capture mode could use cyclic DMA without ALSA period wakeups; it must still manage ownership and stop cleanly. Merely leaving an ordinary `arecord` stream open does not remove host wakeups or ALSA overrun behavior.

The live device tree has a `shared-dma-pool` named `lpm@6d40000`, physical base `0x06d40000`, size `0x3000` (12 KiB), attached to `lpaif@6d00000` (VA). The [Deckard soundcard link](https://github.com/evlaV/deckard-linux-no-oopsies/blob/sm8650-7.2/arch/arm64/boot/dts/qcom/sm8650-deckard-sndcard.dtsi) currently lists both `<&lpass_core_lpaif 2>` and `<&lpass_va_lpaif LPAIF_VA_CODEC_TX>`. Its comment says removing core LPAIF and its clock makes the VA interface use LPASS LPM directly instead of chaining through normal memory. In the current chained setup, `lpaif_open` allocates an 8 KiB VA-side bridge buffer but uses only 64 bytes of it, plus an 8 KiB ALSA ring on the core side.

**Can the DMA ring grow?** Yes, the driver limit can be raised. A direct VA configuration has room in its dedicated pool for *at most* 12 KiB, subject to allocation/alignment and device testing; changing the master ring allocation and `buffer_bytes_max` from `0x2000` to `0x3000` would give 384 ms at 16 kHz mono S16. Raising both allocations in the current chained configuration is not the desired low power change. A ring larger than 12 KiB needs another verified memory region or a revised reservation; the nearby 64 KiB LPM pool belongs to the WSA interface and should not be treated as spare VA memory. DDR can hold a larger ring but forfeits the direct LPM advantage. A wake word model can keep a one-to-two-second **feature history or state** on the aDSP while consuming a smaller cyclic PCM ring, so the DMA ring need not cover the full phrase.

## Proposed on-headset implementation

1. Add a Deckard soundcard configuration that routes `VA Capture` directly through VA LPAIF, then verify live PCM capture and the DMA address in the `0x06d40000` pool. Keep an easy rollback to the existing link.
2. Add a kernel armed-capture interface that owns the direct VA ring, disables period interrupts, reports the producer pointer to the aDSP, and restores the normal mic path on disarm. Use the existing driver clock and DMA lifecycle rather than leaving a userspace PCM stream running. Test coexistence with PipeWire microphone use.
3. Map the VA LPM pool or its allocated buffer into the aDSP IOMMU at a nonoverlapping IOVA. The existing mapping ioctl demonstrates the mechanism, but `dsp_service` owns `/dev/adsp`, so this needs a kernel/service integration point. Do not pass an ALSA DMA address to `DSP_IOCTL_MAP_DIRECT` as though it were a physical address. Map the backing physical pages or a DMA-BUF through the correct API.
4. Add a small aDSP worker to Valve's image (or a compatible extension) that reads the ring, maintains model state, and uses the existing aDSP notification helper with a reserved event bit on a match. Preserve Valve's audio filter, charger, version contract, and cache behavior. Runtime writable firmware memory is visible to service clients, but a safe live code patch and instruction-cache invalidation have **not** been demonstrated; an offline patched-image test may be easier to validate.
5. Measure on-device current/power with matched headset state for idle, ordinary ALSA capture, direct armed capture, and detector active. Validate false accepts, misses, latency, repeated arm/disarm, audio coexistence, and service/firmware recovery. No power saving or wake word accuracy number is established yet.

This route uses the Qualcomm aDSP and LPASS hardware while the headset is on. It does **not** currently invoke Qualcomm's stock SVA CAPI. If use of that exact SVA implementation is required, its binary/model loading and compatibility with Valve's active image remain an independent investigation.

## What was and was not changed

The investigation used read-only firmware copies/disassembly, live device-tree and ALSA reads, short microphone captures to `/dev/null`, a service client that received DMA-BUF descriptors, and one host-simulated notification bit. It did not replace firmware, restart `dsp_service`, install a module, modify headset files, or leave microphone capture armed. No repository runtime or DSP source was changed by this investigation.

Source references: [Valve kernel branch](https://github.com/evlaV/deckard-linux-no-oopsies/tree/sm8650-7.2), [bare-metal DSP driver](https://github.com/evlaV/deckard-linux-no-oopsies/blob/sm8650-7.2/drivers/soc/qcom/hexagon-baremetal.c), [LPAIF driver](https://github.com/evlaV/deckard-linux-no-oopsies/blob/sm8650-7.2/sound/soc/qcom/lpaif.c), [Deckard soundcard link](https://github.com/evlaV/deckard-linux-no-oopsies/blob/sm8650-7.2/arch/arm64/boot/dts/qcom/sm8650-deckard-sndcard.dtsi), and a [recovered live device tree](https://gist.github.com/strongtz/178dd7484ba556a6e82674f9dbf8083a) used only as a cross-check against `/proc/device-tree`.
