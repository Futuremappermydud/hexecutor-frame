# Frame NPU Chat

Qwen3-0.6B Q4_0 chat panel for the Steam Frame. The model is the [Qualcomm AI Hub Qwen3-0.6B](https://aihub.qualcomm.com/models/qwen3_0_6b) GGUF. Inference runs entirely on the CDSP; there is no CPU or GPU inference backend. The ARM host tokenizes the text, passes raw Q4_0 embedding rows and reads back token ids.

The launcher opens an X11 window on Gamescope's `:0` display and tags it with `STEAM_GAME=2000600601`, producing a SteamVR dashboard panel named **Frame NPU Chat**. The panel can be selected from the headset's dashboard. **Eye tracking is paused for as long as the chat app is open**, because the DSP session and its KV cache stay open across turns.

## Performance

**Oct 28, 2026: AXI-master clock fix.** Valve's firmware leaves the CDSP's Q6 AXI-master clock (`turing_cc+0x254`) on the 19.2 MHz crystal, with the Turing core PLLs off. The payload now raises it to QuRT's 750 MHz operating point for the session and restores it on exit. Measured with `dsp_qwen.py --generate 32`, identical tokens in both runs:

| | AXI on XO (`FRAME_QWEN_AXI=0`) | AXI at 750 MHz (default) |
| --- | ---: | ---: |
| Decode, short context | 3.58 tok/s | **53.1 tok/s** |
| DSP time per decode token | 516 Mcycles | 35 Mcycles |
| Prefill, 3 tokens | 0.29 s | 0.034 s |

Reports: `build/qwen-axi-xo.json`, `build/qwen-axi-750.json`.

**Oct 28, 2026: HVX for the per-element phases.** RMSNorm, quantize, SiLU, RoPE, the attention softmax, the residual adds and the greedy argmax were scalar. They now use HVX; quantize is split across the three workers (a decode row used to run on worker 0 alone), and attention copies each 64-position block of K^T and V from DDR into VTCM before its inner loops (dependent HVX loads from DDR cost ~6x their TCM time). Both columns run the AXI master at 750 MHz:

| `dsp_qwen.py` | Scalar phases | HVX phases (default) |
| --- | ---: | ---: |
| Decode, 64 tokens | 55.0 tok/s | **91.6 tok/s** |
| Decode, 512 tokens | 40.3 tok/s | **74.6 tok/s** |
| DSP time per token at ~515 context | 59.1 Mcycles | 29.4 Mcycles |
| Mean next-token NLL, 35-token check | 1.3980 | 1.3971 |

Quantize is bit-exact against the scalar code; the other phases round differently in the last bits (qf32 arithmetic), which can flip near-ties in greedy decoding (one seen with a top-2 logit gap of 0.0016). `bench/qwen_ops_test.c` checks every HVX phase against its scalar reference on the DSP. Reports: `build/qwen-final-*.json`. The payload is now 32,264 bytes of the loader's 32 KB region.

The tables below predate both changes and were measured on XO; speculation and chat-panel numbers have not been re-measured.

Oct 26, 2026 (AXI on XO):

| | Old path (`npu_qwen.py`) | Whole-token engine (`dsp_qwen.py`) |
| --- | ---: | ---: |
| Decode, short context | 0.36 tok/s, plus 1.7 s session overhead per 8 tokens | **3.5 tok/s** |
| Decode, 600-token context | n/a (257-token limit) | 2.8 tok/s |
| Decode with speculation, text that repeats context | n/a | up to **13.6 tok/s** |
| Prompt, 20–40 tokens | ~10 s | 0.6–1.2 s |
| Context limit | 257 tokens | 2048 tokens |

Speculative decoding is greedy-exact: it produces the same tokens as plain greedy decoding (checked token-for-token over multi-turn chats). It gains 25% on code and nothing on fresh prose, and costs at most about 1%.

**Before the clock fix, decode was bound by DDR bandwidth.** Each token reads 335 MB of Q4_0 weights (248 MB of layers, 87.5 MB for the output head), plus 4 KB of fp16 KV per position per layer. With the AXI master on XO the CDSP sustained only ~1.25 GB/s from DDR, so ~3.7 tok/s was the ceiling for one token per weight pass:

- That ceiling held for HVX loads, one user-DMA engine, and both together.
- Two DMA engines briefly burst to ~2.4 GB/s, then fell back to the same total.
- Raising the CDSP interconnect vote (`kmod/cdsp_bw.c`), the compute-NoC BCM (`kmod/cdsp_noc.c`) or the GCC NSP-NoC clock (`kmod/cdsp_nocclk.c`) made no difference; the AXI-master clock was the limit (`bench/axim_tune.c`).

Speculation gets past the ceiling because verifying 6 tokens costs only 13% more than 1 token.

## How it works

- `kmod/cdsp_mem.c` is a kernel module. It allocates 768 MB in 2 MB contiguous chunks, maps them at CDSP SMMU IOVA `0xa0000000` (in the domain Valve's `hexagon-baremetal` driver attached), and exposes them as `/dev/cdsp_mem`. dsp_service only shares a 16 MB DMA buffer, so before this module the old path had to re-copy every weight through it on every token.
- `pack_dsp.py` builds `~/frame-chat/models/Qwen3-0.6B.dspq4` once (~10 s). It re-arranges the GGUF's Q4_0 blocks into a decode layout (see `qwen_dsp.c`), and adds the RoPE table and norm weights. The engine copies the image into `/dev/cdsp_mem` when its fingerprint differs (0.1 s). Otherwise it stays resident while the module is loaded.
- `qwen_dsp.c` is the DSP payload, loaded by the existing `qd_loader` through the FLOPS callback. It installs JTLB entries for the buffer, then runs a whole command on all three workers: embedding, 28 layers and the output head for up to 32 tokens.
  - Matrices: Q4 × int8 `vrmpy` with per-block fp16 scales in qf32.
  - Attention: HVX over an fp16 KV cache, with K stored transposed.
  - Weights stream through worker 0's user-DMA engine into a 3-slot VTCM ring, which runs ahead of the compute.
  - Barriers are flag words in L2-TCM. Uncached-DDR barriers cost up to 25% while DMA was running.
- `dsp_qwen.py` is the host engine. `step()` runs a batch; `verify()` returns the greedy token after every position (up to 6). `chat_session.py` does ChatML, 32-token prompt chunks and prompt-lookup speculative decoding (n-grams 3 and 2).

## On the Frame

The development checkout is `/home/steamos/frame-dsp/depth-model/dev/frame`. The model and tokenizer are under `/home/steamos/frame-chat/models`. The Python environment is `../.venv` from the Frame directory.

```sh
cd /home/steamos/frame-dsp/depth-model/dev/frame
make HEX_CC=clang HEX_LD=ld.lld build/qd_loader.elf build/qwen_dsp.elf
../.venv/bin/python llm/pack_dsp.py      # once per model
kmod/load.sh                             # after every reboot (sudo)
./llm/launch.sh
```

`kmod/build.sh` builds the modules against Valve's `deckard-linux` source in `~/work/kernel/deckard-linux`, branch `sm8650-6.18`. The script pins the module's release string to the running kernel. Because pahole is missing, it also has to restore the BTF fields of `struct module`.

`launch.sh` requires an active Gamescope session. The app holds up to 2048 context tokens and up to 512 generated tokens per request. **New chat** rewinds the context without reopening the DSP session.

Environment knobs:

- `FRAME_QWEN_SPECULATE=0` turns off drafting.
- `FRAME_QWEN_DRAFT_NGRAMS=3,2,1` also drafts from unigram matches; that measured slower.
- `FRAME_QWEN_AXI=0` leaves the Q6 AXI-master clock on XO (the old, ~15× slower behavior), for comparisons. Otherwise the payload raises it to 750 MHz on the first command and restores it when the session exits; `axi_cfg` (`0x100503` when raised) and `axi_error` appear in the `dsp_qwen.py` report.
- `FRAME_QWEN_DMA` selects how weights reach the DSP: `2` (default) uses one DMA engine, `1` uses two (no faster sustained, rare ~0.4 s stalls), `0` uses HVX loads straight from DDR.

Copy `frame-npu-chat.desktop` to `~/.local/share/applications/` to make **Frame NPU Chat** available from the headset's Launch Program menu after a reboot.

## Verification and profiling

- `dsp_qwen.py --tokens ... --generate N` reports decode tok/s, DSP cycles and per-phase cycle counts.
- `trace_stream.py` prints the per-chunk DMA issue and completion times.
- Batch sizes 1, 8 and 32 produce identical tokens.
- A whole conversation with and without speculation produces identical KV histories.
- The output matches the old HMX path and llama.cpp (`build-ref`, used only to compare outputs).

`bench/bench_dma.c` and `bench/run_dma.py` hold the DDR-bandwidth and user-DMA experiments: DM2 config, TLB mapping, stall stress tests. `bench/probe_qf.c` checks which HVX qfloat/convert instructions work on this firmware; `vcvt` does not.

Hazards are recorded in the project memory. In short:

- Never `dmpause` a running DMA engine.
- Never let the DSP touch the `0xa0000000` range without its TLB entries.
- Never read NoC or Turing registers through `/dev/mem`; a gem_noc read reset the SoC.

## Next steps

- Prefill is compute-bound at 15–37 tok/s. RMSNorm/quantize, SiLU, RoPE and the softmax are still scalar; vectorizing them, and sharing K/V reads across a batch's tokens in attention, would help.
- Better drafting for prose, for example a tiny draft model, would carry the speculation gains beyond repetitive text.

## Old path

`npu_qwen.py` and `hmx_q4_matvec.c` (per-operator HMX commands, FP16 weights copied through the arena) and their probes remain for reference.

- `probe_q4.py` and `probe_ops.py` compare HMX and scalar operators with NumPy.
- `compare_batch.py` and `compare_prefill.py` compare batch splits.
- `profile_speed.py` times one token on that path.
