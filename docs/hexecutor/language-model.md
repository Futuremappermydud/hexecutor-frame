# Language-model DSP client

The Qwen3-0.6B Q4_0 example under `hexecutor/examples/llm/` demonstrates a second client of the loader, transport, private-memory module, HVX, and clock infrastructure. It is not part of the spatial depth pipeline and cannot run concurrently with a client claiming the same DSP session.

The separate language-model example's ARM host tokenizes text, supplies embedding rows, and reads token IDs. `qwen_dsp.c` performs the model command on the DSP: quantized matrix products, attention, normalization, residuals, and output selection. Weights stream through a three-slot DMA/VTCM ring; the KV cache is fp16 with transposed K storage. Synchronization uses TCM-resident flags rather than high-latency uncached-DDR barriers.

`pack_dsp.py` creates the local DSP image from the supplied GGUF. `dsp_qwen.py` owns the host engine; `chat_session.py` implements ChatML, prompt chunks, and prompt-lookup speculative decoding. The Gamescope/X11 launcher presents a SteamVR dashboard panel. The retained session pauses eye tracking for the life of the app; starting a new chat rewinds context rather than reopening the session.

The example uses a larger private-memory allocation than the depth engine's 128 MB contract. Do not reuse its module configuration for depth without checking the caller's required size. Model/tokenizer/packed-image assets are local and must remain outside Git.

The AXI fix increased a short-context comparison from 3.58 to 53.1 tokens/s. Vectorizing per-element phases later measured 91.6 tokens/s at 64 generated tokens and 74.6 at 512 in the recorded setup. These are dated model/workload measurements, not the dashboard's guaranteed rate. HVX qfloat rounding can change near-tied greedy decisions even when quantization matches its scalar reference.

`FRAME_QWEN_AXI=0` leaves the early low-clock setting unchanged for controlled comparisons. `FRAME_QWEN_DMA` selects the documented DMA/HVX loading alternatives. Prompt speculation is a separate algorithmic choice; its results should not be mixed with the plain decode clock comparison.

Build and launch from the configured Frame development package using `make ... build/qwen_dsp.elf`, the local packer, and `llm/launch.sh`. Check the launcher's expected package/model paths and active Gamescope display first. The guide does not provision weights or resize/load the kernel module automatically.
