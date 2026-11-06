#!/usr/bin/env python3
"""Qwen3-0.6B on the Frame CDSP: whole tokens per DSP command (qwen_dsp.c).

The model image (pack_dsp.py) stays resident in the cdsp_mem buffer, so a
token costs one command: the host passes raw Q4_0 embedding rows and reads
back the greedy token (and logits on request). The KV cache lives on the DSP
side; `reset()` only rewinds the position.
"""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "depth-model/python"),
                             str(_source_root / "depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import argparse
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np

HERE = (PATHS.repo / 'hexecutor/examples/llm')
FRAME = (PATHS.native)
sys.path.insert(0, str(FRAME))
from hexecutor.offline.frame_transport import FrameTransport

MODEL_DEFAULT = Path.home() / "frame-chat/models/Qwen3-0.6B-Q4_0.gguf"
IMAGE_DEFAULT = MODEL_DEFAULT.parent / "Qwen3-0.6B.dspq4"
GGUF_PY_DEFAULT = Path.home() / "frame-chat/vendor/llama.cpp/gguf-py"
CDSP_MEM = Path("/dev/cdsp_mem")
CDSP_MEM_MB = Path("/sys/module/cdsp_mem/parameters/size_mb")
MAGIC = 0x334E5751
EMB_OFF = 0x20000
LOGITS_OFF = 0x100000
OUT_OFF = 2048
MAX_TOK = 32
MAX_VERIFY = 6
ARGMAX_OFF = 0x5000
VOCAB = 151936


class QwenDSP:
    def __init__(self, model=MODEL_DEFAULT, image=IMAGE_DEFAULT, gguf_python=GGUF_PY_DEFAULT):
        sys.path.insert(0, str(gguf_python))
        import gguf
        reader = gguf.GGUFReader(model, mode="r")
        self.embedding = next(t for t in reader.tensors if t.name == "token_embd.weight").data
        self.image = Path(image)
        meta = json.loads(Path(str(image) + ".json").read_text())
        self.ctx = meta["ctx"]
        self._load_image(meta)
        self.position = 0
        self.transport = None
        self.calls = 0
        self.dsp_seconds = 0.0
        # Weight stream: 2 = worker 0's DMA engine (default), 1 = engines of
        # workers 0 and 1 (no faster sustained, rare stalls), 0 = HVX loads.
        self.dma_mode = int(os.environ.get("FRAME_QWEN_DMA", "2"))
        self.trace = os.environ.get("FRAME_QWEN_TRACE") == "1"
        # The payload raises the Q6 AXI-master clock from XO to 750 MHz (DDR
        # reads ~1.2 -> ~9 GB/s per thread); FRAME_QWEN_AXI=0 keeps it on XO.
        self.raise_axi = os.environ.get("FRAME_QWEN_AXI", "1") != "0"
        self.axi_cfg = self.axi_error = None

    def _load_image(self, meta):
        if not CDSP_MEM.exists():
            raise RuntimeError("cdsp_mem module is not loaded (sudo insmod kmod/cdsp_mem.ko size_mb=768)")
        size = int(CDSP_MEM_MB.read_text()) << 20
        if meta["total_bytes"] > size:
            raise RuntimeError(f"cdsp_mem holds {size >> 20} MB; the image needs {meta['total_bytes'] >> 20} MB")
        fd = os.open(CDSP_MEM, os.O_RDWR)
        try:
            self.mem = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        want = bytes.fromhex(meta["fingerprint"])
        if bytes(self.mem[44:60]) == want:
            return
        started = time.monotonic()
        self.mem[0:4] = bytes(4)        # invalid until the copy completes
        with open(self.image, "rb") as stream:
            stream.seek(4096)
            offset = 4096
            while True:
                chunk = stream.read(16 << 20)
                if not chunk:
                    break
                self.mem[offset:offset + len(chunk)] = chunk
                offset += len(chunk)
            stream.seek(0)
            header = stream.read(4096)
        self.mem[4:4096] = header[4:]
        self.mem[0:4] = header[:4]
        print(f"loaded DSP image in {time.monotonic() - started:.1f} s", flush=True)

    def __enter__(self):
        payload = Path(os.environ.get("FRAME_QWEN_PAYLOAD", FRAME / "build/qwen_dsp.elf"))
        self.transport = FrameTransport(FRAME / "build/qd_loader.elf", payload, 30)
        self.transport.__enter__()
        return self

    def __exit__(self, *exc):
        if self.transport is not None:
            self.transport.__exit__(*exc)
            self.transport = None

    def reset(self):
        self.position = 0

    def verify(self, token_ids):
        """Run up to MAX_VERIFY tokens and return the greedy next token after
        each of them (for speculative decoding)."""
        if not 1 <= len(token_ids) <= MAX_VERIFY:
            raise ValueError(f"1..{MAX_VERIFY} tokens per verification")
        self.step(token_ids, verify=True)
        return list(struct.unpack_from(f"<{len(token_ids)}I", self.transport.arena, ARGMAX_OFF))

    def step(self, token_ids, *, logits=True, full_logits=False, verify=False):
        """Run up to MAX_TOK tokens; returns (greedy next token, logits or None)."""
        ids = [int(i) for i in token_ids]
        if not 1 <= len(ids) <= MAX_TOK:
            raise ValueError(f"1..{MAX_TOK} tokens per DSP command")
        if self.position + len(ids) > self.ctx:
            raise ValueError(f"context limit is {self.ctx} tokens")
        if self.transport is None:
            raise RuntimeError("DSP session is not open")
        t = self.transport
        raw = b"".join(self.embedding[i].tobytes() for i in ids)
        flags = ((1 if logits or full_logits or verify else 0) | (2 if full_logits else 0) |
                 (4 if self.trace else 0) | (8 if verify else 0) | (0 if self.raise_axi else 16))
        position = self.position

        def prepare(seq):
            t.arena[128:176] = bytes(48)
            t.arena[EMB_OFF:EMB_OFF + len(raw)] = raw
            struct.pack_into("<10I", t.arena, 0, MAGIC, 2, seq - 1, 1, len(ids), position, flags,
                             self.dma_mode, 0, (~seq) & 0xffffffff)
            t.fence()
            struct.pack_into("<I", t.arena, 8, seq)
            t.fence()

        def collect():
            status = struct.unpack_from("<16I", t.arena, OUT_OFF)
            values = (np.frombuffer(bytes(t.arena[LOGITS_OFF:LOGITS_OFF + VOCAB * 4]), "<f4")
                      if full_logits else None)
            return status, values

        start = time.monotonic()
        status, values = t.submit(prepare, collect)
        self.dsp_seconds += time.monotonic() - start
        self.calls += 1
        if status[0] != 6 or status[1]:
            raise RuntimeError(f"DSP command failed: status={status[:5]}")
        self.position += len(ids)
        self.last_cycles = status[2] | (status[3] << 32)
        self.dma_active = status[5]
        self.axi_cfg, self.axi_error = status[6], status[7]
        if self.axi_error and self.calls == 1:
            print(f"AXI clock not raised (error {self.axi_error}, CFG {self.axi_cfg:#x}); running at XO speed",
                  file=sys.stderr, flush=True)
        if self.trace:
            n = struct.unpack_from("<I", t.arena, 0x60000)[0]
            self.stream_trace = np.frombuffer(bytes(t.arena[0x60004:0x60004 + n * 20]), "<u4").reshape(n, 5)
        phases = struct.unpack_from("<16I", t.arena, OUT_OFF + 64)
        self.phase_mcycles = [round(v * 16 / 1e6, 2) for v in phases[:10]]
        self.attn_mcycles = [round(v * 16 / 1e6, 2) for v in phases[10:14]]
        prof = struct.unpack_from("<12I", t.arena, OUT_OFF + 128)
        self.dma_profile = [[round(prof[w * 4 + i] * 16 / 1e6, 1) for i in range(3)] for w in range(3)]
        self.sync_mcycles = [round(v * 16 / 1e6, 1) for v in struct.unpack_from("<3I", t.arena, OUT_OFF + 192)]
        return (status[4] if flags & 1 else None), values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="9707,0,1207")
    p.add_argument("--generate", type=int, default=16)
    p.add_argument("--batch", type=int, default=MAX_TOK)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    prompt = [int(v) for v in args.tokens.split(",")]
    engine = QwenDSP()
    with engine:
        started = time.monotonic()
        nxt = None
        for i in range(0, len(prompt), args.batch):
            chunk = prompt[i:i + args.batch]
            nxt, _ = engine.step(chunk, logits=i + len(chunk) == len(prompt))
        prefill = time.monotonic() - started
        out = []
        started = time.monotonic()
        for _ in range(args.generate):
            out.append(nxt)
            nxt, _ = engine.step([nxt])
        decode = time.monotonic() - started
    report = {"prompt": prompt, "generated": out, "prefill_seconds": prefill,
              "decode_seconds": decode, "decode_tps": args.generate / decode,
              "last_dsp_mcycles": engine.last_cycles / 1e6, "dma": [engine.dma_mode, engine.dma_active], "axi_cfg": hex(engine.axi_cfg), "axi_error": engine.axi_error, "direct/wait/vtcm per worker": engine.dma_profile, "sync": engine.sync_mcycles,
              "phases_mcycles(embed,norm/quant,qkv,qk_prep,attn,o,gateup,silu,down,head)": engine.phase_mcycles,
              "attn_worker0_mcycles(q_fp16,qk,softmax,v)": engine.attn_mcycles}
    print(json.dumps(report), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
