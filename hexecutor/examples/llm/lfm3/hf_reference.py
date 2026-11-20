#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""HF reference for LFM3-1.7B: greedy tokens and the final-norm hidden state of every step (float32 CPU).

Usage: hf_reference.py MODEL_DIR OUT_DIR "prompt text" [NEW_TOKENS]
Writes OUT_DIR/ref.json (prompt ids, generated ids) and OUT_DIR/ref_hidden.f32 ([steps, 2048] for every
position fed, prompt included: the input to lm_head after embedding_norm).
"""
import json, sys, pathlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, out_dir, prompt = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
new = int(sys.argv[4]) if len(sys.argv) > 4 else 8
out_dir.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32).eval()
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
hidden = []
model.model.embedding_norm.register_forward_hook(lambda m, i, o: hidden.append(o[0].detach().numpy().copy()))
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=new, do_sample=False)
    gen = out[0, ids.shape[1]:].tolist()
    # one full forward over prompt+generated gives hidden states for every position
    hidden.clear()
    model(out)
h = np.concatenate(hidden, axis=0).astype("<f4")
(out_dir / "ref_hidden.f32").write_bytes(h.tobytes())
(out_dir / "ref.json").write_text(json.dumps({"prompt": prompt, "prompt_ids": ids[0].tolist(), "generated": gen,
                                              "text": tok.decode(gen)}))
print(ids.shape, gen, repr(tok.decode(gen)), h.shape)
