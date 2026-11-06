#!/usr/bin/env python3
"""Compare one large prompt batch to equivalent smaller causal batches."""
import numpy as np
from tokenizers import Tokenizer
from npu_qwen import QwenNPU
from pathlib import Path

PROMPT='<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
tokenizer=Tokenizer.from_file(str(Path.home()/'frame-chat/models/tokenizer.json'))
ids=tokenizer.encode(PROMPT,add_special_tokens=False).ids
engine=QwenNPU()
with engine:
    complete=engine.run_batch(ids)
    print('one batch',int(np.argmax(complete)),float(np.max(complete)),flush=True)
    engine.history=[[] for _ in range(28)]
    engine.run_batch(ids[:7],logits=False)
    split=engine.run_batch(ids[7:])
    print('split batch',int(np.argmax(split)),float(np.max(split)),flush=True)
    print('max_abs_difference',float(np.max(np.abs(split-complete))),flush=True)
    print('top_one',np.argsort(complete)[-8:][::-1].tolist(),flush=True)
    print('top_split',np.argsort(split)[-8:][::-1].tolist(),flush=True)
