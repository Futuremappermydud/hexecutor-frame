#!/usr/bin/env python3
"""Print weight-stream timing for one decode token (FRAME_QWEN_TRACE)."""
import os
import numpy as np
os.environ["FRAME_QWEN_TRACE"] = "1"
from dsp_qwen import QwenDSP

engine = QwenDSP()
with engine:
    nxt, _ = engine.step([151644, 872, 198])
    nxt, _ = engine.step([nxt])
tr = engine.stream_trace.astype(np.int64) * 64
t0 = tr[:, :4].min()
tr -= t0
print("chunk issue0 issue1 done0 done1 wait   (k-cycles)")
for i, row in enumerate(tr):
    if i < 8 or i > len(tr) - 46:
        print(i, *(row // 1000))
