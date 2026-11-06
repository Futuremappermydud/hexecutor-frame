#!/usr/bin/env python3
"""Generate Qwen3-0.6B's static NeoX RoPE coefficients for 257 positions."""
import math
from pathlib import Path
import struct

target=Path(__file__).with_name('rope_table.f32')
with target.open('wb') as output:
    for position in range(257):
        for index in range(64):
            angle=position/(1000000.0 ** (index/64.0))
            output.write(struct.pack('<ff',math.cos(angle),math.sin(angle)))
print(target,target.stat().st_size)
