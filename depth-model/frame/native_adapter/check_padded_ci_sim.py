#!/usr/bin/env python3
"""Independent integer oracle and CPU-prefix comparison for padded-CI leaf."""
import json
import argparse
import hashlib
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--check-poison', action='store_true')
args = parser.parse_args()
fixture = (HERE / 'padded-ci.bin').read_bytes()
output = (HERE / 'padded-ci-out.bin').read_bytes()
h = struct.unpack_from('<32I', fixture)
if h[0] != 0x31585151 or h[1] not in (115, 3416) or h[14] != 32:
    raise ValueError('bad padded-CI fixture header')
_, sequence, ih, iw, imh, imw, islice, icount, oh, ow, omh, omw, oslice, ocount, ci32, co, wfmt, wb, fb = h[:19]
xpad_h, xpad_w, ypad_h, ypad_w, input_zp, original_ci, original_wb, ref_bytes = h[19:27]
if wfmt != 1 or original_wb != original_ci * co or wb != 32 * co:
    raise ValueError('unexpected Xpb expansion geometry')
offset = 128
xb = fixture[offset:offset + icount * 2048]; offset += len(xb)
yb = fixture[offset:offset + ocount * 2048]; offset += len(yb)
weights32 = fixture[offset:offset + wb]; offset += wb
fi_raw = fixture[offset:offset + fb]; offset += fb
weights_original = fixture[offset:offset + original_wb]; offset += original_wb
cpu_ref = fixture[offset:offset + ref_bytes]; offset += ref_bytes
if offset != len(fixture):
    raise ValueError('fixture has truncated or trailing bytes')

out_header = output[:128]
out_offset = 128
got = output[out_offset:out_offset + len(yb)]; out_offset += len(yb)
cpu_output = output[out_offset:out_offset + ref_bytes]; out_offset += ref_bytes
out_weights = output[out_offset:out_offset + original_wb]; out_offset += original_wb
out_fi = output[out_offset:out_offset + fb]; out_offset += fb
if (out_header != fixture[:128] or len(got) != len(yb) or out_offset != len(output) or
        cpu_output != cpu_ref or out_weights != weights_original or out_fi != fi_raw):
    raise ValueError('sim output packet/prefix source mismatch')
poison_identical = None
poison_sha256 = None
if args.check_poison:
    poisoned = (HERE / 'padded-ci-poison-out.bin').read_bytes()
    poison_identical = poisoned == output
    poison_sha256 = hashlib.sha256(poisoned).hexdigest()
    if not poison_identical:
        raise ValueError('poisoned physical input tail changed native output')
fi = struct.unpack('<%di' % (fb // 4), fi_raw)
effective_h = min(((oh + 7) // 8) * 8, omh - ypad_h, imh - xpad_h)
effective_w = min(((ow + 7) // 8) * 8, omw - ypad_w, imw - xpad_w)
min_delta, max_delta, different, cpu_diff, first = 255, -255, 0, 0, None
active_blocks = set()
for y in range(effective_h):
    for x in range(effective_w):
        iy, ix = y + xpad_h, x + xpad_w
        input_block = ((iy // 8) * (imw // 8) + ix // 8) + islice
        output_base = ((y // 8) * (omw // 8) + x // 8) * (co // 32) + oslice
        for out_ch in range(co):
            output_block = output_base + out_ch // 32
            active_blocks.add(output_block)
            acc = fi[(out_ch // 32) * 64 + 32 + out_ch % 32]
            for in_ch in range(original_ci):
                value = xb[input_block * 2048 + (iy % 8) * 256 + (ix % 8) * 32 + in_ch]
                weight_index = ((out_ch // 32) * (original_ci // 4) + in_ch // 4) * 128
                weight_index += (out_ch % 32) * 4 + in_ch % 4
                weight = weights_original[weight_index]
                acc += (value - input_zp) * (weight if weight < 128 else weight - 256)
            multiplier_word = fi[(out_ch // 32) * 64 + out_ch % 32]
            multiplier = multiplier_word & 0xffff
            preshift = 34 - ((multiplier >> 10) & 31) - 14
            expected = max(0, min(255,
                (((acc >> preshift) * (1024 + (multiplier & 1023)) + 8192) >> 14)))
            lane = output_block * 2048 + (y % 8) * 256 + (x % 8) * 32 + out_ch % 32
            actual = got[lane]
            reference = cpu_ref[lane]
            delta = actual - expected
            min_delta, max_delta = min(min_delta, delta), max(max_delta, delta)
            different += delta != 0
            cpu_diff += actual != reference
            if delta and first is None:
                first = {'y': y, 'x': x, 'out': out_ch, 'acc': int(acc),
                         'multiplier_word': int(multiplier_word), 'bias': int(fi[(out_ch // 32) * 64 + 32 + out_ch % 32]),
                         'expected': int(expected), 'actual': int(actual),
                         'cpu': int(reference), 'delta': int(delta)}

outside = [i for i in range(ocount) if i not in active_blocks and
           got[i * 2048:(i + 1) * 2048] != yb[i * 2048:(i + 1) * 2048]]
passed = min_delta >= 0 and max_delta <= 1 and not outside
report = {'passed': passed, 'offline_only': True, 'sequence': sequence,
          'input_ci': original_ci, 'native_padded_ci': ci32, 'output_co': co,
          'geometry': [oh, ow], 'effective_geometry': [effective_h, effective_w],
          'slices': [islice, oslice], 'native_minus_integer_bound': [min_delta, max_delta],
          'native_different_values': different, 'native_different_from_cpu': cpu_diff,
          'first_difference': first, 'inactive_output_blocks_changed': outside,
          'expanded_weights_bytes': len(weights32), 'input_padding_zero_weighted': original_ci < ci32,
          'output_sha256': hashlib.sha256(output).hexdigest(),
          'poisoned_tail_full_output_identical': poison_identical,
          'poisoned_output_sha256': poison_sha256}
report_path = HERE / f'padded-ci-{sequence}-sim.json'
report_path.write_text(json.dumps(report, indent=2) + '\n')
if not passed:
    print(f'FAIL padded CI sequence {sequence}: {json.dumps(report)}')
    raise SystemExit(1)
print(f"PASS padded CI{original_ci}->32 sequence {sequence}: delta [{min_delta},{max_delta}], CPU-different {cpu_diff}, output guards intact")
