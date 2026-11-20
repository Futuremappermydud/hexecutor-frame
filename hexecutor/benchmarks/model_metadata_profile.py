# SPDX-License-Identifier: LGPL-3.0-or-later
"""Decode the Model vendor's 352-byte NNETPerfInfo records (Oct 30 firmware).

Metadata offsets were recovered from libhexagon_skel.so's
QnnModelBase::parseNodeQnnProfileEvent and HTPEventTracer::log_profiling_delegate.
QNN event IDs 5001..5005 are named in the local SDK QnnHtpProfile.h.
Node values are cycles; the two ExecuTorch wrapper values are microseconds.
The background op ID array is truncated by the vendor to ten entries.
"""
import argparse
import json
from pathlib import Path
import re
import struct


def decode(data):
    if len(data) % 352:
        raise ValueError('NNETPerfInfo buffer must contain complete 352-byte records')
    rows = []
    for offset in range(0, len(data), 352):
        record = data[offset:offset + 352]
        value = struct.unpack_from('<i', record, 256)[0]
        if value == -1:
            return rows
        name = record[:256].split(b'\0', 1)[0].decode('utf8')
        if not name:
            raise ValueError('missing profile sentinel or empty event name')
        row = {'name': name, 'value': value, 'unit': 'cycles' if name.endswith('(cycles)') else 'us'}
        if row['unit'] == 'cycles':
            match = re.search(r'OpId_(\d+)', name)
            row['op_id'] = int(match[1]) if match else None
            # The vendor's node_id field at +264 is often zero; the name
            # carries the original QNN ID reliably for these two contexts.
            row.update(zip(('cycles', 'wait_cycles', 'overlap_cycles',
                            'wait_overlap_cycles', 'resource_mask'),
                           struct.unpack_from('<5Q', record, 272)))
            row['critical_background_op_ids'] = [v for v in struct.unpack_from('<10I', record, 312) if v]
        rows.append(row)
    raise ValueError('profile buffer has no terminating sentinel; it may have overrun')


def summary(rows):
    nodes = [r for r in rows if r['unit'] == 'cycles']
    return {'events': len(rows), 'nodes': len(nodes),
            'sum_node_cycles': sum(r['cycles'] for r in nodes),
            'sum_wait_cycles': sum(r['wait_cycles'] for r in nodes),
            'sum_overlap_cycles': sum(r['overlap_cycles'] for r in nodes),
            'sum_wait_overlap_cycles': sum(r['wait_overlap_cycles'] for r in nodes),
            'nodes_with_overlap': sum(r['overlap_cycles'] > 0 for r in nodes),
            'top_nodes': sorted(nodes, key=lambda r: -r['cycles'])[:20],
            'wrapper_events': [r for r in rows if r['unit'] == 'us'],
            'caveat': 'Logical QNN op aggregates, not individual compiled tiles or a timestamped hardware timeline. Overlap sums are not additive wall time.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = decode(args.input.read_bytes())
    report = {'summary': summary(rows), 'events': rows}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))
