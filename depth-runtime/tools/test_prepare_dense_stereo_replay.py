#!/usr/bin/env python3
"""Independent fixture checks for dense selection, provenance, and transfer safety."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_dense_stereo_replay as dense


def fixture():
    rows = [{'pair': i, 'role': 'reference', 'left_file': f'pair-{i}-left.u8',
             'right_file': f'pair-{i}-right.u8', 'left_raw_ns': i*100+1,
             'right_raw_ns': i*100+2} for i in range(600)]
    frames = [{'ordinal': i, 'split': 'train' if i < 15 or 30 <= i < 45 else 'heldout',
               'left': '/local/' + rows[i*10]['left_file'],
               'right': '/local/' + rows[i*10]['right_file']} for i in range(60)]
    return {'frames': frames}, rows


class SelectionTests(unittest.TestCase):
    def test_longer_reference_windows_preserve_heldout_boundaries(self):
        manifest, rows = fixture()
        windows = dense.select_windows(manifest, rows, 30)
        self.assertTrue(all(len(w['pairs']) == 30 for w in windows))
        self.assertEqual(windows[2]['pairs'][-1]['pair'], 329)
        with self.assertRaisesRegex(ValueError, 'crosses held-out boundary'):
            dense.select_windows(manifest, rows, 60)
        for size in (True, 2, 61, 12.5):
            with self.assertRaises(ValueError):
                dense.select_windows(manifest, rows, size)

    def test_interleaved_after_selection_uses_cycles_and_preserves_boundary(self):
        manifest,rows=fixture();combined=[]
        for r in rows:
            ref=dict(r,cycle=r['pair']);combined.append(ref)
            combined.append(dict(r,pair=10000+r['pair'],cycle=r['pair'],role='after',
                left_file=f"after-{r['pair']}-l.u8",right_file=f"after-{r['pair']}-r.u8",
                left_raw_ns=r['left_raw_ns']+40,right_raw_ns=r['right_raw_ns']+40))
        windows=dense.select_windows(manifest,combined,30,include_after=True)
        self.assertTrue(all(len(w['pairs'])==60 for w in windows))
        self.assertEqual([r['role'] for r in windows[0]['pairs'][:4]],['reference','after']*2)
        # A dropped after capture stays explicit, never replaced by lit pixels.
        dropped=[r for r in combined if r['pair']!=10000]
        first=dense.select_windows(manifest,dropped,30,include_after=True)[0]
        self.assertEqual(len(first['pairs']),59);self.assertEqual(first['missing_after_reference_pair_ids'],[0])
        with self.assertRaises(ValueError):dense.select_windows(manifest,combined,31,include_after=True)
        combined[1]['left_raw_ns']=combined[300]['left_raw_ns']+1
        with self.assertRaises(ValueError):dense.select_windows(manifest,combined,30,include_after=True)

    def test_frozen_selection(self):
        manifest, rows = fixture()
        windows = dense.select_windows(manifest, rows)
        self.assertEqual([w['anchor_v2_ordinal'] for w in windows], [0, 10, 30, 35, 40])
        self.assertEqual([[p['pair'] for p in w['pairs']] for w in windows],
                         [list(range(i, i+12)) for i in (0, 100, 300, 350, 400)])
        self.assertTrue(all(all(p['role'] == 'reference' for p in w['pairs']) for w in windows))

    def test_path_traversal_rejected(self):
        for value in ('../steal', '/tmp/a', 'x/y', 'x\\y', 'a\nb', '..'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                dense.safe_name(value)

    def test_duplicate_pair_rejected(self):
        m, rows = fixture(); rows[20]['pair'] = rows[19]['pair']
        with self.assertRaises(ValueError): dense.select_windows(m, rows)

    def test_missing_role_rejected(self):
        m, rows = fixture(); del rows[20]['role']
        with self.assertRaises(ValueError): dense.select_windows(m, rows)

    def test_training_boundary_rejected(self):
        m, rows = fixture(); m['frames'][15]['left'] = '/local/' + rows[105]['left_file']; m['frames'][15]['right'] = '/local/' + rows[105]['right_file']
        with self.assertRaisesRegex(ValueError, 'crosses held-out boundary'): dense.select_windows(m, rows)

    def test_nonreference_filter_and_split_preservation(self):
        m, rows = fixture()
        rows.insert(1, dict(rows[0], pair=9999, role='lit', left_file='lit-l.u8', right_file='lit-r.u8'))
        windows = dense.select_windows(m, rows)
        self.assertEqual(windows[0]['pairs'][1]['pair'], 1)
        m['frames'][30]['split'] = 'heldout'
        with self.assertRaises(ValueError): dense.select_windows(m, rows)

    def test_materialize_requires_present_and_records_unknown_identity(self):
        m, rows = fixture()
        with tempfile.TemporaryDirectory() as directory, patch.object(dense, 'RAW_SIZE', 8):
            root = Path(directory)
            for f in m['frames']:
                for side in ('left','right'):
                    f[side] = str(root / Path(f[side]).name)
            (root/'pairs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            (root/'observations.json').write_text(json.dumps([{'ordinal': 0, 'tag': 1}, {'ordinal': 15, 'tag': 2}]))
            for name in ('rectifier.py','calib.json'): (root/name).write_text('test')
            m.update(pairs=str(root/'pairs.jsonl'), observations=str(root/'observations.json'),
                     rectifier_source=str(root/'rectifier.py'), calibration=str(root/'calib.json'), tag_size_m=.12)
            manifest = root/'manifest.json'; manifest.write_text(json.dumps(m))
            local = root/'existing'; local.mkdir()
            (local/'pair-1-left.u8').write_bytes(b'12345678')
            plan_path = dense.build(manifest, root/'out', 'frame:dataset/ir/', local_source_dirs=[local])
            with self.assertRaisesRegex(ValueError, 'missing/invalid'): dense.materialize(plan_path)
            plan = json.loads(plan_path.read_text())
            self.assertEqual(plan['files']['pair-1-left.u8']['local_path'], str((local/'pair-1-left.u8').resolve()))
            self.assertEqual(plan['files']['pair-1-left.u8']['known_local_sha256'], dense.sha(local/'pair-1-left.u8'))
            for info in plan['files'].values():
                p = Path(info['local_path']); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'12345678')
            out = dense.materialize(plan_path)
            dm = json.loads((out/'window-0.json').read_text())
            obs = json.loads((out/'window-0-observations.json').read_text())
            self.assertEqual(dm['frames'][0]['source_pair_id'], 0)
            self.assertEqual(dm['frames'][0]['original_v2_ordinal'], 0)
            self.assertEqual(obs[0]['ordinal'], 0)
            self.assertEqual(obs[0]['source_pair_id'], 0)
            self.assertEqual(len(obs), 1)  # heldout tag remains external
            self.assertEqual(dm['frames'][0]['left_identity_status'], 'size_ok_remote_hash_unknown')
            self.assertFalse(json.loads((out/'integrity.json').read_text())['remote_content_verified'])


if __name__ == '__main__': unittest.main()
