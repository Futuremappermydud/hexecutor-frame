
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

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str((PATHS.native)))
import hexecutor.offline.frame_transport as transport
class EyeReplacement(unittest.TestCase):
    def exercise(self,outputs,expected,states=None):
        clock=[0.0];position=[0]
        def read(*args,**kwargs):
            value=outputs[min(position[0],len(outputs)-1)];position[0]+=1
            return SimpleNamespace(stdout=value)
        def sleep(seconds):clock[0]+=seconds
        with patch.object(transport.subprocess,'run',side_effect=read), \
             patch.object(transport,'_process_state',side_effect=lambda pid:(states or {}).get(int(pid),'S')), \
             patch.object(transport.time,'monotonic',side_effect=lambda:clock[0]), \
             patch.object(transport.time,'sleep',side_effect=sleep):
            self.assertEqual(transport._wait_for_single_eye(0.3),expected)
        self.assertLessEqual(clock[0],0.3)

    def test_supervisor_startup_gap(self):self.exercise(['','17'],17)
    def test_supervisor_transient_duplicate(self):self.exercise(['17 18','18'],18)
    def test_missing_or_persistent_duplicate_rejected(self):
        self.exercise([''],None);self.exercise(['17 18'],None)
    def test_zombie_not_accepted(self):self.exercise(['17 18'],18,{17:'Z'})


if __name__=='__main__':unittest.main()
