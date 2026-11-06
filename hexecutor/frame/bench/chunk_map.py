"""Read-only: which host 2 MB chunk does each DSP IOVA chunk resolve to?"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cdsp_mem_check import chunk_map

bad=chunk_map()
print(json.dumps({'bad_chunks':{str(k):v for k,v in bad.items()},'n_bad':len(bad)}))
sys.exit(1 if bad else 0)
