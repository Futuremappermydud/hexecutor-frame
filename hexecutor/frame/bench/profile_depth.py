#!/usr/bin/env python3
"""Profile cold/warm direct HVX inference without changing the transport."""
import cProfile
import pstats
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'frame'),str(ROOT/'tools')]
import run_frame
if __name__=='__main__':
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True)
    sys.argv=['run_frame.py','--smoke','--compute-engine','hmx','--repeat','3','--output',str(output)]
    prof=cProfile.Profile()
    try:prof.runcall(run_frame.main)
    finally:
        prof.dump_stats(str(output/'profile.pstats'))
        with (output/'profile.txt').open('w') as stream:
            pstats.Stats(prof,stream=stream).strip_dirs().sort_stats('tottime').print_stats(60)
