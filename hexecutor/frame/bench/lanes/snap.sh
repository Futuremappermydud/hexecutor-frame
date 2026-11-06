#!/bin/sh
# usage: snap.sh NAME : sources-only snapshot of the candidate
M=/Users/owenelliott/Documents/projects/spatial-stuff/depth-model
cd /Users/owenelliott/Documents/projects/spatial-stuff/depth-model && find frame -type f \( -name '*.c' -o -name '*.h' -o -name '*.S' -o -name '*.inc' -o -name '*.py' -o -name '*.sh' -o -name 'Kbuild' \) -not -path 'frame/build/*' -size -2000k | tar czf $M/frame/build/ab/planes-20261105/source-$1.tar.gz -T - && ls -la $M/frame/build/ab/planes-20261105/source-$1.tar.gz
