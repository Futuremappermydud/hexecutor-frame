#!/bin/sh
# One-command A/B of prepared-reference engine variants on the Frame.
#
#   sh frame/bench/ab.sh [-q|-p] [-n] [-r REPEAT] [-t TAG] VARIANT...
#
# VARIANT is a name below or NAME:KEY=VAL[,KEY=VAL...] (extra QDS_ flags over
# the base set, e.g. try:QDS_PREPARED_THREADS=3,QDS_PREPARED_FUSE=1).
# For each variant: build (frame/build_resident_dsp.sh), keep the engine and
# bootstrap under frame/build/ab/TAG/, copy them to the Frame, run
# verify_l2_short (primary command 0..1) and then verify_prepared_full_model
# for both models (smoke + random, --repeat REPEAT). Reports land in
# frame/build/ab/TAG/ (fetched back) and the table is printed by ab_table.py.
#   -q  primary smoke only (no alternate, no random): a fast timing pass
#   -p  primary only, smoke + random (no alternate): the primary gate
#   -n  no build: reuse frame/build/ab/TAG/eng-NAME.elf
# Gate (PLAN_100MS.md): every report "passed": exact on both models, lease
# and guards clean. A failing variant stops the run.
set -eu
cd "$(dirname "$0")/../.."
QUICK=0 BUILD=1 REPEAT=2 TAG=$(date +%Y%m%d-%H%M)
while getopts qpnr:t: o; do
  case $o in q) QUICK=1;; p) QUICK=2;; n) BUILD=0;; r) REPEAT=$OPTARG;; t) TAG=$OPTARG;; *) exit 2;; esac
done
shift $((OPTIND-1))
[ $# -ge 1 ] || { sed -n 2,17p "$0"; exit 2; }
REMOTE=${AB_REMOTE:-frame}
RDIR=${AB_REMOTE_DIR:-frame-dsp/depth-model/wave1-20261101}
PY=${AB_PYTHON:-/home/steamos/frame-dsp/depth-model/.venv/bin/python}
OUT=frame/build/ab/$TAG
mkdir -p "$OUT"

base() {
  echo QDS_BLOCKED_GLUE=1 QDS_DYNAMIC_TCM=1 QDS_FALLBACK_TCM=2 QDS_FAST_SPILL=1 QDS_NATIVE_1X1=1 \
       QDS_NATIVE_3X3=1 QDS_NATIVE_DIRECT_RANGE=1 QDS_NATIVE_PREPARED=1 QDS_PREPARED_ELEMENTWISE=1 \
       QDS_PREPARED_REFERENCE=1 QDS_PRESERVE_VTCM=1 QDS_VECTORIZE=0 QDS_PREPARED_PROFILE=1
}
flags() {
  case $1 in
    *:*) echo "$(base) $(echo "${1#*:}" | tr , ' ')";;
    base|ctl) base;;
    thr3) echo "$(base) QDS_PREPARED_THREADS=3";;
    fuse) echo "$(base) QDS_PREPARED_FUSE=1";;
    thr3fuse) echo "$(base) QDS_PREPARED_THREADS=3 QDS_PREPARED_FUSE=1";;
    hmx) echo "$(base) QDS_PREPARED_THREADS=3 QDS_PREPARED_FUSE=1 QDS_PREPARED_HMX=1";;
    hw) echo "$(base) QDS_PREPARED_THREADS=3 QDS_PREPARED_FUSE=1 QDS_PREPARED_HMX=1 QDS_EXTRA_CFLAGS=-DNPRG2_HW_ROUNDING=1";;
    l2hmx) echo "$(base) QDS_PREPARED_THREADS=3 QDS_PREPARED_FUSE=1 QDS_PREPARED_HMX=1 QDS_L2_CACHE=1";;
    l2hw) echo "$(base) QDS_PREPARED_THREADS=3 QDS_PREPARED_FUSE=1 QDS_PREPARED_HMX=1 QDS_L2_CACHE=1 QDS_EXTRA_CFLAGS=-DNPRG2_HW_ROUNDING=1+-DNPRG2_SLIDE3=1";;
    l2hw0) echo "$(base) QDS_PREPARED_THREADS=3 QDS_PREPARED_FUSE=1 QDS_PREPARED_HMX=1 QDS_L2_CACHE=1 QDS_EXTRA_CFLAGS=-DNPRG2_HW_ROUNDING=1";;
    l2) echo "$(base) QDS_L2_CACHE=5";;
    l2fuse) echo "$(base) QDS_L2_CACHE=5 QDS_PREPARED_FUSE=1";;
    *) echo "unknown variant $1" >&2; exit 2;;
  esac
}

names=""
for v in "$@"; do
  n=${v%%:*}; names="$names $n"
  if [ $BUILD = 1 ]; then
    f=$(flags "$v")
    echo "== build $n: $f"
    log=$OUT/build-$n.log
    if ! env $f sh frame/build_resident_dsp.sh >"$log" 2>&1; then tail -30 "$log"; exit 1; fi
    eng=frame/$(sed -n 's/^Experimental engine: \([^;]*\);.*/\1/p' "$log")
    boot=frame/$(sed -n 's/.*bootstrap: \(.*\)$/\1/p' "$log")
    [ -f "$eng" ] && [ -f "$boot" ] || { echo "no engine/bootstrap in $log" >&2; exit 1; }
    cp "$eng" "$OUT/eng-$n.elf"; cp "$boot" "$OUT/boot-$n.elf"
    echo "$f" >"$OUT/flags-$n.txt"
    git rev-parse HEAD >"$OUT/rev-$n.txt"; git status --short >>"$OUT/rev-$n.txt"
  fi
  shasum -a 256 "$OUT/eng-$n.elf" "$OUT/boot-$n.elf"
done

# Harness sources follow the engine (host guards, profile decoders).
ssh "$REMOTE" "mkdir -p $RDIR/$OUT $RDIR/results/ab/$TAG"
rsync -c frame/*.py "$REMOTE:$RDIR/frame/"
rsync -c frame/bench/*.py "$REMOTE:$RDIR/frame/bench/"
rsync -c "$OUT"/*.elf "$REMOTE:$RDIR/$OUT/"
for m in primary alternate; do rsync -c recovered/$m/hw_*.json "$REMOTE:$RDIR/recovered/$m/" 2>/dev/null || true; done

for n in $names; do
  common="--engine $OUT/eng-$n.elf --bootstrap $OUT/boot-$n.elf"
  expect=""; if grep -q "NPRG2_HW_ROUNDING=1" "$OUT/flags-$n.txt" 2>/dev/null; then expect="--expect hw"; fi
  R=results/ab/$TAG
  echo "== $n: short (primary 0..1)"
  ssh "$REMOTE" "cd $RDIR && $PY frame/bench/verify_l2_short.py --model primary --first 0 --last 1 --case smoke $common \
      --prepared-program frame/native_program/examples/primary-v2.nprg --report $R/${n}_short.json" | tail -3
  if [ $QUICK = 1 ]; then models=primary; cases="--case smoke"; elif [ $QUICK = 2 ]; then models=primary; cases=""; else models="primary alternate"; cases=""; fi
  for m in $models; do
    echo "== $n: full $m"
    ssh "$REMOTE" "cd $RDIR && $PY frame/bench/verify_prepared_full_model.py --model $m $cases --repeat $REPEAT $common \
        --prepared-program frame/native_program/examples/$m-v2.nprg --report $R/${n}_$m.json $expect" \
      | python3 -c 'import sys,json
for l in sys.stdin:
  try: d=json.loads(l)
  except Exception: print(l,end=""); continue
  print(" ",d["case"],d["run"],"passed" if d["passed"] else "FAILED","%.1f ms"%d["dsp_ms"])' || {
        rsync -r "$REMOTE:$RDIR/$R/" "$OUT/" >/dev/null 2>&1 || true; echo "$n $m FAILED" >&2; exit 1; }
  done
done
rsync -r "$REMOTE:$RDIR/results/ab/$TAG/" "$OUT/"
python3 frame/bench/ab_table.py "$OUT" $names | tee "$OUT/table.txt"
