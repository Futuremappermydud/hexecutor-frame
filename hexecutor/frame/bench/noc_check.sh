#!/bin/sh
# NSP NoC clock check: read-only GCC and Turing clock dumps, DMA bandwidth at the
# current clock, then again with cdsp_nocclk mhz=300 (restored on rmmod).
# Run from the frame directory with SteamVR idle; needs sudo for the module.
# Stop after the read-only steps with: bench/noc_check.sh readonly
set -eu
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
OUT=build/noc-check-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
make HEX_CC=clang HEX_LD=ld.lld build/qd_loader.elf build/clk_dump.elf build/bench_dma.elf
[ -f kmod/cdsp_nocclk.ko ] || kmod/build.sh

lsmod | grep -q '^cdsp_nocclk' && sudo rmmod cdsp_nocclk
sudo insmod kmod/cdsp_nocclk.ko
sudo dmesg | grep cdsp_nocclk | tail -2 | tee "$OUT/gcc-readonly.txt"
sudo rmmod cdsp_nocclk
$PY bench/run_clk_dump.py "$OUT/turing-clocks.json"
[ "${1:-}" = readonly ] && { echo "read-only results in $OUT"; exit 0; }

[ -e /dev/cdsp_mem ] || kmod/load.sh  # 'big' DMA cases use the cdsp_mem buffer
$PY bench/run_dma.py big | tee "$OUT/dma-before.txt"
sudo insmod kmod/cdsp_nocclk.ko mhz=300
sudo dmesg | grep cdsp_nocclk | tail -4 | tee "$OUT/gcc-300.txt"
$PY bench/run_dma.py big | tee "$OUT/dma-300.txt" || true
sudo rmmod cdsp_nocclk
sudo dmesg | grep cdsp_nocclk | tail -2 | tee "$OUT/gcc-restored.txt"
echo "results in $OUT"
