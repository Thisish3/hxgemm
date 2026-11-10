#!/usr/bin/env bash
# usage: scripts/sweep.sh <date>   (run on the desktop, from the repo root)
set -euo pipefail
d=$1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -l 1 > docs/results/${d}_sweep_clocks.csv &
mon=$!
trap 'kill $mon' EXIT
NVIDIA_TF32_OVERRIDE=0 ./build/hxgemm_bench --kernel k04 --kernel k05 --kernel cublas --sweep \
  --shape 4096,4096,4096 --shape 16,4096,4096 --shape 128,4096,4096 --shape 2048,4096,4096 \
  --csv docs/results/${d}_sweep.csv | tee docs/results/${d}_sweep.txt
