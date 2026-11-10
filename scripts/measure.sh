#!/usr/bin/env bash
# usage: scripts/measure.sh <date> <bench args...>   (desktop, repo root)
# Runs hxgemm_bench with a 1 s clock log next to it. Refuses to measure without a clock log:
# waits (up to 5 s) for the first clock sample before running, and takes one more after.
set -euo pipefail
d=$1; shift
log=docs/results/${d}_clocks.csv
: > "$log"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -l 1 > "$log" &
mon=$!
trap 'kill $mon 2>/dev/null || true' EXIT
for _ in $(seq 50); do [ "$(wc -l < "$log")" -ge 2 ] && break; sleep 0.1; done
[ "$(wc -l < "$log")" -ge 2 ] || { echo "no clock samples after 5 s, not measuring: $log" >&2; exit 1; }
NVIDIA_TF32_OVERRIDE=0 ./build/hxgemm_bench "$@" --csv docs/results/${d}_final.csv | tee docs/results/${d}_final.txt
sleep 1.2
kill $mon 2>/dev/null || true; trap - EXIT
[ "$(wc -l < "$log")" -ge 3 ] || { echo "clock log missing or too short: $log" >&2; exit 1; }
echo "clock log: $log ($(($(wc -l < "$log") - 1)) samples; SM MHz: $(tail -n +2 "$log" | cut -d, -f2 | sort | uniq -c | tr -s ' ' | tr '\n' ';'))"
