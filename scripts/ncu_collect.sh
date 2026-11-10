#!/usr/bin/env bash
# usage: scripts/ncu_collect.sh <date> <kernel> [tile]   (desktop, repo root) — one launch at 4096^3
# The bench runs with --check-only, so the target kernel launches exactly once; the kernel-name
# filter skips the FP64 reference kernel (ref_gpu_fp64). Clocks are left as set by
# `nvidia-smi -lgc` (Windows admin), so ncu runs with --clock-control none (LEDGER R5).
set -euo pipefail
d=$1; k=$2; tile=${3:-}
declare -A NAME=([k01]=k01_naive [k02]=k02_coalesce [k03]=k03_smem [k04]=k04_reg [k05]=k05_pipe [cublas]=gemm)
[ -n "${NAME[$k]:-}" ] || { echo "unknown kernel $k" >&2; exit 64; }
args=(--kernel "$k" --shape 4096,4096,4096 --check-only)
[ -n "$tile" ] && args+=(--tile "$tile")
NCU=${NCU:-/usr/local/cuda/bin/ncu}
out=docs/results/${d}_${k}
NVIDIA_TF32_OVERRIDE=0 "$NCU" --clock-control none -k "regex:${NAME[$k]}" --launch-count 1 \
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --section Occupancy --section LaunchStats --section WarpStateStats --section SourceCounters \
  --metrics dram__bytes_read.sum,lts__t_bytes.sum,l1tex__t_bytes.sum,smsp__inst_executed.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_mio_throttle.ratio,smsp__average_warp_latency_issue_stalled_lg_throttle.ratio,smsp__average_warp_latency_issue_stalled_barrier.ratio \
  -o "$out" -f ./build/hxgemm_bench "${args[@]}" > "${out}_ncu.log" 2>&1
rep=$(ls "${out}".ncu-rep* | head -1)
"$NCU" --import "$rep" --csv --page details --print-units base > "${out}_ncu.csv"
echo "$k: $(grep -c . "${out}_ncu.csv") rows -> ${out}_ncu.csv (report $rep)"
