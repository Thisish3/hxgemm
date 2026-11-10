"""Collect key metrics from scripts/ncu_collect.sh outputs into one CSV.

usage: python3 scripts/ncu_summary.py <date> k01 k02 ... > docs/results/<date>_ncu_summary.csv
"""
import csv
import sys

KEYS = [
    ("Duration", "duration_ns"),
    ("Compute (SM) Throughput", "sm_throughput_pct"),
    ("Memory Throughput", "mem_throughput_pct"),
    ("L1/TEX Cache Throughput", "l1tex_throughput_pct"),
    ("DRAM Throughput", "dram_throughput_pct"),
    ("Executed Ipc Active", "ipc_active"),
    ("Registers Per Thread", "regs_per_thread"),
    ("Achieved Occupancy", "achieved_occupancy_pct"),
    ("Warp Cycles Per Issued Instruction", "warp_cycles_per_issue"),
    ("dram__bytes_read.sum", "dram_read_bytes"),
    ("lts__t_bytes.sum", "l2_bytes"),
    ("l1tex__t_bytes.sum", "l1tex_bytes"),
    ("smsp__inst_executed.sum", "inst_executed"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "smem_ld_conflicts"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "smem_st_conflicts"),
    ("smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio", "stall_long_sb"),
    ("smsp__average_warp_latency_issue_stalled_mio_throttle.ratio", "stall_mio"),
    ("smsp__average_warp_latency_issue_stalled_lg_throttle.ratio", "stall_lg"),
    ("smsp__average_warp_latency_issue_stalled_barrier.ratio", "stall_barrier"),
    ("SM Frequency", "sm_freq_hz"),
]


def main():
    d, kernels = sys.argv[1], sys.argv[2:]
    w = csv.writer(sys.stdout)
    w.writerow(["kernel", "kernel_name"] + [k for _, k in KEYS])
    for k in kernels:
        vals, name = {}, ""
        with open(f"docs/results/{d}_{k}_ncu.csv", newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("Kernel Name", name)
                m = row.get("Metric Name")
                if m and m not in vals:
                    vals[m] = row.get("Metric Value", "").replace(",", "")
        w.writerow([k, name.split("(")[0]] + [vals.get(m, "") for m, _ in KEYS])


if __name__ == "__main__":
    main()
