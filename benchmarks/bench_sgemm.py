"""Time hxgemm kernels and torch.matmul (TF32 off) with CUDA events.

Same method as the C++ bench: warm-up 10, then 50 individually timed runs, each preceded by
an L2 flush (write 2 x L2 bytes). Writes a CSV and, when matplotlib is available, a PNG.

usage: NVIDIA_TF32_OVERRIDE=0 python benchmarks/bench_sgemm.py --out docs/results/<date>_py_bench
"""
import argparse
import csv
import os
import statistics

import torch

import hxgemm_ext  # noqa: F401  registers torch.ops.hxgemm

torch.backends.cuda.matmul.allow_tf32 = False

SQUARES = [256, 512, 1024, 2048, 4096, 8192]
ODD = [1000, 2047, 4095]
LLM = [(16, 4096, 4096), (128, 4096, 4096), (2048, 4096, 4096)]
NAMES = {1: "k01", 2: "k02", 3: "k03", 4: "k04", 5: "k05", 0: "torch"}


def shapes():
    return [(n, n, n) for n in SQUARES] + [(n, n, n) for n in ODD] + LLM


def time_fn(fn, flush, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)  # us
    return statistics.median(ts), min(ts), max(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="path prefix for .csv/.png")
    p.add_argument("--kernels", default="1,2,3,4,5")
    p.add_argument("--naive-max", type=int, default=2048,
                   help="skip k01/k02 when max(M,N,K) exceeds this (they take seconds per call)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        print("warning: NVIDIA_TF32_OVERRIDE is not 0")

    kernels = [int(k) for k in args.kernels.split(",")]
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.empty(l2 * 2, dtype=torch.uint8, device="cuda")
    rows = []
    for (m, n, k) in shapes():
        g = torch.Generator(device="cuda").manual_seed(0)
        a = torch.rand(m, k, device="cuda", generator=g) * 2 - 1
        b = torch.rand(k, n, device="cuda", generator=g) * 2 - 1
        c = torch.empty(m, n, device="cuda")
        flops = 2.0 * m * n * k
        t_ref, _, _ = time_fn(lambda: torch.matmul(a, b, out=c), flush, args.warmup, args.iters)
        rows.append(dict(kernel="torch", M=m, N=n, K=k, median_us=t_ref, gflops=flops / t_ref / 1e3,
                         pct_torch=100.0))
        for kn in kernels:
            if kn <= 2 and max(m, n, k) > args.naive_max:
                continue
            fn = lambda kn=kn: torch.ops.hxgemm.sgemm_out(c, a, b, kn, [], 1.0, 0.0)
            med, mn, mx = time_fn(fn, flush, args.warmup, args.iters)
            rows.append(dict(kernel=NAMES[kn], M=m, N=n, K=k, median_us=med, gflops=flops / med / 1e3,
                             pct_torch=100.0 * t_ref / med))
        for r in rows[-(1 + len(kernels)):]:
            if (r["M"], r["N"], r["K"]) == (m, n, k):
                print(f"{r['kernel']:6s} {m:5d}x{n:5d}x{k:5d} {r['median_us']:11.1f} us "
                      f"{r['gflops']:8.1f} GFLOPS {r['pct_torch']:6.1f}% of torch")

    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "M", "N", "K", "median_us", "gflops", "pct_torch"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", args.out + ".csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["torch"] + [NAMES[k] for k in kernels]:
        pts = [(r["M"], r["gflops"]) for r in rows if r["kernel"] == name and r["M"] == r["N"] == r["K"]
               and r["M"] in SQUARES]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M = N = K")
    ax.set_ylabel("GFLOPS (FP32, TF32 off)")
    ax.set_title("hxgemm vs torch.matmul, RTX 3060 Ti")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(args.out + ".png", dpi=120, bbox_inches="tight")
    print("wrote", args.out + ".png")


if __name__ == "__main__":
    main()
