"""Roofline ESTIMATE from measured times — not Nsight Compute measurements.

    python bench/roofline.py results/bench_<gpu>_<time>.csv

For every "ok" row of a bench CSV this computes, from the measured median
time and an analytical byte model:

  achieved TFLOP/s   = 4 B H N^2 D / t                      (same as bench.py)
  pct_peak_flops     = achieved / peak of the unit the kernel actually uses
  model_bytes        = HBM bytes the algorithm would move with NO cache reuse
  model_GBps         = model_bytes / t,   pct_peak_bw = model_GBps / peak
  AI_model           = FLOPs / model_bytes   vs   ridge = peak_flops / peak_bw

model_bytes is an UPPER bound on DRAM traffic (L2 hits make the real number
smaller), so pct_peak_bw is an upper bound too. ncu's dram__bytes on a
profiling machine replaces this guess.

Verdict per row:
  * l2-reuse        if pct_peak_bw > 100 % (the no-reuse model can't be right)
  * memory-bound    if AI_model < ridge and pct_peak_bw >= 50 %
  * compute-bound   if AI_model >= ridge and pct_peak_flops >= 50 %
  * neither saturated (latency / occupancy) if both percentages < 30 %
  * otherwise "mixed"
"""

import csv
import math
import os
import sys

# RTX 4090 peaks, NVIDIA Ada Lovelace architecture whitepaper (dense, boost clock).
# The scalar kernels (naive, fa1, fa2 v2a) run on FP32 CUDA cores for both
# dtypes; cuBLAS / SDPA use tensor cores for fp16 inputs.
PEAKS = {
    "NVIDIA GeForce RTX 4090": {
        "fp32_cuda_core_tflops": 82.6,
        "fp16_tensor_fp32acc_tflops": 165.2,
        "dram_GBps": 1008.0,
    }
}
SCALAR_IMPLS = {"naive", "fa1", "fa2"}
BYTES = {"fp16": 2, "fp32": 4}
BR = BC = 32  # tile sizes in csrc/fa1.cu, csrc/fa2.cu


def model_bytes(impl, dtype, B, H, N, D):
    e = BYTES[dtype]
    bh = B * H
    qkvo = 4 * N * D * e  # read Q, K, V, write O once: the IO lower bound
    if impl == "naive":
        # S write (fp32) + softmax read & write + PV read of P
        return bh * (qkvo + 16 * N * N)
    if impl == "torch_naive":
        # fp16: S write(e), read for .float()(e), write fp32(4), softmax r/w(8),
        # .to(dtype) r(4)+w(e), PV read(e). fp32: S w, softmax r/w, PV r = 16.
        per = 16 if dtype == "fp32" else 3 * e + 16 + e
        return bh * (qkvo + per * N * N)
    if impl == "fa1":
        tc = math.ceil(N / BC)
        k_v = 2 * N * D * e                # each K/V tile read once
        q = tc * N * D * e                 # Q re-read for every K/V tile
        o = 2 * tc * N * D * 4             # fp32 O read + write per (i, j)
        lm = 2 * 2 * tc * N * 4            # l and m read + write per (i, j)
        cast = N * D * (4 + e)             # final fp32 -> dtype cast
        return bh * (k_v + q + o + lm + cast)
    if impl == "fa2":
        tr = math.ceil(N / BR)
        return bh * (2 * N * D * e + 2 * tr * N * D * e)  # Q, O once; K, V per Q tile
    return bh * qkvo  # SDPA: unknown internals, use the lower bound


def main(path):
    rows = [r for r in csv.DictReader(open(path)) if r["status"] == "ok"]
    out_path = os.path.join(os.path.dirname(path),
                            os.path.basename(path).replace("bench_", "roofline_", 1))
    fields = ["impl", "dtype", "B", "H", "N", "D", "ms_median", "tflops", "peak_tflops",
              "pct_peak_flops", "model_GB", "model_GBps", "pct_peak_bw", "AI_model",
              "ridge", "verdict", "source"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            peaks = PEAKS[r["gpu"]]
            B, H, N, D = (int(r[k]) for k in ("B", "H", "N", "D"))
            t = float(r["ms_median"]) * 1e-3
            flops = 4 * B * H * N * N * D
            tensor = r["dtype"] == "fp16" and r["impl"] not in SCALAR_IMPLS
            peak_f = peaks["fp16_tensor_fp32acc_tflops" if tensor else "fp32_cuda_core_tflops"]
            peak_bw = peaks["dram_GBps"]
            mb = model_bytes(r["impl"], r["dtype"], B, H, N, D)
            tflops = flops / t / 1e12
            gbps = mb / t / 1e9
            pf, pb = 100 * tflops / peak_f, 100 * gbps / peak_bw
            ai, ridge = flops / mb, peak_f * 1e12 / (peak_bw * 1e9)
            if pb > 100:
                # Moving model_bytes in time t would exceed DRAM peak, so most of
                # the modelled re-reads must have hit L2: the no-reuse model is wrong.
                verdict = "l2-reuse(model-bytes>peak)"
            elif ai < ridge and pb >= 50:
                verdict = "memory-bound"
            elif ai >= ridge and pf >= 50:
                verdict = "compute-bound"
            elif pf < 30 and pb < 30:
                verdict = "neither-saturated"
            else:
                verdict = "mixed"
            w.writerow(dict(impl=r["impl"], dtype=r["dtype"], B=B, H=H, N=N, D=D,
                            ms_median=r["ms_median"], tflops=f"{tflops:.2f}",
                            peak_tflops=peak_f, pct_peak_flops=f"{pf:.1f}",
                            model_GB=f"{mb / 1e9:.3f}", model_GBps=f"{gbps:.0f}",
                            pct_peak_bw=f"{pb:.1f}", AI_model=f"{ai:.1f}",
                            ridge=f"{ridge:.1f}", verdict=verdict,
                            source=os.path.basename(path)))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
