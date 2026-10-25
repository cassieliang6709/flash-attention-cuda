"""Forward-pass benchmark over seq_len x head_dim. Writes one raw CSV per run.

    python bench/bench.py                       # default sweep
    python bench/bench.py --impls fa2 sdpa_flash --seq 1024 4096 --dims 64

Timing: CUDA events around each call, after warmup; L2 is flushed before
every timed call (writes a 256 MB buffer) so K/V never start out hot in L2
from the previous iteration. We report the median and the 20th/80th percentile.

Memory: extra_mem_MiB = peak allocated during one call minus what was
allocated just before it (q, k, v and the L2-flush buffer), i.e. the output
O plus any workspace (naive's [B*H, N, N] S matrix, fa1's fp32 O/l/m).

TFLOP/s counts the two matmuls only: 4 * B * H * N^2 * D (non-causal),
the same convention as the FlashAttention papers. Softmax exps are not counted.
"""

import argparse
import csv
import datetime as dt
import math
import os
import statistics
import subprocess

import torch

from flash_cuda import IMPLS
from flash_cuda.reference import REF_IMPLS

ALL = {**IMPLS, **REF_IMPLS}
FP32_UNSUPPORTED = {"sdpa_flash"}
# naive.cu keeps one fp32 [B*H, N, N] score matrix (softmax is in place);
# torch_naive holds ~12 bytes per score at peak (fp16 S, fp32 copy, fp32 P, fp16 P).
S_BYTES_PER_ELEM = {"naive": 4, "torch_naive": 12}
S_MATRIX_LIMIT_BYTES = 10 * 2**30

_l2_flush = None


def flush_l2():
    global _l2_flush
    if _l2_flush is None:
        _l2_flush = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
    _l2_flush.zero_()


def time_ms(fn, warmup, iters):
    # The first (probe) call also measures memory. Slow configs (e.g. fa1 at
    # N=16k is a single sequential loop per head) would take minutes at full
    # iteration count, so if the probe is slow we back off.
    flush_l2()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    extra_mib = (torch.cuda.max_memory_allocated() - base) / 2**20
    if start.elapsed_time(end) > 1000:
        warmup, iters = 1, 3
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        flush_l2()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    q = lambda p: times[min(len(times) - 1, int(p * len(times)))]
    return statistics.median(times), q(0.2), q(0.8), extra_mib


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impls", nargs="+", default=list(ALL))
    ap.add_argument("--seq", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--dims", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--dtypes", nargs="+", default=["fp16", "fp32"])
    ap.add_argument("--batch", type=int, default=4)
    # Fixed total model width B*H*D-ish: heads = 2048 // D (64 -> 32, 128 -> 16).
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gpu = torch.cuda.get_device_name()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or os.path.join(
        "results", f"bench_{gpu.replace(' ', '_')}_{stamp}.csv"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dtypes = {"fp16": torch.float16, "fp32": torch.float32}

    fields = ["impl", "dtype", "B", "H", "N", "D", "ms_median", "ms_p20", "ms_p80",
              "tflops", "extra_mem_MiB", "status", "gpu", "torch", "cuda", "git_rev", "time"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for dname in args.dtypes:
            dtype = dtypes[dname]
            for D in args.dims:
                H = args.width // D
                for N in args.seq:
                    B = args.batch
                    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=dtype) for _ in range(3))
                    flops = 4 * B * H * N * N * D
                    for name in args.impls:
                        row = dict(impl=name, dtype=dname, B=B, H=H, N=N, D=D, gpu=gpu,
                                   torch=torch.__version__, cuda=torch.version.cuda,
                                   git_rev=git_rev(), time=stamp)
                        s_bytes = B * H * N * N * S_BYTES_PER_ELEM.get(name, 0)
                        if dtype == torch.float32 and name in FP32_UNSUPPORTED:
                            row["status"] = "unsupported_dtype"
                        elif s_bytes > S_MATRIX_LIMIT_BYTES:
                            row["status"] = "skipped_S_too_big"
                        else:
                            try:
                                med, p20, p80, mem = time_ms(lambda: ALL[name](q, k, v),
                                                        args.warmup, args.iters)
                                row.update(ms_median=f"{med:.4f}", ms_p20=f"{p20:.4f}",
                                           ms_p80=f"{p80:.4f}",
                                           tflops=f"{flops / (med * 1e-3) / 1e12:.2f}",
                                           extra_mem_MiB=f"{mem:.1f}",
                                           status="ok")
                            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                                row["status"] = f"error:{type(e).__name__}"
                                torch.cuda.empty_cache()
                        w.writerow(row)
                        f.flush()
                        print(f"{name:13s} {dname} N={N:6d} D={D:3d} "
                              f"{row.get('ms_median', '-'):>10s} ms  "
                              f"{row.get('tflops', '-'):>7s} TFLOP/s  "
                              f"{row.get('extra_mem_MiB', '-'):>9s} MiB  {row['status']}")
                    del q, k, v
                    torch.cuda.empty_cache()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
