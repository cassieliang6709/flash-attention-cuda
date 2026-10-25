"""Run one attention call so Nsight Compute can profile it.

    ncu ... python profile/run_one.py --impl fa2 --N 4096 --D 64

One warmup call happens first; only kernels inside the NVTX range
"profiled" are captured (ncu.sh passes --nvtx --nvtx-include profiled/).
"""

import argparse

import torch

from flash_cuda import IMPLS
from flash_cuda.reference import REF_IMPLS

ALL = {**IMPLS, **REF_IMPLS}

ap = argparse.ArgumentParser()
ap.add_argument("--impl", required=True, choices=list(ALL))
ap.add_argument("--N", type=int, default=4096)
ap.add_argument("--D", type=int, default=64)
ap.add_argument("--B", type=int, default=4)
ap.add_argument("--width", type=int, default=2048)
ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
args = ap.parse_args()

dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
H = args.width // args.D
q, k, v = (torch.randn(args.B, H, args.N, args.D, device="cuda", dtype=dtype) for _ in range(3))
fn = ALL[args.impl]
fn(q, k, v)  # warmup (outside the NVTX range, not profiled)
torch.cuda.synchronize()
with torch.cuda.nvtx.range("profiled"):
    fn(q, k, v)
torch.cuda.synchronize()
