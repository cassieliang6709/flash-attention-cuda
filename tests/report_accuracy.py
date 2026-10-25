"""Write max-abs-error numbers behind the accuracy tests to a CSV.

    python tests/report_accuracy.py      -> results/accuracy_<gpu>.csv

pytest only says pass/fail; this records the actual errors (vs a float64
reference, and PyTorch SDPA's error on the same inputs) so the README can
quote them.
"""

import csv
import os

import torch

from flash_cuda import IMPLS
from flash_cuda.reference import attention_fp64, sdpa_default

SHAPES = [(2, 4, 1000, 64), (2, 4, 1024, 128), (1, 8, 4096, 64)]

gpu = torch.cuda.get_device_name()
out = os.path.join("results", f"accuracy_{gpu.replace(' ', '_')}.csv")
os.makedirs("results", exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["impl", "dtype", "B", "H", "N", "D", "max_abs_err", "sdpa_max_abs_err", "ratio_to_sdpa"])
    for dtype, dname in [(torch.float32, "fp32"), (torch.float16, "fp16")]:
        for shape in SHAPES:
            g = torch.Generator(device="cuda").manual_seed(0)
            q, k, v = (torch.randn(shape, device="cuda", dtype=dtype, generator=g) for _ in range(3))
            ref = attention_fp64(q, k, v)
            sdpa_err = (sdpa_default(q, k, v).double() - ref).abs().max().item()
            for name, fn in IMPLS.items():
                if name == "naive" and shape[2] > 2048:
                    continue
                err = (fn(q, k, v).double() - ref).abs().max().item()
                w.writerow([name, dname, *shape, f"{err:.3e}", f"{sdpa_err:.3e}", f"{err / sdpa_err:.2f}"])
            del q, k, v, ref
print(f"wrote {out}")
