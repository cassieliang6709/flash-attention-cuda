"""Accuracy tests against a float64 reference.

Tolerances (why these numbers — see README "精度测试"):
  * fp32: every element within atol = rtol = 1e-4 of the fp64 result.
  * fp16: (a) within atol = rtol = 1e-2 of fp64, and
          (b) max abs error <= 2 * (max abs error of PyTorch SDPA) + 1e-5.
    (b) is the stricter, relative check FlashAttention's own test suite
    uses: "no worse than twice the production kernel on the same inputs".
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

if torch.cuda.is_available():
    from flash_cuda import IMPLS
    from flash_cuda.reference import attention_fp64, sdpa_default

KERNELS = ["naive", "fa1", "fa2"]
# 1000 and 77 are not multiples of the 32-row tile: they exercise padding/masking.
SHAPES = [(2, 4, 128, 64), (2, 4, 1000, 64), (1, 2, 77, 128), (2, 4, 1024, 128)]
TOL = {torch.float32: 1e-4, torch.float16: 1e-2}


def make_qkv(shape, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(shape, device="cuda", dtype=dtype, generator=g) for _ in range(3)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("impl", KERNELS)
def test_matches_fp64(impl, shape, dtype):
    q, k, v = make_qkv(shape, dtype)
    ref = attention_fp64(q, k, v)
    out = IMPLS[impl](q, k, v)
    torch.cuda.synchronize()

    assert out.shape == q.shape and out.dtype == dtype
    assert torch.isfinite(out).all()
    tol = TOL[dtype]
    torch.testing.assert_close(out.double(), ref, atol=tol, rtol=tol)

    if dtype == torch.float16:
        err = (out.double() - ref).abs().max().item()
        sdpa_err = (sdpa_default(q, k, v).double() - ref).abs().max().item()
        assert err <= 2 * sdpa_err + 1e-5, f"{impl}: max err {err:.2e} vs SDPA {sdpa_err:.2e}"


@pytest.mark.parametrize("impl", KERNELS)
def test_large_logits(impl):
    """Scores up to ~±100: a softmax without max-subtraction overflows exp()."""
    q, k, v = make_qkv((1, 2, 256, 64), torch.float32, seed=1)
    q = q * 10
    out = IMPLS[impl](q, k, v)
    torch.testing.assert_close(out.double(), attention_fp64(q, k, v), atol=1e-4, rtol=1e-4)
