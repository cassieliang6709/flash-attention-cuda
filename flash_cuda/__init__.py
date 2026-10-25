"""Python entry points. All take q, k, v of shape [B, H, N, D] (fp32 or fp16,
contiguous, CUDA) and return O of the same shape. Non-causal, scale = 1/sqrt(D)."""

import torch

from flash_cuda import _C


def _prep(*ts):
    return [t.contiguous() for t in ts]


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _C.naive_forward(*_prep(q, k, v))


def flash_attn_v1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _C.fa1_forward(*_prep(q, k, v))


def flash_attn_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _C.fa2_forward(*_prep(q, k, v))


IMPLS = {
    "naive": naive_attention,
    "fa1": flash_attn_v1,
    "fa2": flash_attn_v2,
}
