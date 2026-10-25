"""Reference implementations the CUDA kernels are checked and timed against."""

import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def attention_fp64(q, k, v):
    """Ground truth: plain softmax(QK^T / sqrt(d)) V in float64."""
    q, k, v = q.double(), k.double(), v.double()
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return torch.softmax(s, dim=-1) @ v


def torch_naive(q, k, v):
    """Materialises S in HBM like naive.cu, but with cuBLAS GEMMs."""
    s = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(q.shape[-1]))
    return torch.softmax(s.float(), dim=-1).to(q.dtype) @ v


def sdpa_default(q, k, v):
    """Whatever backend PyTorch picks for these inputs."""
    return F.scaled_dot_product_attention(q, k, v)


def sdpa_flash(q, k, v):
    """PyTorch's built-in FlashAttention-2 kernel (fp16/bf16 only)."""
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v)


REF_IMPLS = {
    "torch_naive": torch_naive,
    "sdpa_default": sdpa_default,
    "sdpa_flash": sdpa_flash,
}
