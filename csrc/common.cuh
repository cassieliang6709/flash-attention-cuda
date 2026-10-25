#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CUDA_CHECK_LAUNCH()                                              \
  do {                                                                   \
    cudaError_t err__ = cudaGetLastError();                              \
    TORCH_CHECK(err__ == cudaSuccess, "CUDA launch failed: ",            \
                cudaGetErrorString(err__));                              \
  } while (0)

// Inputs are [B, H, N, D] contiguous; every kernel treats (B, H) as one flat
// "bh" index, so the base pointer of head bh is  ptr + bh * N * D.
inline void check_qkv(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  TORCH_CHECK(q.dim() == 4, "expected q of shape [B, H, N, D]");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "q, k, v must have the same shape (self-attention, no GQA)");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q, k, v must have the same dtype");
  TORCH_CHECK(q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf,
              "only float32 and float16 are supported");
}

// Load/store helpers so a kernel templated on scalar_t (float or __half)
// can always do its math in float.
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ float from_float<float>(float x) { return x; }
template <>
__device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }

// torch's at::Half and CUDA's __half have the same bit layout; kernels use
// __half so the intrinsics above apply.
template <typename T>
struct cuda_type { using type = T; };
template <>
struct cuda_type<at::Half> { using type = __half; };

// Dispatch head_dim at compile time: the kernels keep D-sized arrays in
// registers / shared memory, so D must be a template parameter.
#define DISPATCH_HEAD_DIM(d, D_CONST, ...)                         \
  [&] {                                                            \
    if ((d) == 64) {                                               \
      constexpr int D_CONST = 64;                                  \
      return __VA_ARGS__();                                        \
    } else if ((d) == 128) {                                       \
      constexpr int D_CONST = 128;                                 \
      return __VA_ARGS__();                                        \
    } else {                                                       \
      TORCH_CHECK(false, "head_dim must be 64 or 128, got ", (d)); \
    }                                                              \
  }()
