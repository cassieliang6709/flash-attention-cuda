// FlashAttention v2 forward (Dao 2023), step "v2a": scalar FP32 math.
//
// Two changes relative to fa1.cu:
//
// 1. Loop order swapped:
//
//      for each Q tile i:             <- one thread block per Q tile
//        load Q_i once
//        o, l, m = 0, 0, -inf         <- running state in REGISTERS
//        for each K/V tile j:         <- inner loop
//          load K_j, V_j into SMEM
//          online-softmax update of (o, l, m)
//        write O_i = o / l once       <- one HBM write of O, no reads
//
//    O traffic drops from ~2 * Tc * N * D floats (v1) to N * D elements.
//    The price: K/V are now re-read once per Q tile (Tr times), but K/V
//    reads are what tiling is good at and they hit L2 across blocks.
//
//    v2 also delays the division by l to the very end ("un-normalised O"),
//    removing the per-tile rescale-by-1/l_new of v1.
//
// 2. Sequence-dimension parallelism: grid = (Tr, B*H) instead of (B*H).
//    With N = 4096, BR = 32 that is 128 blocks per head, so all SMs get work
//    even when B*H is small.
//
// Thread mapping is the same as v1: BR threads, thread r owns query row r.
// The next step (v2b, tensor cores via WMMA) changes this mapping; it is
// written after v2a passes the tests and ncu has told us what's limiting it.

#include "common.cuh"

namespace {

constexpr int BR = 32;
constexpr int BC = 32;

template <typename T, int ROWS, int D>
__device__ __forceinline__ void load_tile(float* __restrict__ dst, const T* __restrict__ src,
                                          int row0, int N) {
  for (int e = threadIdx.x; e < ROWS * D; e += blockDim.x) {
    const int r = e / D, c = e % D;
    const int g = row0 + r;
    dst[e] = (g < N) ? to_float(src[(size_t)g * D + c]) : 0.f;
  }
}

template <typename T, int D>
__global__ void fa2_kernel(const T* __restrict__ Q, const T* __restrict__ K,
                           const T* __restrict__ V, T* __restrict__ O, int N, float scale) {
  extern __shared__ float smem[];
  float* Qs = smem;         // [BR][D]
  float* Ks = Qs + BR * D;  // [BC][D]
  float* Vs = Ks + BC * D;  // [BC][D]

  const int q0 = blockIdx.x * BR;  // this block's Q tile
  const int bh = blockIdx.y;
  const int tx = threadIdx.x;
  const int row = q0 + tx;
  const T* k_bh = K + (size_t)bh * N * D;
  const T* v_bh = V + (size_t)bh * N * D;

  load_tile<T, BR, D>(Qs, Q + (size_t)bh * N * D, q0, N);

  // Running state, in registers. o[] is indexed only with compile-time
  // indices inside unrolled loops — otherwise nvcc puts it in local memory
  // (check "spill" / "lmem" in the --ptxas-options=-v build output and ncu).
  float o[D];
#pragma unroll
  for (int d = 0; d < D; ++d) o[d] = 0.f;
  float m = -INFINITY;
  float l = 0.f;

  const int Tc = (N + BC - 1) / BC;
  for (int j = 0; j < Tc; ++j) {
    const int kv0 = j * BC;
    __syncthreads();  // previous tile's Ks/Vs are no longer being read
    load_tile<T, BC, D>(Ks, k_bh, kv0, N);
    load_tile<T, BC, D>(Vs, v_bh, kv0, N);
    __syncthreads();

    if (row < N) {
      // ================= TODO(Cassie): FA v2 inner loop =================
      // Available: Qs[tx * D + d], Ks[c * D + d], Vs[c * D + d], scale,
      //            kv0 + c >= N -> key c is padding (score = -INFINITY),
      //            running state o[D], m, l (o is NOT normalised).
      //
      // Exactly the body of the j-loop in flash_pytorch.py:
      //   s[c]  = scale * dot(Q_row, K_c)
      //   m_new = max(m, max_c s[c])
      //   p[c]  = exp(s[c] - m_new)
      //   alpha = exp(m - m_new)
      //   l     = alpha * l + sum_c p[c]
      //   o[d]  = alpha * o[d] + sum_c p[c] * Vs[c][d]
      //   m     = m_new
      // ===================================================================
    }
  }

  if (row < N) {
    // ================= TODO(Cassie): finalise =================
    // O[bh, row, d] = o[d] / l   (convert with from_float<T>)
    // ===========================================================
    (void)O;
    (void)scale;
  }
}

}  // namespace

torch::Tensor fa2_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  check_qkv(q, k, v);
  const int B = q.size(0), H = q.size(1), N = q.size(2), d = q.size(3);
  const int BH = B * H;
  const float scale = 1.0f / std::sqrt((float)d);
  auto o = torch::zeros_like(q);  // zeros so an unfinished kernel fails the tests loudly

  DISPATCH_FLOAT_HALF(q.scalar_type(), T, [&] {
    DISPATCH_HEAD_DIM(d, D, [&] {
      const size_t smem = (size_t)(BR + 2 * BC) * D * sizeof(float);
      auto kernel = fa2_kernel<T, D>;
      C10_CUDA_CHECK(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      dim3 grid((N + BR - 1) / BR, BH);
      kernel<<<grid, BR, smem, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const T*>(q.data_ptr()), reinterpret_cast<const T*>(k.data_ptr()),
          reinterpret_cast<const T*>(v.data_ptr()), reinterpret_cast<T*>(o.data_ptr()), N,
          scale);
      CUDA_CHECK_LAUNCH();
    });
  });
  return o;
}
