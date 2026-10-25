// FlashAttention v1 forward (Dao et al. 2022, Algorithm 1), scalar FP32 math.
//
// Loop order (this is what makes it "v1"):
//
//   for each K/V tile j:            <- OUTER loop, K_j / V_j loaded once
//     load K_j, V_j into SMEM
//     for each Q tile i:            <- INNER loop
//       load Q_i            from HBM
//       load O_i, l_i, m_i  from HBM    <- running state lives in HBM
//       S_ij = scale * Q_i K_j^T        (on chip only, never written out)
//       update m_i, l_i, O_i with online softmax
//       write O_i, l_i, m_i back to HBM
//
// Because the running state (O, l, m) is read and written once per (i, j)
// pair, O traffic is ~2 * Tc * N * D floats, Tc = ceil(N / Bc). v2 fixes that.
//
// Parallelism: one thread block per (batch, head). Every block walks all
// (j, i) pairs sequentially, so with B*H = 16 heads only 16 SMs have work.
// v2 fixes that too.
//
// Thread mapping inside a block: BR threads (one warp); thread r owns query
// row r of the current Q tile. Loads into SMEM are cooperative (all threads
// stride over the tile) so global reads are coalesced.
//
// Precision: O is kept in an FP32 workspace (o_buf) between iterations,
// and cast to the input dtype at the very end. Rounding O to fp16 after every
// K/V tile would add Tc rounding errors.

#include "common.cuh"

namespace {

constexpr int BR = 32;  // query rows per tile  (= threads per block)
constexpr int BC = 32;  // key rows per tile

// Copy rows [row0, row0 + ROWS) of a [N, D] matrix into SMEM as float,
// zero-filling rows >= N (so the last, partial tile is padded with zeros).
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
__global__ void fa1_kernel(const T* __restrict__ Q, const T* __restrict__ K,
                           const T* __restrict__ V, float* __restrict__ o_buf,
                           float* __restrict__ l_buf, float* __restrict__ m_buf, int N,
                           float scale) {
  extern __shared__ float smem[];
  float* Qs = smem;            // [BR][D]
  float* Ks = Qs + BR * D;     // [BC][D]
  float* Vs = Ks + BC * D;     // [BC][D]

  const int bh = blockIdx.x;
  const int tx = threadIdx.x;
  const T* q_bh = Q + (size_t)bh * N * D;
  const T* k_bh = K + (size_t)bh * N * D;
  const T* v_bh = V + (size_t)bh * N * D;
  float* o_bh = o_buf + (size_t)bh * N * D;
  float* l_bh = l_buf + (size_t)bh * N;
  float* m_bh = m_buf + (size_t)bh * N;

  const int Tc = (N + BC - 1) / BC;
  const int Tr = (N + BR - 1) / BR;

  for (int j = 0; j < Tc; ++j) {
    const int kv0 = j * BC;
    load_tile<T, BC, D>(Ks, k_bh, kv0, N);
    load_tile<T, BC, D>(Vs, v_bh, kv0, N);
    __syncthreads();

    for (int i = 0; i < Tr; ++i) {
      const int q0 = i * BR;
      load_tile<T, BR, D>(Qs, q_bh, q0, N);
      __syncthreads();

      const int row = q0 + tx;  // global query index owned by this thread
      if (row < N) {
        // Running state of this row, read from HBM.
        const float m_old = m_bh[row];   // -inf before the first K/V tile
        const float l_old = l_bh[row];   // 0    before the first K/V tile
        float* o_row = o_bh + (size_t)row * D;  // FP32, zero-initialised

        // ================= TODO(Cassie): FA v1 core =================
        // Available:
        //   Qs[tx * D + d]   this thread's query row           (d < D)
        //   Ks[c * D + d]    key   c of the tile               (c < BC)
        //   Vs[c * D + d]    value c of the tile
        //   kv0 + c >= N  ->  key c is padding, its score must be -INFINITY
        //   m_old, l_old, o_row[d]  the running state (o_row is already
        //                            normalised by l_old — that's v1)
        //   scale
        //
        // Compute (paper Algorithm 1, lines 9-12; same symbols as
        // flash_pytorch.py except v1 keeps O normalised):
        //   s[c]    = scale * dot(Q_row, K_c)          float s[BC] in registers
        //   m_tilde = max_c s[c]
        //   p[c]    = exp(s[c] - m_tilde),  l_tilde = sum_c p[c]
        //   m_new   = max(m_old, m_tilde)
        //   l_new   = exp(m_old - m_new) * l_old + exp(m_tilde - m_new) * l_tilde
        //   o_row[d] = ( l_old * exp(m_old - m_new) * o_row[d]
        //              + exp(m_tilde - m_new) * sum_c p[c] * Vs[c][d] ) / l_new
        //   m_bh[row] = m_new;  l_bh[row] = l_new;
        //
        // Hints: loops over c with a compile-time bound (BC) can be
        // `#pragma unroll`ed so s[] stays in registers. Write o_row directly
        // (it's global memory — that read/write is exactly v1's cost).
        (void)m_old;
        (void)l_old;
        (void)o_row;
        (void)scale;
        // =============================================================
      }
      __syncthreads();  // Qs is overwritten by the next Q tile
    }
  }
}

}  // namespace

torch::Tensor fa1_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  check_qkv(q, k, v);
  const int B = q.size(0), H = q.size(1), N = q.size(2), d = q.size(3);
  const int BH = B * H;
  const float scale = 1.0f / std::sqrt((float)d);
  auto f32 = q.options().dtype(torch::kFloat);
  auto o_buf = torch::zeros({B, H, N, d}, f32);
  auto l_buf = torch::zeros({BH, N}, f32);
  auto m_buf = torch::full({BH, N}, -INFINITY, f32);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "fa1_forward", [&] {
    using T = typename cuda_type<scalar_t>::type;
    DISPATCH_HEAD_DIM(d, D, [&] {
      const size_t smem = (size_t)(BR + 2 * BC) * D * sizeof(float);
      auto kernel = fa1_kernel<T, D>;
      C10_CUDA_CHECK(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      kernel<<<BH, BR, smem, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const T*>(q.data_ptr()), reinterpret_cast<const T*>(k.data_ptr()),
          reinterpret_cast<const T*>(v.data_ptr()), o_buf.data_ptr<float>(),
          l_buf.data_ptr<float>(), m_buf.data_ptr<float>(), N, scale);
      CUDA_CHECK_LAUNCH();
    });
  });
  // For fp32 inputs this is a no-op; for fp16 it's one extra cast kernel,
  // which is part of v1's cost and is included in the benchmark.
  return o_buf.to(q.scalar_type());
}
