// Naive attention: three separate kernels, the [N, N] score matrix S (and
// then P = softmax(S)) is written to and read back from HBM.
//
//   1. S = scale * Q K^T          writes  BH * N * N floats
//   2. P = softmax(S) row-wise    reads + writes BH * N * N floats (in place)
//   3. O = P V                    reads   BH * N * N floats
//
// The matmuls are also naive (one thread per output element, no tiling), so
// this is the "how slow can it be" baseline. bench.py also times
// "torch_naive" (cuBLAS matmul + torch.softmax), which still materialises S
// but uses fast GEMMs: comparing FA against *that* isolates the benefit of
// fusion from the benefit of a good GEMM.

#include "common.cuh"

namespace {

constexpr int TILE = 16;

// grid (ceil(N/16), ceil(N/16), BH), block (16, 16).
// threadIdx.x walks the key index j, so consecutive threads write
// consecutive S[i, j] (coalesced store).
template <typename T, int D>
__global__ void naive_scores_kernel(const T* __restrict__ Q, const T* __restrict__ K,
                                    float* __restrict__ S, int N, float scale) {
  const int bh = blockIdx.z;
  const int i = blockIdx.y * TILE + threadIdx.y;
  const int j = blockIdx.x * TILE + threadIdx.x;
  if (i >= N || j >= N) return;
  const T* q = Q + ((size_t)bh * N + i) * D;
  const T* k = K + ((size_t)bh * N + j) * D;
  float acc = 0.f;
#pragma unroll 8
  for (int d = 0; d < D; ++d) acc += to_float(q[d]) * to_float(k[d]);
  S[((size_t)bh * N + i) * N + j] = acc * scale;
}

// One block per row of S. Three passes over the row: max, sum of exp, write.
constexpr int SOFTMAX_THREADS = 256;

__device__ float block_reduce(float val, bool is_max) {
  __shared__ float warp_vals[SOFTMAX_THREADS / 32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int off = 16; off > 0; off /= 2) {
    float other = __shfl_xor_sync(0xffffffff, val, off);
    val = is_max ? fmaxf(val, other) : val + other;
  }
  if (lane == 0) warp_vals[warp] = val;
  __syncthreads();
  val = warp_vals[0];
  for (int w = 1; w < SOFTMAX_THREADS / 32; ++w)
    val = is_max ? fmaxf(val, warp_vals[w]) : val + warp_vals[w];
  __syncthreads();  // warp_vals is reused by the next call
  return val;
}

__global__ void naive_softmax_kernel(float* __restrict__ S, int N) {
  float* row = S + (size_t)blockIdx.x * N;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < N; j += blockDim.x) m = fmaxf(m, row[j]);
  m = block_reduce(m, true);
  float l = 0.f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) l += expf(row[j] - m);
  l = block_reduce(l, false);
  for (int j = threadIdx.x; j < N; j += blockDim.x) row[j] = expf(row[j] - m) / l;
}

// grid (ceil(D/16), ceil(N/16), BH), block (16, 16). threadIdx.x walks d, so
// reads of V[j, d] and writes of O[i, d] are coalesced.
template <typename T, int D>
__global__ void naive_pv_kernel(const float* __restrict__ P, const T* __restrict__ V,
                                T* __restrict__ O, int N) {
  const int bh = blockIdx.z;
  const int i = blockIdx.y * TILE + threadIdx.y;
  const int d = blockIdx.x * TILE + threadIdx.x;
  if (i >= N || d >= D) return;
  const float* p = P + ((size_t)bh * N + i) * N;
  const T* v = V + (size_t)bh * N * D + d;
  float acc = 0.f;
  for (int j = 0; j < N; ++j) acc += p[j] * to_float(v[(size_t)j * D]);
  O[((size_t)bh * N + i) * D + d] = from_float<T>(acc);
}

}  // namespace

torch::Tensor naive_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  check_qkv(q, k, v);
  const int B = q.size(0), H = q.size(1), N = q.size(2), d = q.size(3);
  const int BH = B * H;
  auto o = torch::empty_like(q);
  auto s = torch::empty({BH, N, N}, q.options().dtype(torch::kFloat));
  const float scale = 1.0f / std::sqrt((float)d);

  DISPATCH_FLOAT_HALF(q.scalar_type(), T, [&] {
    DISPATCH_HEAD_DIM(d, D, [&] {
      const T* Q = reinterpret_cast<const T*>(q.data_ptr());
      const T* K = reinterpret_cast<const T*>(k.data_ptr());
      const T* V = reinterpret_cast<const T*>(v.data_ptr());
      T* O = reinterpret_cast<T*>(o.data_ptr());
      float* S = s.data_ptr<float>();
      cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      dim3 block(TILE, TILE);

      dim3 grid1((N + TILE - 1) / TILE, (N + TILE - 1) / TILE, BH);
      naive_scores_kernel<T, D><<<grid1, block, 0, stream>>>(Q, K, S, N, scale);
      CUDA_CHECK_LAUNCH();

      naive_softmax_kernel<<<BH * N, SOFTMAX_THREADS, 0, stream>>>(S, N);
      CUDA_CHECK_LAUNCH();

      dim3 grid3((D + TILE - 1) / TILE, (N + TILE - 1) / TILE, BH);
      naive_pv_kernel<T, D><<<grid3, block, 0, stream>>>(S, V, O, N);
      CUDA_CHECK_LAUNCH();
    });
  });
  return o;
}
