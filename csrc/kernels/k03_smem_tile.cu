#include "common.cuh"
#include "launch.h"
// Each block loads a T x T tile of A and of B into shared memory per K step and reuses it
// T times; out-of-range elements are loaded as 0 so the inner loop needs no bounds checks.
template <int T>
__global__ void k03_smem(int M, int N, int K, float alpha, const float* A, int lda,
                         const float* B, int ldb, float beta, float* C, int ldc) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += T) {
    As[ty][tx] = (row < M && k0 + tx < K) ? A[row * lda + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * ldb + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N)
    C[row * ldc + col] = hx_out(acc, alpha, beta, beta == 0.f ? 0.f : C[row * ldc + col]);
}
void launch_k03(int M, int N, int K, float alpha, const float* A, int lda, const float* B,
                int ldb, float beta, float* C, int ldc, cudaStream_t s) {
  dim3 b(32, 32), g((N + 31) / 32, (M + 31) / 32);
  k03_smem<32><<<g, b, 0, s>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
