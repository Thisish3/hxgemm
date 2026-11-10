#include "common.cuh"
#include "launch.h"
// Same body as k01, but threadIdx.x -> column: a warp reads one broadcast A element and 32
// consecutive floats of B, and writes 32 consecutive floats of C.
__global__ void k02_coalesce(int M, int N, int K, float alpha, const float* A, int lda,
                             const float* B, int ldb, float beta, float* C, int ldc) {
  const int row = blockIdx.y * 32 + threadIdx.y, col = blockIdx.x * 32 + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * lda + k] * B[k * ldb + col];
  C[row * ldc + col] = hx_out(acc, alpha, beta, beta == 0.f ? 0.f : C[row * ldc + col]);
}
void launch_k02(int M, int N, int K, float alpha, const float* A, int lda, const float* B,
                int ldb, float beta, float* C, int ldc, cudaStream_t s) {
  dim3 b(32, 32), g((N + 31) / 32, (M + 31) / 32);
  k02_coalesce<<<g, b, 0, s>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
