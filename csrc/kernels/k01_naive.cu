#include "common.cuh"
#include "launch.h"
// threadIdx.x -> row on purpose: a warp reads 32 different rows of A and writes 32 rows of C.
__global__ void k01_naive(int M, int N, int K, float alpha, const float* A, int lda,
                          const float* B, int ldb, float beta, float* C, int ldc) {
  const int row = blockIdx.x * 32 + threadIdx.x, col = blockIdx.y * 32 + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * lda + k] * B[k * ldb + col];
  C[row * ldc + col] = hx_out(acc, alpha, beta, beta == 0.f ? 0.f : C[row * ldc + col]);
}
void launch_k01(int M, int N, int K, float alpha, const float* A, int lda, const float* B,
                int ldb, float beta, float* C, int ldc, cudaStream_t s) {
  dim3 b(32, 32), g((M + 31) / 32, (N + 31) / 32);
  k01_naive<<<g, b, 0, s>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
