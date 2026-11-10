#include "common.cuh"
__global__ void scale_c(int M, int N, float beta, float* C, int ldc) {
  int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < M && c < N) C[r * ldc + c] = beta == 0.f ? 0.f : beta * C[r * ldc + c];
}
