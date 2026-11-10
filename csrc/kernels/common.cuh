#pragma once
__device__ __forceinline__ float hx_out(float acc, float alpha, float beta, float c) {
  return beta == 0.f ? alpha * acc : alpha * acc + beta * c;
}
__global__ void scale_c(int M, int N, float beta, float* C, int ldc);
