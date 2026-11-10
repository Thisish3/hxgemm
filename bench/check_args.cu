#include <cstdio>
#include <cuda_runtime.h>
#include "../csrc/gemm.h"
static int fails = 0;
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
int main() {
  float *A, *B, *C;
  cudaMalloc(&A, 64 * 4); cudaMalloc(&B, 64 * 4); cudaMalloc(&C, 64 * 4);
  hxgemm_tile_t d = {0, 0, 0, 0, 0, 0};
  for (int k = HXGEMM_K01; k <= HXGEMM_K05; ++k) {
    auto kk = static_cast<hxgemm_kernel_t>(k);
    EXPECT(hxgemm_sgemm(kk, d, -1, 4, 4, 1, A, 4, B, 4, 0, C, 4, 0) == HXGEMM_EINVAL);
    EXPECT(hxgemm_sgemm(kk, d, 4, 4, 4, 1, A, 3, B, 4, 0, C, 4, 0) == HXGEMM_EINVAL);   // lda < K
    EXPECT(hxgemm_sgemm(kk, d, 4, 4, 4, 1, A, 4, B, 3, 0, C, 4, 0) == HXGEMM_EINVAL);   // ldb < N
    EXPECT(hxgemm_sgemm(kk, d, 4, 4, 4, 1, A, 4, B, 4, 0, C, 3, 0) == HXGEMM_EINVAL);   // ldc < N
    EXPECT(hxgemm_sgemm(kk, d, 4, 4, 4, 1, nullptr, 4, B, 4, 0, C, 4, 0) == HXGEMM_EINVAL);
    EXPECT(hxgemm_sgemm(kk, d, 0, 4, 4, 1, nullptr, 4, nullptr, 4, 0, nullptr, 4, 0) == HXGEMM_OK);
  }
  // K = 0: C = beta * C
  float h[4] = {1, 2, 3, 4};
  cudaMemcpy(C, h, 16, cudaMemcpyHostToDevice);
  EXPECT(hxgemm_sgemm(HXGEMM_K01, d, 2, 2, 0, 1, nullptr, 0, nullptr, 2, 2.f, C, 2, 0) == HXGEMM_OK);
  cudaMemcpy(h, C, 16, cudaMemcpyDeviceToHost);
  EXPECT(h[0] == 2 && h[3] == 8);
  EXPECT(hxgemm_sgemm(HXGEMM_K04, hxgemm_tile_t{96, 96, 8, 8, 8, 0}, 4, 4, 4, 1, A, 4, B, 4, 0, C, 4, 0) == HXGEMM_ETILE);
  std::printf(fails ? "check_args: %d failures\n" : "check_args: ok\n", fails);
  return fails != 0;
}
