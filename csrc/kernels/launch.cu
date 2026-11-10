#include "launch.h"

#define HX_LAUNCH_ARGS                                                                    \
  int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, \
      float beta, float *C, int ldc, cudaStream_t s
void launch_k01(HX_LAUNCH_ARGS);
void launch_k02(HX_LAUNCH_ARGS);

namespace {
bool is_default(const hxgemm_tile_t& t) {
  return t.bm == 0 && t.bn == 0 && t.bk == 0 && t.tm == 0 && t.tn == 0 && t.stages == 0;
}
}  // namespace

int hx_launch(hxgemm_kernel_t k, hxgemm_tile_t t, int M, int N, int K, float alpha,
              const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc,
              cudaStream_t s) {
  switch (k) {
    case HXGEMM_K01:
      if (!is_default(t)) return HXGEMM_ETILE;
      launch_k01(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
      return HXGEMM_OK;
    case HXGEMM_K02:
      if (!is_default(t)) return HXGEMM_ETILE;
      launch_k02(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
      return HXGEMM_OK;
    default:
      return HXGEMM_ETILE;
  }
}
extern "C" int hxgemm_list_tiles(hxgemm_kernel_t, hxgemm_tile_t*, int) { return 0; }
