#include "launch.h"

#define HX_LAUNCH_ARGS                                                                    \
  int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, \
      float beta, float *C, int ldc, cudaStream_t s
void launch_k01(HX_LAUNCH_ARGS);
void launch_k02(HX_LAUNCH_ARGS);
void launch_k03(HX_LAUNCH_ARGS);
int launch_k04(hxgemm_tile_t t, HX_LAUNCH_ARGS);
int list_k04(hxgemm_tile_t* out, int cap);
int launch_k05(hxgemm_tile_t t, HX_LAUNCH_ARGS);
int list_k05(hxgemm_tile_t* out, int cap);

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
    case HXGEMM_K03:
      if (!is_default(t)) return HXGEMM_ETILE;
      launch_k03(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
      return HXGEMM_OK;
    case HXGEMM_K04:
      return launch_k04(t, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
    case HXGEMM_K05:
      return launch_k05(t, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
    default:
      return HXGEMM_ETILE;
  }
}

extern "C" hxgemm_tile_t hxgemm_default_tile(hxgemm_kernel_t k, int M, int N, int) {
  if (k == HXGEMM_K04) return hxgemm_tile_t{64, 128, 16, 8, 4, 0};  // 4096^3 sweep winner (R10)
  if (k == HXGEMM_K05) {
    const long long big_blocks = (long long)((M + 127) / 128) * ((N + 255) / 256);
    return big_blocks >= 256 ? hxgemm_tile_t{128, 256, 16, 16, 8, 3}   // R15: best at 4096^3/8192^3
                             : hxgemm_tile_t{128, 128, 32, 8, 4, 2};  // R10: better when few blocks
  }
  return hxgemm_tile_t{0, 0, 0, 0, 0, 0};
}

extern "C" int hxgemm_list_tiles(hxgemm_kernel_t k, hxgemm_tile_t* out, int cap) {
  if (!out || cap <= 0) return 0;
  if (k == HXGEMM_K04) return list_k04(out, cap);
  if (k == HXGEMM_K05) return list_k05(out, cap);
  return 0;
}
