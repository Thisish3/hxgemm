#include <cstdint>
#include "common.cuh"
#include "launch.h"
#include "tiles.h"

// Thread computes a TM x TN block of C. A tile stored transposed (As[BK][BM]) so the inner
// loop reads TM consecutive floats; B tile stored as is (Bs[BK][BN]).
template <int BM, int BN, int BK, int TM, int TN, bool VEC>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
k04_reg(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
        float beta, float* C, int ldc) {
  constexpr int NT = (BM / TM) * (BN / TN);
  extern __shared__ float smem[];
  float* As = smem;            // [BK][BM]
  float* Bs = smem + BK * BM;  // [BK][BN]
  const int tid = threadIdx.x;
  const int bm0 = blockIdx.y * BM, bn0 = blockIdx.x * BN;
  const int tr = (tid / (BN / TN)) * TM, tc = (tid % (BN / TN)) * TN;
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    if (VEC) {
      for (int i = tid; i < BM * BK / 4; i += NT) {
        const int r = i / (BK / 4), c = (i % (BK / 4)) * 4, gr = bm0 + r, gc = k0 + c;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (gr < M && gc + 3 < K) v = *reinterpret_cast<const float4*>(&A[gr * lda + gc]);
        else if (gr < M) {
          float* p = &v.x;
          for (int q = 0; q < 4; ++q) p[q] = gc + q < K ? A[gr * lda + gc + q] : 0.f;
        }
        As[(c + 0) * BM + r] = v.x; As[(c + 1) * BM + r] = v.y;
        As[(c + 2) * BM + r] = v.z; As[(c + 3) * BM + r] = v.w;
      }
      for (int i = tid; i < BK * BN / 4; i += NT) {
        const int r = i / (BN / 4), c = (i % (BN / 4)) * 4, gr = k0 + r, gc = bn0 + c;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (gr < K && gc + 3 < N) v = *reinterpret_cast<const float4*>(&B[gr * ldb + gc]);
        else if (gr < K) {
          float* p = &v.x;
          for (int q = 0; q < 4; ++q) p[q] = gc + q < N ? B[gr * ldb + gc + q] : 0.f;
        }
        *reinterpret_cast<float4*>(&Bs[r * BN + c]) = v;
      }
    } else {
      for (int i = tid; i < BM * BK; i += NT) {
        const int r = i / BK, c = i % BK;
        As[c * BM + r] = (bm0 + r < M && k0 + c < K) ? A[(bm0 + r) * lda + k0 + c] : 0.f;
      }
      for (int i = tid; i < BK * BN; i += NT) {
        const int r = i / BN, c = i % BN;
        Bs[r * BN + c] = (k0 + r < K && bn0 + c < N) ? B[(k0 + r) * ldb + bn0 + c] : 0.f;
      }
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
      for (int i = 0; i < TM; ++i) ra[i] = As[kk * BM + tr + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) rb[j] = Bs[kk * BN + tc + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int r = bm0 + tr + i, c = bn0 + tc + j;
      if (r < M && c < N)
        C[r * ldc + c] = hx_out(acc[i][j], alpha, beta, beta == 0.f ? 0.f : C[r * ldc + c]);
    }
}

namespace {
bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }
int smem_optin() {
  int dev = 0, v = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  return v;
}
template <int BM, int BN, int BK, int TM, int TN>
int run(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
        float beta, float* C, int ldc, cudaStream_t s) {
  constexpr int NT = (BM / TM) * (BN / TN);
  const int bytes = (BM * BK + BK * BN) * 4;
  if (NT < 64 || NT > 1024 || bytes > smem_optin()) return HXGEMM_ETILE;
  const bool vec = aligned16(A) && aligned16(B) && lda % 4 == 0 && ldb % 4 == 0;
  dim3 g((N + BN - 1) / BN, (M + BM - 1) / BM);
  if (vec) {
    auto k = k04_reg<BM, BN, BK, TM, TN, true>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    k<<<g, NT, bytes, s>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  } else {
    auto k = k04_reg<BM, BN, BK, TM, TN, false>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    k<<<g, NT, bytes, s>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  return HXGEMM_OK;
}
}  // namespace

int launch_k04(hxgemm_tile_t t, int M, int N, int K, float alpha, const float* A, int lda,
               const float* B, int ldb, float beta, float* C, int ldc, cudaStream_t s) {
  if (t.bm == 0 && t.bn == 0 && t.bk == 0 && t.tm == 0 && t.tn == 0 && t.stages == 0)
    t = hxgemm_tile_t{64, 128, 16, 8, 4, 0};  // 4096^3 sweep winner (docs/results/2026-11-10_best_tiles.md)
  if (t.stages != 0) return HXGEMM_ETILE;
#define HX_CASE(BM, BN, BK, TM, TN)                                            \
  if (t.bm == BM && t.bn == BN && t.bk == BK && t.tm == TM && t.tn == TN)     \
    return run<BM, BN, BK, TM, TN>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
  HX_TILES(HX_CASE)
#undef HX_CASE
  return HXGEMM_ETILE;
}

int list_k04(hxgemm_tile_t* out, int cap) {
  int n = 0;
  const int lim = smem_optin();
#define HX_ADD(BM, BN, BK, TM, TN)                                                         \
  if ((BM / TM) * (BN / TN) >= 64 && (BM / TM) * (BN / TN) <= 1024 &&                     \
      (BM * BK + BK * BN) * 4 <= lim && n < cap) out[n++] = hxgemm_tile_t{BM, BN, BK, TM, TN, 0};
  HX_TILES(HX_ADD)
#undef HX_ADD
  return n;
}
