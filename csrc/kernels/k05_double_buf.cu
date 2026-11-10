#include <cstdint>
#include "common.cuh"
#include "cp_async.cuh"
#include "launch.h"
#include "tiles.h"

// k04's compute loop fed by an S-stage cp.async pipeline. The A tile is kept row-major
// (As[BM][BK]) because cp.async copies contiguous bytes and cannot transpose.
template <int BM, int BN, int BK, int TM, int TN, int S, bool VEC>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
k05_pipe(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
         float beta, float* C, int ldc) {
  constexpr int NT = (BM / TM) * (BN / TN), SA = BM * BK, SB = BK * BN;
  extern __shared__ float smem[];
  const int tid = threadIdx.x, bm0 = blockIdx.y * BM, bn0 = blockIdx.x * BN;
  const int tr = (tid / (BN / TN)) * TM, tc = (tid % (BN / TN)) * TN;
  const int ntile = (K + BK - 1) / BK;
  auto load = [&](int t, int slot) {
    float* As = smem + slot * (SA + SB);
    float* Bs = As + SA;
    const int k0 = t * BK;
    if (VEC) {
      for (int i = tid; i < SA / 4; i += NT) {
        const int r = i / (BK / 4), c = (i % (BK / 4)) * 4, gr = bm0 + r, gc = k0 + c;
        const int n = (gr < M) ? max(0, min(4, K - gc)) : 0;
        cp_async16(&As[r * BK + c], n ? &A[gr * lda + gc] : A, n * 4);
      }
      for (int i = tid; i < SB / 4; i += NT) {
        const int r = i / (BN / 4), c = (i % (BN / 4)) * 4, gr = k0 + r, gc = bn0 + c;
        const int n = (gr < K) ? max(0, min(4, N - gc)) : 0;
        cp_async16(&Bs[r * BN + c], n ? &B[gr * ldb + gc] : B, n * 4);
      }
    } else {
      for (int i = tid; i < SA; i += NT) {
        const int r = i / BK, c = i % BK, ok = bm0 + r < M && k0 + c < K;
        cp_async4(&As[r * BK + c], ok ? &A[(bm0 + r) * lda + k0 + c] : A, ok ? 4 : 0);
      }
      for (int i = tid; i < SB; i += NT) {
        const int r = i / BN, c = i % BN, ok = k0 + r < K && bn0 + c < N;
        cp_async4(&Bs[r * BN + c], ok ? &B[(k0 + r) * ldb + bn0 + c] : B, ok ? 4 : 0);
      }
    }
  };
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
#pragma unroll
  for (int p = 0; p < S - 1; ++p) {  // prologue: tiles 0..S-2
    if (p < ntile) load(p, p);
    cp_async_commit();
  }
  for (int t = 0; t < ntile; ++t) {
    cp_async_wait<S - 2>();  // tile t has landed
    __syncthreads();
    const int nt = t + S - 1;  // refill the slot freed by tile t-1
    if (nt < ntile) load(nt, nt % S);
    cp_async_commit();
    const float* As = smem + (t % S) * (SA + SB);
    const float* Bs = As + SA;
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
      for (int i = 0; i < TM; ++i) ra[i] = As[(tr + i) * BK + kk];
#pragma unroll
      for (int j = 0; j < TN; ++j) rb[j] = Bs[kk * BN + tc + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
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
template <int BM, int BN, int BK, int TM, int TN, int S>
int run(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
        float beta, float* C, int ldc, cudaStream_t s) {
  constexpr int NT = (BM / TM) * (BN / TN);
  const int bytes = S * (BM * BK + BK * BN) * 4;
  if (NT < 64 || NT > 1024 || bytes > smem_optin()) return HXGEMM_ETILE;
  const bool vec = aligned16(A) && aligned16(B) && lda % 4 == 0 && ldb % 4 == 0;
  dim3 g((N + BN - 1) / BN, (M + BM - 1) / BM);
  if (vec) {
    auto k = k05_pipe<BM, BN, BK, TM, TN, S, true>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    k<<<g, NT, bytes, s>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  } else {
    auto k = k05_pipe<BM, BN, BK, TM, TN, S, false>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    k<<<g, NT, bytes, s>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  return HXGEMM_OK;
}
}  // namespace

int launch_k05(hxgemm_tile_t t, int M, int N, int K, float alpha, const float* A, int lda,
               const float* B, int ldb, float beta, float* C, int ldc, cudaStream_t s) {
  if (t.bm == 0 && t.bn == 0 && t.bk == 0 && t.tm == 0 && t.tn == 0 && t.stages == 0)
    t = hxgemm_tile_t{128, 128, 32, 8, 4, 2};  // 4096^3 sweep winner (docs/results/2026-11-10_best_tiles.md)
#define HX_CASE(BM, BN, BK, TM, TN)                                                      \
  if (t.bm == BM && t.bn == BN && t.bk == BK && t.tm == TM && t.tn == TN) {             \
    if (t.stages == 2) return run<BM, BN, BK, TM, TN, 2>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s); \
    if (t.stages == 3) return run<BM, BN, BK, TM, TN, 3>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s); \
    return HXGEMM_ETILE;                                                                 \
  }
  HX_TILES(HX_CASE)
#undef HX_CASE
  return HXGEMM_ETILE;
}

int list_k05(hxgemm_tile_t* out, int cap) {
  int n = 0;
  const int lim = smem_optin();
  for (int st = 2; st <= 3; ++st) {
#define HX_ADD(BM, BN, BK, TM, TN)                                                         \
    if ((BM / TM) * (BN / TN) >= 64 && (BM / TM) * (BN / TN) <= 1024 &&                   \
        st * (BM * BK + BK * BN) * 4 <= lim && n < cap)                                    \
      out[n++] = hxgemm_tile_t{BM, BN, BK, TM, TN, st};
    HX_TILES(HX_ADD)
#undef HX_ADD
  }
  return n;
}
