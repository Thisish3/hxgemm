#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

inline void fill_uniform(std::vector<float>& v, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  for (auto& x : v) x = d(g);
}

// row-major, C = alpha*A*B + beta*C0 in double
inline void ref_cpu_fp64(int M, int N, int K, float alpha, const std::vector<float>& A, int lda,
                         const std::vector<float>& B, int ldb, float beta,
                         const std::vector<float>& C0, int ldc, std::vector<double>& out) {
  out.assign(static_cast<size_t>(M) * ldc, 0.0);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += double(A[i * lda + k]) * double(B[k * ldb + j]);
      double c = beta == 0.f ? 0.0 : double(beta) * C0[i * ldc + j];
      out[i * ldc + j] = double(alpha) * acc + c;
    }
}

inline bool check(const float* got, const double* ref, int M, int N, int ldc, int K,
                  double* max_abs, double* max_rel) {
  const double rtol = 1e-5 * std::sqrt(double(K)), atol = 1e-6 * K;
  bool ok = true;
  *max_abs = *max_rel = 0;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double r = ref[i * ldc + j], g = got[i * ldc + j];
      double err = std::fabs(g - r);
      if (std::isnan(g) || err > atol + rtol * std::fabs(r)) ok = false;
      *max_abs = std::max(*max_abs, err);
      if (r != 0) *max_rel = std::max(*max_rel, err / std::fabs(r));
    }
  return ok;
}

// Each iteration: flush L2 (write flush_bytes), then time one launch with events.
inline float time_kernel(const std::function<void()>& launch, int warmup, int iters,
                         float* min_us, float* max_us, void* flush_buf, size_t flush_bytes) {
  for (int i = 0; i < warmup; ++i) launch();
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  std::vector<float> t(iters);
  for (int i = 0; i < iters; ++i) {
    CK(cudaMemsetAsync(flush_buf, i & 0xff, flush_bytes));
    CK(cudaEventRecord(a));
    launch();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    t[i] = ms * 1000.f;
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  std::sort(t.begin(), t.end());
  *min_us = t.front();
  *max_us = t.back();
  return t[iters / 2];
}
