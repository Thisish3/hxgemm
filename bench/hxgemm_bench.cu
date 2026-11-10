// hxgemm_bench: verify each kernel against a reference, then time it with CUDA events.
// Reference: CPU FP64 when max(M,N,K) <= 512, otherwise cuBLAS (CUBLAS_DEFAULT_MATH).
// A kernel that fails verification is never timed.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include "../csrc/gemm.h"
#include "harness.h"

#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  std::fprintf(stderr, "%s:%d cublas status %d\n", __FILE__, __LINE__, int(s_)); std::exit(2); } } while (0)

namespace {

constexpr int KCUBLAS = 0;  // pseudo kernel id for cuBLAS

struct Shape { int M, N, K; };

struct Opts {
  std::vector<int> kernels;
  std::vector<Shape> shapes;
  bool have_tile = false, sweep = false, check_only = false, nan_c = false;
  hxgemm_tile_t tile = {0, 0, 0, 0, 0, 0};
  int warmup = 10, iters = 50;
  float alpha = 1.f, beta = 0.f;
  std::string csv;
};

[[noreturn]] void usage(const char* msg) {
  std::fprintf(stderr,
               "error: %s\nusage: hxgemm_bench --kernel k01|k02|k03|k04|k05|cublas|all ... "
               "--shape M,N,K ... | --squares LO:HI  [--tile bm,bn,bk,tm,tn,stages] [--sweep] "
               "[--warmup 10] [--iters 50] [--alpha 1] [--beta 0] [--check-only] [--nan-c] "
               "[--csv path]\n",
               msg);
  std::exit(64);
}

std::string kname(int k) {
  if (k == KCUBLAS) return "cublas";
  char b[8];
  std::snprintf(b, sizeof b, "k%02d", k);
  return b;
}

void add_kernel(Opts& o, const std::string& s) {
  auto push = [&](int k) {
    for (int x : o.kernels) if (x == k) return;
    o.kernels.push_back(k);
  };
  if (s == "all") {
    for (int k = HXGEMM_K01; k <= HXGEMM_K05; ++k) push(k);
    push(KCUBLAS);
  } else if (s == "cublas") {
    push(KCUBLAS);
  } else if (s.size() == 3 && s[0] == 'k' && s[1] == '0' && s[2] >= '1' && s[2] <= '5') {
    push(s[2] - '0');
  } else {
    usage(("unknown kernel " + s).c_str());
  }
}

Opts parse(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> const char* {
      if (i + 1 >= argc) usage((a + " needs a value").c_str());
      return argv[++i];
    };
    if (a == "--kernel") add_kernel(o, val());
    else if (a == "--shape") {
      Shape s;
      if (std::sscanf(val(), "%d,%d,%d", &s.M, &s.N, &s.K) != 3) usage("bad --shape");
      o.shapes.push_back(s);
    } else if (a == "--squares") {
      int lo, hi;
      if (std::sscanf(val(), "%d:%d", &lo, &hi) != 2 || lo <= 0 || hi < lo) usage("bad --squares");
      for (int n = lo; n <= hi; n *= 2) o.shapes.push_back({n, n, n});
    } else if (a == "--tile") {
      hxgemm_tile_t& t = o.tile;
      if (std::sscanf(val(), "%d,%d,%d,%d,%d,%d", &t.bm, &t.bn, &t.bk, &t.tm, &t.tn, &t.stages) != 6)
        usage("bad --tile");
      o.have_tile = true;
    } else if (a == "--sweep") o.sweep = true;
    else if (a == "--warmup") o.warmup = std::atoi(val());
    else if (a == "--iters") o.iters = std::atoi(val());
    else if (a == "--alpha") o.alpha = std::strtof(val(), nullptr);
    else if (a == "--beta") o.beta = std::strtof(val(), nullptr);
    else if (a == "--check-only") o.check_only = true;
    else if (a == "--nan-c") o.nan_c = true;
    else if (a == "--csv") o.csv = val();
    else usage(("unknown argument " + a).c_str());
  }
  if (o.kernels.empty()) usage("no --kernel");
  if (o.shapes.empty()) usage("no --shape/--squares");
  if (o.iters < 1 || o.warmup < 0) usage("bad --iters/--warmup");
  if (o.nan_c) o.beta = 0.f;
  return o;
}

std::string tile_str(int k, const hxgemm_tile_t& t) {
  if (k != HXGEMM_K04 && k != HXGEMM_K05) return "-";
  char b[64];
  std::snprintf(b, sizeof b, "%dx%dx%dx%dx%dx%d", t.bm, t.bn, t.bk, t.tm, t.tn, t.stages);
  return b;
}

std::vector<hxgemm_tile_t> tiles_for(const Opts& o, int k) {
  if (k != HXGEMM_K04 && k != HXGEMM_K05) return {hxgemm_tile_t{0, 0, 0, 0, 0, 0}};
  if (o.sweep) {
    std::vector<hxgemm_tile_t> v(512);
    int n = hxgemm_list_tiles(static_cast<hxgemm_kernel_t>(k), v.data(), int(v.size()));
    v.resize(n > 0 ? n : 0);
    return v;  // empty when the kernel offers no tiles yet
  }
  return {o.have_tile ? o.tile : hxgemm_tile_t{0, 0, 0, 0, 0, 0}};
}

struct Row {
  std::string kernel, tile, status;
  Shape s;
  float median = 0, mn = 0, mx = 0;
  double gflops = 0, pct = 0, max_abs = 0, max_rel = 0;
  bool pass = false, timed = false;
};

void print_row(const Row& r, FILE* csv) {
  if (r.timed)
    std::printf("kernel=%-6s tile=%-16s M=%-5d N=%-5d K=%-5d median_us=%10.2f min_us=%10.2f "
                "max_us=%10.2f gflops=%9.1f pct_cublas=%6.1f max_abs_err=%.3e max_rel_err=%.3e "
                "pass=%d\n",
                r.kernel.c_str(), r.tile.c_str(), r.s.M, r.s.N, r.s.K, r.median, r.mn, r.mx,
                r.gflops, r.pct, r.max_abs, r.max_rel, int(r.pass));
  else
    std::printf("kernel=%-6s tile=%-16s M=%-5d N=%-5d K=%-5d max_abs_err=%.3e max_rel_err=%.3e "
                "pass=%d%s%s\n",
                r.kernel.c_str(), r.tile.c_str(), r.s.M, r.s.N, r.s.K, r.max_abs, r.max_rel,
                int(r.pass), r.status.empty() ? "" : " status=", r.status.c_str());
  if (csv) {
    if (r.timed)
      std::fprintf(csv, "%s,%s,%d,%d,%d,%.3f,%.3f,%.3f,%.2f,%.2f,%.6e,%.6e,%d\n", r.kernel.c_str(),
                   r.tile.c_str(), r.s.M, r.s.N, r.s.K, r.median, r.mn, r.mx, r.gflops, r.pct,
                   r.max_abs, r.max_rel, int(r.pass));
    else
      std::fprintf(csv, "%s,%s,%d,%d,%d,,,,,,%.6e,%.6e,%d\n", r.kernel.c_str(), r.tile.c_str(),
                   r.s.M, r.s.N, r.s.K, r.max_abs, r.max_rel, int(r.pass));
    std::fflush(csv);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Opts o = parse(argc, argv);

  FILE* csv = nullptr;
  if (!o.csv.empty()) {
    csv = std::fopen(o.csv.c_str(), "w");
    if (!csv) { std::perror(o.csv.c_str()); return 2; }
    std::fprintf(csv, "kernel,tile,M,N,K,median_us,min_us,max_us,gflops,pct_cublas,max_abs_err,"
                      "max_rel_err,pass\n");
  }

  int l2 = 0;
  CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  const size_t flush_bytes = size_t(l2) * 2;
  void* flush = nullptr;
  CK(cudaMalloc(&flush, flush_bytes));

  cublasHandle_t h;
  CB(cublasCreate(&h));
  CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

  int failures = 0;
  for (const Shape& s : o.shapes) {
    const int M = s.M, N = s.N, K = s.K, lda = K, ldb = N, ldc = N;
    std::vector<float> hA(size_t(M) * lda), hB(size_t(K) * ldb), hC0(size_t(M) * ldc), hC(hC0.size());
    fill_uniform(hA, 1234);
    fill_uniform(hB, 5678);
    if (o.nan_c) std::fill(hC0.begin(), hC0.end(), std::numeric_limits<float>::quiet_NaN());
    else fill_uniform(hC0, 9012);

    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, std::max<size_t>(hA.size(), 1) * 4));
    CK(cudaMalloc(&dB, std::max<size_t>(hB.size(), 1) * 4));
    CK(cudaMalloc(&dC, std::max<size_t>(hC0.size(), 1) * 4));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
    auto reset_c = [&] { CK(cudaMemcpy(dC, hC0.data(), hC0.size() * 4, cudaMemcpyHostToDevice)); };

    // row-major C(MxN) = A(MxK) B(KxN)  <=>  col-major C^T(NxM) = B^T(NxK) A^T(KxM)
    auto run_cublas = [&] {
      CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &o.alpha, dB, ldb, dA, lda, &o.beta, dC,
                     ldc));
    };

    // Reference values.
    std::vector<double> ref;
    const char* ref_kind;
    if (std::max({M, N, K}) <= 512) {
      ref_cpu_fp64(M, N, K, o.alpha, hA, lda, hB, ldb, o.beta, hC0, ldc, ref);
      ref_kind = "cpu_fp64";
    } else {
      reset_c();
      run_cublas();
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hC.data(), dC, hC.size() * 4, cudaMemcpyDeviceToHost));
      ref.assign(hC.begin(), hC.end());
      ref_kind = "cublas";
    }
    std::printf("# shape M=%d N=%d K=%d ref=%s alpha=%g beta=%g%s\n", M, N, K, ref_kind, o.alpha,
                o.beta, o.nan_c ? " nan_c=1" : "");

    // cuBLAS median for pct_cublas (always measured when timing).
    float cublas_us = 0;
    if (!o.check_only) {
      float mn, mx;
      reset_c();
      cublas_us = time_kernel(run_cublas, o.warmup, o.iters, &mn, &mx, flush, flush_bytes);
    }

    for (int k : o.kernels) {
      for (const hxgemm_tile_t& t : tiles_for(o, k)) {
        Row r;
        r.kernel = kname(k);
        r.tile = k == KCUBLAS ? "-" : tile_str(k, t);
        r.s = s;
        std::function<void()> launch;
        int rc = 0;
        reset_c();
        if (k == KCUBLAS) {
          launch = run_cublas;
          launch();
        } else {
          const auto kk = static_cast<hxgemm_kernel_t>(k);
          rc = hxgemm_sgemm(kk, t, M, N, K, o.alpha, dA, lda, dB, ldb, o.beta, dC, ldc, 0);
          launch = [&, kk, t] {
            int e = hxgemm_sgemm(kk, t, M, N, K, o.alpha, dA, lda, dB, ldb, o.beta, dC, ldc, 0);
            if (e != 0) { std::fprintf(stderr, "launch failed rc=%d\n", e); std::exit(2); }
          };
        }
        if (rc == HXGEMM_ETILE) r.status = "unsupported";
        else if (rc == HXGEMM_EINVAL) r.status = "einval";
        else if (rc > 0) r.status = std::string("cuda:") + cudaGetErrorString(cudaError_t(rc));
        if (rc == 0) {
          cudaError_t e = cudaDeviceSynchronize();
          if (e != cudaSuccess) {
            std::fprintf(stderr, "%s failed: %s\n", r.kernel.c_str(), cudaGetErrorString(e));
            return 2;
          }
          CK(cudaMemcpy(hC.data(), dC, hC.size() * 4, cudaMemcpyDeviceToHost));
          r.pass = check(hC.data(), ref.data(), M, N, ldc, K, &r.max_abs, &r.max_rel);
        }
        if (r.pass && !o.check_only) {
          r.median = time_kernel(launch, o.warmup, o.iters, &r.mn, &r.mx, flush, flush_bytes);
          r.gflops = 2.0 * M * N * K / (double(r.median) * 1e3);
          r.pct = cublas_us > 0 ? 100.0 * cublas_us / r.median : 0;
          r.timed = true;
        }
        if (!r.pass) ++failures;
        print_row(r, csv);
      }
      if ((k == HXGEMM_K04 || k == HXGEMM_K05) && o.sweep && tiles_for(o, k).empty()) {
        Row r;
        r.kernel = kname(k);
        r.tile = "sweep";
        r.s = s;
        r.status = "no_tiles";
        ++failures;
        print_row(r, csv);
      }
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
  }
  cublasDestroy(h);
  cudaFree(flush);
  if (csv) std::fclose(csv);
  if (failures) std::printf("# %d result(s) failed or unsupported\n", failures);
  return failures ? 1 : 0;
}
