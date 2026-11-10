#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "gemm.h"

namespace {
hxgemm_tile_t to_tile(c10::IntArrayRef t) {
  if (t.empty()) return {0, 0, 0, 0, 0, 0};
  TORCH_CHECK_VALUE(t.size() == 6, "tile must have 6 entries");
  return {int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5])};
}
void check_mat(const at::Tensor& x, const char* name) {
  TORCH_CHECK_VALUE(x.is_cuda() && x.scalar_type() == at::kFloat && x.dim() == 2, name,
                    " must be a 2-D float32 CUDA tensor");
  TORCH_CHECK_VALUE(x.stride(1) == 1 && x.stride(0) >= x.size(1), name,
                    " must be row-major (unit column stride)");
}
void run(at::Tensor& c, const at::Tensor& a, const at::Tensor& b, int64_t kernel,
         c10::IntArrayRef tile, double alpha, double beta) {
  check_mat(a, "a"); check_mat(b, "b"); check_mat(c, "c");
  TORCH_CHECK_VALUE(a.size(1) == b.size(0) && c.size(0) == a.size(0) && c.size(1) == b.size(1),
                    "shape mismatch");
  int r = hxgemm_sgemm(static_cast<hxgemm_kernel_t>(kernel), to_tile(tile), a.size(0), b.size(1),
                       a.size(1), float(alpha), a.data_ptr<float>(), a.stride(0),
                       b.data_ptr<float>(), b.stride(0), float(beta), c.data_ptr<float>(),
                       c.stride(0), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK_VALUE(r != HXGEMM_EINVAL && r != HXGEMM_ETILE, "hxgemm rejected the arguments (", r, ")");
  TORCH_CHECK(r == 0, "hxgemm CUDA error ", r);
}
at::Tensor sgemm(const at::Tensor& a, const at::Tensor& b, int64_t kernel, c10::IntArrayRef tile) {
  auto c = at::empty({a.size(0), b.size(1)}, a.options());
  run(c, a, b, kernel, tile, 1.0, 0.0);
  return c;
}
void sgemm_out(at::Tensor& c, const at::Tensor& a, const at::Tensor& b, int64_t kernel,
               c10::IntArrayRef tile, double alpha, double beta) {
  run(c, a, b, kernel, tile, alpha, beta);
}
}  // namespace

TORCH_LIBRARY(hxgemm, m) {
  m.def("sgemm(Tensor a, Tensor b, int kernel, int[] tile) -> Tensor");
  m.def("sgemm_out(Tensor(a!) c, Tensor a, Tensor b, int kernel, int[] tile, float alpha, float beta) -> ()");
}
TORCH_LIBRARY_IMPL(hxgemm, CUDA, m) {
  m.impl("sgemm", &sgemm);
  m.impl("sgemm_out", &sgemm_out);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
