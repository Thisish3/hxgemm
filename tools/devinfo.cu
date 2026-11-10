#include <cstdio>
#include <cuda_runtime.h>
int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int clk = 0, smem_optin = 0, l2 = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
  std::printf("name=%s cc=%d.%d sms=%d clock_khz=%d l2_bytes=%d smem_per_block=%zu "
              "smem_per_block_optin=%d smem_per_sm=%zu regs_per_sm=%d\n",
              p.name, p.major, p.minor, p.multiProcessorCount, clk, l2, p.sharedMemPerBlock,
              smem_optin, p.sharedMemPerMultiprocessor, p.regsPerMultiprocessor);
  std::printf("fp32_peak_gflops_at_reported_clock=%.1f (SMs x 128 FP32 lanes x 2 x clock)\n",
              p.multiProcessorCount * 128.0 * 2 * clk / 1e6);
}
