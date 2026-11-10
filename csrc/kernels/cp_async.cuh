#pragma once
// src_bytes < size zero-fills the rest; src_bytes = 0 reads nothing.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem, int src_bytes) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
