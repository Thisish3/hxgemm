#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { HXGEMM_K01 = 1, HXGEMM_K02, HXGEMM_K03, HXGEMM_K04, HXGEMM_K05 } hxgemm_kernel_t;
typedef struct { int bm, bn, bk, tm, tn, stages; } hxgemm_tile_t;  /* all 0 = default */
enum { HXGEMM_OK = 0, HXGEMM_EINVAL = -1, HXGEMM_ETILE = -2 };
int hxgemm_sgemm(hxgemm_kernel_t k, hxgemm_tile_t t, int M, int N, int K, float alpha,
                 const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc,
                 cudaStream_t stream);
/* fills up to cap supported tiles for k04/k05 on the current device; returns count */
int hxgemm_list_tiles(hxgemm_kernel_t k, hxgemm_tile_t* out, int cap);
#ifdef __cplusplus
}
#endif
