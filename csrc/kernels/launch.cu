#include "launch.h"
int hx_launch(hxgemm_kernel_t, hxgemm_tile_t, int, int, int, float, const float*, int,
              const float*, int, float, float*, int, cudaStream_t) {
  return HXGEMM_ETILE;
}
extern "C" int hxgemm_list_tiles(hxgemm_kernel_t, hxgemm_tile_t*, int) { return 0; }
