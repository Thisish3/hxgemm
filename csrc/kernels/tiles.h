#pragma once
// Tile configurations compiled for k04/k05: BM,BN in {64,128}, BK in {8,16,32}, TM,TN in {4,8}.
#define HX_TMTN(X, BM, BN, BK) X(BM, BN, BK, 4, 4) X(BM, BN, BK, 4, 8) X(BM, BN, BK, 8, 4) X(BM, BN, BK, 8, 8)
#define HX_BK(X, BM, BN) HX_TMTN(X, BM, BN, 8) HX_TMTN(X, BM, BN, 16) HX_TMTN(X, BM, BN, 32)
#define HX_TILES(X) HX_BK(X, 64, 64) HX_BK(X, 64, 128) HX_BK(X, 128, 64) HX_BK(X, 128, 128)
// k05 only (Task 10 round 1, LEDGER R12): 256 threads with a 16x8 / 8x16 register tile, the shape
// cuBLAS picks at 8192^3 (cutlass_80_simt_sgemm_256x128_8x4: 256x128 block, 216 registers).
#define HX_BIG_BK(X, BK) X(256, 128, BK, 16, 8) X(128, 256, BK, 8, 16) X(256, 128, BK, 8, 16) X(128, 256, BK, 16, 8)
#define HX_TILES_BIG(X) HX_BIG_BK(X, 8) HX_BIG_BK(X, 16)
