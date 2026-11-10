#pragma once
// Tile configurations compiled for k04/k05: BM,BN in {64,128}, BK in {8,16,32}, TM,TN in {4,8}.
#define HX_TMTN(X, BM, BN, BK) X(BM, BN, BK, 4, 4) X(BM, BN, BK, 4, 8) X(BM, BN, BK, 8, 4) X(BM, BN, BK, 8, 8)
#define HX_BK(X, BM, BN) HX_TMTN(X, BM, BN, 8) HX_TMTN(X, BM, BN, 16) HX_TMTN(X, BM, BN, 32)
#define HX_TILES(X) HX_BK(X, 64, 64) HX_BK(X, 64, 128) HX_BK(X, 128, 64) HX_BK(X, 128, 128)
