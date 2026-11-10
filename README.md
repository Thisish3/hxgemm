# hxgemm

FP32 SGEMM(C = αAB + βC, 행 우선)을 naive부터 cp.async 파이프라인까지 다섯 단계로 직접 만들고, 같은 조건에서 cuBLAS와 비교한 프로젝트(G1). 단계마다 "바꾼 코드 → 바뀐 Nsight 지표 → 바뀐 GFLOPS"를 측정으로 남긴다.

- 설계: `docs/specs/2026-11-08-hxgemm-g1-design.md` · 계획: `docs/plans/2026-11-08-hxgemm-g1.md`
- 실행 기록(계획과 달라진 결정 R1~R16): `docs/LEDGER.md` · 미해결 현상: `docs/KNOWN_ISSUES.md`
- 블로그용 Nsight 캡처: `docs/blog/2026-11-10_nsight_{k02_k03,k03_k04,k04_k05}/` (옛 기본 타일 128x128x8x8x8로 잰 것, 아래 참고)

## 결과 (RTX 3060 Ti, 4096³, SM 1500 MHz 고정)

k01~k04·cuBLAS는 `docs/results/2026-11-10_final.csv`, k05(추가 개선 후)는 `docs/results/2026-11-10_auto_final.csv`에서 옮겼다. 각 값은 warm-up 10회 후 50회 CUDA event 중앙값, 매 회 L2 비우기. 이론 최고는 38 SM × 128 FP32 lane × 2 × 1.5 GHz = 14,592 GFLOPS.

| 커널 | 핵심 변경 | 타일 | 시간 (ms) | GFLOPS | cuBLAS 대비 | 이론 최고 대비 |
|---|---|---|---|---|---|---|
| k01 naive | 스레드 1개 = C 1개, `threadIdx.x` → 행(일부러 비연속) | – | 1260.13 | 109.07 | 1.17 % | 0.75 % |
| k02 coalescing | `threadIdx.x` → 열 | – | 170.13 | 807.86 | 8.69 % | 5.54 % |
| k03 shared tiling | 32×32 타일을 shared에 올려 재사용 | 32×32×32 | 129.25 | 1063.40 | 11.44 % | 7.29 % |
| k04 register tiling | 스레드당 TM×TN 누적, float4 적재 | 64x128x16, 8x4 | 16.70 | 8231.19 | 88.53 % | 56.41 % |
| k05 cp.async (Task 6~9) | k04 + 2단 `cp.async` 파이프라인 | 128x128x32, 8x4, 2단 | 16.33 | 8417.01 | 90.53 % | 57.68 % |
| **k05 추가 개선 후** | + 큰 레지스터 타일, B 열 묶음 배치(bank conflict 0), 모양별 타일 자동 선택 | 128x256x16, 16x8, 3단 | 15.66 | **8775.27** | **94.38 %** | 60.14 % |
| cuBLAS | `cublasSgemm`, `CUBLAS_DEFAULT_MATH` | – | 14.78 | 9298.08 | 100 % | 63.72 % |

### 목표 판정

- **성공 기준 2 (4096³에서 마지막 커널 ≥ cuBLAS 중앙값 속도의 80 %): 통과.** 자동 탐색 후 k05 90.53 %(독립 실행 2회 같음). 계획 Task 10의 종료 조건을 이미 만족해 계획상 수동 개선은 0회였다(LEDGER R11).
- 이후 사용자 요청으로 추가 개선 3회를 했다(LEDGER R12~R16). 1회 큰 레지스터 타일(8192³ 74 → 78 %), 2회 B 열 묶음 배치로 shared load bank conflict 10.7억 → 0(4096³ **94.4 %**, 8192³ 81.3 %), 3회 C 저장 float4는 느려져 되돌림. 큰 타일은 작은 모양에서 크게 느려서(128×4096² −45 %) 기본 타일을 모양별로 고르게 했다(128x256 블록 ≥ 256개면 큰 타일). 그 결과 이전 기본 타일 대비 모든 모양에서 같거나 빠르다.

### 다른 모양 (기본 타일, cuBLAS 대비, `docs/results/2026-11-10_auto_final.csv`)

| 모양 | k04 (64x128x16x8x4) | k05 자동 선택 타일 | k05 | cuBLAS GFLOPS |
|---|---|---|---|---|
| 256³ | 31.82 % | 128x128x32x8x4x2 | 31.11 % | 2340.57 |
| 512³ | 58.87 % | 128x128x32x8x4x2 | 61.11 % | 5171.77 |
| 1024³ | 78.68 % | 128x128x32x8x4x2 | 82.75 % | 7973.96 |
| 2048³ | 87.76 % | 128x128x32x8x4x2 | 90.23 % | 8883.15 |
| 4096³ | 88.50 % | 128x256x16x16x8x3 | 94.38 % | 9297.43 |
| 8192³ | 72.20 % | 128x256x16x16x8x3 | 80.14 % | 11298.38 |
| 1000³ | 75.91 % | 128x128x32x8x4x2 | 78.09 % | 7719.86 |
| 2047³ (ld가 4의 배수 아님) | 64.10 % | 128x128x32x8x4x2 | 75.84 % | 8859.15 |
| 4095³ (ld가 4의 배수 아님) | 62.42 % | 128x256x16x16x8x3 | 82.93 % | 9288.69 |
| 16×4096×4096 (디코드) | 31.90 % | 128x128x32x8x4x2 | 31.74 % | 2818.75 |
| 128×4096×4096 | 64.48 % | 128x128x32x8x4x2 | 82.33 % | 8542.37 |
| 2048×4096×4096 | 86.64 % | 128x256x16x16x8x3 | 92.88 % | 9424.08 |

모양별 최적 타일은 `docs/results/2026-11-10_best_tiles.md`(예: 16×4096²에서는 `64x64x32x8x4x2`가 55.8 %).

## 단계별 Nsight 지표 (4096³, 기본 타일)

`docs/results/2026-11-10_ncu_summary.csv`와 추가 개선 후 `2026-11-10_ncu_summary_tuned.csv`(ncu 2026.3.1, `--clock-control none`, 커널당 1회 실행)에서 옮겼다. stall 값은 `smsp__average_warp_latency_issue_stalled_*.ratio` 원시값으로, 커널끼리 상대 비교용이다.

| 커널 | DRAM read | L1/TEX 바이트 | 실행 명령 | IPC | 레지스터 | occupancy | 명령 간 cycle | 가장 큰 stall |
|---|---|---|---|---|---|---|---|---|
| k01 | 10.46 GB | 2268.2 GB | 9.42 G | 0.13 | 40 | 65.7 % | 239.4 | LG throttle 4,040,425 |
| k02 | 8.97 GB | 343.7 GB | 9.42 G | 0.97 | 40 | 66.7 % | 32.9 | LG throttle 357,969 |
| k03 | 9.04 GB | 17.2 GB | 6.39 G | 0.87 | 38 | 66.7 % | 36.7 | MIO throttle 258,359 |
| k04 | 4.32 GB | 6.6 GB | 2.76 G | 2.92 | 64 | 65.4 % | 10.7 | long scoreboard 325,482 |
| k05 | 2.08 GB | 4.6 GB | 2.65 G | 2.87 | 120 | 33.3 % | 5.57 | barrier 123,495 |
| k05 추가 개선 후 (128x256x16x16x8x3) | 1.92 GB | 3.5 GB | 2.49 G | 2.89 | 255 | 16.7 % | 2.76 | barrier 46,057 |
| cuBLAS | 1.83 GB | 6.8 GB | 2.52 G | 2.99 | 122 | 33.2 % | 5.31 | barrier 24,252 |

단계마다 확인한 것(자세한 수치와 캡처는 `docs/blog/` 각 폴더의 README):

- k01 → k02: 같은 명령 수(9.42 G)에서 L1/TEX 바이트가 1/6.6로 줄고, 명령 간 cycle이 239 → 33.
- k02 → k03 (2048³): 스펙 가설 "DRAM 바이트가 크게 준다"는 **틀렸다**. DRAM·L2 바이트는 거의 같고, L1/TEX 바이트가 1/20, 명령이 −32 %. 병목이 L1 global load(LG throttle)에서 shared(MIO throttle)·barrier로 옮겨 가서 1.32배에 그쳤다.
- k03 → k04: 명령 −57 %, IPC 0.87 → 2.92. (옛 기본 타일에서는 레지스터 115, occupancy 31.9 %로도 6.3배 빨라졌다.)
- k04 → k05: global load 대기(long scoreboard)가 거의 사라졌다(스펙 가설 확인). DRAM read도 줄었다(새 기본 타일 기준 4.32 → 2.08 GB. 옛 기본 타일에서는 반대로 2배가 됐다, KNOWN_ISSUES #6).
- k05 vs cuBLAS(추가 개선 전): occupancy·레지스터·IPC가 비슷하고 큰 차이는 barrier stall(5.1배)과 MIO throttle(60,144 vs 1,737). 원인인지는 **확인하지 않음**.
- 추가 개선 2회차: 큰 타일(TN=8)에서 이웃 스레드가 B를 32 B 간격으로 읽어 shared load bank conflict가 10.7억 회였다. 열을 4개 묶음으로 흩어 0이 됐고 4096³ 8387 → 8777 GFLOPS. 출력 비트는 같다.

## 정확성

- 작은 크기(max(M,N,K) ≤ 512)는 CPU FP64, 큰 크기는 GPU FP64 naive 커널과 비교한다. 허용 오차 |x − ref| ≤ 1e−6·K + 1e−5·√K·|ref|.
- C++ 벤치: 커널·타일 전부 × {1³, 7×33×5, 33×47×129, 1000³, 64×64×3, 128×128×8} × {기본, 포인터 1 float 어긋남 + ld+1, C에 NaN·β=0, α=0.5·β=2}. 최종 측정 72건, sweep 580건 모두 통과.
- pytest(`tests/test_sgemm.py`, `torch.matmul` FP64 기준): 54 passed, 2 skipped(k01·k02의 4096).
- k01~k05의 출력은 1000³·2047³·4096³에서 **비트 단위로 같다**(FNV 해시). 다섯 커널 모두 k = 0..K−1 순서로 하나씩 누적하기 때문이다(KNOWN_ISSUES #4).

## 측정 조건

- GPU: RTX 3060 Ti 8 GB (sm_86, 38 SM, L2 3 MiB), 드라이버 581.80(Windows) / 580.105.07(WSL), WSL2 Ubuntu 22.04.
- CUDA 12.2(`/usr/local/cuda`, nvcc 12.2.140), g++ 11.4, cmake 3.22.1, Nsight Compute 2026.3.1, PyTorch 2.9.1+cu128.
- 클럭: Windows 관리자 cmd에서 `nvidia-smi -lgc 1500,1500`(WSL 안에서는 권한 없음). 최종 측정 동안 1초 간격 클럭 로그 1,040개가 모두 SM 1500 MHz, 온도 38~68 °C(`docs/results/2026-11-10_clocks.csv`).
- TF32 끔: cuBLAS `CUBLAS_DEFAULT_MATH`, `NVIDIA_TF32_OVERRIDE=0`, PyTorch `allow_tf32 = False`.

## 빌드와 실행

```bash
# C++ 라이브러리·벤치 (CMakeLists가 /usr/local/cuda/bin/nvcc를 우선 사용, LEDGER R4)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/check_args                       # 인자 검사
./build/devinfo                          # 장비 정보
NVIDIA_TF32_OVERRIDE=0 ./build/hxgemm_bench --kernel all --shape 4096,4096,4096
./build/hxgemm_bench --kernel k05 --sweep --shape 1000,1000,1000 --offset 1 --check-only
./build/hxgemm_bench --kernel k05 --tile 128,128,32,8,4,2 --tile 128,256,16,16,8,3 --shape 8192,8192,8192   # --tile 여러 번

# 클럭 로그와 함께 측정 / Nsight 수집 / 타일 탐색
bash scripts/measure.sh <date> --kernel all --squares 256:8192
bash scripts/ncu_collect.sh <date> k05 && python3 scripts/ncu_summary.py <date> k05 > docs/results/<date>_ncu_summary.csv
bash scripts/sweep.sh <date>

# PyTorch 확장 (venv: ~/venvs/hxgemm, LEDGER R9)
CUDA_HOME=/usr/local/cuda ~/venvs/hxgemm/bin/pip install -e . --no-build-isolation
NVIDIA_TF32_OVERRIDE=0 ~/venvs/hxgemm/bin/python -m pytest tests -q
NVIDIA_TF32_OVERRIDE=0 ~/venvs/hxgemm/bin/python benchmarks/bench_sgemm.py --out docs/results/<date>_py_bench
```

Python에서는 `import hxgemm_ext` 후 `torch.ops.hxgemm.sgemm(a, b, kernel, [])` 또는 `torch.ops.hxgemm.sgemm_out(c, a, b, kernel, tile, alpha, beta)`.

## 구조

```
csrc/gemm.h               C 인터페이스 (hxgemm_sgemm, hxgemm_default_tile, hxgemm_list_tiles)
csrc/dispatch.cu          인자 검사, K=0 경로
csrc/kernels/             k01~k05, scale(K=0), launch(커널 선택), tiles.h(타일 목록), cp_async.cuh
csrc/torch_bindings.cpp   TORCH_LIBRARY(hxgemm) 등록
bench/                    hxgemm_bench(검증·측정·sweep), check_args, harness.h
tests/, benchmarks/       pytest, Python 벤치
scripts/                  measure.sh, ncu_collect.sh, ncu_summary.py, sweep.sh
tools/                    devinfo.cu, nsight_gui.py(WSLg Nsight GUI 캡처)
docs/results/             원본 측정 출력(.ncu-repz는 git 제외)
```

## 확인하지 않은 것

`docs/KNOWN_ISSUES.md`에 정리했다. 요약: k05와 cuBLAS의 남은 차이(4096³ 5.6 %, 8192³ 20 %)의 원인, 3회차 C 저장 float4가 느려진 이유, 디코드 모양(M=16)에서 cuBLAS와 차이가 큰 이유(31.7 %), 자동 선택 규칙이 남기는 약 7 % 손해(1536³ 등), 옛 기본 타일 k05에서 DRAM read가 2배가 된 이유.
