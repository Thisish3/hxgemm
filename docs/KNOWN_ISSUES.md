# Known issues

미뤄 둔 문제와 확인하지 못한 현상. 리뷰할 때마다 갱신한다. (Task 9에서 형식을 hxccl과 맞춘다.)

## 열림

1. **데스크탑에 CUDA 툴킷이 두 벌 있고, 잘못 고르면 GPU를 못 찾는다** (2026-11-10, LEDGER R4)
   - apt `nvidia-cuda-toolkit` 11.5(`/usr/bin/nvcc`, `libcublas.so.11`)와 `/usr/local/cuda` 12.2가 함께 있다. apt `libnvidia-compute-{495,510,535,580}`도 설치되어 있어 `/lib/x86_64-linux-gnu/libcuda.so`(580.178.04, WSL용 아님)가 있다.
   - cuBLAS 11에 링크하면 이 `libcuda.so`가 로드되어 `no CUDA-capable device is detected`. 지금은 `CMakeLists.txt`가 `/usr/local/cuda/bin/nvcc`를 우선해 피한다.
   - 남은 위험: 다른 도구가 `libcuda.so`(버전 없는 이름)를 dlopen하면 같은 문제가 날 수 있다. PyTorch 확장(Task 8, `CUDA_HOME=/usr/local/cuda`로 빌드)은 문제없이 동작했다(pytest 54 passed). 시스템 패키지 정리는 사용자 결정으로 남김.
2. **클럭 고정은 Windows 관리자 권한에서만 된다** (LEDGER R3)
   - WSL 안에서는 `nvidia-smi -lgc`가 권한 오류. 측정 전 Windows 관리자 cmd에서 `nvidia-smi -lgc 1500,1500`, 끝나면 `nvidia-smi -rgc`. Mac에서 `ssh desk`로 측정할 때는 고정 여부를 클럭 로그로만 확인할 수 있다.
   - ncu `--clock-control` 값과 `-lgc`의 관계는 Task 9에서 확인한다.
3. **k02 → k03가 2048³에서 1.32배뿐** (804.4 → 1063.1 GFLOPS, 2026-11-10)
   - ncu 2026.3.1, `--clock-control none`(SM 1.498 GHz로 보고), 2048³, 원본 `docs/results/2026-11-10_ncu_k02_k03.txt`:

     | 지표 | k02 | k03 |
     |---|---|---|
     | Duration | 21.35 ms | 16.18 ms |
     | DRAM read | 1.126 GB | 1.144 GB |
     | L2 (lts__t_bytes) | 2.191 GB | 2.190 GB |
     | L1/TEX (l1tex__t_bytes) | 42.97 GB | 2.16 GB |
     | L1/TEX throughput | 88.75 % | 80.42 % |
     | DRAM throughput | 12.33 % | 16.51 % |
     | 실행 명령 수 (smsp__inst_executed) | 1.181 G | 0.801 G |
     | shared bank conflict | 0 | 4.30 M |
     | 주요 stall (warp당 누적 비율) | lg_throttle 178,904 / long_scoreboard 45,238 | mio_throttle 129,152 / long_scoreboard 33,719 / barrier 15,200 |
     | Achieved occupancy | 66.65 % | 66.65 % |

   - 측정으로 확인한 것: 스펙 가설 "k03에서 DRAM 바이트가 크게 준다"는 **틀렸다.** DRAM·L2 바이트는 k02와 k03가 거의 같다. k02는 이미 재사용을 L1에서 받고 있었고(L1/TEX 43 GB), 병목은 L1/TEX 파이프(88.75 %, lg_throttle stall)였다. k03는 global 접근을 shared로 바꿔 L1/TEX 바이트가 1/20이 되고 명령 수가 32 % 줄었지만, 병목이 shared(MIO throttle)와 barrier로 옮겨 갔다. shared 처리량이 L1/TEX 80 %로 높게 남아 있는 것이 1.32배에 그친 이유와 맞는다.
   - 남은 것: 이 해석을 k04(레지스터 타일링으로 shared 읽기 횟수 감소)에서 확인한다. bank conflict 4.3 M이 시간에 얼마나 기여하는지는 원인 미확인.
   - `--set full` 재수집(`docs/results/2026-11-10_ncu_full_k02_k03.txt`, GUI 캡처 `docs/blog/2026-11-10_nsight_k02_k03/`)에서 추가로 본 것:
     - bank conflict는 shared **store**에서 보고된다: "평균 1.3-way, 16,777,216 store 요청 중 4,331,909 conflict, store wavefront의 20.53 %"(Est. Speedup 16.54 %). `As[ty][tx]`·`Bs[ty][tx]` store는 warp 안에서 연속 주소인데 왜 conflict가 나는지 원인 미확인. Source Counters에서 줄 단위로 확인할 것.
     - k02 rule: "global load에서 sector 32 B 중 평균 26.4 B만 사용"(Est. Speedup 15.47 %). 2048은 32의 배수인데 왜 100 %가 아닌지 원인 미확인.
     - k03 Source: 18행 내부 루프가 not-issued stall 샘플의 76.06 %, 타일 적재 14·15행이 10.25 %·10.19 %.
5. **k04(옛 기본 타일 128x128x8x8x8)의 Nsight 진단: shared load conflict, C store 비연속** (4096³, 2026-11-10, `docs/blog/2026-11-10_nsight_k03_k04/`)
   - shared load: 평균 5.0-way bank conflict, 268,435,456 conflict = load wavefront의 39.93 %(Est. Speedup 27.69 %).
   - shared store: 평균 2.4-way, 16,777,216 conflict = store wavefront의 33.33 %(Est. 23.12 %).
   - C global store: sector 32 B 중 평균 4.0 B만 사용(Est. 61.48 %). 스레드가 TN=8개 연속 열을 스칼라로 쓰므로 warp 안 스레드 간 간격이 8 float인 것과 맞지만, 그것이 원인이라고 아직 측정으로 확인하지 않음.
   - occupancy 33.3 %(레지스터 115, Est. 29.74 %).
   - 이 중 무엇이 시간에 가장 큰지는 원인 미확인. Task 7(자동 탐색) 후 Task 10 수동 개선에서 하나씩 바꿔 잰다.
6. **k05(옛 기본 타일)에서 DRAM read가 k04의 2배, L2 hit rate 하락** (4096³, 2026-11-10, `docs/blog/2026-11-10_nsight_k04_k05/`)
   - DRAM read 1.06 → 2.14 GB, L2 hit 77.13 → 57.46 %. 그래도 시간은 −11 %(20.80 → 18.49 ms).
   - 새 기본 타일(R10)에서는 반대로 k04 4.32 GB → k05 2.08 GB로 줄었다(최종 ncu). 옛 타일에서 늘어난 원인은 미확인. 후보: `cp.async.cg`(L1 우회)로 A/B 재사용이 L1에서 L2로 옮겨 감, 블록 실행 순서 변화. `cp.async.ca`로 바꿔 보거나 L1/L2 sector 지표를 비교해 확인한다.
   - k05의 55행 `load()` 호출이 stall 샘플 29.2 %를 차지하는 이유도 미확인(SASS로 확인).
7. **Python 벤치에서 본 비배수·작은 모양 성능** (2026-11-10, `docs/results/2026-11-10_py_bench.csv`, 기본 타일)
   - k05가 2047³(4914.2 GFLOPS)·4095³(5207.0)에서 k04(5494.5·6083.6)보다 느리다. 4096³에서는 k05가 빠르다(7459.6 vs 6618.6). 코드상 ld가 4의 배수가 아니면 k04·k05 모두 float4/16 B 경로 대신 스칼라 경로를 쓰고, k05의 스칼라 경로는 4 B `cp.async`를 원소마다 발행한다. 이것이 느린 이유인지는 측정하지 않았다(원인 미확인).
   - k01이 1000³(287.0)·2047³(366.2)에서 2의 거듭제곱 크기(약 109)보다 2.6~3.4배 빠르다. 원인 미확인.
   - 8192³에서 k05는 torch 대비 66.3 %(4096³ 80.2 %보다 낮음), torch는 8192³에서 11402.8 GFLOPS로 4096³(9297.4)보다 높다. 원인 미확인.
   - 디코드 모양 16×4096²에서 k05 기본 타일은 torch 대비 24.2 %(sweep 최고 타일도 55.8 %).
8. **k05와 cuBLAS의 남은 차이** (2026-11-10, 추가 개선 3회 후, LEDGER R12~R16)
   - 4096³: k05(자동 선택 → 128x256x16x16x8x3) 15.75 ms vs cuBLAS 14.79 ms(94.4 %). ncu(`docs/results/2026-11-10_ncu_summary_tuned.csv`): IPC 2.89 vs 2.99, 명령 간 2.76 vs 5.31 cycle, 실행 명령 2.49 G vs 2.52 G, 레지스터 255 vs 122, occupancy 16.7 % vs 33.2 %. 큰 차이는 여전히 원인 미확인.
   - 8192³: 9053 vs 11270 GFLOPS(80.1 %). cuBLAS 커널 `cutlass_80_simt_sgemm_256x128_8x4`는 IPC 3.48, 명령 간 2.30 cycle, k05 2.92·2.74. MIO throttle 53,477 vs 16,807. 원인 미확인.
   - 3회차(C 저장 float4)는 느려져 되돌렸다(R14, 원인 미확인).
   - k05 기본 타일 자동 선택 규칙(R16)은 1536³·256~768×4096²에서 최적 대비 약 7 % 손해를 남긴다(블록 수 32·64·96에서는 큰 타일이 빠른데 규칙이 작은 타일을 고름).

## 해결

9. **8192³에서 k04·k05가 4096³보다 낮아 보였다** (2026-11-10 열림 → 같은 날 해결, LEDGER R12)
   - k05 자체는 4096³ 8417 → 8192³ 8487 GFLOPS로 떨어지지 않았다. cuBLAS가 8192³에서 다른 커널(`cutlass_80_simt_sgemm_256x128_8x4`, 레지스터 216)로 바꿔 9298 → 11397 GFLOPS로 빨라진 것이 비율이 떨어진 이유다(ncu로 커널 이름과 지표 확인). 추가 개선 후 8192³ k05는 80.1 %.

4. **큰 크기 검증이 cuBLAS 기준이라 자명해질 수 있었다** (2026-11-10 열림 → 같은 날 해결, LEDGER R7)
   - 당시 현상: 2048³·2047³에서 k01/k02/k03 결과가 cuBLAS와 비트 단위로 같았고(max_abs_err 0), 4096³에서는 k03과 k04의 오차가 완전히 같았다.
   - 조치: 벤치의 큰 크기(max(M,N,K) > 512) 기준을 cuBLAS에서 **GPU FP64 naive 커널**(`ref_gpu_fp64`, 원소당 스레드 1개, FP64 누적)로 바꿨다. cuBLAS도 이제 이 기준으로 검증된다. 결과 행마다 출력 비트의 FNV-1a 해시(`fnv=`)를 찍어 커널끼리 비트 단위로 비교한다.
   - 결과(측정, α=1·β=0, offset 0):

     | 크기 | k01~k05 오차(max_abs) | k01~k05 해시 | cuBLAS 오차 | cuBLAS 해시 |
     |---|---|---|---|---|
     | 1000³ | 7.847e-05 (5개 같음) | 모두 `a7af9f3f29ac5922` | 2.034e-05 | 다름 |
     | 2047³ | 1.607e-04 | 모두 `f3fb52c6b3029f43` | 1.607e-04 | **같음** |
     | 4096³ (k03~k05) | 3.868e-04 | 모두 `45e5410fa35be654` | 1.200e-04 | 다름 |
     | 4096³ α=0.5·β=2·offset 1 (k04, k05) | 1.850e-04 | 같음 | 5.111e-05 | 다름 |
     | 2048³ NaN-C (k04, k05) | 1.577e-04 | 같음 | – | – |

   - 결론: **k01~k05(k03과 k04 포함)의 출력은 비트 단위로 같다**(측정). 코드상 다섯 커널 모두 원소마다 `acc = 0`에서 시작해 k = 0, 1, …, K−1 순서로 `acc += a*b`를 하나씩 누적하고 마지막에 `hx_out`을 적용하므로, FP32 연산 순서가 같다. 타일링은 데이터를 어디서 읽는지만 바꾸고 누적 순서는 바꾸지 않는다. cuBLAS는 2047³에서만 같고 1000³·4096³에서는 다른 순서로 누적한다(4096³에서 오차가 3.2배 작음). cuBLAS 내부 순서는 미확인.
   - 독립 기준에서 모든 커널이 통과하므로 "자명한 검증" 문제는 닫는다. 대신 다섯 커널이 같은 결과를 내므로, 커널끼리 비교하는 검증은 누적 순서를 바꾸는 최적화(split-K 등, G3)에서만 의미가 있다.
