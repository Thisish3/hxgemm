# Nsight Compute 캡처: k03(shared tiling) → k04(register tiling), 4096³

Tistory 글 "k04 register tiling과 occupancy" 편에 쓸 그림. 2026-11-10 데스크탑(RTX 3060 Ti, WSL2)에서 Nsight Compute 2026.3.1 GUI(WSLg `ncu-ui`)로 캡처했다. 앞 편은 `../2026-11-10_nsight_k02_k03/`.

## 측정 조건

- 리포트: `docs/results/2026-11-10_ncu_full_k03_k04.ncu-repz`(`.gitignore`, 데스크탑 `~/hxgemm`에만 있음) / 텍스트: `docs/results/2026-11-10_ncu_full_k03_k04.txt`
- 명령(저장소 루트):
  ```bash
  NVIDIA_TF32_OVERRIDE=0 /usr/local/cuda/bin/ncu --clock-control none --set full -k regex:'k0[34]' \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,l1tex__t_bytes.sum,smsp__inst_executed.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    -o docs/results/2026-11-10_ncu_full_k03_k04 -f \
    ./build/hxgemm_bench --kernel k03 --kernel k04 --shape 4096,4096,4096 --check-only
  ```
- 클럭은 Windows 관리자 `nvidia-smi -lgc 1500,1500`(ncu `--clock-control none`). k04는 기본 타일 BM=BN=128, BK=8, TM=TN=8, 256 스레드, float4 경로.
- 벤치 숫자(같은 날, CUDA event 중앙값, `docs/results/2026-11-10_k04.csv`): k03 1047.2 GFLOPS(cuBLAS 대비 11.3 %), **k04 6603.6 GFLOPS(71.0 %)**, cuBLAS 9297.6 GFLOPS.
- `vs_k03` 그림은 k03이 Baseline(초록), k04가 Current(파랑)이다. 괄호 안 %는 k03 대비 변화.

## 그림 목록

| 파일 | 화면 | 글에서 보여 줄 것 |
|---|---|---|
| `01_summary.png` | Summary | Duration 129.52 → 20.87 ms. 레지스터 38 → 115. k03 진단은 MIO throttle |
| `02_k04_summary_rules.png` | k04 Summary 규칙 | Nsight가 제안하는 다음 개선: ① C global store가 sector 32 B 중 4 B만 사용(Est. 61.48 %), ② shared 접근 초과 wavefront 40 %(Est. 40.05 %), ③ 레지스터로 occupancy 33.3 % 제한(Est. 29.74 %) |
| `03_k04_vs_k03_overview.png` | 상단 지표 비교 | DRAM read 9.04 → 1.06 GB(−88 %), L1/TEX bytes −73 %, 명령 −59.5 %, IPC 0.87 → 2.19(+152 %), shared **load** bank conflict 0 → 268 M |
| `04_k04_vs_k03_sol.png` | SOL 차트 | Compute 80.1 → 55.6 %, Memory 80.1 → 70.3 %. 둘 다 내려갔는데 시간은 −84 % |
| `05_k04_vs_k03_roofline.png` | FP32 Roofline | k03 ≈ 1 TFLOP/s, k04 ≈ 6.5 TFLOP/s(이론 최고의 46 %). L1/L2/DRAM 연산 강도 점이 오른쪽으로 이동 |
| `06_k04_vs_k03_compute_memory.png` | Compute·Memory 비교 | FMA 파이프 46.7 %, 가장 많이 실행된 명령은 64행 FFMA 2.15e9회 |
| `07_k04_vs_k03_memory_chart.png` | Memory Chart 비교 | Shared 명령 −94.5 %, Global 명령 −93.4 %, L2→L1 4.11 GB(−76 %), DRAM 1.06 GB(−88 %). 레지스터 재사용 효과가 모든 계층에 보인다 |
| `08_k04_vs_k03_scheduler.png` | Scheduler 비교 | 스케줄러당 active warp 7.99 → 3.88, issued warp 0.22 → 0.56/cycle. 명령 간격 36.6 → 6.97 cycle |
| `09_k04_vs_k03_warp_state.png` | Warp State 비교 | k03의 MIO Throttle(약 21 cycle)이 k04에서 사라짐. 남은 stall은 모두 수 cycle |
| `10_k04_vs_k03_launch_occupancy.png` | Launch/Occupancy | 레지스터 115 → 블록 한도 2 → theoretical occupancy 33.3 %, achieved 31.87 %. 그래도 6.3배 빨라짐 = occupancy보다 명령 수·재사용이 컸다 |
| `11_k04_source.png` | k04 Source(줄별) | 64행 FFMA가 stall 샘플 19.92 %, 74행 C store 6.52 %, 54행 `__syncthreads` 7.89 %. live register 최대 98 |

## 확인한 것과 다음에 볼 것

- 확인(측정): 스펙 가설 "FFMA 비중↑, 레지스터↑, occupancy↓"가 그대로 나왔다(IPC 2.19, 레지스터 115, occupancy 31.87 %). occupancy가 절반이 됐는데도 6.3배 빨라졌다.
- 4096³에서는 k03의 DRAM read가 9.04 GB였다(2048³에서는 1.14 GB). k04는 1.06 GB.
- 다음 수동 개선 후보(Task 10, Nsight 근거): shared load bank conflict 평균 5.0-way(268 M, load wavefront의 39.9 %), C store 비연속(sector 활용 4/32 B), shared store conflict 2.4-way. 어느 것을 먼저 할지는 Task 7 자동 탐색 결과를 보고 정한다.
