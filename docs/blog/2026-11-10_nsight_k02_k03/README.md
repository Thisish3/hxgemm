# Nsight Compute 캡처: k02(coalescing) → k03(shared tiling), 2048³

Tistory 글 "k03 shared tiling" 편에 쓸 그림. 2026-11-10 데스크탑(RTX 3060 Ti, WSL2)에서 Nsight Compute 2026.3.1 GUI(WSLg `ncu-ui`)를 띄워 캡처했다.

## 측정 조건

- 리포트: `docs/results/2026-11-10_ncu_full_k02_k03.ncu-repz` (10 MB, `.gitignore`로 커밋 안 함. 데스크탑 `~/hxgemm`에 있음) / 텍스트 요약: `docs/results/2026-11-10_ncu_full_k02_k03.txt`(커밋됨)
- 명령(저장소 루트):
  ```bash
  NVIDIA_TF32_OVERRIDE=0 /usr/local/cuda/bin/ncu --clock-control none --set full -k regex:'k0[23]' \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,l1tex__t_bytes.sum,smsp__inst_executed.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    -o docs/results/2026-11-10_ncu_full_k02_k03 -f \
    ./build/hxgemm_bench --kernel k02 --kernel k03 --shape 2048,2048,2048 --check-only
  ```
- 클럭: Windows 관리자 `nvidia-smi -lgc 1500,1500`으로 고정. ncu는 `--clock-control none`이라 모든 화면 상단에 "Data collection happened without GPU frequencies fixed by the profiler" 경고가 뜬다. 고정은 우리가 직접 했고, 리포트의 SM Frequency는 1.50 GHz다.
- 벤치 숫자(같은 날, CUDA event 중앙값): k02 804.4 GFLOPS, k03 1063.1 GFLOPS, cuBLAS 8820.8 GFLOPS (`docs/results/2026-11-10_k03.csv`).
- `vs_k02` 그림은 k02를 Baseline(초록), k03을 Current(파랑)로 둔 비교 화면이다. 괄호 안 %는 k02 대비 변화.

## 그림 목록

| 파일 | 화면 | 글에서 보여 줄 것 |
|---|---|---|
| `01_summary.png` | Summary | 두 커널 Duration 21.34 → 16.17 ms. Compute/Memory Throughput 88.4 → 80.3 % |
| `02_k02_sol.png` | k02 Details · GPU Speed Of Light | k02는 Compute·Memory 88.4 %, L1/TEX 88.77 %인데 DRAM은 12.33 %. DRAM이 아니라 L1이 바쁘다 |
| `10_k02_memory_chart.png` | k02 Memory Chart | Global load 537 M 명령, L1/TEX hit 94.96 %, L2→L1 2.15 GB, DRAM→L2 1.13 GB. 재사용을 이미 L1이 받아 주고 있다 |
| `09_k02_warp_state.png` | k02 Warp State | 주된 stall이 LG Throttle(약 20 cycle/inst). L1 load 큐가 가득 참 |
| `03_k03_vs_k02_sol.png` | SOL 비교 | Duration −24.25 %, L1/TEX bytes −94.96 %, 명령 −32.14 %, DRAM read +1.64 %(거의 그대로), shared bank conflict 0 → 4.33 M |
| `05_k03_vs_k02_memory_chart.png` | Memory Chart 비교 | Global 명령 −96.85 %(16.9 M), Shared 352 M 명령 신규, L2→L1 2.15 GB와 DRAM 1.14 GB는 변화 없음 |
| `04_k03_vs_k02_compute_memory.png` | Compute/Memory Workload 비교 | IPC 0.87(−10 %), L1/TEX hit 0 %(shared로 옮겨 감), "Shared Store Bank Conflicts: 평균 1.3-way, shared store wavefront의 20.53 %" |
| `06_k03_vs_k02_warp_state.png` | Warp State 비교 | 초록(k02) LG Throttle이 사라지고 파랑(k03) MIO Throttle(약 21 cycle)과 Barrier가 생긴다. 병목이 L1 global load에서 shared/동기화로 이동 |
| `07_k03_vs_k02_launch_occupancy.png` | Launch/Occupancy 비교 | Occupancy 66.67 %로 동일(블록 1024 스레드, SM당 1블록). shared 8 KiB/블록 추가 |
| `08_k03_source_stalls.png` | k03 Source(줄별) | 18행 `acc += As[ty][k] * Bs[k][tx]`가 실행 명령의 75 %, stall 샘플(not-issued)의 76.06 %. 타일 적재 14·15행이 각 약 10 % |

## 이 캡처로 확인한 것과 아직 모르는 것

- 확인(측정): 스펙 가설 "k03에서 DRAM 바이트가 크게 준다"는 틀렸다. DRAM·L2 바이트는 거의 같고, 바뀐 것은 L1/TEX 바이트(−95 %)와 명령 수(−32 %)다. 병목은 L1 global load(LG throttle)에서 shared 접근(MIO throttle)과 barrier로 옮겨 갔다. 그래서 1.32배에 그쳤다.
- 미확인: shared **store** bank conflict(1.3-way, 4.33 M)가 어느 store에서 나는지. 코드상 `As[ty][tx]`/`Bs[ty][tx]`는 warp 안에서 연속 주소라 conflict가 없어 보이는데 Nsight는 conflict를 보고한다. 원인 미확인(`docs/KNOWN_ISSUES.md` #3).
- 다음 글(k04 register tiling)에서 볼 것: 스레드당 출력 개수를 늘려 shared 읽기 횟수를 줄이면 MIO throttle이 줄어드는가.
