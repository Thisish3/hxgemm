# Nsight Compute 캡처: k04(register tiling) → k05(cp.async 2단 파이프라인), 4096³

Tistory 글 "k05 cp.async" 편에 쓸 그림. 2026-11-10 데스크탑(RTX 3060 Ti, WSL2)에서 Nsight Compute 2026.3.1 GUI(WSLg `ncu-ui`)로 캡처했다. 앞 편은 `../2026-11-10_nsight_k03_k04/`.

## 측정 조건

- 리포트: `docs/results/2026-11-10_ncu_full_k04_k05.ncu-repz`(`.gitignore`, 데스크탑에만 있음) / 텍스트: `docs/results/2026-11-10_ncu_full_k04_k05.txt`
- 명령(저장소 루트):
  ```bash
  NVIDIA_TF32_OVERRIDE=0 /usr/local/cuda/bin/ncu --clock-control none --set full -k regex:'k0[45]' \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,l1tex__t_bytes.sum,smsp__inst_executed.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    -o docs/results/2026-11-10_ncu_full_k04_k05 -f \
    ./build/hxgemm_bench --kernel k04 --kernel k05 --shape 4096,4096,4096 --check-only
  ```
- 클럭은 Windows 관리자 `nvidia-smi -lgc 1500,1500`(ncu `--clock-control none`). 두 커널 모두 기본 타일 128×128×8, TM=TN=8, 256 스레드. k05는 2단(stages=2), float4 + `cp.async.cg` 16 B 경로.
- 벤치 숫자(같은 날, CUDA event 중앙값, `docs/results/2026-11-10_k05.csv`): k04 6589.6 GFLOPS(cuBLAS 대비 70.9 %), **k05 2단 7458.5 GFLOPS(80.2 %)**, cuBLAS 9298.1 GFLOPS. 3단은 7478.1 GFLOPS(80.4 %, `2026-11-10_k05_s3.csv`).
- `vs_k04` 그림은 k04가 Baseline(초록), k05가 Current(파랑)이다.

## 그림 목록

| 파일 | 화면 | 글에서 보여 줄 것 |
|---|---|---|
| `01_summary.png` | Summary | Duration 20.80 → 18.49 ms, 레지스터 115 → 128 |
| `02_k05_summary_rules.png` | k05 Summary 규칙 | 남은 개선 후보: C store sector 활용 4/32 B(Est. 66.92 %), shared 초과 wavefront 44 %(Est. 43.91 %), shared load 5.0-way conflict(Est. 30.49 %) |
| `03_k05_vs_k04_overview.png` | 상단 지표 비교 | Duration −11.08 %, IPC 2.22 → 2.63(+18.6 %), shared **store** conflict 16.8 M → 0, DRAM read 1.06 → 2.14 GB(+102.55 %), 명령 +6.5 % |
| `04_k05_vs_k04_sol.png` | SOL 차트 | Compute 54.8 → 65.7 %, Memory 69.3 → 76.5 % |
| `05_k05_vs_k04_roofline.png` | FP32 Roofline | FP32 최고 대비 45 % → 51 % |
| `06_k05_vs_k04_compute_memory.png` | Compute·Memory 비교 | FMA 파이프 52.2 %, L2 hit rate 77.13 → 57.46 % |
| `07_k05_vs_k04_memory_chart.png` | Memory Chart 비교 | 새 경로 "Load Global Store Shared" 8.39 M 명령(= cp.async), global load 명령 −94 %, shared store 0, DRAM→L2 2.14 GB |
| `08_k05_vs_k04_scheduler.png` | Scheduler 비교 | eligible warp 1.31 → 1.97(+49.6 %), issued 0.56 → 0.66, 명령 간격 7.00 → 6.00 cycle |
| `09_k05_vs_k04_warp_state.png` | Warp State 비교 | **Long Scoreboard(global load 대기)가 k04 약 1.4 cycle → k05 거의 0**. Short Scoreboard도 감소. 대신 Not Selected(일할 warp가 남음)가 1위 |
| `10_k05_source_cp_async.png` | k05 Source: `load()` 람다 | `cp_async16` 줄들의 stall 샘플은 각 1~3 % |
| `11_k05_source_pipeline.png` | k05 Source: 메인 루프 | `cp_async_wait`(52행) < 0.01 %, `__syncthreads`(53행) 0.2 %, `load()` 호출(55행) 29.2 %, FFMA(68행) 11.25 %, `ra` shared load(62행) 8.96 % |

## 확인한 것과 모르는 것

- 확인(측정): 스펙 가설 "메모리 대기 stall이 줄어든다"는 맞았다. Long Scoreboard가 사실상 사라지고 eligible warp가 50 % 늘었다. cp.async가 shared에 직접 쓰므로 shared store bank conflict도 0이 됐다.
- 미확인: DRAM read가 2배(1.06 → 2.14 GB), L2 hit rate가 77 → 57 %로 떨어진 이유. `cp.async.cg`(L1 우회)와 관련이 있을 수 있지만 측정하지 않았다(`docs/KNOWN_ISSUES.md` #6).
- 미확인: 55행 `load()` 호출 지점이 stall 샘플의 29 %를 차지하는 이유(주소 계산인지 cp.async 발행 대기인지). SASS 단위로 봐야 한다.
- occupancy는 두 커널 모두 레지스터 한도로 33.3 %(k05 레지스터 128, shared 16 KiB/블록).
