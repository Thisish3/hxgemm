# Known issues

미뤄 둔 문제와 확인하지 못한 현상. 리뷰할 때마다 갱신한다. (Task 9에서 형식을 hxccl과 맞춘다.)

## 열림

1. **데스크탑에 CUDA 툴킷이 두 벌 있고, 잘못 고르면 GPU를 못 찾는다** (2026-11-10, LEDGER R4)
   - apt `nvidia-cuda-toolkit` 11.5(`/usr/bin/nvcc`, `libcublas.so.11`)와 `/usr/local/cuda` 12.2가 함께 있다. apt `libnvidia-compute-{495,510,535,580}`도 설치되어 있어 `/lib/x86_64-linux-gnu/libcuda.so`(580.178.04, WSL용 아님)가 있다.
   - cuBLAS 11에 링크하면 이 `libcuda.so`가 로드되어 `no CUDA-capable device is detected`. 지금은 `CMakeLists.txt`가 `/usr/local/cuda/bin/nvcc`를 우선해 피한다.
   - 남은 위험: PyTorch 확장(Task 8)이나 다른 도구가 `libcuda.so`(버전 없는 이름)를 dlopen하면 같은 문제가 날 수 있다. 시스템 패키지 정리는 사용자 결정으로 남김.
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
4. **2048³·2047³에서 k01/k02/k03 결과가 cuBLAS와 비트 단위로 같다** (max_abs_err 0, 1000³에서는 8.0e-05)
   - 큰 크기의 정확성 기준이 cuBLAS라서 검증이 자명해 보일 수 있다. 작은 크기(≤ 512)는 CPU FP64와 비교해 통과했다. 같은 이유는 원인 미확인(누적 순서가 같다는 추측은 미확인).
