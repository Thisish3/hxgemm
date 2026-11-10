# hxgemm G1: FP32 SGEMM을 단계별로 만들기

- 작성일: 2026-11-08
- 상태: 사용자 리뷰 대기
- 전체 프로젝트: NVIDIA 스택 로드맵의 CUDA/cuBLAS 층. G1(FP32) → G2(Tensor Core, FP16/BF16) → G3(INT8, FP8 에뮬레이션, split-K) → G4(HIP 이식). 이 문서는 G1만 다룬다.

## 0. 목적과 성공 기준 (사용자 합의)

- 목적: 학습 목차 05(GEMM, tiling), 06(성능 분석), 11(프로파일링)을 직접 만들고 재면서 익힌다. 포트폴리오와 블로그에 쓴다.
- 성공 기준 1 (설명): 커널 단계마다 "바꾼 코드 → 바뀐 Nsight 지표 → 바뀐 GFLOPS"를 실측으로 설명한다. 예상과 다르게 나오면 그대로 기록한다.
- 성공 기준 2 (숫자): 마지막 커널이 4096³ FP32에서 cuBLAS SGEMM 중앙값 속도의 80% 이상(같은 프로그램, 같은 조건, TF32 끔).
- 튜닝 기한: 5단계 완료 후 자동 탐색 1회 + 수동 개선 3회. 그 안에 80%에 못 닿으면 남은 차이를 Nsight 근거로 설명하고 `docs/KNOWN_ISSUES.md`에 남기고 G1을 닫는다.

## 1. 환경

- 개발: 맥(이 세션)에서 코드를 쓰고, SSH로 데스크탑 WSL2(RTX 3060 Ti 8 GB)에 접속해 빌드·실행·측정한다.
- 사용자가 할 일(1회): WSL2에 sshd 켜기, Tailscale 로그인(집 밖 접속용), 맥 공개 키 등록, 저장소를 데스크탑과 공유(비공개 GitHub 또는 rsync). 외부 접속을 여는 설정이라 Claude는 하지 않는다. 단계별 명령은 구현 계획 Task 0에 적는다.
- SSH 연결 직후 확인해서 이 문서 끝 "환경 기록"에 적을 것: CUDA 툴킷 버전, 드라이버 버전, `nvidia-smi` 출력, device query(SM 수, 클럭, L2 크기, SM당·블록당 shared memory 최대치), WSL2에서 `ncu`가 GPU 카운터를 읽는지, `nvidia-smi -lgc`(클럭 고정)가 되는지, PyTorch 버전과 CUDA 빌드.

## 2. 저장소 구조

```
hxgemm/
  csrc/
    gemm.h               C 인터페이스 (아래 3절)
    kernels/k01_naive.cu, k02_coalesce.cu, k03_smem_tile.cu, k04_reg_tile.cu, k05_double_buf.cu
    dispatch.cu          kernel 번호(+템플릿 설정) → 실행 설정 선택
    torch_bindings.cpp   TORCH_LIBRARY로 torch.ops.hxgemm.sgemm 등록 (vLLM csrc/torch_bindings.cpp와 같은 방식)
  bench/hxgemm_bench.cu  C++ 벤치: 검증 후 CUDA event로 측정, CSV 출력. ncu로 커널 하나만 잴 때 쓴다
  tests/test_sgemm.py    pytest: torch.matmul(TF32 끔)과 비교
  benchmarks/bench_sgemm.py  크기별 GFLOPS, cuBLAS 대비 %, 그래프
  scripts/ncu_collect.sh     정해 둔 섹션을 커널별로 수집
  docs/specs, plans, KNOWN_ISSUES.md, results/ (원본 측정 출력)
  CMakeLists.txt (CUDA, sm_86), setup.py (torch.utils.cpp_extension)
```

실무 근거: vLLM은 커널을 `csrc/`의 .cu로 쓰고 `csrc/torch_bindings.cpp`에서 `TORCH_LIBRARY_FRAGMENT`로 등록하며, 정확성은 `tests/kernels/`(pytest), 성능은 `benchmarks/kernels/`(Python)에서 본다(2026-11-08 로컬 vLLM 저장소에서 확인).

## 3. 인터페이스

```c
typedef enum { HXGEMM_K01 = 1, HXGEMM_K02, HXGEMM_K03, HXGEMM_K04, HXGEMM_K05 } hxgemm_kernel_t;
typedef struct { int bm, bn, bk, tm, tn, stages; } hxgemm_tile_t;   /* k04/k05만 사용, 0이면 기본값 */

/* C = alpha * A(MxK) * B(KxN) + beta * C(MxN), 모두 행 우선 FP32, 기기 메모리 포인터 */
int hxgemm_sgemm(hxgemm_kernel_t k, hxgemm_tile_t t, int M, int N, int K, float alpha,
                 const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc,
                 cudaStream_t stream);
```

- 반환값: 0 성공, 음수는 인자 오류(M/N/K < 0, ld가 너무 작음, null 포인터, 지원하지 않는 타일 설정)이며 커널을 실행하지 않는다. CUDA 실행 오류는 `cudaGetLastError` 값을 그대로 돌려준다.
- M, N, K 중 하나가 0이면 아무것도 하지 않고 성공(K = 0이면 C = βC를 계산한다. BLAS 의미론).
- cuBLAS 비교: cuBLAS는 열 우선이므로 C^T = B^T·A^T로 인자를 바꿔 `cublasSgemm`을 부른다. 이 변환은 테스트로 고정한다.

## 4. 커널

공통: 행 우선, C = αAB + βC. M, N, K가 타일 배수가 아니어도 맞아야 한다(타일 밖은 0으로 읽고 C에는 쓰지 않음). 숫자는 시작값이며 k04·k05는 자동 탐색으로 정한다.

| 커널 | 블록/스레드 | 핵심 | 확인할 가설 (Nsight) |
|---|---|---|---|
| k01 naive | 32×32, 스레드 1개 = C 1개 | `threadIdx.x`를 **행**에 대응(일부러 비연속 접근) | DRAM·L2를 반복해서 읽어 메모리에 묶인다 |
| k02 coalescing | k01과 같음 | `threadIdx.x`를 **열**에 대응 | 요청당 sector 수가 줄고 메모리 효율이 오른다 |
| k03 shared tiling | BM=BN=BK=32, 1024 스레드 | A·B 타일을 shared에 올려 재사용, `__syncthreads()` | DRAM 바이트가 크게 줄고 shared 접근·동기화가 병목이 된다 |
| k04 register tiling | 시작 BM=BN=128, BK=8, TM=TN=8, 256 스레드 | 스레드당 64개 누적, A 타일 전치 저장, float4 경로(정렬 안 맞으면 스칼라) | FFMA 비중↑, 레지스터↑, occupancy↓ |
| k05 cp.async | k04 + 2단 버퍼 | `cp.async`로 다음 타일 선적재, 경계는 크기 지정으로 0 채움 | 메모리 대기 stall이 줄어든다 |

자동 탐색: k04·k05를 템플릿으로 만들어 BM,BN ∈ {64,128}, BK ∈ {8,16,32}, TM,TN ∈ {4,8}, stages ∈ {2,3}(k05)의 조합 중 shared memory 한도(device query) 안의 것을 미리 컴파일한다. 4096³과 LLM 모양 3개에서 모두 재고 모양별 최적 조합을 표로 남긴다.

수동 개선 3회: 자동 탐색 결과의 Nsight 지표에서 가장 큰 병목부터 고친다. 후보는 warptiling, shared 패딩·swizzle(bank conflict), C 저장 벡터화이며, 무엇을 할지는 측정 후 정한다.

## 5. 측정

- 시간: warm-up 10회, 이후 50회를 각각 CUDA event로 재서 중앙값(최소·최대 함께 기록). 매 측정 직전 L2 크기 × 2 버퍼에 써서 L2를 비운다(L2 크기는 device query).
- 클럭: 고정이 되면 고정하고, 안 되면 측정 중 `nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu`를 1초 간격으로 기록한다.
- 크기: 정사각 256~8192(2배씩), 비배수 1000·2047·4095, LLM 모양 N=K=4096 × M ∈ {16, 128, 2048}.
- 지표: GFLOPS = 2MNK/t, cuBLAS 대비 %, 이론 최고 대비 %(SM 수 × SM당 FP32 유닛 128(Ampere GA10x) × 2 × 측정 중 SM 클럭).
- 공정성: cuBLAS는 `CUBLAS_DEFAULT_MATH`, PyTorch는 `torch.backends.cuda.matmul.allow_tf32 = False`, 실행 환경에 `NVIDIA_TF32_OVERRIDE=0`.
- 결과 파일: `docs/results/<date>_<kernel>.csv`와 원본 출력. README와 블로그 표는 이 파일에서만 옮긴다.

## 6. 정확성 검증

- 입력: 고정 seed 난수, U(−1, 1).
- 작은 크기(≤ 512): CPU FP64 결과와 비교. 큰 크기: GPU FP64 naive 커널 결과와 비교(2026-11-10 변경, 원래는 cuBLAS. cuBLAS 기준은 우리 커널과 비트 단위로 같아지는 경우가 있어 독립 기준으로 바꿈. LEDGER R7, KNOWN_ISSUES #4).
- 허용 오차: |x − ref| ≤ atol + rtol·|ref|, rtol = 1e−5·√K, atol = 1e−6·K(FP32 누적 오차가 K에 따라 커지는 것을 반영). 실제 최대 오차를 함께 기록한다. 이 기준으로 cuBLAS 자신도 FP64 대비 통과하는지 먼저 확인하고, 통과하지 못하면 기준을 조정하고 그 근거를 적는다.
- pytest: 커널 5개 × 크기 {1, 7, 33, 128, 1000, 2047, 4096} × (α, β) ∈ {(1,0), (0.5, 2), (1,1)} × 정렬 안 맞는 포인터(오프셋 1 float) × ld > 열 수인 경우. 4096은 k01에서 느리므로 k01·k02는 2047까지만.
- C++ 벤치는 검증에 실패한 커널의 시간을 출력하지 않는다.

## 7. Nsight Compute

- `scripts/ncu_collect.sh <kernel> <size>`: C++ 벤치를 `ncu`로 실행해 커널 하나만 잰다. 섹션 `SpeedOfLight`, `MemoryWorkloadAnalysis`, `ComputeWorkloadAnalysis`, `Occupancy`, `LaunchStats`, `WarpStateStats`, `SourceCounters`.
- 산출물: `.ncu-rep` 원본과 핵심 지표 CSV(`docs/results/`).
- WSL2에서 카운터 접근이 안 되면 Windows용 Nsight Compute로 같은 바이너리(또는 Windows 빌드)를 측정하고, 이 사실과 방법을 README에 적는다.

## 8. 오류 처리

- 인자 오류는 커널 실행 전에 음수 반환(3절). 바인딩은 이를 Python 예외(`ValueError`)로 바꾼다.
- 바인딩 입력 검사: CUDA 텐서, float32, 2차원, 내부 차원 일치, 행 우선 연속이거나 행 stride가 있는 경우(`lda`로 전달). 아니면 `ValueError`.
- CUDA 실행 오류는 C++ 벤치에서 메시지와 함께 종료, 바인딩에서는 `RuntimeError`.

## 9. 범위 밖 (G1)

FP16/BF16, Tensor Core(G2), 양자화와 split-K(G3), HIP(G4), batched GEMM, 열 우선 입력, 전치 플래그(A^T·B 등). 전치는 LLM 연산에서 필요해지는 G2에서 다룬다.

## 10. 블로그

단계마다 개발 일지(Tistory 비공개 먼저): k01→k02(접근 패턴), k03(shared tiling), k04(register tiling과 occupancy), k05(cp.async), 튜닝과 cuBLAS와의 차이. 각 글에 "코드 diff → Nsight 지표 → GFLOPS" 표를 넣는다.

## 환경 기록

2026-11-10, 데스크탑 WSL2(Ubuntu 22.04)에서 직접 실행해 확인했다.

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti, 8192 MiB, Bus 00000000:01:00.0, 디스플레이 연결됨(Disp.A On, Xwayland 프로세스) |
| 드라이버 | Windows 581.80 / WSL `nvidia-smi` 580.105.07, 드라이버가 지원하는 CUDA 13.0 |
| CUDA 툴킷 | nvcc 12.2 (V12.2.140), `/usr/local/cuda` → 12.2 |
| 컴파일러·빌드 | g++ 11.4.0, cmake 3.22.1, ninja `/usr/bin/ninja` |
| Python / PyTorch | Python 3.10.12(`/usr/bin/python3`), torch 2.9.1+cu128, `torch.version.cuda` 12.8, `cuda.is_available()` True |
| Nsight Compute | `/usr/local/cuda/bin/ncu`, 2026.3.1.0 (build 38829034) |
| WSL2에서 ncu 카운터 | **읽힌다.** 작은 테스트 커널에서 `--section SpeedOfLight`가 7 pass로 수집됨(권한 오류 없음). ncu 기본값(`--clock-control base`)에서 SM 1.66 GHz, DRAM 6.78 GHz로 보고됨 |
| 클럭 고정 `nvidia-smi -lgc 1500,1500` | WSL 안(일반 사용자)에서는 실패: `The current user does not have permission to change clocks for GPU 00000000:01:00.0.` / `Terminating early due to previous errors.`(`-rgc`도 같음). **Windows 관리자 cmd에서는 성공:** `GPU clocks set to "(gpuClkMin 1500, gpuClkMax 1500)" for GPU 00000000:01:00.0`. WSL에서 4096² matmul을 돌리며 1초 간격으로 확인하니 부하 100%, 36→66 °C 동안 SM 1500 MHz, 메모리 6800 MHz로 유지됐다(clocks.max.sm 2130 MHz). 해제는 Windows 관리자 권한으로 `nvidia-smi -rgc`. 측정 중 클럭 로그(5절)는 고정해도 계속 남긴다 |
| ncu와 클럭 | ncu 기본값은 `--clock-control base`(ncu가 클럭을 직접 설정)이고, 위 테스트에서는 SM 1.66 GHz로 보고됐다(이때는 `-lgc` 전이었음, 1.66 GHz가 나온 원인 미확인). `-lgc` 1500과 함께 쓸 때 ncu가 보고하는 클럭은 Task 9에서 확인하고 `--clock-control` 값을 정한다 |
| 메모리 | WSL2에 보이는 RAM 7 GiB(`free -g`), 호스트 16 GB |

device query(`build/devinfo`, Task 1, 2026-11-10):

```
name=NVIDIA GeForce RTX 3060 Ti cc=8.6 sms=38 clock_khz=1680000 l2_bytes=3145728 smem_per_block=49152 smem_per_block_optin=101376 smem_per_sm=102400 regs_per_sm=65536
fp32_peak_gflops_at_reported_clock=16343.0 (SMs x 128 FP32 lanes x 2 x clock)
```

- L2는 3 MiB(3,145,728 B)로 보고된다(인계 메모에 적었던 "L2 4MB"와 다름). L2 비우기 버퍼는 이 값 × 2 = 6 MiB.
- 이론 최고는 SM 클럭을 1500 MHz로 고정하면 38 × 128 × 2 × 1.5 GHz = 14,592 GFLOPS다(5절 공식, 측정 중 클럭 로그로 확인).
- shared memory: 블록당 기본 48 KiB, opt-in 최대 99 KiB(101,376 B), SM당 100 KiB.
