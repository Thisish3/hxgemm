# hxgemm G1 실행 기록 (ledger)

계획(`docs/plans/2026-11-08-hxgemm-g1.md`)과 다르게 한 결정(Ruling)과 Task 진행을 적는다.

## Rulings

- **R1 (2026-11-10, Task 0~): 원격 단계를 로컬에서 실행.** 데스크탑 Claude가 WSL2 안에서 직접 작업하므로 "원격:" 단계는 `rsync`/`ssh desk`를 빼고 저장소 루트(`~/hxgemm`)에서 `<cmd>`만 실행한다. 결과 파일을 맥으로 옮기는 `rsync`는 `git push`로 대신한다. Task 0 Step 1~3(sshd, Tailscale, 맥 키)은 `docs/DESKTOP_HANDOFF.md` 0순위에서 사용자가 직접 끝냈다.

- **R2 (2026-11-10, Task 1): `cmake_minimum_required`를 3.24 → 3.22로 낮춤.** 데스크탑 cmake가 3.22.1(Ubuntu 22.04 apt)이다. 계획의 CMakeLists가 쓰는 기능(`CMAKE_CUDA_ARCHITECTURES`, `FindCUDAToolkit`/`CUDA::cublas`, `file(GLOB CONFIGURE_DEPENDS)`)은 3.22에 모두 있어, 새 cmake를 설치하지 않고 최소 버전만 낮췄다.
- **R3 (2026-11-10): 클럭은 Windows 관리자 권한으로 1500 MHz 고정.** WSL 안에서는 권한이 없어 `-lgc`가 실패하고, Windows 관리자 cmd의 `nvidia-smi -lgc 1500,1500`은 성공한다(스펙 환경 기록). 측정 전에 사용자가 고정하고, 클럭 로그는 그대로 같이 남긴다.

- **R4 (2026-11-10, Task 2): CUDA 컴파일러를 `/usr/local/cuda/bin/nvcc`(12.2)로 고정.** 처음 configure에서 CMake가 apt `nvidia-cuda-toolkit`의 `/usr/bin/nvcc`(11.5)와 `/usr/lib/x86_64-linux-gnu/libcublas.so.11`을 골랐다. 이렇게 빌드한 `hxgemm_bench`는 첫 CUDA 호출에서 `no CUDA-capable device is detected`로 끝났고, `LD_DEBUG=libs`로 보니 WSL용 `/usr/lib/wsl/lib/libcuda.so.1`이 아니라 apt `libnvidia-compute-580`의 `/lib/x86_64-linux-gnu/libcuda.so`(580.178.04)가 로드됐다. `CMakeLists.txt`에서 `CMAKE_CUDA_COMPILER`/`CUDACXX`가 지정되지 않았고 `/usr/local/cuda/bin/nvcc`가 있으면 그것을 쓰도록 바꾸고 `build/`를 새로 만들었다. 이후 cuBLAS는 `/usr/local/cuda/lib64/libcublas.so.12`로 링크되고 정상 동작한다. Task 1의 `check_args`/devinfo도 12.2로 다시 빌드해 같은 결과를 확인했다. 시스템 패키지(nvcc 11.5, libnvidia-compute-*)는 건드리지 않았다 → KNOWN_ISSUES 후보.

## Task 진행

- Task 0 Step 4: 환경 기록을 스펙 끝에 채움(ncu 카운터 읽힘, 클럭 고정은 Windows 관리자 권한에서만 됨).
- Task 1: `check_args: ok`, devinfo 출력을 환경 기록에 추가(L2 3 MiB).
- Task 2: cuBLAS가 CPU FP64 기준을 통과(오차 기준 변경 없음). `--shape 33,47,129`: max_abs 1.632e-06, max_rel 2.531e-04 / `512³`: max_abs 1.100e-05, max_rel 4.013e-02(max_rel은 |ref|가 0에 가까운 원소에서 커지며 판정은 atol+rtol·|ref|로 함). α=0.5, β=2와 `--nan-c`도 pass=1. 벤치에 `--alpha`/`--beta`를 추가했다(계획 인자 목록에 없음, 검증용).
- Task 3: k01·k02가 {1,1,1}, {7,33,5}, {33,47,129}, {128³}, {1000³}, α=0.5·β=2, `--nan-c`에서 모두 pass=1. 2048³(SM 1500 MHz 고정, 클럭 로그 13개 모두 1500 MHz): k01 108.6 GFLOPS(cuBLAS 대비 1.2%), k02 804.8 GFLOPS(9.1%), cuBLAS 8881.5 GFLOPS. 원본: `docs/results/2026-11-10_k01_k02.{csv,txt,clocks.csv}`. 2048³에서 k01·k02의 cuBLAS 대비 max_abs_err가 정확히 0이었다(1000³에서는 8.0e-05). 누적 순서가 같아서인지는 원인 미확인.
- Task 4: k03가 {1,1,1}, {7,33,5}, {33,47,129}, {64³ NaN-C}, {1000³}(NaN-C 포함), {2047³ α=0.5·β=2}에서 모두 pass=1. 2048³(클럭 로그 4개 모두 1500 MHz): k02 804.4 GFLOPS(9.1%), k03 1063.1 GFLOPS(12.0%), cuBLAS 8820.8 GFLOPS. k02 → k03 1.32배. 차이가 작은 이유는 Nsight(Task 9) 전이라 원인 미확인(KNOWN_ISSUES #3). 원본: `docs/results/2026-11-10_k03.{csv,txt,clocks.csv}`.
- **R5 (2026-11-10, Task 4 후): Task 9 전에 ncu로 k02/k03을 먼저 쟀다**(사용자 요청). `ncu --clock-control none -k regex:'k0[23]'`, 섹션 SpeedOfLight/MemoryWorkloadAnalysis/Occupancy/LaunchStats/WarpStateStats + DRAM·L2·L1 바이트, stall 지표. `--clock-control none`에서 ncu가 보고한 SM 클럭은 1.498 GHz로 `-lgc 1500`과 맞았다. 결과: k03에서 DRAM 바이트는 줄지 않았고(1.126 → 1.144 GB), 병목이 L1/TEX(lg_throttle)에서 shared(mio_throttle)·barrier로 옮겨 갔다(KNOWN_ISSUES #3). 리포트는 `.ncu-repz`(ncu 2026.3 기본 압축 형식)로 저장되며 `.gitignore`에 추가해 커밋하지 않는다. 텍스트 요약(`.txt`)만 커밋.
- **R6 (2026-11-10): 블로그용 Nsight GUI 캡처를 `docs/blog/2026-11-10_nsight_k02_k03/`에 저장**(사용자 요청, Tistory 글에 맥 Claude와 같이 씀). 메모리 차트 등이 나오도록 `--set full`로 다시 수집했다(`docs/results/2026-11-10_ncu_full_k02_k03.{txt,log}`). GUI는 WSLg의 `ncu-ui`(2026.3.1)이고, python-xlib(XTEST)로 조작해 창 단위로 캡처했다. 그림 목록과 각 그림의 숫자는 그 폴더의 `README.md`에 있다.
- Task 5: 벤치에 `--offset N`(A/B/C를 N float 밀고 ld += N)을 추가. k04는 `HX_TILES` 48개 조합이 모두 `hxgemm_list_tiles`에 나온다(스레드 64~1024, shared ≤ 32 KiB < 99 KiB). 48 타일 × {1,1,1}, {7,33,5}, {33,47,129}, {1000³} = 192건이 기본/`--offset 1`/`--nan-c`/α=0.5·β=2/offset+α·β에서 모두 pass=1. 기본 타일로 2047³(offset 포함), 4096³도 pass=1. 4096³(클럭 로그 14개 모두 1500 MHz): k03 1047.2 GFLOPS(11.3 %), **k04 128×128×8×8×8 6603.6 GFLOPS(71.0 %)**, cuBLAS 9297.6 GFLOPS. 원본 `docs/results/2026-11-10_k04.*`. 벤치의 기본 타일 표기를 `0x0x...`에서 실제 기본값으로 바꿈.
- Task 5 후 Nsight(R6과 같은 방식): k03 → k04 4096³ `--set full` 수집과 GUI 캡처 11장을 `docs/blog/2026-11-10_nsight_k03_k04/`에 저장. GUI 조작 스크립트를 `tools/nsight_gui.py`로 저장소에 넣음(WSLg X11 창 캡처·클릭·스크롤·드래그, python-xlib·Pillow 필요).
- Task 6: k05 타일 96개(48 × stages {2,3}, 최대 shared 96 KiB ≤ opt-in 99 KiB) 전부 `hxgemm_list_tiles`에 나옴. 96 × {1,1,1}, {7,33,5}, {33,47,129}, {1000³}, {64,64,3}(K < BK), {128,128,8}(타일 수 < S) = 576건이 기본/`--offset 1`/`--nan-c`/α=0.5·β=2/offset+α·β에서 모두 pass=1. 2047³(3단 offset 포함), 4096³ pass=1. 빌드 시간 82 s(k04+k05 템플릿 288개). 4096³(클럭 로그 전부 1500 MHz): k04 6589.6 GFLOPS(70.9 %), **k05 2단 7458.5 GFLOPS(80.2 %)**, 3단 7478.1 GFLOPS(80.4 %, 별도 실행이며 pct는 그 실행의 cuBLAS 기준), cuBLAS 9298.1 GFLOPS. 성공 기준 80 %에 0.2~0.4 %p 차이라 반복 측정 전에는 통과로 판정하지 않는다(Task 9에서 반복). 원본 `docs/results/2026-11-10_k05*.{csv,txt,clocks.csv}`.
