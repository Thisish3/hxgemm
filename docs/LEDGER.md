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
