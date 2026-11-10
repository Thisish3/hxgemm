# hxgemm G1 실행 기록 (ledger)

계획(`docs/plans/2026-11-08-hxgemm-g1.md`)과 다르게 한 결정(Ruling)과 Task 진행을 적는다.

## Rulings

- **R1 (2026-11-10, Task 0~): 원격 단계를 로컬에서 실행.** 데스크탑 Claude가 WSL2 안에서 직접 작업하므로 "원격:" 단계는 `rsync`/`ssh desk`를 빼고 저장소 루트(`~/hxgemm`)에서 `<cmd>`만 실행한다. 결과 파일을 맥으로 옮기는 `rsync`는 `git push`로 대신한다. Task 0 Step 1~3(sshd, Tailscale, 맥 키)은 `docs/DESKTOP_HANDOFF.md` 0순위에서 사용자가 직접 끝냈다.

- **R2 (2026-11-10, Task 1): `cmake_minimum_required`를 3.24 → 3.22로 낮춤.** 데스크탑 cmake가 3.22.1(Ubuntu 22.04 apt)이다. 계획의 CMakeLists가 쓰는 기능(`CMAKE_CUDA_ARCHITECTURES`, `FindCUDAToolkit`/`CUDA::cublas`, `file(GLOB CONFIGURE_DEPENDS)`)은 3.22에 모두 있어, 새 cmake를 설치하지 않고 최소 버전만 낮췄다.
- **R3 (2026-11-10): 클럭은 Windows 관리자 권한으로 1500 MHz 고정.** WSL 안에서는 권한이 없어 `-lgc`가 실패하고, Windows 관리자 cmd의 `nvidia-smi -lgc 1500,1500`은 성공한다(스펙 환경 기록). 측정 전에 사용자가 고정하고, 클럭 로그는 그대로 같이 남긴다.

## Task 진행

- Task 0 Step 4: 환경 기록을 스펙 끝에 채움(ncu 카운터 읽힘, 클럭 고정은 Windows 관리자 권한에서만 됨).
- Task 1: `check_args: ok`, devinfo 출력을 환경 기록에 추가(L2 3 MiB).
