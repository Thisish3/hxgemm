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
   - 스펙 가설은 "DRAM 바이트가 크게 줄고 shared 접근·동기화가 병목". Nsight로 재지 않았으므로 원인 미확인. Task 9 `ncu_collect.sh`로 k02/k03의 DRAM 바이트, shared 처리량, warp stall을 비교한다.
4. **2048³·2047³에서 k01/k02/k03 결과가 cuBLAS와 비트 단위로 같다** (max_abs_err 0, 1000³에서는 8.0e-05)
   - 큰 크기의 정확성 기준이 cuBLAS라서 검증이 자명해 보일 수 있다. 작은 크기(≤ 512)는 CPU FP64와 비교해 통과했다. 같은 이유는 원인 미확인(누적 순서가 같다는 추측은 미확인).
