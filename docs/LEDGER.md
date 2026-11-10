# hxgemm G1 실행 기록 (ledger)

계획(`docs/plans/2026-11-08-hxgemm-g1.md`)과 다르게 한 결정(Ruling)과 Task 진행을 적는다.

## Rulings

- **R1 (2026-11-10, Task 0~): 원격 단계를 로컬에서 실행.** 데스크탑 Claude가 WSL2 안에서 직접 작업하므로 "원격:" 단계는 `rsync`/`ssh desk`를 빼고 저장소 루트(`~/hxgemm`)에서 `<cmd>`만 실행한다. 결과 파일을 맥으로 옮기는 `rsync`는 `git push`로 대신한다. Task 0 Step 1~3(sshd, Tailscale, 맥 키)은 `docs/DESKTOP_HANDOFF.md` 0순위에서 사용자가 직접 끝냈다.

## Task 진행

- Task 0 Step 4: 환경 기록을 스펙 끝에 채움(클럭 고정은 일반 사용자 권한 없음, ncu 카운터는 읽힘).
