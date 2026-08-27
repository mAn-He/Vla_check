# VERIFY.md — GPU 환경 검증 체크리스트

사람이 GPU 환경에서 순서대로 실행하며 체크한다. (코드 작성 시점에는 GPU/LIBERO
환경이 없어 아래 중 "로컬 검증됨" 표시 항목만 실제 실행으로 확인되었다 —
DOC_TRACE.md의 ✅ 목록과 일치.)

## 0. GPU 없이 어디서든 (로컬 검증됨 항목 포함)

- [ ] `bash setup/00_common.sh` → `[OK] common ready`
- [ ] `bash setup/10_libero.sh` → `[OK] libero eval-client env ready`
- [ ] `python setup/check_env.py` → 전 항목 PASS (GPU 없는 머신에서는 gpu 행만 FAIL이 정상)
- [x] `python servers/serve_random.py --port 8000 &` 후 `curl localhost:8000/healthz` → `OK` *(로컬 검증됨)*
- [x] `python client/run_eval.py --model random --suite libero_spatial --limit-episodes 2 --dry-run` → 태스크/에피소드 계획 출력 *(로컬 검증됨, LIBERO 미설치 시 정적 suite 맵 사용 경고 포함)*
- [ ] 위를 `--dry-run` 없이 실행 → `results/random__libero_spatial.jsonl` 생성, 각 줄이 스키마와 일치
      (LIBERO 설치 환경 필요; 프로토콜 왕복 자체는 로컬 검증됨 — DOC_TRACE §0 참조)
- [x] `python viz/build_heatmap.py --demo` → `viz/figures/`에 PNG 2개 + SVG 2개 *(로컬 검증됨)*

## 1. LIBERO 스모크 (GPU 권장, CPU 가능)

- [ ] `.venvs/libero/bin/python -c "from libero.libero import benchmark; d=benchmark.get_benchmark_dict(); print(sorted(d))"`
      → `libero_10, libero_90, libero_goal, libero_object, libero_spatial` (+ libero_100)
- [ ] `python client/run_eval.py --model random --suite libero_spatial --limit-episodes 1`
      → MUJOCO_GL 백엔드 로그 출력, 에피소드 완주, JSONL 1줄 이상
- [ ] JSONL의 `success`가 random 정책에서 대부분 `false`인지 (전부 true면 성공 판정 버그)

## 2. π0.5 (GPU)

- [ ] `bash setup/20_openpi.sh` → `[OK] openpi server env ready`
- [ ] `cd third_party/openpi && uv run scripts/serve_policy.py --env LIBERO --port 8000`
      (첫 실행 시 gs://openpi-assets/checkpoints/pi05_libero 자동 다운로드)
- [ ] `curl -s localhost:8000/healthz` → `OK`
- [ ] 1개 태스크 스모크: `python client/run_eval.py --model pi05_libero --suite libero_spatial --limit-episodes 2`
      → 성공률이 0%가 아님 (공식 수치: spatial 98.8%)

## 3. GR00T (GPU)

- [ ] `bash setup/30_groot.sh` → `[OK] groot server env ready`
- [ ] HF에서 `nvidia/Cosmos-Reason2-2B` 접근 승인 + `huggingface-cli login`
- [ ] 공식 서버 기동: `uv run python gr00t/eval/run_gr00t_server.py --model-path nvidia/GR00T-N1.7-LIBERO --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper --port 5555`
      (repo id 로딩 실패 시 DOC_TRACE의 중첩 폴더 주의사항 참조)
- [ ] 어댑터 기동: `uv run python <repo>/servers/serve_groot.py --port 8600`
- [ ] Gr00tSimPolicyWrapper의 obs horizon assert가 나면 메시지의 기대값으로 `--obs-horizon` 조정
      → 실제 값을 DOC_TRACE.md의 TODO(verify)에 기록
- [ ] 1개 태스크 스모크 → 성공률이 0%가 아님 (공식: spatial 97.65%)

## 4. OpenVLA-OFT (GPU)

- [ ] `bash setup/40_openvla.sh` → `[OK] openvla server env ready`
- [ ] `OFT_DIR=third_party/openvla-oft python servers/serve_openvla.py --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --task-suite libero_spatial --port 8700`
- [ ] `curl -s localhost:8700/healthz` → `OK`
- [ ] 1개 태스크 스모크 → 성공률이 0%가 아님 (공식: 97%+)

## 5. 전체 배치

- [ ] suite 4개 × 모델 3개, `--num-episodes 50` (공식 설정) — 세션이 끊겨도
      재실행하면 이어서 진행되는지 (resume 확인)
- [ ] `python viz/build_heatmap.py` → 실제 결과로 heatmap 2종 생성
- [ ] 요약 heatmap의 모델별 Average가 공개 수치(π0.5 96.85 / GR00T ≈97 / OFT 97.1)와
      크게 어긋나지 않는지 — 어긋나면 결과 사용 전에 원인 규명
