# vla-libero-bench

세 개의 오픈 VLA 모델(π0.5, GR00T N1.7, OpenVLA-OFT)을 동일한 LIBERO 벤치마크로
평가하고 `(모델 × 태스크)` 성공률 heatmap을 만드는 하네스.

> **상태**: 코드는 공식 문서·소스 대조로 작성되었고 GPU 실행 검증은 아직이다.
> 무엇이 문서로 확인됐고 무엇이 추측인지는 `DOC_TRACE.md`, 사람이 GPU에서 밟을
> 절차는 `VERIFY.md` 참조.

## 설계

세 모델은 의존성이 충돌한다(Python 3.8/3.10/3.12, torch 1.11/2.2/2.9,
flash-attn 2.5.5/2.8.3). 하나의 venv에 설치하지 않고, venv를 분리한 뒤
**openpi의 policy-server 프로토콜**(WebSocket + msgpack-numpy)로 통신한다.

```
┌────────────────────────┐   WebSocket(msgpack-numpy)   ┌─────────────────────────────┐
│ client/run_eval.py     │ ── observation ───────────▶  │ policy server (모델별 venv) │
│ venv: .venvs/libero    │ ◀── {"actions": (T,7)} ───   │                             │
└────────────────────────┘                              └─────────────────────────────┘
```

- **π0.5**: openpi 공식 서버를 그대로 사용 (`servers/serve_openpi.py`는 실행 래퍼).
- **GR00T**: NVIDIA 공식 ZMQ 서버(`run_gr00t_server.py --use-sim-policy-wrapper`)를
  그대로 띄우고, `servers/serve_groot.py`가 WebSocket↔ZMQ 어댑터를 맡는다.
- **OpenVLA-OFT**: `servers/serve_openvla.py`가 openvla-oft의 공식 로딩/추론
  헬퍼(`initialize_model`, `get_action`, `process_action`)를 그대로 import해서 서빙.

관측 스키마(와이어 공통, openpi LIBERO 클라이언트와 동일 규약):

```
observation/image        uint8 (H,W,3)  agentview, 180도 회전
observation/wrist_image  uint8 (H,W,3)  eye-in-hand, 180도 회전
observation/state        float (8,)     [eef_pos(3), eef axis-angle(3), gripper_qpos(2)]
prompt                   str            LIBERO task.language 원문
```

응답 `{"actions": (T,7)}`는 **env-ready** action chunk다(그리퍼 부호/정규화 변환은
서버 쪽에서 끝낸다). 헬스체크는 같은 포트의 HTTP `GET /healthz`(openpi 서버 내장).

## 체크포인트

| 모델 | 체크포인트 | 출처 |
|---|---|---|
| π0.5 | `gs://openpi-assets/checkpoints/pi05_libero` | openpi `pi05_libero` config로 학습된 공식 체크포인트 (공식 성공률 평균 96.85%) |
| GR00T N1.7 | `nvidia/GR00T-N1.7-LIBERO` (+ `nvidia/Cosmos-Reason2-2B` 백본) | Isaac-GR00T examples/LIBERO (공식 평균 ≈97%) |
| OpenVLA-OFT | `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}` | openvla-oft LIBERO.md (suite별 평균 97.1%) |

원본 `openvla/openvla-7b`는 LIBERO 파인튜닝이 없어 그대로 쓰면 불공정하므로
OFT-LIBERO 체크포인트를 쓴다. 같은 이유로 GR00T도 base 3B가 아닌 LIBERO
파인튜닝 체크포인트를 쓴다(세 모델 모두 LIBERO 파인튜닝판으로 통일).

## 실행

### 1) 환경 구성

```bash
bash setup/00_common.sh
bash setup/10_libero.sh          # eval client (py3.8: LIBERO + openpi-client)
bash setup/20_openpi.sh          # π0.5 서버 (uv 관리, GPU)
bash setup/30_groot.sh           # GR00T 서버 (py3.12, GPU, HF gated 모델 접근 필요)
bash setup/40_openvla.sh         # OpenVLA-OFT 서버 (py3.10, GPU)
python setup/check_env.py        # 사전 점검 (전 항목 PASS 확인)
```

### 2) 파이프라인 검증 (GPU 불필요)

```bash
source .venvs/libero/bin/activate
python servers/serve_random.py --port 8000 &
curl -s localhost:8000/healthz                       # -> OK
python client/run_eval.py --model random --suite libero_spatial --limit-episodes 2 --dry-run
python client/run_eval.py --model random --suite libero_spatial --limit-episodes 2
python viz/build_heatmap.py --demo                   # viz/figures/ 에 PNG+SVG
```

### 3) 실제 평가 (GPU)

터미널 A(서버) / 터미널 B(클라이언트)로 나눠 실행. 예: π0.5

```bash
# A
cd third_party/openpi && uv run scripts/serve_policy.py --env LIBERO --port 8000
# B
source .venvs/libero/bin/activate
python client/run_eval.py --model pi05_libero --suite libero_spatial
```

GR00T (서버 2개: 공식 ZMQ 서버 + 어댑터):

```bash
# A
cd third_party/Isaac-GR00T && uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.7-LIBERO --embodiment-tag LIBERO_PANDA \
  --use-sim-policy-wrapper --port 5555
# A'
cd third_party/Isaac-GR00T && uv run python ../../servers/serve_groot.py --port 8600
# B
python client/run_eval.py --model groot_n17_libero --suite libero_spatial
```

OpenVLA-OFT (suite마다 체크포인트가 다르므로 suite 단위로 서버 재시작):

```bash
# A
source .venvs/openvla/bin/activate
OFT_DIR=third_party/openvla-oft python servers/serve_openvla.py \
  --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --task-suite libero_spatial --port 8700
# B
python client/run_eval.py --model openvla_oft_libero --suite libero_spatial
```

### Colab 셀 예시

```python
!bash setup/00_common.sh && bash setup/10_libero.sh
# 백그라운드 서버 + 로그 tail 패턴
!nohup .venvs/libero/bin/python servers/serve_random.py --port 8000 > server.log 2>&1 &
!sleep 3 && tail -5 server.log && curl -s localhost:8000/healthz
!.venvs/libero/bin/python client/run_eval.py --model random --suite libero_spatial --limit-episodes 2
```

Drive에 캐시를 두려면 셀 최상단에서
`%env VENV_ROOT=/content/drive/MyDrive/vla/.venvs` 등으로 지정한다
(`CKPT_CACHE_DIR`, `HF_HOME` 동일).

### RunPod / EC2 (SSH)

```bash
export CKPT_CACHE_DIR=/workspace/checkpoints HF_HOME=/workspace/hf   # volume/EBS
bash setup/00_common.sh && bash setup/20_openpi.sh && bash setup/10_libero.sh
tmux new -s server 'cd third_party/openpi && uv run scripts/serve_policy.py --env LIBERO'
tmux new -s eval   'source .venvs/libero/bin/activate && python client/run_eval.py --model pi05_libero'
```

## 평가 규약

- 에피소드당 JSONL 1줄, 즉시 flush. 실패/예외 에피소드도 기록 (`success:false`, `error`).
- **resume이 기본**: 기존 `results/*.jsonl`의 (model, suite, task_id, episode_idx)는
  스킵. 처음부터 다시는 `--fresh`.
- 첫 3 에피소드 후 ETA 출력, 첫 20 에피소드 성공률 0%면 경고 후 종료
  (체크포인트-레포 불일치 사고 방어).
- max_steps는 openpi/OFT 공식 하네스의 suite별 값(220/280/300/520/400) 고정 —
  LIBERO 자체에는 suite별 값이 없다(자체 eval은 전역 600). 변경 금지.
- 성공 판정: step의 `done`이 스텝 예산 내 True → 성공 (openpi·OFT·LIBERO
  metric.py 공통 규약).
- 렌더링: `MUJOCO_GL`을 egl → osmesa → glx 순으로 프로브해서 첫 성공 백엔드
  사용, 로그에 남김.

## 비교군 선정 근거

포함: 오픈 가중치 + LIBERO 파인튜닝 체크포인트 + 액션 출력이 있는 3종.

제외:
- **Genie 3** (DeepMind) — 클로즈드 모델로 공개 API가 없고, 월드 모델이라 출력이
  액션이 아니라 영상/환경 시뮬레이션이어서 LIBERO 성공률로 비교 불가.
- **Gemini Robotics VLA** — trusted tester 전용으로 일반 접근 불가.
- **Gemini Robotics ER** — 구현/공간 추론 계층으로 액션을 직접 출력하지 않아
  같은 벤치마크에 넣을 수 없음.

## 디렉토리

```
setup/          환경 구성 스크립트 + check_env.py
client/         평가 루프(run_eval.py), LIBERO 래퍼, 프로토콜 클라이언트
servers/        base_server + random/openpi/groot/openvla 서버
common/         msgpack-numpy 와이어 코덱 (openpi에서 채택)
viz/            build_heatmap.py (--demo 로 GPU 없이 검증 가능)
configs/        eval.yaml (모델별 host/port/전처리)
results/        에피소드 JSONL (git 미추적)
DOC_TRACE.md    모든 외부 API 호출의 근거 대조표
VERIFY.md       GPU 환경 검증 체크리스트
```
