# DOC_TRACE.md — 공식 문서 대조 검증

- 최종 갱신: 2026-08-27 (코드 골격 작성 + GPU 없이 가능한 실행 검증까지 반영)
- 표기 규칙:
  - ✅ = **실제로 실행해서** 확인 (이 저장소의 코드가 이 환경에서 돌았음)
  - 📄 = 공식 저장소의 **문서/소스 코드에서** 확인 (실행은 안 함)
  - ⚠️ = 문서에서도 확인 못함 (코드에 `# TODO(verify)` 주석 존재)
- GPU/LIBERO 실행 환경이 없으므로 모델·환경 관련 항목은 📄가 상한이다.
  ✅는 아래 §0의 로컬 실행 항목에만 붙어 있다.

## 접근 제한 (정직성 기록)

- huggingface.co, arxiv.org 직접 fetch가 이 작업 환경에서 차단됨. HF 체크포인트
  존재는 웹 검색 결과(실제 HF 페이지로 연결)로 교차 확인 → 해당 항목은
  "📄(검색 교차확인)"으로 구분.
- GitHub의 README·raw 소스는 전부 직접 읽음(§4 URL 체크리스트).
- **주의**: 이번 작업 지시가 전제한 기존 골격(`vla-libero-bench.zip`)은 이
  저장소에 존재하지 않았다. 골격은 지시서 사양대로 이 세션에서 새로 작성했고,
  지시서의 ⚠️ 우선순위 항목(1~7)은 모두 아래 표의 근거로 해소했다.

---

## 0. 실제로 실행해서 확인한 것 (✅)

이 환경(Python 3.11 venv: numpy/msgpack/websockets/pyyaml/pillow/matplotlib/seaborn/pandas)에서:

| 실행한 것 | 결과 |
|---|---|
| `python -m py_compile` — 저장소의 모든 `.py` | ✅ 전부 통과 |
| `bash -n setup/*.sh` (5개) | ✅ 전부 통과 |
| `python servers/serve_random.py --port 8123` + `curl localhost:8123/healthz` | ✅ HTTP 200 "OK" |
| `client/policy_client.py`(번들 폴백) ↔ serve_random 왕복: 핸드셰이크 metadata 수신, `infer()` → `{"actions": (10,7)}` + `server_timing` | ✅ |
| 서버 에러 경로: 필수 키 누락 obs → 서버가 traceback 문자열 프레임 전송 → 클라이언트 `RuntimeError` | ✅ |
| `run_eval.run_episode` — 성공 경로(대기 10스텝 후 done→success, num_steps 정확), 타임아웃 경로(max_steps 소진→failure), 7-dim 액션으로 env.step 호출 | ✅ (LIBERO step 계약의 mock env + 실서버) |
| `run_eval.load_completed` — resume 튜플 파싱, 깨진 줄 경고 후 무시 | ✅ |
| `python client/run_eval.py --model random --suite libero_spatial --limit-episodes 2 --dry-run` | ✅ (LIBERO 미설치 → 정적 suite 맵 경고 후 계획 출력) |
| `python viz/build_heatmap.py --demo` (global/suite 두 정렬 모드) | ✅ PNG+SVG 4파일 생성, 눈으로 확인 |

실행하지 **못한** 것(GPU/대용량 설치 필요, "테스트 완료" 아님): LIBERO 실제 설치
(`setup/10_libero.sh` — torch cu113 수 GB 다운로드), 세 모델 서버 구동, 실제
에피소드 평가. 이들은 VERIFY.md의 체크리스트로 넘긴다.

---

## 1. 지시서와 실제(공식 저장소)가 달랐던 점 — 저장소 우선 적용

### 1.1 [중대] 통신 프로토콜: HTTP+JSON+base64가 아니라 WebSocket+msgpack

- 지시서 §4는 `GET /health` / `POST /infer` + base64 PNG의 HTTP 계약을 제시하지만
  동시에 "openpi의 프로토콜을 채택"하라고 지시. openpi의 실제 프로토콜은(📄):
  WebSocket + **msgpack_numpy**(numpy 배열 그대로, base64 없음), 연결 직후 서버가
  metadata를 먼저 push, 에러는 str 프레임(traceback) 후 close. 같은 포트에 HTTP
  `GET /healthz`(200 "OK")가 내장되어 지시서의 /health 요구도 사실상 충족.
  - 근거: `packages/openpi-client/src/openpi_client/websocket_client_policy.py`,
    `src/openpi/serving/websocket_policy_server.py`, `packages/openpi-client/src/openpi_client/msgpack_numpy.py`
- 적용: `servers/base_server.py`가 이 프로토콜을 그대로 구현(✅ 왕복 검증됨),
  π0.5는 openpi 공식 서버를 무수정 사용.

### 1.2 [중대] observation 키: "LIBERO 키 그대로"가 아니라 openpi 규약 + 180도 회전

- 지시서 §4 "camera_key는 LIBERO가 주는 키를 그대로"와 달리, openpi 공식
  클라이언트는 `observation/image`·`observation/wrist_image`·`observation/state`(8-dim)·
  `prompt`로 리매핑하고 이미지를 **180도 회전**한다(📄, examples/libero/main.py:115-141).
- 이 회전은 세 하네스 공통 규약이다(📄): OFT `libero_utils.get_libero_image`
  ("rotate 180 degrees to match train preprocessing"), GR00T
  `gr00t/eval/sim/LIBERO/libero_env.py:148-149`(`[::-1, ::-1]`)도 동일.
  8-dim state 구성 `[eef_pos, quat→axis-angle, gripper_qpos]`도 셋 다 동일.
- 적용: 클라이언트가 회전+리매핑(`client/libero_env.py:119-131`), openpi에만
  224 resize_with_pad를 클라이언트에서 적용(공식 클라이언트 재현), GR00T/OFT는
  원본 256을 보내고 서버가 자기 방식대로 리사이즈(각 공식 하네스와 동일 분담).

### 1.3 [중대] `max_steps`: LIBERO에는 "태스크별 기본값"이 없다

- LIBERO 자체에는 suite별 max_steps가 없다(📄 — benchmark 모듈에 horizon 없음,
  env 기본은 robosuite `horizon=1000`, LIBERO 자체 eval은 전역
  `max_steps: 600`/`n_eval: 20`, `libero/configs/eval/default.yaml`).
- suite별 220/280/300/520/400은 **openpi와 openvla-oft 두 공식 하네스가 동일하게
  쓰는 값**이다(📄 — openpi examples/libero/main.py:60-71, openvla-oft
  run_libero_eval.py:63-69, 주석까지 동일). 적용: `client/libero_env.py:38-44`에
  고정, 출처 명기. 변경 금지.

### 1.4 성공 판정: `done`이 성공 신호 (지시서에 미정의였던 항목)

- 세 근거가 일치(📄): openpi main.py:153-157(`if done: successes += 1`),
  OFT run_libero_eval.py:350-353(동일), LIBERO 자체 metric.py:136-155
  (`dones[k] = dones[k] or done[k]` → `num_success += int(dones[k])`).
  `env.check_success()`(env_wrapper.py:103, `self.env._check_success()` 위임)도
  같은 신호의 직접 조회 경로다. GR00T의 LIBERO env는 `ignore_done=True`로 두고
  `info["success"] = env.check_success()`를 쓴다(📄) — 신호 자체는 동일.
- 적용: `client/run_eval.py`의 `run_episode`는 스텝 예산 내 `done` → 성공
  (✅ mock env로 성공/타임아웃 경로 실행 확인).

### 1.5 [중대] GR00T N1.7: 모델·체크포인트 실존, 단 API가 구버전과 전혀 다름

- `nvidia/GR00T-N1.7-3B` 실존(📄(검색 교차확인); Isaac-GR00T main 브랜치 = N1.7 GA,
  N1.5/N1.6은 `n1d5`/`n1d6` 브랜치). 지시서의 모델 표는 옳다.
- 구버전(N1/N1.5) API는 main에서 소멸(📄): `gr00t/model/policy.py` → 404, 현재는
  `gr00t/policy/gr00t_policy.py`의
  `Gr00tPolicy(embodiment_tag, model_path, *, device, strict=True)`.
  `modality_config`/`modality_transform` 인자와 `DATA_CONFIG_MAP`은 없음 —
  modality config는 체크포인트 내장 AutoProcessor에서. obs는 중첩 dict
  (`{"video": {...}, "state": {...}, "language": {"task": [[str]]}}`), 출력은
  비정규화 `(action_dict, info)`, base `action_horizon=40`.
- **LIBERO는 first-class**(📄): `nvidia/GR00T-N1.7-LIBERO` 체크포인트 +
  `EmbodimentTag.LIBERO_PANDA = "libero_sim"`(embodiment_tags.py) +
  `examples/LIBERO/`(공식 성공률 97.65/97.5/98.45/94.35) + 전용 sim env
  (`gr00t/eval/sim/LIBERO/libero_env.py`).
- 적용: 공정 비교(3모델 전부 LIBERO 파인튜닝판)를 위해 base 3B 대신
  **`nvidia/GR00T-N1.7-LIBERO`** 사용 — 지시서 표의 체크포인트와 다름을 명시 보고.
  서빙은 NVIDIA 공식 서버(`gr00t/eval/run_gr00t_server.py`, ZMQ :5555,
  `--use-sim-policy-wrapper`)를 그대로 쓰고 `servers/serve_groot.py`는
  WebSocket↔ZMQ 어댑터만 담당.
- 주의(📄): VLM 백본 `nvidia/Cosmos-Reason2-2B`는 HF **gated**(모든 GR00T
  체크포인트가 첫 로드 시 당김); N1.7-LIBERO는 HF에서 중첩 폴더 구조
  (`scripts/deployment/README.md` 참조).

### 1.6 OpenVLA: OFT-LIBERO 체크포인트 실존, 추론 인터페이스는 vanilla와 다름

- OFT 레시피 `moojink/openvla-oft`(arXiv:2502.19645) + LIBERO 체크포인트 4종
  `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}` + 통합
  `…-spatial-object-goal-10` 전부 실존(📄 LIBERO.md 명기 + 📄(검색 교차확인);
  suite별 평균 97.1% vs 통합 96.8%).
- 지시서 §2의 `vla.predict_action(**inputs, unnorm_key=…)`는 vanilla openvla
  경로다. OFT는(📄, run_libero_eval.py + robot_utils.py):
  `GenerateConfig` + `initialize_model()`(L1-regression action head,
  proprio projector `proprio_dim=8`, processor, unnorm_key `_no_noops` 폴백) →
  `get_action(...)` → **action chunk** → 액션마다
  `normalize_gripper_action(binarize=True)` + `invert_gripper_action`.
- 의존성(📄): Python 3.10, PyTorch 2.2.0, **커스텀 transformers 4.40.1 fork**
  (`github.com/moojink/transformers-openvla-oft.git`), flash-attn 2.5.5
  (`--no-build-isolation`); LIBERO를 같은 env에 설치(LIBERO.md). →
  `setup/40_openvla.sh`에 그대로 반영.
- vanilla 참고(📄): `openvla/openvla-7b`는 OXE 970K 학습, LIBERO 파인튜닝 아님.
  openvla org의 LoRA 파인튜닝판 `openvla/openvla-7b-finetuned-libero-*`도
  존재하나 OFT 쪽 수치가 높아 OFT를 1순위로 채택.

### 1.7 heatmap §8(b): "전역 오름차순 정렬"과 "suite 경계 구분선"은 양립 불가

- 전 태스크를 전역 평균 오름차순으로 정렬하면 suite가 섞여 경계선이 무의미해진다.
- 적용: 기본(`--sort global`)은 전역 오름차순 + suite를 색 밴드로 표시,
  `--sort suite`는 suite 블록 유지 + 경계 구분선 + 블록 내 오름차순.
  둘 다 구현·실행 확인(✅).

### 1.8 경미한 차이·확인 사항

- LIBERO suite는 5개(+`libero_100` 등록): spatial/object/goal/10 각 10태스크,
  90은 90태스크(📄 libero_suite_task_map.py). "libero_long"이라는 이름은 없음
  (장기 스위트는 `libero_10`). 평가 대상 4개는 관례(두 공식 하네스 동일)와 일치.
- env seed 관행이 하네스마다 다름(📄): openpi는 `env.seed(args.seed=7)`,
  OFT는 `env.seed(0)` 고정(둘 다 "seed가 고정 init state에서도 물체 위치에 영향"
  주석). 우리는 `--seed`(기본 7)를 쓰고 JSONL에 기록.
- `pi05_libero` config 실존(📄, config.py ~743행):
  `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)`,
  action_dim은 기본 32; 체크포인트 `gs://openpi-assets/checkpoints/pi05_libero`
  (serve_policy.py의 LIBERO 기본값이기도 함); 공식 성공률 98.8/98.2/98.0/92.4.
  클라이언트 replan_steps=5, chunk 길이 assert(📄 main.py:144-148) → 동일 구현.
- π0.5 PyTorch 포팅 실존(📄, README "PyTorch Support", `src/openpi/models_pytorch/`).
- openpi 이슈 #849(`pi0_fast_libero` 불일치) 원문은 확인 못함(⚠️) — README들에
  언급 없음. 20-에피소드 0% 가드레일은 이슈 진위와 무관하게 유지.
- Python/의존성 충돌 실측(📄): LIBERO 3.8 / openpi LIBERO client 3.8 /
  OFT 3.10 / GR00T 3.12 전용(torch 2.9.0, flash-attn 2.8.3 prebuilt wheel 고정).
  → venv 분리 지시가 타당함을 확인.
- MUJOCO_GL(📄): openpi compose.yml 기본 `egl`(+PYOPENGL_PLATFORM=egl), 문서화된
  폴백은 `glx`뿐. 지시서의 egl→osmesa→glx 3단 폴백은 초집합이라 채택하되
  osmesa 단계는 공식 문서에 없음(⚠️, `client/libero_env.py:pick_mujoco_gl`).

---

## 2. 코드의 모든 외부 API 호출 — 근거 대조표

### 2.1 LIBERO (`Lifelong-Robot-Learning/LIBERO`, 기본 브랜치 `master`)

| 파일:줄 | 호출 | 근거 URL | 문서상 시그니처 / 사실 | 확인 |
|---|---|---|---|---|
| `client/libero_env.py:142` | `benchmark.get_benchmark_dict()` | raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/benchmark/__init__.py | `def get_benchmark_dict(help=False)` → `BENCHMARK_MAPPING`(소문자 suite명→클래스) | 📄 |
| `client/libero_env.py:144-153` | `benchmark_dict[name]()`, `.n_tasks`, `.get_task(i)`, `.get_task_init_states(i)` | 동일 | `Task` NamedTuple(`name, language, problem, problem_folder, bddl_file, init_states_file`); `get_task_init_states`는 `torch.load(init_states 경로)` | 📄 |
| `client/libero_env.py:60-66` | suite별 태스크 수(정적 맵, dry-run 폴백용) | raw…/master/libero/libero/benchmark/libero_suite_task_map.py | spatial/object/goal/10 각 10, 90은 90 | 📄 |
| `client/libero_env.py:161-166` | `get_libero_path("bddl_files")`, `OffScreenRenderEnv(bddl_file_name=…, camera_heights=…, camera_widths=…)` | raw…/master/libero/libero/envs/env_wrapper.py + openpi examples/libero/main.py:189-196 + OFT libero_utils.py:18-25 | `OffScreenRenderEnv(ControlEnv)`; `ControlEnv.__init__(bddl_file_name, robots=["Panda"], controller="OSC_POSE", …, camera_names=["agentview","robot0_eye_in_hand"], camera_heights=128, camera_widths=128, horizon=1000, …)`; 두 공식 하네스 모두 resolution 256으로 생성 | 📄 |
| `client/libero_env.py:167` | `env.seed(seed)` | env_wrapper.py + 두 공식 하네스(생성 직후 호출) | `def seed(self, seed)` — "seed seems to affect object positions even when using fixed initial state" | 📄 |
| `client/run_eval.py:246-247` | `env.reset()`; `obs = env.set_init_state(init_states[episode_idx])` | env_wrapper.py(`set_init_state` → `regenerate_obs_from_state`) + openpi main.py:93-97 + OFT run_libero_eval.py:293-297 | 두 공식 하네스와 동일 순서(reset → set_init_state → dummy 안정화 스텝) | 📄 |
| `client/run_eval.py:110` (run_episode) | `env.step(a7)` → `(obs, reward, done, info)` | openpi main.py:109,153 + OFT:319,350 + LIBERO metric.py:136 | gym-classic 4-tuple; 7-dim 액션(OSC_POSE 6+gripper); dummy `[0]*6+[-1]` | 📄 (mock env로 루프 로직은 ✅) |
| `client/run_eval.py` (성공 판정) | 스텝 예산 내 `done` → success | §1.4의 세 근거 | `env.check_success()`(env_wrapper.py:103)도 동일 신호 | 📄 |
| `client/libero_env.py:126-130` | obs 키: `agentview_image`, `robot0_eye_in_hand_image`, `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos` | libero/configs/data/default.yaml(obs_key_mapping) + 세 공식 하네스의 실사용 | 세 하네스 전부 이 키를 직접 인덱싱 | 📄 |
| `client/libero_env.py:101-113` | `quat2axisangle` | openpi main.py:199-214(robosuite transform_utils에서 복사) + OFT libero_utils | 동일 구현 복사 | 📄 |
| `client/libero_env.py:38-44` | suite별 max_steps | §1.3 | openpi·OFT 공식값 220/280/300/520/400 | 📄 |
| `client/libero_env.py:73-97` | MUJOCO_GL egl→osmesa→glx 프로브 | openpi compose.yml(egl 기본, glx 폴백 문서화) | osmesa 중간 단계는 우리 추가분 | ⚠️ TODO(verify: osmesa 실동작) |
| `setup/10_libero.sh` | 소스 설치 + torch 1.11 cu113 | LIBERO README + setup.py(패키지명 `libero`, 권장 py3.8.13) | PyPI 배포 여부 미확인 → 소스 설치 채택 | 📄 |
| (액션 범위) | [-1,1] 가정 없음 — 서버 출력을 그대로 step | robosuite OSC 관례(LIBERO repo에 명시 없음) | 코드가 범위에 의존하지 않도록 작성 | ⚠️ (robosuite 소스 미확인) |

### 2.2 openpi (`Physical-Intelligence/openpi`, main)

| 파일:줄 | 호출 | 근거 URL | 문서상 시그니처 / 사실 | 확인 |
|---|---|---|---|---|
| `servers/base_server.py` 전체 | WebSocket 서버 프로토콜(metadata 선전송, msgpack, str-traceback 에러, `/healthz`) | raw…/openpi/main/src/openpi/serving/websocket_policy_server.py | `WebsocketPolicyServer(policy, host, port, metadata)`; `_health_check`: `GET /healthz` → 200 "OK" | 📄 (우리 구현 왕복은 ✅) |
| `common/msgpack_numpy.py` | 와이어 코덱 | raw…/packages/openpi-client/src/openpi_client/msgpack_numpy.py | `b"__ndarray__"`/`b"__npgeneric__"` 엔벨로프, dtype/shape 필드 — 동일 사본 | 📄 (왕복 ✅) |
| `client/policy_client.py:24` | `openpi_client.websocket_client_policy.WebsocketClientPolicy` (우선 사용) | raw…/packages/openpi-client/src/openpi_client/websocket_client_policy.py | `__init__(host="0.0.0.0", port=None, api_key=None)`; `infer(obs)->dict`; `compression=None, max_size=None`; 연결 직후 metadata unpack; str 응답=에러 | 📄 (번들 폴백 동작은 ✅) |
| `client/run_eval.py:117-131` (run_episode) | 요청 obs 키·응답 `"actions"` chunk·deque 소비 | raw…/openpi/main/examples/libero/main.py:127-158 | `observation/image`, `observation/wrist_image`, `observation/state`(8-dim), `prompt`; `client.infer(element)["actions"]`; `replan_steps=5`, chunk 길이 assert | 📄 (mock 왕복 ✅) |
| `client/image_utils.py` | `resize_with_pad`, `convert_to_uint8` | raw…/packages/openpi-client/src/openpi_client/image_tools.py | PIL bilinear + zero-pad, tf.image.resize_with_pad 재현 — 동일 사본(공식 패키지 있으면 그것 사용) | 📄 |
| `servers/serve_openpi.py:46-52` | `uv run scripts/serve_policy.py --env LIBERO` / `policy:checkpoint --policy.config=… --policy.dir=…` | raw…/openpi/main/scripts/serve_policy.py | `EnvMode.LIBERO`; LIBERO 기본 = `Checkpoint(config="pi05_libero", dir="gs://openpi-assets/checkpoints/pi05_libero")`; `port=8000` | 📄 |
| `configs/eval.yaml` (pi05_libero) | resize 224, replan 5, chunk 10 | config.py(`action_horizon=10`) + main.py(`resize_size=224`, `replan_steps=5`) | §1.8 참조 | 📄 |
| `setup/20_openpi.sh` | `GIT_LFS_SKIP_SMUDGE=1 uv sync` (+`uv pip install -e .`) | openpi README | 공식 설치 명령 그대로 | 📄 |

### 2.3 GR00T (`NVIDIA/Isaac-GR00T`, main = N1.7)

| 파일:줄 | 호출 | 근거 URL | 문서상 시그니처 / 사실 | 확인 |
|---|---|---|---|---|
| `setup/30_groot.sh` (서버 기동 명령) | `run_gr00t_server.py --model-path … --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper --port 5555` | raw…/Isaac-GR00T/main/gr00t/eval/run_gr00t_server.py | `ServerConfig(model_path, embodiment_tag="new_embodiment", device="cuda", host="0.0.0.0", port=5555, strict=True, use_sim_policy_wrapper=False)`; 내부에서 `Gr00tPolicy(embodiment_tag, model_path, device, strict)` 생성, `use_sim_policy_wrapper`면 `Gr00tSimPolicyWrapper(policy)` | 📄 |
| `servers/serve_groot.py:86` | `PolicyClient(host, port)` + `get_action(obs) -> (action_dict, info)` | raw…/main/gr00t/policy/server_client.py + run_gr00t_server.py | `PolicyClient(host="localhost", port=5555, timeout_ms=15000, api_token=None, strict=False)` — ZMQ tcp, msgpack_numpy; 로컬 policy와 동일 인터페이스 | 📄 |
| `servers/serve_groot.py:99-115` | flat obs 스키마: `video.image`/`video.wrist_image` uint8 (B,T,H,W,3), `state.{x,y,z,roll,pitch,yaw}` (B,T,1), `state.gripper` (B,T,2), `annotation.human.action.task_description` [str] | raw…/main/gr00t/eval/sim/LIBERO/libero_env.py:111-157 + gr00t_policy.py의 `Gr00tSimPolicyWrapper.check_observation`(uint8/float32·5차원/3차원·T=len(delta_indices) assert) | GR00T 자체 LIBERO env의 `_process_observation`과 동일 구성(180도 회전 포함) | 📄 |
| `servers/serve_groot.py:119-122` | 액션 flat 키 concat 순서 + 그리퍼 변환 | 동일 libero_env.py:171-186 | `action.x…action.gripper` concat → `normalize_gripper_action(binarize)` → `invert_gripper_action` → env.step | 📄 |
| `servers/serve_groot.py:70-77` | obs horizon T 기본 1 | (체크포인트 modality config 필요) | N1.7-LIBERO의 실제 delta_indices 길이 미확인 | ⚠️ TODO(verify: 체크포인트 processor에서 확인, assert 메시지가 기대값 알려줌) |
| `setup/30_groot.sh` | `uv sync`(flash-attn 2.8.3 prebuilt wheel, torch 2.9.0, py3.12 전용), gated `Cosmos-Reason2-2B` 로그인 | raw…/main/pyproject.toml + README(FAQ: uv의 URL-pinned wheel 재검증 메시지는 재빌드 아님) | §1.5·§1.8 참조 | 📄 |
| `configs/eval.yaml` (groot) | 체크포인트 `nvidia/GR00T-N1.7-LIBERO` | github…/Isaac-GR00T/blob/main/examples/LIBERO/README.md | 공식 성공률 97.65/97.5/98.45/94.35; HF 중첩 폴더 주의 | 📄 (HF 페이지는 검색 교차확인) |

### 2.4 OpenVLA-OFT (`moojink/openvla-oft`) / vanilla (`openvla/openvla`)

| 파일:줄 | 호출 | 근거 URL | 문서상 시그니처 / 사실 | 확인 |
|---|---|---|---|---|
| `servers/serve_openvla.py:60-74` | `GenerateConfig(pretrained_checkpoint=…, task_suite_name=…, center_crop=…)` + `initialize_model(cfg)` | raw…/openvla-oft/main/experiments/robot/libero/run_libero_eval.py:81-176 | 기본값 `use_l1_regression=True, num_images_in_input=2, use_proprio=True, center_crop=True, num_open_loop_steps=8`; `initialize_model` → (model, action_head, proprio_projector(proprio_dim=8), noisy_action_projector, processor) + `check_unnorm_key`(`_no_noops` 폴백) | 📄 |
| `servers/serve_openvla.py:81,93-94` | `get_image_resize_size(cfg)`, `resize_image_for_policy(img, size)` | raw…/main/experiments/robot/robot_utils.py:77 + openvla_utils | openvla family → 224 | 📄 |
| `servers/serve_openvla.py:98-108` | `get_action(cfg, model, observation, task_label, processor=…, action_head=…, proprio_projector=…, …)` → chunk | robot_utils.py:99-146 | obs = `{"full_image","wrist_image","state","task_description"}`(run_libero_eval.py:243-262와 동일 구성; 이미지는 이미 180도 회전본) | 📄 |
| `servers/serve_openvla.py:109` | `process_action(a, "openvla")` | run_libero_eval.py:265-275 + robot_utils.py:149-201 | `normalize_gripper_action(binarize=True)` → `invert_gripper_action` (env-ready) | 📄 |
| `configs/eval.yaml` (oft) | 체크포인트 `moojink/openvla-7b-oft-finetuned-libero-*` | github…/openvla-oft/blob/main/LIBERO.md | 4종 + 통합 1종; 97.1%/96.8% | 📄 (HF 페이지는 검색 교차확인) |
| `setup/40_openvla.sh` | py3.10 + torch 2.2.0 + transformers fork + flash-attn 2.5.5 + LIBERO 동일 env 설치 | SETUP.md + LIBERO.md("custom transformers v4.40.1 fork … stick to these versions") | §1.6 참조 | 📄 |
| (참고, 미사용) | vanilla `predict_action` | raw…/openvla/main/prismatic/extern/hf/modeling_prismatic.py:506 + README | `predict_action(self, input_ids=None, unnorm_key=None, **kwargs) -> np.ndarray`; prompt `"In: What action should the robot take to {<INSTRUCTION>}?\nOut:"`; zero-shot `unnorm_key="bridge_orig"` | 📄 |

---

## 3. URL fetch 체크리스트 (지시서 §2 전체 + 추가 소스)

| URL | 상태 |
|---|---|
| github.com/Lifelong-Robot-Learning/LIBERO (+benchmark/__init__.py, env_wrapper.py, libero_suite_task_map.py, metric.py, configs, setup.py, requirements.txt) | 📄 직접 읽음 |
| github.com/Physical-Intelligence/openpi (+examples/libero/{README,main.py,compose.yml}, training/config.py, models/pi0_config.py, scripts/serve_policy.py, serving/websocket_policy_server.py, openpi-client 소스 전부) | 📄 직접 읽음 |
| github.com/NVIDIA/Isaac-GR00T (+gr00t/policy/*, eval/run_gr00t_server.py, eval/sim/LIBERO/libero_env.py, data/embodiment_tags.py, pyproject.toml, getting_started/*, examples/LIBERO) | 📄 직접 읽음 |
| huggingface.co/nvidia/GR00T-N1.7-3B | ⚠️ 직접 fetch 차단 — 존재는 검색 결과의 실제 HF 페이지로 교차 확인 |
| github.com/openvla/openvla (+prismatic/extern/hf/modeling_prismatic.py) | 📄 직접 읽음 |
| huggingface.co/openvla/openvla-7b | ⚠️ 직접 fetch 차단 — README 동일 스니펫 + 검색 교차 확인 |
| github.com/moojink/openvla-oft (+SETUP.md, LIBERO.md, run_libero_eval.py, robot_utils.py, libero_utils.py) | 📄 직접 읽음 |

## 4. 남은 ⚠️ / TODO(verify) — GPU 환경에서 확인할 것

1. **GR00T obs horizon**(`serve_groot.py --obs-horizon`): N1.7-LIBERO 체크포인트의
   video/state delta_indices 길이. 서버의 assert 메시지가 기대값을 알려주므로
   스모크 1회로 확정 가능. (사람이 가장 먼저 확인할 것 #1)
2. **GR00T-N1.7-LIBERO의 HF 중첩 폴더 로딩**: repo id 직접 로딩 실패 시
   `scripts/deployment/README.md` 절차. (#2)
3. **OFT 서버의 `GenerateConfig`/`initialize_model` import가 서빙 프로세스에서
   그대로 동작하는지**(wandb/tensorflow 등 무거운 모듈 import 포함) — 코드는
   공식 eval 스크립트와 동일 경로지만 실행은 안 해봄. (#3)
4. robosuite 유래 사실 2건: `{camera}_image` 키 생성 라인, OSC_POSE 액션 범위.
5. `MUJOCO_GL=osmesa` 폴백 실동작(공식 문서에 없는 우리 추가 단계).
6. openpi 이슈 #849 원문(0% 가드레일의 일화적 근거).
7. HF 모델카드 원문 4건(네트워크 허용 환경에서).

---

## Phase 1 로컬 검증 결과 (WSL2 / GPU 없음)

실행 환경: WSL2 Ubuntu, Python 3.8.20, torch 1.11.0+cu113

| 항목 | 상태 | 확인 내용 |
|---|---|---|
| LIBERO benchmark API | ✅ | `libero_spatial` 10 tasks, instruction 원문 정상 조회 |
| observation dict 키 | ✅ | 에피소드 220스텝 완주, 키 오류 없음 |
| 성공 판정 | ✅ | random 정책에서 `success=false` 정상 기록 |
| max_steps (libero_spatial) | ✅ | 220 확인 |
| MUJOCO_GL | ✅ | WSL2에서 **egl** 로 동작 (osmesa 폴백 불필요) |
| WebSocket 프로토콜 | ✅ | `action_horizon: 10` 메타데이터 수신 정상 |
| JSONL 스키마 | ✅ | 실패 에피소드 포함 기록, 즉시 flush 확인 |
| resume | ✅ | 재실행 시 `10 planned, 10 already done` 스킵 |
| heatmap | ✅ | summary / per_task PNG+SVG 생성 |

### 발견된 문제와 우회 (10_libero.sh에 반영함)

1. **LIBERO editable 설치가 MAPPING을 비운 채 완료**
   `__editable___libero_0_1_0_finder.py`의 `MAPPING = {}` 로 `import libero` 실패.
   → site-packages에 `libero_path.pth` 수동 생성으로 우회.

2. **첫 import 시 대화형 프롬프트**
   "Do you want to specify a custom path for the dataset folder? (Y/N)" 에서 멈춤.
   자동화 스크립트를 블로킹하므로 `echo "N" |` 로 사전 처리.

3. **datasets 경로 경고는 무시 가능**
   `[Warning]: datasets path ... does not exist!` — 파인튜닝된 체크포인트를 쓰므로
   LIBERO 원본 데이터셋은 불필요.

### 정정

- **`nvidia/Cosmos-Reason2-2B`는 gated가 아니다.**
  인증 없이 `config.json` 다운로드 성공 확인. HF 토큰 불필요.
  README의 "gated" 표기는 오류이므로 수정 대상.


---

## RunPod 실행 검증 (RTX 4090 24GB, $0.74/hr)

### 재현 결과

| 모델 | suite | 성공률 | 공식 수치 | 판정 |
|---|---|---|---|---|
| OpenVLA-OFT | libero_spatial | **50/50 (100%)** | 97.1% | ✅ 재현 성공 |

에피소드당 약 4초 (첫 에피소드만 12초, 이후 3.6~5.1초).

### 이로써 검증된 항목 (⚠️ → ✅)

| 항목 | 근거 |
|---|---|
| observation 스키마 (image/wrist_image 180도 회전, state 8차원) | 실제 추론 성공 |
| action chunk 실행 (NUM_ACTIONS_CHUNK=8, ACTION_DIM=7) | 서버 로그 + 정상 동작 |
| action 정규화 (bounds_q99, PROPRIO_DIM=8) | 공식 수치와 일치 |
| 성공 판정 (`done` within budget) | 성공/실패 정상 기록 |
| WebSocket + msgpack-numpy 프로토콜 | 50 에피소드 무오류 |
| OpenVLA-OFT prompt 포맷 / unnorm_key | 100% 성공률이 곧 증거 |

### 환경 구성에서 발견된 문제 (setup/*.sh 에 반영함)

1. **모델 서버 venv 에도 LIBERO 가 필요하다.**
   `serve_openvla.py` → `experiments.robot.libero.run_libero_eval` 이 최상단에서
   `from libero.libero import benchmark` 를 호출한다.
   GR00T 서버도 동일할 가능성이 높으므로 `30_groot.sh` 확인 필요.

2. **LIBERO editable 설치가 MAPPING 을 비운 채 완료된다.**
   `__editable___libero_0_1_0_finder.py` 의 `MAPPING = {}`.
   → site-packages 에 `libero_path.pth` 직접 생성으로 우회. venv 마다 필요.

3. **첫 import 시 대화형 프롬프트.**
   "Do you want to specify a custom path for the dataset folder? (Y/N)"
   → 자동화 스크립트를 블로킹. `echo "N" |` 로 사전 처리.

4. **protobuf 3자 충돌.**
   - `tensorflow_datasets` → `tensorflow_metadata` 는 `runtime_version`(protobuf 5.27+) 요구
   - `tensorflow` 본체는 `MessageFactory.GetPrototype`(protobuf 6.x 에서 제거) 요구
   - 두 요구는 동시에 만족 불가
   → **`tensorflow-metadata==1.13.1`** 로 낮춰 `runtime_version` 요구를 제거.

   근본 원인: `run_libero_eval` import 사슬이 학습용 데이터 로더
   (`prismatic.vla.datasets.rlds` → `dlimp` → `tensorflow_datasets`)를 끌어온다.
   추론에는 불필요하므로, 필요한 함수만 하위 모듈에서 직접 import 하면 사슬을 끊을 수 있다.

5. **`HF_HUB_ENABLE_HF_TRANSFER=1`.**
   RunPod 이미지가 켜두는데 `hf_transfer` 패키지는 없다.
   → 패키지 설치, 또는 `export HF_HUB_ENABLE_HF_TRANSFER=0`.

6. **`10_libero.sh` 순서 버그.**
   `.pth` 생성이 `python -c "import libero"` 검증 줄보다 뒤에 있어
   `set -e` 로 검증이 먼저 죽으면서 `.pth` 가 만들어지지 않았다. → 순서 수정.

### 무시해도 되는 경고

`datasets path ... does not exist` / cuDNN·cuFFT·cuBLAS 중복 등록 / TF-TRT TensorRT 없음 /
robosuite private macro / OpenGL_accelerate / gym unmaintained — 전부 동작에 영향 없음.
