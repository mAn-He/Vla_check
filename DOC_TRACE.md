# DOC_TRACE.md — 공식 문서 대조 검증 (초안, §9-2단계)

- 검증 일자: 2026-08-27
- 검증 방법: 지시서 §2의 URL 8개 전부와, 각 저장소 내부의 실제 소스 파일(raw.githubusercontent.com)을 fetch하여 대조. GPU/실행 검증 없음 — **모든 항목은 문서·소스 코드 근거만으로 표기**.
- 표기 규칙: ✅ = 저장소/문서에서 직접 확인. ⚠️ = 확인 못했거나 간접 확인(추후 코드에 `# TODO(verify)` 주석 필요).
- 코드가 아직 작성되지 않았으므로 "파일:줄" 컬럼은 **사용 예정 위치**로 기입했다. 코드 작성 후 실제 줄 번호로 갱신한다.

## 접근 제한 (정직성 기록)

- huggingface.co, arxiv.org 직접 fetch가 이 환경에서 차단됨. HF 체크포인트 존재 여부는 웹 검색 결과(실제 HF 페이지로 연결되는 결과)로 교차 확인했다. 해당 항목은 "✅(검색 교차확인)"으로 구분 표기.
- GitHub 저장소(README·raw 소스)는 전부 직접 읽음.

---

## 1. 지시서와 저장소가 달랐던 부분 (저장소 우선 적용)

지시서 §2의 "저장소가 우선" 원칙에 따라, 아래 항목은 **저장소 쪽을 따른다.**

### 1.1 [중대] 통신 프로토콜: HTTP+JSON+base64가 아니라 WebSocket+msgpack

- 지시서 §4는 `GET /health` / `POST /infer` + base64 PNG의 HTTP+JSON 계약을 제시하면서, 동시에 §2에서 "openpi의 프로토콜을 조사해서 채택"하라고 지시한다. 두 지시가 충돌하며, **openpi의 실제 프로토콜은 HTTP가 아니다.**
- 실제(✅): `WebsocketPolicyServer(policy, host="0.0.0.0", port=args.port)` ↔ `WebsocketClientPolicy(host, port)` — `ws://{host}:{port}` 위에서 **msgpack_numpy** 직렬화로 obs dict를 통째로 주고받는다. base64 인코딩 없음(numpy 배열 그대로 pack). 연결 직후 서버가 메타데이터를 먼저 push한다. 서버 오류는 `str` 응답으로 신호.
  - 근거: `packages/openpi-client/src/openpi_client/websocket_client_policy.py`, `scripts/serve_policy.py`
- 적용 결정: 서버 3종은 openpi의 WebSocket+msgpack 프로토콜을 구현한다(`openpi-client` 패키지를 client 의존성으로 사용). §4의 HTTP 스키마는 폐기. `/health` 대응물은 연결 시 서버 메타데이터 push로 대체.

### 1.2 [중대] observation 키: "LIBERO 키를 그대로" 쓰지 않는다

- 지시서 §4: "camera_key는 LIBERO가 주는 키를 그대로 쓴다. 임의로 이름을 바꾸지 말 것."
- 실제(✅): openpi LIBERO 클라이언트(`examples/libero/main.py`)는 LIBERO 키를 그대로 보내지 않고 다음으로 **리매핑**한다:
  - `"observation/image"` ← `obs["agentview_image"]`, `"observation/wrist_image"` ← `obs["robot0_eye_in_hand_image"]`
  - `"observation/state"` ← `np.concatenate((robot0_eef_pos(3), quat→axis-angle(3), robot0_gripper_qpos(2)))` = **8-dim**
  - `"prompt"` ← task language instruction
  - 이미지는 **180도 회전 필수**: `np.ascontiguousarray(img[::-1, ::-1])` ("rotate 180 degrees to match train preprocessing") 후 `resize_with_pad(…, 224, 224)` + uint8 변환.
- 적용 결정: openpi 프로토콜을 채택하므로 obs 키는 openpi 규약(`observation/*`)을 따르고, LIBERO 원본 키→openpi 키 매핑과 180도 회전은 client의 LIBERO 어댑터 계층에 명시적으로 둔다.

### 1.3 [중대] `max_steps`: LIBERO에는 "태스크별 기본값"이 없다

- 지시서 §2는 "태스크별 max_steps 기본값"을 LIBERO에서 확인하라고 하지만, **LIBERO 자체에는 태스크별/스위트별 max_steps가 존재하지 않는다**(✅ — benchmark 모듈에 horizon 없음; env는 robosuite 기본 `horizon=1000`; LIBERO 자체 eval 설정은 전역 `max_steps: 600`, `libero/configs/eval/default.yaml`).
- 스위트별 값은 **openpi 하네스의 선택**이다(✅, `examples/libero/main.py`): `libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520, libero_90=400`. 지시서 §5 예시의 `max_steps: 520`은 이 중 libero_10 값과 일치.
- 적용 결정: 공개 재현 수치(openpi 결과표)와 비교 가능해야 하므로 openpi의 스위트별 값을 기본값으로 채택하고, 출처를 openpi로 명기한다(LIBERO로 인용하지 않음). openpi의 `num_steps_wait=10`(초기 dummy action으로 물리 안정화 대기) 관례도 함께 채택.

### 1.4 [중대] GR00T N1.7: 모델·체크포인트는 실존하나, API가 구버전(N1/N1.5)과 전혀 다름

- `nvidia/GR00T-N1.7-3B`는 실존(✅(검색 교차확인) — Isaac-GR00T main 브랜치 = N1.7 GA; N1.5/N1.6은 `n1d5`/`n1d6` 브랜치). 지시서의 모델 표 자체는 옳다.
- 그러나 널리 알려진 N1.5식 API는 main에서 전부 사라졌다(✅):
  - `gr00t/model/policy.py` → **404**. 현재는 `gr00t/policy/gr00t_policy.py`.
  - 생성자: `Gr00tPolicy(embodiment_tag, model_path, *, device, strict=True)` — `modality_config`/`modality_transform` 인자와 `DATA_CONFIG_MAP`은 **존재하지 않음**. modality config는 체크포인트 내장 `AutoProcessor`에서 나온다(`policy.get_modality_config()`).
  - `get_action(observation, options=None) -> (action_dict, info_dict)` — obs는 flat `"video.ego_view"`식이 아니라 **중첩 dict**: `{"video": {name: uint8 (B,T,H,W,3)}, "state": {name: float32 (B,T,D)}, "language": {"task": [[str]]}}`. 출력 action은 물리 단위(비정규화), base 모델 `action_horizon=40`.
  - 서버/클라이언트: `RobotInferenceServer/Client`가 아니라 `PolicyServer`/`PolicyClient`(**ZeroMQ** tcp 기본 5555, msgpack_numpy) — `gr00t/policy/server_client.py`. 우리 하네스에서는 이를 openpi 프로토콜로 감싸는 어댑터 서버(`serve_groot.py`)를 둔다.
- **LIBERO는 first-class 지원**(✅): 파인튜닝 체크포인트 `nvidia/GR00T-N1.7-LIBERO` + `EmbodimentTag.LIBERO_PANDA = "libero_sim"` + `examples/LIBERO/`(공식 성공률: spatial 97.65 / goal 97.5 / object 98.45 / 10: 94.35).
- 적용 결정: 공정 비교(다른 두 모델은 LIBERO 파인튜닝 체크포인트 사용)를 위해 base `GR00T-N1.7-3B`가 아니라 **`nvidia/GR00T-N1.7-LIBERO` + `LIBERO_PANDA` 태그를 사용**한다. 지시서 표의 체크포인트 컬럼과 다르므로 명시 보고. 주의(⚠️): 이 체크포인트는 HF에서 중첩 폴더 구조라 로딩 시 특수 처리 필요(`scripts/deployment/README.md` 참조), VLM 백본 `nvidia/Cosmos-Reason2-2B`는 **gated**라 HF 인증 필요.

### 1.5 OpenVLA: OFT-LIBERO 체크포인트 실존, 단 인터페이스가 vanilla와 다름

- OFT 레시피: `moojink/openvla-oft` (paper: "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success", arXiv:2502.19645). LIBERO 체크포인트 전부 실존(✅ LIBERO.md 명기 + ✅(검색 교차확인)): `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}` + 통합 `…-libero-spatial-object-goal-10`(스위트별 평균 97.1%, 통합 96.8%).
- 지시서 §2가 전제한 `vla.predict_action(**inputs, unnorm_key=…)` 호출 패턴은 **vanilla openvla용**이다. OFT 추론은 다른 경로(✅): `get_vla(cfg)` + `get_action_head(cfg, llm_dim=…)`(L1-regression MLP 액션 헤드) + `get_proprio_projector(…)` + `get_vla_action(cfg, vla, processor, observation, task_description, action_head, proprio_projector)` → **action chunk 반환**. obs는 `{"full_image", "wrist_image", "state", "task_description"}`. 설정: `use_l1_regression=True, num_images_in_input=2, use_proprio=True, center_crop=True, unnorm_key="libero_spatial_no_noops"` 등.
- 의존성 주의(✅): OFT는 커스텀 transformers fork(`github.com/moojink/transformers-openvla-oft.git`, v4.40.1 기반) 필요 — `40_openvla.sh`에 반영해야 함.
- 참고: openvla org 자체에도 LoRA 파인튜닝 체크포인트 `openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}`가 존재(✅ README). 적용 결정: 성능·속도 근거로 **OFT 체크포인트를 1순위**, openvla LoRA 체크포인트를 대안으로 기록.

### 1.6 경미한 차이·확인 사항

- **task suite 개수**: LIBERO는 suite 5개(`libero_spatial/object/goal/10/90`, 각 10/10/10/10/90 태스크; `libero_100`은 90+10 합산 등록). 지시서 §8의 "suite 4개"는 관례적 평가 대상(spatial/object/goal/10)과 일치하므로 그대로 두되, heatmap 코드가 suite 목록을 하드코딩하지 않고 결과 JSONL에서 유도하게 한다.
- **π0.5 PyTorch 포팅**: 실존 확인(✅) — openpi README "PyTorch Support"(2025-09), `src/openpi/models_pytorch/`, `pi05_libero` config에 `pytorch_weight_path` 필드 존재. 지시서 표와 일치.
- **`pi05_libero` config**: 실존(✅). `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)`, `action_dim`은 `Pi0Config` 기본값 32. 체크포인트 `gs://openpi-assets/checkpoints/pi05_libero` — serve_policy.py의 LIBERO 기본 체크포인트로도 확인. 공식 성공률: spatial 98.8 / object 98.2 / goal 98.0 / 10: 92.4 (평균 96.85).
- **openpi 이슈 #849 (`pi0_fast_libero` 체크포인트 불일치)**: README·LIBERO README에서 언급 확인 못함(⚠️). `pi0_fast_libero` train config는 존재하나 공개 체크포인트는 README 표에 없음. 20-에피소드 0% 가드레일(§6) 설계 근거로는 그대로 유지(가드레일 자체는 이슈 진위와 무관하게 타당).
- **Python 버전 충돌 실측치**: LIBERO(권장 3.8.13) / openpi LIBERO client(uv venv --python 3.8) / GR00T(>=3.12,<3.13 전용) / OpenVLA·OFT(3.10). venv 분리 지시(§1)가 저장소들 요구사항과 정합함을 확인(✅).
- **flash-attn**: GR00T `flash-attn==2.8.3`(cu12·torch2.9·cp312 prebuilt wheel 고정), OpenVLA `flash-attn==2.5.5`(vanilla 추론 경로에서는 optional). 버전이 서로 달라 §1의 "venv 분리" 근거 재확인(✅).
- **MUJOCO_GL**: openpi compose.yml은 `MUJOCO_GL=egl`(+`PYOPENGL_PLATFORM=egl`, `MUJOCO_EGL_DEVICE_ID=0`)이 기본이고 문서화된 폴백은 `glx`뿐(✅). 지시서의 egl→osmesa→glx 3단 폴백은 openpi보다 넓은 초집합이므로 그대로 채택(osmesa 단계는 ⚠️ — openpi 문서에 없음, 우리 추가분).

---

## 2. 확인된 API 시그니처 (저장소별)

### 2.1 LIBERO (`Lifelong-Robot-Learning/LIBERO`, 기본 브랜치 `master`)

| 파일:줄(예정) | 호출 | 근거 URL | 문서상 시그니처 / 사실 | 확인 |
|---|---|---|---|---|
| `client/libero_env.py` | `benchmark.get_benchmark_dict()` | raw…/LIBERO/master/libero/libero/benchmark/__init__.py | `def get_benchmark_dict(help=False)` → `BENCHMARK_MAPPING` (소문자 suite명 → benchmark 클래스; `get_benchmark(name)`도 존재) | ✅ |
| `client/libero_env.py` | `task_suite = benchmark_dict["libero_10"]()` … `task_suite.get_task(i)` | 동일 + README | `Task` NamedTuple: `name, language, problem, problem_folder, bddl_file, init_states_file`; `get_num_tasks()→n_tasks`, `get_task_names()`, `get_task_bddl_file_path(i)`, `get_task_init_states(i)` | ✅ |
| `client/libero_env.py` | suite 구성 | raw…/master/libero/libero/benchmark/libero_suite_task_map.py | `libero_spatial`:10, `libero_object`:10, `libero_goal`:10, `libero_10`:10, `libero_90`:90 ("libero_long"이라는 이름은 없음 — 장기 스위트는 `libero_10`) | ✅ |
| `client/libero_env.py` | `OffScreenRenderEnv(bddl_file_name=…, camera_heights=…, camera_widths=…)` | raw…/master/libero/libero/envs/env_wrapper.py | `class OffScreenRenderEnv(ControlEnv)`: `has_renderer=False, has_offscreen_renderer=True` 강제 후 `ControlEnv.__init__(bddl_file_name, robots=["Panda"], controller="OSC_POSE", …, camera_names=["agentview","robot0_eye_in_hand"], camera_heights=128, camera_widths=128, horizon=1000, …)` | ✅ |
| `client/libero_env.py` | `env.seed(s)`, `env.reset()`, `env.step(a)` | 동일 + README | `step` → gym-classic 4-tuple `(obs, reward, done, info)`; `reset`은 RandomizationError 재시도 포함 | ✅ |
| `client/libero_env.py` | `env.set_init_state(init_states[k])` | 동일 + benchmark/__init__.py | `get_task_init_states(i)`: `torch.load(init_states 경로)`; `set_init_state` → `regenerate_obs_from_state(init_state)`. LIBERO 자체 eval(metric.py)은 set_init_state 후 dummy zero-action으로 물리 안정화 | ✅ |
| `client/libero_env.py` | obs 키 | raw…/master/libero/configs/data/default.yaml (obs_key_mapping) | `agentview_image`, `robot0_eye_in_hand_image`, `robot0_gripper_qpos`, `robot0_joint_pos`; `robot0_eef_pos`/`robot0_eef_quat`는 robosuite `SingleArmEnv` 상속분 | ✅ |
| `client/libero_env.py` | `{camera_name}_image` 키 생성 규칙 | robosuite (LIBERO repo 밖) | LIBERO repo 안에서 해당 라인 미확인 — 결과 키는 위 obs_key_mapping으로 검증됨 | ⚠️ TODO(verify: robosuite 소스) |
| `client/run_eval.py` | action 차원 | README(`dummy_action=[0.]*7`), libero/lifelong/metric.py(`np.zeros((env_num,7))`) | 7-dim (OSC_POSE 6 + gripper 1) | ✅ |
| `client/run_eval.py` | action 범위 [-1,1] | (LIBERO repo에 명시 없음 — robosuite OSC 관례) | 미확인 | ⚠️ TODO(verify: robosuite controller) |
| `client/run_eval.py` | max_steps | raw…/master/libero/configs/eval/default.yaml | LIBERO 자체: 전역 `max_steps: 600`, `n_eval: 20`. 태스크별 기본값 없음 → §1.3 결정 참조 | ✅ |
| `setup/10_libero.sh` | 설치 | README + setup.py | 소스 설치(`pip install -r requirements.txt && pip install -e .`), 패키지명 `libero` v0.1.0, 권장 Python 3.8.13. PyPI 배포 여부 미확인 | ✅ (PyPI는 ⚠️) |

### 2.2 openpi (`Physical-Intelligence/openpi`, main)

| 파일:줄(예정) | 호출 | 근거 URL | 문서상 시그니처 / 사실 | 확인 |
|---|---|---|---|---|
| `servers/serve_openpi.py` | `scripts/serve_policy.py` 인자 | raw…/openpi/main/scripts/serve_policy.py | tyro CLI. `EnvMode: ALOHA/ALOHA_SIM/DROID/LIBERO`; `default_prompt`, `port=8000`, `record`; `policy:checkpoint --policy.config=pi05_libero --policy.dir=…`; LIBERO 기본 = `Checkpoint(config="pi05_libero", dir="gs://openpi-assets/checkpoints/pi05_libero")` | ✅ |
| `servers/base_server.py` | 서버 프로토콜 | 동일 | `WebsocketPolicyServer(policy=…, host="0.0.0.0", port=…, metadata=…)` + `serve_forever()` — WebSocket+msgpack (§1.1) | ✅ |
| `client/policy_client.py` | 클라이언트 | raw…/packages/openpi-client/src/openpi_client/websocket_client_policy.py | `WebsocketClientPolicy(host="0.0.0.0", port=None, api_key=None)`; `infer(obs: Dict) -> Dict`; msgpack_numpy, `compression=None, max_size=None`; 연결 시 서버 metadata 수신; `BasePolicy`: abstract `infer`, no-op `reset()` | ✅ |
| `client/policy_client.py` | 요청/응답 포맷 | raw…/openpi/main/examples/libero/main.py | 요청 키: `observation/image`, `observation/wrist_image`, `observation/state`(8-dim, §1.2), `prompt`. 응답: `client.infer(element)["actions"]` = action chunk(각 action 7-dim; dummy는 `[0.0]*6+[-1.0]`) | ✅ |
| `configs/eval.yaml` | action horizon / replan | config.py + main.py | `pi05_libero`: `action_horizon=10`; client `replan_steps=5` (`action_plan.extend(chunk[:replan_steps])`, chunk 길이 ≥ replan_steps assert) | ✅ |
| `client/libero_env.py` | 이미지 전처리 | main.py | 렌더 256(`LIBERO_ENV_RESOLUTION`), 모델 입력 `resize_size=224`, 180도 회전 + `resize_with_pad` + uint8 (§1.2); `num_steps_wait=10`, `num_trials_per_task=50`, `seed=7` | ✅ |
| `client/run_eval.py` | max_steps per suite | main.py | spatial 220 / object 280 / goal 300 / 10: 520 / 90: 400 | ✅ |
| `setup/20_openpi.sh` | venv 구성 | README + examples/libero/README.md + compose.yml | 본체: `GIT_LFS_SKIP_SMUDGE=1 uv sync`. LIBERO client는 별도 **Python 3.8** venv(`uv venv --python 3.8 examples/libero/.venv` + `uv pip sync … --extra-index-url cu113 …` + `openpi-client`, `third_party/libero` editable + PYTHONPATH). 권장 경로는 docker compose(`SERVER_ARGS="--env LIBERO"`), `MUJOCO_GL=egl` 기본·`glx` 폴백 | ✅ |
| `servers/serve_openpi.py` | PyTorch 포팅 | README "PyTorch Support" | π₀/π₀.₅ PyTorch 구현 존재(`src/openpi/models_pytorch/`), transformers 4.53.2, 체크포인트 자동 감지 | ✅ |

### 2.3 GR00T (`NVIDIA/Isaac-GR00T`, main = N1.7)

| 파일:줄(예정) | 호출 | 근거 URL | 문서상 시그니처 / 사실 | 확인 |
|---|---|---|---|---|
| `servers/serve_groot.py` | 정책 로딩 | raw…/Isaac-GR00T/main/gr00t/policy/gr00t_policy.py | `Gr00tPolicy(embodiment_tag: EmbodimentTag \| str, model_path: str, *, device: int \| str, strict: bool = True)` — modality config는 체크포인트 내장 `AutoProcessor`에서(`get_modality_configs()`); `DATA_CONFIG_MAP` 없음 (§1.4) | ✅ |
| `servers/serve_groot.py` | `EmbodimentTag` | raw…/main/gr00t/data/embodiment_tags.py | LIBERO용: `LIBERO_PANDA = "libero_sim"`. 기타: `NEW_EMBODIMENT="new_embodiment"`, `SIMPLER_ENV_GOOGLE/WIDOWX`, `UNITREE_G1…` 등. 구버전 `GR1` 태그 없음 | ✅ |
| `servers/serve_groot.py` | `get_action()` | raw…/main/gr00t/policy/policy.py + getting_started/policy.md | `get_action(observation: dict, options: dict \| None = None) -> tuple[action_dict, info_dict]`; obs 중첩 dict(video uint8 (B,T,H,W,3) / state float32 (B,T,D) / language `{"task": [[str]]}`); action은 비정규화 float32 (B,T,D), base `action_horizon=40`, 실행 간격은 `--execution-horizon` | ✅ |
| `servers/serve_groot.py` | 자체 서버/클라이언트 | raw…/main/gr00t/policy/server_client.py | `PolicyServer(policy, host="*", port=5555, api_token=None)` / `PolicyClient(host="localhost", port=5555, timeout_ms=15000, …)` — ZeroMQ+msgpack_numpy. 런처: `gr00t/eval/run_gr00t_server.py --model-path … --embodiment-tag …` | ✅ |
| `servers/serve_groot.py` | LIBERO 체크포인트 | github…/Isaac-GR00T/blob/main/examples/LIBERO/README.md | `nvidia/GR00T-N1.7-LIBERO` + `LIBERO_PANDA`; 데이터 `IPEC-COMMUNITY/libero_*_no_noops_1.0.0_lerobot`; 공식 성공률 97.65/97.5/98.45/94.35; HF상 중첩 폴더 구조 → 로딩 특수 처리(`scripts/deployment/README.md`) | ✅ (HF 페이지는 검색 교차확인) |
| `setup/30_groot.sh` | 의존성 | raw…/main/pyproject.toml + README | `requires-python = ">=3.12,<3.13"`, `torch==2.9.0`, `flash-attn==2.8.3`(cu12torch2.9 cp312 prebuilt wheel URL 고정), torchcodec 0.8.0(FFmpeg 4–7), 추론 VRAM 16GB+. VLM 백본 `nvidia/Cosmos-Reason2-2B`는 gated(HF 로그인 필요) | ✅ |

### 2.4 OpenVLA (`openvla/openvla` + `moojink/openvla-oft`)

| 파일:줄(예정) | 호출 | 근거 URL | 문서상 시그니처 / 사실 | 확인 |
|---|---|---|---|---|
| `servers/serve_openvla.py` | vanilla 로딩 | github.com/openvla/openvla README | `AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)`; `AutoModelForVision2Seq.from_pretrained(…, attn_implementation="flash_attention_2"(optional), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)` | ✅ |
| `servers/serve_openvla.py` | `predict_action` | raw…/openvla/main/prismatic/extern/hf/modeling_prismatic.py:506 | `def predict_action(self, input_ids=None, unnorm_key: Optional[str] = None, **kwargs) -> np.ndarray` — generate() 래퍼, 비정규화 action 반환(차원은 `get_action_dim(unnorm_key)`), 토큰 29871 자동 보정 | ✅ |
| `servers/serve_openvla.py` | prompt 포맷 | 동일 README | `"In: What action should the robot take to {<INSTRUCTION>}?\nOut:"` | ✅ |
| `servers/serve_openvla.py` | openvla-7b의 LIBERO 학습 여부 | README + HF 모델카드 | OXE 970K 트래젝토리 학습, **LIBERO 파인튜닝 아님** → 지시서 §2의 우려대로 그대로 쓰면 불공정 | ✅ (모델카드 원문은 검색 교차확인) |
| `servers/serve_openvla.py` | OFT 추론 경로 | github.com/moojink/openvla-oft README + LIBERO.md | `get_vla(cfg)`, `get_processor(cfg)`, `get_action_head(cfg, llm_dim=vla.llm_dim)`, `get_proprio_projector(…)`, `get_vla_action(cfg, vla, processor, observation, task_description, action_head, proprio_projector)` → **action chunk**; obs `{"full_image","wrist_image","state","task_description"}`; `use_l1_regression=True, num_images_in_input=2, use_proprio=True, center_crop=True, num_open_loop_steps=NUM_ACTIONS_CHUNK, unnorm_key="libero_spatial_no_noops"`; 평가 스크립트 `experiments/robot/libero/run_libero_eval.py` | ✅ |
| `servers/serve_openvla.py` | OFT-LIBERO 체크포인트 | LIBERO.md | `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}` + 통합 `…-spatial-object-goal-10` (97.1% / 96.8%). 대안: `openvla/openvla-7b-finetuned-libero-{…}` LoRA 체크포인트 | ✅ (HF 페이지는 검색 교차확인) |
| `setup/40_openvla.sh` | 의존성 | openvla README + openvla-oft SETUP.md | Python 3.10, PyTorch 2.2.*, transformers 4.40.1, flash-attn 2.5.5(`--no-build-isolation`). **OFT는 커스텀 transformers fork 필수**: `github.com/moojink/transformers-openvla-oft.git` | ✅ |
| `servers/serve_openvla.py` | `unnorm_key` 값 | README + OFT | zero-shot: `"bridge_orig"`; LIBERO fine-tune: `"libero_spatial_no_noops"` 등 suite별 `no_noops` 키 | ✅ |

---

## 3. §2 URL fetch 체크리스트

| URL | 상태 |
|---|---|
| github.com/Lifelong-Robot-Learning/LIBERO | ✅ 읽음 (+benchmark/__init__.py, env_wrapper.py, libero_suite_task_map.py, configs, metric.py, setup.py) |
| github.com/Physical-Intelligence/openpi | ✅ 읽음 |
| …/openpi/blob/main/examples/libero/README.md | ✅ 읽음 (+examples/libero/main.py, compose.yml) |
| …/openpi/blob/main/src/openpi/training/config.py | ✅ 읽음 (+pi0_config.py, serve_policy.py, openpi-client 소스) |
| github.com/NVIDIA/Isaac-GR00T | ✅ 읽음 (+gr00t/policy/*, embodiment_tags.py, pyproject.toml, getting_started/*, examples/LIBERO) |
| huggingface.co/nvidia/GR00T-N1.7-3B | ⚠️ 직접 fetch 차단 — 존재는 검색 결과의 실제 HF 페이지로 교차 확인 |
| github.com/openvla/openvla | ✅ 읽음 (+modeling_prismatic.py) |
| huggingface.co/openvla/openvla-7b | ⚠️ 직접 fetch 차단 — README의 동일 스니펫 + 검색 결과로 교차 확인 |

## 4. 남은 TODO(verify) 목록 — 코드 작성 시 주석으로 반영할 것

1. robosuite의 `{camera_name}_image` 키 생성 라인, OSC_POSE action 범위 [-1,1] — robosuite 소스에서 확인 필요.
2. HF 모델카드 원문(GR00T-N1.7-3B / N1.7-LIBERO / openvla-7b / moojink OFT 체크포인트) — 네트워크 허용 환경에서 직접 확인.
3. openpi 이슈 #849 원문 — GitHub Issues 열람으로 확인.
4. GR00T `PolicyClient.get_action`에 넘길 LIBERO obs의 정확한 video/state 키 이름 — `nvidia/GR00T-N1.7-LIBERO` 체크포인트의 processor/modality config를 받아서 확인(체크포인트 다운로드 필요, 이번 범위 밖).
5. `MUJOCO_GL=osmesa` 폴백 동작 — 어느 공식 문서에도 없음(우리 추가분), 실행 환경에서 확인.
