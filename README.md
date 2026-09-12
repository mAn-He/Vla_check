# VLA Check — 같은 입력, 다른 반응

오픈 VLA 모델을 동일 조건으로 평가하고, 성공률이 가리는 차이를 인지 · 계획 · 제어 세 층위에서 드러내는 프로젝트.

| 모델 | 체크포인트 | 액션 생성 | 청크 |
|---|---|---|---|
| GR00T N1.7 | `nvidia/GR00T-N1.7-LIBERO` | flow matching (DiT) | 16 |
| OpenVLA-OFT | `moojink/openvla-7b-oft-finetuned-libero-*` | L1 회귀 (MLP head) | 8 |
| π0.5 *(예정)* | `gs://openpi-assets/checkpoints/pi05_libero` | flow matching | 10 |

---

## 배경

LIBERO 표준 벤치마크에서 최신 VLA 성공률이 97~98%로 포화. 직접 재현해도 동일.

| | LIBERO-Spatial | 공식 보고치 |
|---|---|---|
| OpenVLA-OFT | 50/50 (**100%**) | 97.1% |
| GR00T N1.7 | 49/50 (**98.0%**) | 97.65% |

- 표본 50개에서 100%와 98%는 통계적으로 동일 → 이 축으로는 구분 불가
- heatmap을 그려도 전 셀이 동일 색
- 선행 연구도 표준 LIBERO의 한계를 지적. 평가 태스크가 훈련 태스크와 같고 차이는 물체 초기 상태의 미세 섭동뿐
- 즉 97%는 "모델이 좋다"보다 **"벤치마크가 포화됐다"**에 가까움

→ 변별력 있는 축이 필요

---

## 접근: 관측을 얼려 재생

각자 롤아웃 시 모델마다 다른 상태를 지나가므로 **입력 자체가 달라짐**. 반응 차이가 모델 탓인지 상황 탓인지 구분 불가.

에피소드를 한 번만 굴려 관측을 저장 → 그 관측을 모든 모델에 그대로 재생. 환경은 굴리지 않음.

```
┌──────────────────┐     동일한 관측     ┌──────────────────┐
│  episode.npz     │ ─────────────────▶ │  model A / B / C │
│  (얼린 관측)      │ ◀───────────────── │                  │
└──────────────────┘      액션 청크      └──────────────────┘
```

그 위에서 입력을 조작하고 출력 변화를 관찰.

| 층 | 프로브 | 방법 |
|---|---|---|
| **인지** | 무엇을 보는가 | 이미지를 격자로 가리고 액션 변화 측정 |
| | 어느 카메라에 의존하는가 | 카메라를 통째로 가리고 비교 |
| | 언어를 읽는가 | 같은 이미지에 프롬프트만 5종 변형 |
| **계획** | 어디로 가려 하는가 | 액션 청크를 적분해 의도 궤적 복원 |
| | 언어가 계획을 바꾸는가 | 프롬프트별 궤적 중첩 비교 |
| **제어** | 어떻게 실행하는가 | 액션 시계열, jerk, 그리퍼 타이밍 |

> GR00T · OpenVLA-OFT는 모놀리식. 인지·계획·제어 모듈이 분리돼 있지 않음.
> 세 층은 **구조가 아니라 분석 렌즈**이며, 입력 조작에 대한 출력 변화로 간접 관찰.
> 모델 내부에 의존하지 않아 **어떤 VLA에도 동일 적용 가능**.

---

## 발견

전부 아키텍처 차이로 설명 가능. 근거는 [`ANALYSIS.md`](./ANALYSIS.md).

### 1. GR00T는 비결정적

같은 이미지 + 같은 프롬프트를 두 번 입력. 정의상 0이어야 함.

```
                 original
GR00T N1.7        0.085     ← 같은 입력인데 액션이 달라짐
OpenVLA-OFT       0.000     ← 정확히 동일
```

- flow matching 추론은 **가우시안 노이즈 벡터 샘플링으로 시작** → 매 호출마다 다른 출발점
- L1 회귀 헤드는 hidden state → 액션의 결정적 함수
- **0.085 = GR00T의 측정 바닥.** 이보다 작은 변화는 관측 불가

### 2. 언어 — 노이즈 보정 시 격차 확대

```
                original   synonym  target_swap   empty  irrelevant
GR00T N1.7        0.085      0.094      0.096      0.907    0.921
OpenVLA-OFT       0.000      0.031      0.172      0.240    0.233
```

GR00T의 `target_swap`(대상을 bowl→plate로 교체) 반응 0.096은 자기 노이즈 0.085와 구분 불가.

노이즈 바닥을 뺀 언어 민감도:

|  | (target_swap − original) / (empty − original) |
|---|---|
| GR00T N1.7 | **1.3 %** |
| OpenVLA-OFT | **71.7 %** |

- GR00T는 지시문 **유무**는 확실히 감지 (empty 0.907, 노이즈의 10배)
- 지시문 **내용**은 거의 미반영

**구조적 원인**

- GR00T N1.5부터 VLM이 사전학습·파인튜닝 양쪽에서 **동결**
- 언어 경로: 동결 Eagle VLM → cross-attention → DiT. 후속학습에서 갱신되는 건 DiT뿐
- OpenVLA-OFT는 언어 토큰이 Llama-2 시퀀스에 직접 진입, LoRA가 self-attention·MLP 전 선형층에 부착 → **LLM 자체가 파인튜닝**

부가 관찰: OpenVLA-OFT의 LIBERO 체크포인트는 언어 그라운딩 장치 **FiLM이 비활성**(`use_film=False`). 그럼에도 GR00T보다 언어 반영도가 높음 → FiLM 유무가 아니라 **동결 vs 파인튜닝**이 본질이라는 해석을 지지.

### 3. 시각 의존 — OpenVLA-OFT는 wrist 편중

```
              agentview   wrist
GR00T N1.7      1.232     1.224    ← 균형
OpenVLA-OFT     0.218     1.526    ← wrist 7.0배
```

- OpenVLA-OFT는 3인칭 뷰 + **wrist 뷰 + proprio 상태**를 함께 입력받음
- proprio가 "팔의 위치", wrist가 "그리퍼 앞 물체 위치"를 제공 → 3인칭 뷰의 공간 정보가 상당 부분 중복
- 실무 함의: 3인칭 카메라 가림에는 강건, **wrist 카메라 오염에는 취약**
- 실제 로봇에서 wrist는 물체에 가려지고 조명 변화가 심한 위치

### 4. 가장 공정한 프레임에서 최대 불일치

```
frame 0  ‖Δ first action‖ = 2.143   ← 최대
frame 1~4                   0.078 ~ 0.113
```

- 프레임 0은 `set_init_state` 직후 → 어느 정책이 굴렸든 동일 상태 → **유일한 정책 중립 프레임**
- 가장 공정한 지점에서 불일치가 25배

```
GR00T:   거의 정지  | gripper OPEN
OpenVLA: 큰 폭 이동 | gripper CLOSE
```

그리퍼 지령이 정반대. 규약 문제 가능성 있어 확인 필요.

---

## 한계

본 결과는 **태스크 1개 · 프레임 1개 · 시드 1개**. 일반화 주장 아님.

| 교란 변수 | 상태 |
|---|---|
| 청크 길이 16 vs 8 — ‖Δ‖를 다른 크기 텐서에서 계산 | 모델 **내부** 비율 비교는 무관, **간** 절대값 비교는 보정 필요 |
| off-policy 재생 — 프레임 1~4는 GR00T 궤적 | 교차 재생(2×2)으로 대칭 확보 필요 |
| 노이즈 바닥 0.085가 표본 1개 | N≥30 반복 측정 필요 |
| LIBERO-Spatial은 목적어가 전부 동일 | `libero_object`에서 재확인 필요 |
| jerk를 프레임 간 차분으로 계산 | 청크 **내부** 차분으로 수정 필요 |

다음 실험 우선순위: [`ANALYSIS.md` §7](./ANALYSIS.md)

---

## 구조

```
.
├── ANALYSIS.md              분석 본문 — 아키텍처 근거와 해석
├── DOC_TRACE.md             외부 API 근거 추적 + 환경 구축 이슈
├── VERIFY.md                GPU 환경 검증 절차
├── setup/                   venv 4개 구성 (모델별 의존성 충돌 회피)
│   ├── 00_common.sh         공통 패키지 + uv
│   ├── 10_libero.sh         eval client (py3.8, torch 1.11)
│   ├── 20_openpi.sh         π0 server (py3.11, JAX)
│   ├── 30_groot.sh          GR00T server (py3.12, torch 2.9)
│   ├── 40_openvla.sh        OpenVLA server (py3.10, torch 2.2)
│   └── check_env.py         GPU/VRAM/디스크/렌더링 사전 점검
├── client/
│   ├── libero_env.py        LIBERO 래퍼 (공식 하네스와 동일 전처리)
│   ├── policy_client.py     openpi WebSocket 프로토콜
│   └── run_eval.py          평가 루프 (resume 기본, ETA/0% 가드레일)
├── servers/                 모델별 policy server (공통 인터페이스)
├── probe/
│   ├── dump_episode.py      에피소드 관측 저장
│   ├── compare.py           동일 관측 재생 + 프로브 수집
│   └── storyboard.py        사람이 읽는 카드 렌더링
├── configs/eval.yaml        모델별 전처리 설정 (공식 하네스 근거 주석 포함)
└── results/                 에피소드 단위 JSONL
```

**venv 4분할 이유**: 세 모델의 의존성이 정면충돌 (py3.8/3.10/3.12, torch 1.11/2.2/2.9, JAX vs PyTorch). 단일 환경 구성 시 반드시 파손. WebSocket으로 프로세스 분리.

---

## 재현

환경: Ubuntu 22.04 / CUDA GPU 24GB+ / 디스크 100GB+

```bash
bash setup/00_common.sh
bash setup/10_libero.sh
source .venvs/libero/bin/activate
python setup/check_env.py           # MuJoCo 렌더링 백엔드 확인
```

### 성공률 평가

```bash
# 서버 (모델별 venv)
bash setup/40_openvla.sh
source .venvs/openvla/bin/activate
OFT_DIR=third_party/openvla-oft python servers/serve_openvla.py \
  --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task-suite libero_spatial --port 8700

# 클라이언트 (별도 터미널)
source .venvs/libero/bin/activate
python client/run_eval.py --model openvla_oft_libero --suite libero_spatial
```

### 프로브

```bash
# 1) 에피소드 관측 저장 (한 번만)
python probe/dump_episode.py --model groot_n17_libero --suite libero_spatial --task-id 0

# 2) 모델별 서버를 교체하며 같은 관측 재생
python probe/compare.py --episode probe/episodes/libero_spatial_t0_by-groot_n17_libero.npz \
                        --model groot_n17_libero --grid 8
python probe/compare.py --episode <같은 파일> --model openvla_oft_libero --grid 8

# 3) 렌더
python probe/compare.py --render     # 분석용
python probe/storyboard.py           # 장면+지시문+판단 카드
```

`--grid 0`으로 saliency 생략 시 30초 소요 (파이프라인 확인용).

---

## 비교군 선정 근거

VLA와 월드 모델을 같은 성공률 축에 놓으려면 출력 타입이 일치해야 함.
VLA는 `관측+지시 → 액션`, 월드 모델은 `관측+액션 → 다음 관측`.
후자는 정책이 아니라 학습된 시뮬레이터에 가까워 플래너를 얹어야 하고, 그 경우 비교 대상이 "모델"이 아닌 "모델+플래너"가 됨.

| 제외 | 사유 |
|---|---|
| **Genie 3** | 클로즈드. 공개 API·오픈 웨이트 없음. 구독형 인터랙티브 UI로만 접근 가능해 자동 배치 평가 불가. 월드 모델이라 출력 타입도 상이 |
| **Gemini Robotics (VLA)** | trusted tester 전용 |
| **Gemini Robotics ER** | API 접근은 가능하나 VLM(추론 계층). 액션을 직접 출력하지 않고 VLA를 오케스트레이션하는 상위 계층 |

---

## 환경 구축 이슈

전체 기록은 `DOC_TRACE.md`. 재현자가 같은 시간을 쓰지 않도록 정리.

- LIBERO editable 설치가 `MAPPING = {}` 상태로 완료 → `import libero` 실패 → `.pth` 수동 생성
- LIBERO 첫 import 시 대화형 프롬프트가 자동화 스크립트를 블로킹
- 모델 서버 venv에도 LIBERO 필요 (`run_libero_eval` import 사슬)
- protobuf 3자 충돌: tfds는 `runtime_version`(5.27+) 요구, TF는 `GetPrototype`(6.x 제거) 요구 → `tensorflow-metadata==1.13.1`로 해결
- `GR00T-N1.7-LIBERO`는 suite별 서브폴더 구조. 루트 경로로는 로딩 불가
- 동일 저장소에 DeepSpeed 학습 상태(`global_step*`, 12GB+) 포함 → `ignore_patterns` 필요
- `hf_transfer`가 특정 파일에서 0% hang → `HF_HUB_ENABLE_HF_TRANSFER=0`

---

## 참고

- [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734)
- [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)](https://arxiv.org/abs/2502.19645)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [openpi](https://github.com/Physical-Intelligence/openpi)
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [openvla-oft](https://github.com/moojink/openvla-oft)
