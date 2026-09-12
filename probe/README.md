# probe — 인지 / 계획 / 제어 3층 비교

성공률이 97~98%에서 포화되어 모델을 구분하지 못한다.
이 프로브는 **같은 관측 하나를 얼려놓고** 입력을 조작해, 성공률이 가리는 차이를 드러낸다.

설계 근거와 각 프로브의 해석 방법은 `DESIGN.md` 참조.

## 사용

서버가 이미 떠 있는 상태에서 (평가 때 쓰던 그 서버 그대로):

```bash
source .venvs/libero/bin/activate

# 모델별로 한 번씩. 모델당 2분 이내.
python probe/collect.py --model groot_n17_libero  --suite libero_spatial --task-id 0
python probe/collect.py --model openvla_oft_libero --suite libero_spatial --task-id 0

# 전부 모이면 한 번에 렌더
python probe/render.py
```

GPU 없이 렌더 파이프라인만 확인하려면:
```bash
python probe/render.py --demo
```

## 산출물

`probe/figures/` 에 PNG + SVG.

| 파일 | 층 | 읽는 법 |
|---|---|---|
| `perception_saliency_<model>` | 인지 | 대상 물체에 집중 vs 배경·팔에 반응 |
| `perception_camera_reliance` | 인지 | agentview(공간) vs wrist(근접) 의존 |
| `perception_language_probe` | 인지 | empty/irrelevant ≈ 0 이면 지시를 무시 중 |
| `planning_intent_3d` | 계획 | 같은 장면에서 접근 경로가 갈리는가 |
| `planning_language_conditioning` | 계획 | **핵심.** 지시를 바꿔도 궤적이 겹치면 언어 미반영 |
| `planning_replan_stability` | 계획 | 재계획 시 궤적이 튀는가 |
| `control_action_timeseries` | 제어 | 7차원 액션 + jerk. 덜컥거림 비교 |
| `control_noise_sensitivity` | 제어 | 기울기가 가파르면 카메라 노이즈에 취약 |
| `control_efficiency` | 제어 | **추가 실행 불필요.** 성공 에피소드 평균 스텝 수 |

## 주의: 이 모델들은 모놀리식이다

GR00T 와 OpenVLA-OFT 는 `(image, state, instruction) -> action chunk` 를 한 번에 낸다.
인지·계획·제어 모듈이 따로 있는 게 아니다. 세 층은 **구조가 아니라 분석 렌즈**이며,
각 층을 직접 뜯는 대신 입력 조작에 대한 출력 변화로 간접 관찰한다.

이 방식의 장점은 모델 내부 구현에 의존하지 않아 **어떤 VLA 에도 똑같이 적용**된다는 것이다.
π0.5 를 추가해도 서버만 띄우면 같은 프로브가 그대로 돈다.

## 해석 시 하지 말아야 할 것

- 가림 민감도를 "attention" 이라고 부르지 말 것. 그건 attention 이 아니라
  **출력 민감도**다. 상관은 있지만 같은 것이 아니다.
- 단일 관측 한 장의 결과를 일반화하지 말 것. task-id 를 여러 개 돌려
  패턴이 반복되는지 확인한 뒤에 주장할 것.
- `target_swap` 은 휴리스틱 문자열 치환이라 문장이 어색해질 수 있다.
  그 자체가 교란 변수이므로, 변형된 지시문을 그림 캡션에 같이 적을 것.
