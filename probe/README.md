# probe — 같은 입력, 다른 반응

성공률이 97~98%에서 포화되어 모델을 구분하지 못한다.
같은 관측을 두 모델에 먹여 **인지 / 계획 / 제어**가 어떻게 갈리는지 본다.

## 왜 에피소드를 저장해서 재생하는가

롤아웃을 각자 굴리면 모델마다 다른 상태를 지나가므로 **입력 자체가 달라진다.**
비교가 성립하지 않는다. 그래서 한 번만 굴려 관측을 저장하고,
그 관측을 모든 모델에 그대로 재생한다.

## 사용

```bash
source .venvs/libero/bin/activate

# 1) 에피소드 한 번 굴려서 관측 저장 (아무 모델로나. 서버 떠 있어야 함)
python probe/dump_episode.py --model groot_n17_libero --suite libero_spatial --task-id 0

# 2) 모델마다 서버를 바꿔 띄우면서 같은 에피소드를 재생
python probe/compare.py --episode probe/episodes/libero_spatial_t0_by-groot_n17_libero.npz \
                        --model groot_n17_libero
#    (서버 교체 후)
python probe/compare.py --episode probe/episodes/libero_spatial_t0_by-groot_n17_libero.npz \
                        --model openvla_oft_libero

# 3) 렌더
python probe/compare.py --render
```

빠르게 보려면 `--grid 0` (saliency 생략, 30초), 정밀하게는 `--grid 12`.

## 산출물 — probe/compare/figures/

| 파일 | 층 | 읽는 법 |
|---|---|---|
| `01_action_disagreement` | 종합 | 에피소드 어느 지점에서 두 모델이 갈리는가 |
| `02_action_timeseries` | 제어 | 7차원 액션 + jerk. 덜컥거림 비교 |
| `03_intent_trajectory` | 계획 | 같은 첫 프레임에서 어디로 가려 하는가 |
| `04_language_probe` | 인지 | empty/irrelevant ≈ 0 이면 지시를 무시 중 |
| `05_camera_reliance` | 인지 | agentview(공간) vs wrist(근접) 의존 |
| `06_saliency_<model>` | 인지 | 대상 물체를 보는가, 배경을 보는가 |

## 주의

- 이 모델들은 **모놀리식**이다. 인지·계획·제어 모듈이 따로 없다.
  세 층은 구조가 아니라 **분석 렌즈**이며, 입력 조작에 대한 출력 변화로 간접 관찰한다.
- 가림 민감도를 "attention" 이라 부르지 말 것. attention 이 아니라 **출력 민감도**다.
- 관측 한 장/에피소드 하나의 결과를 일반화하지 말 것. task-id 를 바꿔가며
  패턴이 반복되는지 확인한 뒤 주장할 것.
- `target_swap` 은 휴리스틱 치환이라 문장이 어색해질 수 있다.
  변형된 문장을 그림 캡션에 같이 적을 것. (`lang_variants` 로 저장돼 있다)
- `configs/eval.yaml` 에서 groot/openvla 는 `resize_with_pad: null` 이므로
  원본 256px 를 그대로 보낸다. π0.5 를 추가하려면 224 리사이즈가 필요하다.

---

## storyboard.py — 사람이 읽는 버전

`compare.py --render` 는 분석용 그림이고, 이쪽은 **한 장에 전부 담은 카드**다.

```bash
python probe/storyboard.py            # 전체
python probe/storyboard.py --frame 2  # 특정 프레임만
```

### story_frame<k>.png — 프레임 한 장의 전말

```
제목          : frame k / episode step / INSTRUCTION 원문
윗줄          : SCENE(실제 장면) | WRIST | 모델별 "어디를 봤는가"(saliency)
아랫줄        : 모델별 계획 경로(위에서 본 것, 색=높이)
                first action 막대 (이동/회전 6축 + 그리퍼 분리)
                DECISION — 액션을 말로 푼 것 ("DOWN | gripper OPEN")
```

두 모델의 경로는 **같은 축 범위**로 그린다. 안 그러면 눈으로 비교가 안 된다.

### story_language.png — 같은 이미지, 다른 지시

행마다 프롬프트 원문을 그대로 적고, 그 프롬프트로 나온 경로를 옆에 놓는다.
전 행의 경로가 똑같으면 **모델이 텍스트를 안 읽고 있다**는 뜻이다.

`target_swap`(대상 물체를 바꾼 지시)에서 경로가 안 변하는데
`empty`/`irrelevant`에서는 크게 변한다면 — 지시문의 **유무**는 보지만
**내용**은 안 본다는 해석이 가능하다.

### story_overview.png — 에피소드 전체 흐름

프레임별로 장면과 모델별 계획을 한 줄씩. 어느 시점에서 갈리는지 훑어볼 때.

### 해석 주의

- saliency 는 attention 이 아니라 **출력 민감도**다. 가렸을 때 행동이 바뀌는 곳일 뿐.
- LIBERO spatial 은 태스크가 전부 "black bowl 을 plate 에" 변형이라
  언어 변별이 애초에 덜 필요하다. `libero_object` / `libero_goal` 에서
  같은 패턴이 반복되는지 확인한 뒤에 주장할 것.
- 에피소드 하나의 결과를 일반화하지 말 것.
