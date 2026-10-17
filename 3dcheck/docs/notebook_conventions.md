# 노트북 작성 규약 (실외/실내 공통)

CLAUDE.md의 "노트북 작성 규칙"을 코드 수준으로 구체화한 문서입니다. 모든 노트북은 이 규약을 따릅니다.

## 1. 파일과 빌드

- 소스: `<part>/notebooks/src/NN_주제.py` (jupytext percent 형식). 리뷰와 diff는 이 파일로 합니다.
- 산출물: `<part>/notebooks/NN_주제.ipynb`. `python envs/build_notebooks.py`(3dv env)로 생성하고, 손으로 고치지 않습니다.
- 빌더는 nbformat 스키마, 커널 지정, 첫 셀 필수 항목, 코드 셀 문법을 검사합니다. `--check`는 소스와 ipynb의 동기화 여부만 봅니다.
- 소스 맨 위에 커널을 지정합니다.

```python
# ---
# jupyter:
#   kernelspec:
#     display_name: 3dv (torch 2.1.2, cu121)
#     language: python
#     name: 3dv
# ---
```

`3dv-point` 노트북은 `display_name: 3dv-point (torch 2.5.0, cu124)`, `name: 3dv-point`로 씁니다.
마크다운 셀은 `# %% [markdown]`, 코드 셀은 `# %%`로 시작합니다.

## 2. 첫 셀 (마크다운, 한국어) — 아래 6개 제목 필수

```markdown
# NN. 제목
## 목적
## 비교 모델과 원리      (모델마다 3~5줄)
## 원 논문               (링크)
## 커널                  (3dv 또는 3dv-point)
## 예상 VRAM·시간 (추정)  ("추정치"라고 명시, 프로파일별)
## 이 노트북이 답하려는 질문
```

## 3. 셀 순서

1. 설정 로드: `FORCE_RERUN`, `NOTEBOOK`, config, 시드, 버전 로깅
2. 데이터 로드
3. 모델별 실행 (모델 하나씩: 로드 → 추론 → 결과 저장 → 메모리 해제)
4. 평가
5. 결과 저장 (`summary.json`)
6. 시각화
7. 해석 (마크다운, `> (실행 후 작성)` 자리 표시)

## 4. 표준 설정 셀

```python
# %%
FORCE_RERUN = False            # True: ignore cached results
NOTEBOOK = "02_monocular_depth"

import json, time
from pathlib import Path
import numpy as np
from bench3d.config import load_config
from bench3d.runtime import set_seed, env_report, Measure, free_cuda
from bench3d.io import result_dir, save_summary, is_cached

cfg = load_config("outdoor")                 # profile: config.yaml or $THREEDCHECK_PROFILE
P = cfg.p["depth"]                           # this notebook's profile block
FORCE_RERUN = FORCE_RERUN or cfg.get("force_rerun", False)
set_seed(cfg["seed"])
ENV = env_report(["torch", "numpy", "nuscenes-devkit"])
print(json.dumps(ENV, indent=1)); print("profile:", cfg["profile"], P)
```

- 경로는 `cfg.path(...)`, 고정 커밋 저장소는 `cfg.third_party("mast3r")`로만 얻습니다. 절대 경로를 하드코딩하지 않습니다.
- 데이터: `NuScenes(version=cfg["dataset"]["version"], dataroot=str(cfg.path(cfg["dataset"]["data_root"])), verbose=False)`

## 5. 평가 표본 (모든 노트북이 같은 프레임을 씀)

- `01_data_exploration`이 `results/outdoor/01_data_exploration/eval_samples.json`을 저장합니다.

```json
{
  "version": "v1.0-mini",
  "eval4": ["scene-0103", "..."],
  "probe6": ["scene-0061", "..."],
  "keyframes": {"scene-0103": ["<sample_token>", "..."]},
  "scene_stats": {"scene-0103": {"n_keyframes": 40, "ego_path_m": 0.0, "moving_ann_ratio": 0.0}},
  "recon_scene": "scene-xxxx"
}
```

- 표본 선택은 반드시 `bench3d.sampling`을 씁니다. 결정적이라 노트북끼리 같은 프레임을 고릅니다.
  - `subsample(keyframes, P["sample_stride"])`
  - `select_windows(keyframes, P["n_frames"], P["windows_per_scene"])`
  - `middle(keyframes)`
- 평가는 `eval4` 장면만 씁니다. `probe6`은 탐침이나 분류기 학습에만 씁니다.

## 6. 결과 저장

- 위치: `result_dir(cfg, NOTEBOOK, model_id)` → `results/outdoor/<NOTEBOOK>/<model_id>/`
- 요약: `save_summary(dir, part="outdoor", notebook=NOTEBOOK, profile=cfg["profile"], model=..., variant=..., metrics={평평한 dict}, runtime_s=..., peak_vram_mb=..., checkpoint=..., packages=ENV["packages"], seed=cfg["seed"], extra={...})`
  - `metrics`에는 숫자만 넣습니다. 구간별 표나 곡선은 `extra`나 별도 npz/json에 저장합니다.
  - 한 모델에 변형(variant)이 여럿이면 `filename="summary_<variant>.json"`으로 나눠 저장합니다.
- 큰 산출물(깊이맵, 점군)은 같은 폴더에 npz/ply/bin으로 저장합니다. `results/`는 summary*.json만 커밋됩니다.
- 캐싱: `if is_cached(out_file, FORCE_RERUN): 불러오기 else: 계산 후 저장`
- 실행 시간과 피크 VRAM은 `with Measure() as m:`로 측정하고(`m.seconds`, `m.peak_vram_mb`), 모델 사이에는 `with free_cuda():`로 메모리를 비웁니다.

## 7. 정확성

- 체크포인트, 설정 경로, API 호출은 config.yaml 값이나 고정 커밋 소스에서 확인한 것만 씁니다.
- 확인하지 못한 부분은 코드에 `# TODO(verify): ...`로 남깁니다. 그대로 실행하면 틀린 결과가 나올 수 있는 부분은 `assert`나 명시적 경고로 막습니다.
- 메트릭 정의(스케일 정렬, 평가 영역, 필터)는 평가 셀 바로 위 마크다운에 씁니다.
- 마크다운은 한국어, 코드 주석은 간결한 영어로 씁니다. 코드 스타일은 ruff(line-length 120) 기준입니다.

## 8. 환경 경계

- 노트북은 자기 커널 env의 패키지만 import합니다. 다른 env의 결과는 `results/` 파일로만 읽습니다.
- 다른 env 도구를 꼭 써야 하면 `subprocess`로 그 env의 python을 호출합니다(예: 00에서 두 env 스모크 테스트).
