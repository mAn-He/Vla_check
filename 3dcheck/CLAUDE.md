# 3dcheck — 공통 규칙

## 목적
"카메라만으로 3D를 얼마나 복원·이해할 수 있는가"를 실외(nuScenes-mini)와 실내(ScanNet++)에서
NeRF 이후 주요 모델들로 비교하는 포트폴리오용 벤치마크. 모든 실험은 Jupyter Notebook(.ipynb)으로 실행한다.
데이터와 대부분의 가중치가 비상업 라이선스이므로 결과물은 비상업 용도로만 쓴다.

## 실행 정책 (가장 중요)
- 노트북 셀, 학습, 추론, 평가 스크립트를 **절대 실행하지 않는다.** 실행은 사용자가 직접 한다.
- 허용되는 실행: 환경 점검 명령(nvidia-smi, nvcc --version, conda info 등), 가상환경 생성/패키지 설치,
  `python -c "import ..."` 수준의 import 스모크 테스트, nbformat 스키마 검증, 린트.
- 패키지 설치 등 시간이 오래 걸리거나 디스크를 크게 쓰는 명령은 실행 전에 무엇을 설치하는지 보여주고 확인받는다.
- 데이터셋 다운로드는 실행하지 않는다. 다운로드 스크립트만 작성한다.

## 정확성 규칙
- 패키지 버전, 체크포인트 이름/URL, 설정 파일 경로는 **반드시 공식 저장소/문서에서 확인한 값만** 쓴다.
  확인하지 못한 항목은 코드에 `# TODO(verify): ...`로 표시하고 보고서에 목록으로 남긴다. 추측으로 채우지 않는다.
- 공식 사전학습 체크포인트가 해당 데이터셋용으로 존재하지 않으면, 대안(다른 데이터셋 가중치 + 라벨 매핑,
  선형 탐침, 정성 시각화만)을 제시하고 사용자 선택을 받는다.
- 메트릭 정의(스케일 정렬 방식, 평가 영역, 필터링)는 노트북 마크다운에 명시한다.
- **학습 데이터 누수 점검**: 사전학습 모델로 평가할 때는 평가 장면이 그 모델의 학습 분할에 없는지 확인한다.
  (예: nuScenes mini_train 8개 중 6개는 공식 train 분할 → 공식 체크포인트 평가에 쓰지 않는다. docs/step0_research.md 참고)

## 저장소 구조
3dcheck/
├── CLAUDE.md
├── README.md                  # 프로젝트 개요, 결과 표 템플릿, 재현 방법
├── docs/                      # 단계별 조사·결정 기록 (step0_research.md 등)
├── envs/                      # 환경별 environment.yml + requirements + setup_*.sh + 커널 등록
├── common/                    # 공통 파이썬 패키지 bench3d (pip install -e 로 각 env에 설치)
│   └── bench3d/
│       ├── metrics/           # depth, pose, nvs, pointcloud, segmentation, detection 메트릭
│       ├── io/                # 결과 저장/로드 (json, npz, ply), 경로 해석
│       ├── viz/               # open3d/plotly/matplotlib 시각화 헬퍼
│       └── align.py           # Umeyama Sim(3), median scaling 등 스케일 정렬
├── outdoor/                   # nuScenes-mini
│   ├── config.yaml
│   ├── notebooks/
│   └── scripts/               # 다운로드 안내, 전처리, 포맷 변환
├── indoor/                    # ScanNet++
│   ├── config.yaml
│   ├── notebooks/
│   └── scripts/
├── third_party/               # .gitignore — setup 스크립트가 고정 커밋으로 clone
├── data/                      # .gitignore (데이터는 커밋 금지)
└── results/{outdoor,indoor}/{notebook_id}/{model}/   # .gitignore, 단 summary.json은 커밋

## 가상환경 설계 원칙
- **conda는 바닥층만, 파이썬 패키지는 uv.** conda(mamba/micromamba)로 Python·nvcc(CUDA 툴킷)·gcc·ninja만 설치하고,
  활성화된 env 안에서 `uv pip install`로 나머지를 설치한다. 같은 패키지를 conda와 uv로 중복 설치하지 않는다.
- 환경 수는 최소화하되, **컴파일 확장(mmcv, spconv, pointops, flash-attn)이 요구하는 torch 버전이 다르면 분리한다.**
  순수 PyTorch 모델은 torch 버전 차이에 비교적 관대하므로 같은 env에 넣는다.
- 현재 환경 분할 (근거: docs/step0_research.md):
  - `3dv`       : Python 3.10, torch 2.1.2 + CUDA 12.1. 데이터·메트릭·시각화, DA V2/DA3, DUSt3R/MASt3R, VGGT,
                  mmdet3d(PointPillars, CenterPoint, BEVFusion, FCOS3D), nerfstudio/gsplat/pycolmap. 노트북 00–05, 07, 08, 99.
  - `3dv-point` : Python 3.10, torch 2.5.0 + CUDA 12.4 (Pointcept 공식 environment.yml 기준). PTv3/Sonata/Concerto. 노트북 06.
  - (선택, 미구축) `3dv-drive` : DriveStudio(OmniRe/StreetGS). 공식 요구가 torch 2.0/cu117 + 별도 SegFormer env라서 필요 시 RunPod에서만.
- 환경 간 데이터 교환은 파일로만 한다 (results/ 아래 json/npz/ply). 노트북이 다른 환경의 패키지를 import하지 않는다.
- 모든 환경에 `common/`을 editable 설치한다.
- `envs/README.md`에 환경별 Python/PyTorch/CUDA 버전, 설치 순서, 알려진 문제를 표로 정리한다.

## 실행 하드웨어
- 로컬: RTX 3060 (Ampere sm86, bf16·FlashAttention2 지원). VRAM은 12GB(데스크톱) 기준 `mid`, 6GB(노트북)면 `low`.
- 원격: RunPod(주, 커스텀 Jupyter 커널 사용 가능) / Colab(보조, conda 커널 선택 불가 → subprocess 실행).

## 노트북 작성 규칙
- 파일명: `NN_주제.ipynb` (00 환경점검, 99 종합비교).
- 첫 셀(마크다운, 한국어): 목적 / 비교 모델과 원리 3~5줄씩 / 원 논문 링크 / 필요한 커널 이름 /
  예상 VRAM·시간(추정치임을 명시) / 이 노트북이 답하려는 질문.
- 셀 순서: 설정 로드 → 데이터 로드 → 모델별 실행 → 평가 → 결과 저장 → 시각화 → 해석(마크다운, 결과 채울 자리).
- 모든 하이퍼파라미터·경로는 config.yaml에서 읽는다. 절대 경로 하드코딩 금지.
- 캐싱: 결과 파일이 있으면 재실행을 건너뛰는 `FORCE_RERUN` 플래그를 둔다.
- 하드웨어 적응: config에 `profile: low|mid|high`를 두고 프레임 수·해상도·반복 횟수를 프로파일별로 정의한다.
- 각 모델 결과는 `results/.../summary.json`에 같은 스키마(model, metric 이름, 값, 실행 시간, 피크 VRAM,
  체크포인트, 패키지 버전, seed)로 저장한다.
- 시드 고정, 버전 로깅 셀 포함.
- 마크다운은 한국어, 코드 주석은 간결한 영어.

## 작업 방식
- 큰 작업은 Plan → 사용자 승인 → 구현 순서로 진행한다.
- 단계마다 git 커밋을 남긴다 (메시지 예: `outdoor: add 03 multi-view geometry notebook`).
- 각 단계가 끝나면 생성한 파일, 남은 TODO(verify), 사용자가 직접 해야 할 일(다운로드, 토큰 등)을 짧게 보고한다.
