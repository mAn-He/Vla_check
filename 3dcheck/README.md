# 3dcheck — 카메라만으로 3D를 얼마나 알 수 있는가

같은 장면을 카메라로 본 3D와 LiDAR·레이저로 본 3D를 비교하는 벤치마크입니다.
실외는 nuScenes-mini, 실내는 ScanNet++를 쓰고, NeRF 이후의 주요 모델을 같은 조건에서 정량 비교하고 시각화합니다.

> 진행 상태: 실외 2단계(노트북 작성) 완료, **아직 실행하지 않음**. 1단계 환경은 컨테이너에서 처음부터 빌드해 스모크 테스트를 통과했습니다([envs/README.md](envs/README.md)).
> 노트북 작성 규약은 [docs/notebook_conventions.md](docs/notebook_conventions.md)에 있습니다.
> 조사 근거와 결정 사항은 [docs/step0_research.md](docs/step0_research.md)에 있습니다.

## 구성

| 파트 | 데이터 | 비교 축 |
|---|---|---|
| 실외 | nuScenes v1.0-mini + lidarseg-mini | 카메라 3D vs LiDAR 3D (깊이, 기하, 탐지, 분할, 장면 표현, 엣지 배포) |
| 실내 1차 | ScanNet++ (`nvs_sem_val` 중 2~3장면) | 레이저(정답) vs iPhone LiDAR vs 카메라 (복원, 깊이, 분할) |
| 실내 2차 (예정) | 직접 촬영한 집 내부 영상 | 정답 없는 환경에서의 정성 비교 + 홀드아웃 뷰 NVS + 실측 치수 검증 |

## 가상환경

conda는 Python·CUDA 툴킷·컴파일러만 맡고, 파이썬 패키지는 env 안에서 `uv pip`으로 설치합니다.
자세한 버전과 설치 순서는 [envs/README.md](envs/README.md)를 보세요.

| 커널 | torch / CUDA | 노트북 |
|---|---|---|
| `3dv` | 2.1.2 / 12.1 | 실외 00–05, 07, 08, 99 |
| `3dv-point` | 2.5.0 / 12.4 | 실외 06 |

## 실외 실행 순서

평가 장면은 공식 val 분할에 속한 mini 장면 4개(`eval4`)입니다. mini_train 8개 중 6개는 공식 체크포인트의 학습 분할이라 평가에서 뺐습니다.
선형탐침과 PointNet 학습에는 나머지 6개(`probe6`)를 씁니다.

1. 환경: `bash envs/setup_3dv.sh`(BEVFusion을 쓰려면 `BUILD_EXT=1`), `bash envs/setup_3dv-point.sh`(권장 `FLASH_ATTN=1`, PTv3 공식 가중치 시도에는 `PTV3_V151=1`)
2. 데이터: nuScenes 로그인 후 `v1.0-mini.tgz`, `nuScenes-lidarseg-mini-v1.0.tar.bz2`를 `data/downloads/`에 받고
   `bash outdoor/scripts/place_nuscenes_mini.sh` → `python outdoor/scripts/verify_nuscenes_mini.py`
3. 노트북 `00_env_check` → `01_data_exploration`(평가 표본과 `splits.json` 생성)
4. 탐지 준비(3dv): `bash outdoor/scripts/prepare_mmdet3d_infos.sh`
5. 포인트 준비(3dv-point): `python outdoor/scripts/pointcept_nuscenes_mini.py --eval-samples results/outdoor/01_data_exploration/eval_samples.json`
6. 노트북 02 → 03 → 04 → 05 → 06 → 07 → 08 → 99

| 노트북 | 커널 | 먼저 필요한 것 | 모델 | 주 지표 |
|---|---|---|---|---|
| 00 환경 점검 | 3dv | 1단계 | — | 스모크 테스트, 권장 프로파일 |
| 01 데이터 탐색 | 3dv | 데이터 | — | 평가 표본 고정, 장면 통계 |
| 02 단안 깊이 | 3dv | 01 | Depth Anything V2 metric(VKITTI), DA3 METRIC / NESTED / MONO | 거리 구간별 AbsRel·RMSE·δ1, 오차-거리 기울기 |
| 03 다시점 기하 | 3dv | 01 | DUSt3R, MASt3R, VGGT, DA3 | AUC@30, ATE, RRA/RTA, Chamfer, F-score |
| 04 가짜 LiDAR | 3dv | 01, (02) | 02의 최적 metric 모델 | 거리 링별 점 밀도 |
| 05 3D 탐지 | 3dv | 01, 04, 4번 준비 | PointPillars, CenterPoint, BEVFusion, FCOS3D | FOV 필터 mAP·NDS, 거리 구간 AP |
| 06 포인트 이해 | 3dv-point | 01, 04, 5번 준비 | PointNet, Concerto 선형탐침, PTv3, Sonata | mIoU, 클래스별 IoU |
| 07 장면 표현 | 3dv | 01 | Nerfacto, Instant-NGP, Splatfacto | PSNR·SSIM·LPIPS, 학습 시간, 렌더 FPS |
| 08 엣지 배포 | 3dv | 4번 준비 | PointPillars ONNX / TensorRT | 단계별 지연시간 |
| 99 종합 | 3dv | 02~08 결과 | — | 비교 표와 그래프 |

노트북 소스는 `outdoor/notebooks/src/*.py`(jupytext)이고, `.ipynb`는 `python envs/build_notebooks.py`로 만듭니다.

## 결과 표 (실행 후 99 노트북의 표로 채움)

| 축 | 지표 | 카메라 | LiDAR | 비고 |
|---|---|---|---|---|
| 깊이 0–20m | AbsRel | | (정답) | |
| 깊이 50m+ | AbsRel | | (정답) | |
| 다시점 자세 | AUC@30 (front) | | — | 피드포워드 vs 최적화 |
| 탐지 (FOV) | mAP / NDS | FCOS3D: / · 가짜 LiDAR: / | PointPillars: / · CenterPoint: / | BEVFusion: / |
| 분할 | mIoU | 가짜 LiDAR(대리): | Concerto 탐침: | |
| 장면 표현 | PSNR | | — | 평가 뷰 약 4장 |
| 엣지 | 지연시간(ms) | — | | voxelization 비중 |

## 라이선스 주의

nuScenes 데이터, DUSt3R/MASt3R(CC BY-NC-SA), VGGT-1B, Depth Anything 3 GIANT/LARGE/NESTED, Sonata/Concerto 가중치는
비상업 라이선스입니다. 이 저장소의 결과물은 비상업 연구·포트폴리오 용도로만 사용합니다.
