# 3dcheck — 카메라만으로 3D를 얼마나 알 수 있는가

같은 장면을 카메라로 본 3D와 LiDAR·레이저로 본 3D를 비교하는 벤치마크입니다.
실외는 nuScenes-mini, 실내는 ScanNet++를 쓰고, NeRF 이후의 주요 모델을 같은 조건에서 정량 비교하고 시각화합니다.

> 진행 상태: 실외 1단계(환경 구축) 진행 중. 노트북은 아직 없습니다.
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

## 실외 노트북 계획 (평가셋: 공식 val에 속한 mini 장면 4개)

| 번호 | 주제 | 모델 |
|---|---|---|
| 00 | 환경 점검, VRAM 실측 | — |
| 01 | 데이터 탐색, 평가 샘플 고정 | — |
| 02 | 단안 깊이 | Depth Anything V2 (metric outdoor), Depth Anything 3 (METRIC / NESTED / MONO) |
| 03 | 다시점 기하 | DUSt3R, MASt3R, VGGT, Depth Anything 3 |
| 04 | pseudo-LiDAR | 02/03의 최적 깊이 |
| 05 | 3D 탐지 | PointPillars, CenterPoint, BEVFusion, FCOS3D (카메라 전용) |
| 06 | 포인트 이해 | PointNet(직접 구현), Concerto 선형탐침, PTv3, Sonata(PCA) |
| 07 | 장면 표현 | Nerfacto, Instant-NGP, Splatfacto (07b 선택: OmniRe/StreetGS) |
| 08 | 엣지 배포 | PointPillars ONNX → TensorRT, voxelization CPU/GPU 비교 |
| 99 | 종합 비교 | — |

## 라이선스 주의

nuScenes 데이터, DUSt3R/MASt3R(CC BY-NC-SA), VGGT-1B, Depth Anything 3 GIANT/LARGE/NESTED, Sonata/Concerto 가중치는
비상업 라이선스입니다. 이 저장소의 결과물은 비상업 연구·포트폴리오 용도로만 사용합니다.
