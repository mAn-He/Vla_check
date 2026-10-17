# 0단계 조사·결정 기록 (실외, 2026-10-17)

이 문서는 실외 파트 0단계에서 공식 저장소를 읽고 확인한 사실과, 그에 따라 내린 결정을 남긴다.
"확인됨"은 아래 출처 파일을 직접 읽은 것, "미확인"은 읽지 못했거나 추론한 것이다.
조사 환경에서 huggingface.co, download.pytorch.org, download.openmmlab.com, data.pyg.org, nuscenes.org, arxiv.org는
네트워크 정책으로 막혀 있었다. 그래서 HF 모델 카드와 논문 본문은 읽지 못했고, 해당 내용은 GitHub README 기준이다.

## 1. 모델별 사실 (출처는 각 저장소 `main` 브랜치의 raw 파일)

| 모델 | 확인한 사실 | 출처 | 상태 |
|---|---|---|---|
| Depth Anything V2 metric | 실외 체크포인트 `depth-anything/Depth-Anything-V2-Metric-VKITTI-{Small,Base,Large}`, `max_depth=80`(실내 Hypersim은 20). `run.py`와 생성자 기본값이 **20**이므로 실외에서 반드시 80을 넘긴다. 입력 518(짧은 변), BGR 입력. torch 버전 고정 없음 | DepthAnything/Depth-Anything-V2 `metric_depth/README.md`, `metric_depth/depth_anything_v2/dpt.py` | 확인됨 |
| Depth Anything 3 | metric 출력: `DA3METRIC-LARGE`(단안, `metric = focal·out/300` 수동 변환), `DA3NESTED-GIANT-LARGE-1.1`(이미 미터). `-1.1`이 주행 장면 학습 버그를 고친 버전. `xformers` 필수, `numpy<2`. 출력 깊이는 process_res(기본 504) 해상도 | ByteDance-Seed/Depth-Anything-3 `README.md`, `pyproject.toml`, `src/depth_anything_3/api.py`, `utils/alignment.py` | 확인됨 (focal을 리사이즈 해상도 기준으로 써야 하는지는 미확인) |
| DUSt3R | 체크포인트 `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`. `get_im_poses()`는 cam→world. 학습 명령 예시에 `ScanNetpp(split='train')` 포함. README 데이터셋 목록에 Waymo도 있으나 공개 학습 명령 예시에는 없음(배포 체크포인트의 실제 구성은 미확인). nuScenes는 목록에 없음 | naver/dust3r `README.md`, `dust3r/cloud_opt/` | 확인됨 (배포 체크포인트 학습 구성은 미확인) |
| MASt3R | `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`, `sparse_global_alignment`. metric 회귀로 학습. 학습에 ScanNet++ train, VirtualKitti 포함 | naver/mast3r `README.md`, `mast3r/cloud_opt/sparse_ga.py`, `mast3r/losses.py` | 확인됨 (nuScenes에서 실제 metric 정확도는 미확인) |
| VGGT | `facebook/VGGT-1B`(비상업). requirements.txt는 torch==2.3.1 고정, pyproject는 torch 고정 없음. 외부 파라미터는 world→cam(OpenCV), 첫 프레임 기준·스케일 정규화(비metric). bf16은 Ampere 이상 | facebookresearch/vggt `README.md`, `requirements.txt`, `pyproject.toml`, `training/train_utils/normalization.py` | 확인됨 |
| VGGT 메모리 | 현재 README에서 표가 삭제됨. 과거 표(H100, aggregator 단독): 8장 3.2GB, 20장 5.6GB, 50장 11.4GB | vggt 커밋 44b3afb의 README | 과거 값 확인됨 / 현재 값 미확인 |
| PointPillars | `configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py`, mAP 39.71 / NDS 53.15, sweep 10 | open-mmlab/mmdetection3d `configs/pointpillars/metafile.yml` | 확인됨 |
| CenterPoint | `centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py`, mAP 56.54 / NDS 65.17, sweep 9 | `configs/centerpoint/metafile.yml` | 확인됨 |
| BEVFusion | `projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`, NDS 71.4 / mAP 68.6, 커스텀 연산자 빌드 필요 | `projects/BEVFusion/README.md`, `setup.py` | 확인됨 |
| FCOS3D (추가) | 카메라 단안 3D 탐지, nuScenes 학습. `fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune`, mAP 32.1 / NDS 39.3 | `configs/fcos3d/metafile.yml` | 확인됨 |
| PGD (선택) | 카메라 단안 3D 탐지, nuScenes finetune mAP 34.6 / NDS 41.1 | `configs/pgd/metafile.yml` | 확인됨 |
| mmdet3d 버전 | 1.4.0(2024-01-08 이후 릴리스 없음). `mmcv>=2.0.0rc4,<2.2.0`, `mmdet>=3.0.0rc5,<3.4.0`. mmcv 2.1.0 사전빌드 휠은 torch 2.1 + cu118/cu121까지 | `mmdet3d/__init__.py`, mmcv `docs/en/_static/version.json` | 확인됨 |
| mmdet3d mini 지원 | `tools/create_data.py --version v1.0-mini` 분기 있음. 평가 시 `NuScenesMetric`이 `mini_val`을 자동 사용 | `tools/create_data.py`, `mmdet3d/evaluation/metrics/nuscenes_metric.py` | 확인됨 |
| mmdeploy | 1.3.1, `mmdet3d==1.3.0`에 고정. voxelization은 ONNX로 변환되지 않음. TensorRT ≥ 8.6 요구 | open-mmlab/mmdeploy `docs/en/04-supported-codebases/mmdet3d.md`, `requirements/codebases.txt` | 확인됨 |
| PTv3 nuScenes | README에 nuScenes 가중치 링크가 있으나 "Released model weights are temporarily invalid", v1.5.1용 | Pointcept/PointTransformerV3 `README.md` | 확인됨 (HF 폴더 내용은 미확인) |
| Sonata | 실내 전용 사전학습(ScanNet, ScanNet++, S3DIS, ARKit, HM3D, Structured3D). 가중치 CC-BY-NC | facebookresearch/sonata `README.md`, Pointcept `configs/sonata` | 확인됨 |
| Concerto | `concerto_large_outdoor` 백본 있음, nuScenes 선형탐침 config `configs/concerto/semseg-ptv3-large-v1m1-nuscenes-5a-lin.py` 있음. **nuScenes 헤드 가중치는 없음**. 가중치 CC-BY-NC | Pointcept/Concerto `README.md`, Pointcept `configs/concerto` | 확인됨 |
| Pointcept nuScenes 전처리 | trainval/test만 허용(장면 수 assert). **mini는 비공식 패치 필요** | Pointcept `pointcept/datasets/preprocessing/nuscenes/preprocess_nuscenes_info.py` | 확인됨 |
| nerfstudio | 1.1.5(2024-11). torch 2.1.2/cu118 권장. `instant-ngp` 내장(tcnn 없으면 느린 torch 대체). `gsplat==1.4.0` 고정. nuScenes 파서 `nuscenes-data`: **단일 카메라, 키프레임(2Hz)**, 동적 물체 마스크는 `process_nuscenes_masks.py`로 별도 생성 | nerfstudio-project/nerfstudio `pyproject.toml`, `docs/quickstart/installation.md`, `nerfstudio/data/dataparsers/nuscenes_dataparser.py` | 확인됨 |
| DriveStudio (선택) | nuScenes 6캠·v1.0-mini 공식 지원(OmniRe, StreetGS, DeformableGS, PVG). torch 2.0.0/cu117, gsplat 1.3.0, pytorch3d, nvdiffrast, 하늘 마스크용 SegFormer는 **torch 1.8 별도 env** | ziyc/drivestudio `README.md`, `requirements.txt`, `docs/NuScenes.md` | 확인됨 (VRAM 미확인) |
| nuscenes-devkit | 1.2.0, Python ≥3.9, `numpy<2`. `LidarPointCloud.from_file`은 5열 중 4열(x,y,z,intensity)만 유지 | PyPI, nutonomy/nuscenes-devkit `setup/requirements/requirements_base.txt`, `python-sdk/nuscenes/utils/data_classes.py` | 확인됨 (5번째 열 = ring index는 미확인) |

## 2. 평가 분할: 데이터 누수 점검 (직접 계산)

`nuscenes/utils/splits.py`를 파싱해 mini 장면이 공식 trainval의 어느 쪽에 속하는지 확인했다.

| mini 분할 | 장면 | 공식 train | 공식 val |
|---|---|---|---|
| mini_train | 0061, 0553, 0655, 0757, 0796, 1077, 1094, 1100 | 6 | **2 (0553, 0796)** |
| mini_val | 0103, 0916 | 0 | 2 |

공식 탐지·분할 체크포인트는 공식 train으로 학습됐다. 그래서 다음과 같이 정한다.
- **평가셋 `eval4`** = scene-0103, scene-0916, scene-0553, scene-0796 (모두 공식 val)
- **탐침 학습셋 `probe6`** = 나머지 6개 (Concerto 선형탐침, PointNet 학습에만 사용)
- 4개 장면은 통계적으로 작다. 결과는 공식 수치와 직접 비교할 수 없다고 명시한다.
- mmdet3d의 기본 mini 평가(`mini_val`, 2장면)도 함께 보고해 설정 차이를 투명하게 한다.

## 3. 모델–데이터셋 적합성 점검 (실외)

| 노트북 | 원안 모델 | 적합성 판단 | 조치 |
|---|---|---|---|
| 02 단안 깊이 | DA V2 metric outdoor, DA3 | 적합. DA V2는 VKITTI(주행 합성) 학습 체크포인트, DA3METRIC/NESTED는 범용 metric | 유지. DA3MONO(상대깊이)를 median scaling 대조군으로 추가 |
| 03 다시점 기하 | DUSt3R, MASt3R, VGGT, DA3 | 범용 모델. 주행의 전진 궤적(작은 시차)과 서라운드 6캠(겹침 적음)은 이 모델들의 약점이므로 오히려 비교 가치가 있음 | 유지. 실내와 **같은 가중치**로 돌려 도메인 간 비교의 기준축으로 사용 |
| 05 3D 탐지 | PointPillars, CenterPoint, BEVFusion, pseudo-LiDAR | **누락 있음.** 원안의 "카메라 3D"는 LiDAR로 학습한 탐지기에 pseudo-LiDAR를 넣는 방식뿐이라, 카메라 쪽이 구조적으로 불리함 | **FCOS3D(카메라 전용, nuScenes 학습, mmdet3d 공식) 추가.** "카메라 고유 탐지기 vs pseudo-LiDAR vs LiDAR vs 융합" 4축으로 비교 |
| 06 포인트 이해 | PTv3, Sonata, PointNet | Sonata는 실내 전용이라 실외 분할에 부적합 | Concerto 실외 백본 + 선형탐침을 주 실험으로. PTv3는 v1.5.1 가중치 로드 시도. Sonata는 특징 PCA만 해서 도메인 불일치를 시각적으로 보여줌 |
| 07 장면 표현 | Nerfacto, Instant-NGP, 3DGS | 범용 방법. "주행 데이터에서 어디서 무너지는가"라는 원안 목적에는 적합 | 유지. 주행 전용 방법(OmniRe/StreetGS via DriveStudio)은 별도 env가 2개 필요해 **선택 과제(07b)**로 미룸 |
| 08 엣지 | PointPillars | 적합(엣지용 고전 LiDAR 탐지기) | 4절의 방식으로 구현 |

도메인 간 비교(99번)는 실내·실외에서 **같은 가중치**를 쓰는 모델(DA3METRIC, DA3NESTED, VGGT, MASt3R)을 기준축으로 한다.
DA V2는 실내(Hypersim)와 실외(VKITTI) 체크포인트가 달라서 교차 비교에서 따로 표기한다.

### 실내 쪽에서 미리 확인된 점 (실내 0단계에서 이어서 처리)
- DUSt3R와 MASt3R는 `ScanNetpp(split='train')`으로 학습했다. 평가 장면이 그 train에 들어 있지 않은지 확인해야 한다.
  장면 목록은 `scannetpp_pairs.zip` 안의 `all_metadata.npz`에 있고, 이 환경에서는 다운로드가 막혀(403) 확인하지 못했다. → TODO(verify)
- DA3는 ScanNet++ 20개 장면을 자체 벤치마크 평가에 쓴다(`src/depth_anything_3/utils/constants.py`의 `SCANNETPP_SCENES`).
  이 장면들과 공식 `nvs_sem_val`의 교집합을 평가 장면 후보로 삼는다.
- VGGT의 ScanNet++ 학습 사용 여부는 저장소에서 확인하지 못했다. → TODO(verify)

## 4. PointPillars 엣지 배포 설계 (08)

- **Voxelization이란**: LiDAR 점군은 개수가 매번 다르고 순서가 없는 점의 집합이다. 신경망에 넣으려면 고정 크기 텐서가 필요하다.
  PointPillars는 바닥을 0.25m×0.25m 격자로 나눠 각 칸(기둥, pillar)에 들어간 점을 모은다. 칸당 최대 64점,
  최대 3만~4만 칸으로 자르고 패딩해 `voxels[P,64,C]`, `num_points[P]`, `coors[P,3]`을 만든다.
  그 뒤 PointNet(PFN)이 기둥마다 특징을 내고, 이를 BEV 격자에 흩뿌려(scatter) 2D 이미지처럼 만들고, 2D CNN과 탐지 헤드를 거친다.
- **ONNX/TensorRT에서 막히는 이유**: 출력 크기가 입력 데이터에 따라 달라지고(점이 있는 칸만 생김), 칸 배정이 해시·scatter 연산이라
  표준 ONNX 연산으로 표현되지 않는다. mmdeploy도 이 부분을 변환하지 않는다(1절 출처).
- **구현 방침 (권장안 채택)**:
  1. 전처리(voxelization)와 신경망을 분리한다. 신경망 부분(PFN → scatter → backbone → head)만 고정 크기(최대 기둥 수로 패딩)로
     `torch.onnx.export` 한 뒤 TensorRT FP32/FP16 엔진을 만든다.
  2. voxelization은 두 가지로 구현해 비교한다: (a) numpy CPU, (b) torch GPU 벡터화(좌표 → 선형 인덱스 → `unique`).
  3. 단계별 지연시간(voxelize / 네트워크 / 디코드·NMS)을 따로 잰다. Jetson Orin Nano처럼 CPU가 약한 기기에서는
     CPU voxelization이 병목이 될 수 있으므로 이것을 정량으로 보여주는 것이 이 노트북의 핵심이다.
  4. 참고 기준: NVIDIA 공식 CUDA-PointPillars는 voxelization을 CUDA 커널로 처리하며, Orin에서 voxelization 0.18ms,
     디코드·NMS 1.79ms(FP16)를 보고한다. 단, **KITTI 모델(OpenPCDet) 기준**이라 nuScenes 가중치에 그대로 쓸 수 없다.
     (NVIDIA-AI-IOT/CUDA-PointPillars `README.md`, 확인됨)
  5. 추가 최적화 후보: FP16 → INT8 PTQ(nuScenes 샘플로 보정), 최대 기둥 수 축소, sweep 수 축소(10 → 1~3, 정확도 손실과 함께 보고),
     CUDA Graph, pinned memory.
- mmdeploy는 mmdet3d 1.3.0 고정·정체 상태라 쓰지 않는다. 별도 env도 만들지 않는다.

## 5. 환경 결정과 그 근거

- **의존성 해석 검증**: `uv pip compile`(Python 3.10, manylinux)로 torch 2.1.2 + nerfstudio 1.1.5 + gsplat 1.4.0 +
  nuscenes-devkit 1.2.0 + mmdet 3.3.0 + mmdet3d 1.4.0 + xformers + VGGT + DA3 + spconv-cu118 조합이 **해석됨**을 확인했다.
  (mmcv는 OpenMMLab 휠 서버가 막혀 해석에서 제외)
- **소스 점검**: VGGT, DA3, Sonata, Concerto, PTv3 소스에서 torch 2.1에 없는 API를 grep했고, 발견되지 않았다.
- **분리 기준**: mmdet3d는 mmcv<2.2 때문에 torch ≤2.1에 묶인다. Pointcept는 공식 env가 torch 2.5/cu124이고 spconv·pointops·flash-attn이
  모두 컴파일 확장이다. 그래서 `3dv`(torch 2.1.2)와 `3dv-point`(torch 2.5.0) 두 개로 나눈다.
- **CUDA 12.1 선택**: nerfstudio 문서는 cu118을 권장하지만, PyPI의 torch 2.1.2 기본 휠과 xformers 0.0.23.post1 휠이 cu121 빌드다.
  cu121이면 PyPI만으로 재현할 수 있고 mmcv 2.1.0도 cu121 휠이 있다. RTX 3060은 드라이버만 맞으면 둘 다 지원한다.
- **Colab**: conda 커널을 고를 수 없어서, 노트북이 env의 python을 subprocess로 호출하는 방식이 필요하다. 기본 원격 실행은 RunPod로 한다.

## 6. 남은 TODO(verify)

1. VGGT를 torch 2.1.2에서 실행했을 때의 정상 동작 (공식 고정값은 2.3.1)
2. ~~DA3 metric 변환의 focal 해상도~~ → 2단계에서 소스로 결정: 처리 해상도(504×280) 기준 focal 사용(`model/da3.py`, `utils/alignment.py`, nested 설정). 02 노트북에 근거와 런타임 점검 포함
3. DA3를 torch 2.1.2 + xformers 0.0.23.post1에서 실행했을 때의 정상 동작
4. PTv3 nuScenes 가중치(HF `Pointcept/PointTransformerV3/nuscenes-semseg-pt-v3m1-0-base`)가 Pointcept v1.5.1에서 로드되는지. v1.5.1 import 자체는 `ocnn` + `pointgroup_ops`(`PTV3_V151=1`)로 2단계에서 해결
5. ~~BEVFusion 추론 시 Swin 가중치~~ → 불필요: `Runner.test()`는 `init_weights`를 호출하지 않음(2단계 확인)
6. nuScenes LiDAR `.bin` 5번째 열의 의미(ring index로 알려져 있으나 devkit 코드에 명시 없음)
7. DUSt3R/MASt3R의 ScanNet++ train 장면 목록, VGGT의 ScanNet++ 학습 사용 여부 (실내)
8. 각 모델의 RTX 3060 실측 VRAM (00 노트북에서 측정)
