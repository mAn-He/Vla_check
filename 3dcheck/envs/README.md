# 가상환경

conda(mamba/micromamba)는 Python, CUDA 툴킷(nvcc와 헤더), gcc, ninja만 설치합니다.
파이썬 패키지는 env 안에서 `uv pip install`로 설치합니다. 같은 패키지를 두 도구로 중복 설치하지 않습니다.
환경을 둘로 나눈 근거는 [../docs/step0_research.md](../docs/step0_research.md) 5절에 있습니다.

## 환경 요약

| env / 커널 | Python | PyTorch | CUDA (torch 휠 / conda nvcc) | gcc | 주요 패키지 | 노트북 |
|---|---|---|---|---|---|---|
| `3dv` | 3.10 | 2.1.2 | 12.1 / 12.1 | 12 | xformers 0.0.23.post1, mmcv 2.1.0, mmdet 3.3.0, mmdet3d 1.4.0, nerfstudio 1.1.5, gsplat 1.4.0, nuscenes-devkit 1.2.0, pycolmap, DUSt3R/MASt3R, VGGT, Depth Anything 3, Depth Anything V2 | 실외 00–05, 07, 08, 99 |
| `3dv-point` | 3.10 | 2.5.0 | 12.4 / 12.4 | 13.2 | spconv-cu124 2.3.8, torch-scatter, torch-cluster, Pointcept, Sonata, Concerto, nuscenes-devkit 1.2.0 | 실외 06 |

소스에서 설치하는 저장소의 고정 커밋은 [third_party.lock](third_party.lock)에 있습니다.
DUSt3R와 MASt3R는 MASt3R 저장소의 `dust3r` 서브모듈 하나를 함께 씁니다. 두 저장소를 따로 clone하면 같은 이름의 `dust3r` 패키지가 두 개 생겨 충돌합니다.

## 설치 (로컬 RTX 3060 / RunPod)

```bash
cd 3dcheck
bash envs/setup_3dv.sh          # 3dv 커널
bash envs/setup_3dv-point.sh    # 3dv-point 커널
```

각 스크립트는 다음 순서로 진행하고, 마지막에 import 스모크 테스트(`envs/smoke_test.py`)를 실행합니다.

1. conda 층 생성 (`*.environment.yml`)
2. uv 층 1: torch 스택 먼저 설치
3. uv 층 2: `*.requirements.txt`. `*.constraints.txt`가 torch, numpy 등을 고정합니다.
4. 고정 커밋 저장소 clone 후 설치, `common/`(bench3d) editable 설치
5. (선택) CUDA 확장 빌드
6. Jupyter 커널 등록

| 옵션 (환경변수) | 기본값 | 설명 |
|---|---|---|
| `CONDA_FRONTEND` | mamba → micromamba → conda 순 자동 | 사용할 conda 프런트엔드 |
| `ENV_PREFIX` | (없음, 이름으로 생성) | 경로에 env 생성 |
| `THIRD_PARTY` | `3dcheck/third_party` | 저장소 clone 위치 |
| `MMCV_INDEX` | OpenMMLab cu121/torch2.1 휠 인덱스 | `""`이면 mmcv를 소스 빌드 |
| `BUILD_EXT=1` | 0 | BEVFusion 연산자, DUSt3R curope, tiny-cuda-nn 빌드 |
| `PYG_INDEX` | PyG torch-2.5.0+cu124 휠 인덱스 | `""`이면 torch-scatter/cluster 소스 빌드 |
| `FLASH_ATTN=1` | 0 | flash-attn 설치 (Ampere 이상, RTX 3060 가능) |
| `POINTOPS=1` | 0 | Pointcept pointops 빌드 (PTv3에는 불필요) |
| `TORCH_CUDA_ARCH_LIST` | 8.6 | 확장 빌드 대상 GPU 아키텍처 (RTX 3060 = 8.6, A100 = 8.0, L4/4090 = 8.9) |

## Colab에서 쓰기

Colab은 노트북 커널로 conda env를 고를 수 없습니다. 그래서 env를 만든 뒤 그 env의 python을 subprocess로 호출해야 합니다.
노트북 쪽 실행 헬퍼는 2단계에서 추가합니다. RunPod은 JupyterLab에서 등록된 커널을 바로 고를 수 있어서 기본 원격 실행 환경으로 권장합니다.

## 알려진 문제와 주의

| 항목 | 내용 |
|---|---|
| torch 버전 상한 | mmdet3d 1.4.0이 `mmcv<2.2.0`을 요구하므로 `3dv`는 torch 2.1에 묶입니다. torch를 올리면 mmdet3d import 단계에서 assert로 실패합니다. |
| VGGT | 공식 requirements.txt는 torch==2.3.1입니다. `3dv`에서는 torch 2.1.2로 쓰며 실행 검증은 TODO(verify)입니다. |
| CUDA 12.1 선택 | nerfstudio 문서는 cu118을 권장합니다. PyPI의 torch 2.1.2와 xformers 0.0.23.post1 기본 휠이 cu121이라서 cu121로 통일했습니다. |
| conda `cuda-toolkit` 미사용 | 메타패키지가 Nsight GUI 의존성(Qt, gstreamer)까지 설치해 약 4GB가 늘어납니다. nvcc와 dev 헤더만 설치합니다. |
| conda-forge CUDA 헤더 경로 | 헤더가 `$PREFIX/targets/x86_64-linux/include`에 있습니다. 확장 빌드 시 `_lib.sh`의 `cuda_env`가 `CPATH`를 설정합니다. |
| Sonata/Concerto 설치 | upstream `setup.py`가 `pkg_resources`를 import하므로 `setuptools<81`을 깔고 build isolation 없이 설치합니다. |
| Pointcept 두 버전 | 기본은 main 커밋(.pth로 경로 등록)이고, PTv3 nuScenes 가중치 시도용 v1.5.1은 해당 셀에서 `PYTHONPATH`로 우선시킵니다. |
| DriveStudio (07b) | 공식 요구가 torch 2.0/cu117 + torch 1.8용 SegFormer env라서 이 두 env에 넣지 않았습니다. 필요 시 RunPod에서 별도 구축합니다. |

## 스모크 테스트 결과

(1단계 실측 후 기록)
