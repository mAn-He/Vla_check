# ---
# jupyter:
#   kernelspec:
#     display_name: 3dv (torch 2.1.2, cu121)
#     language: python
#     name: 3dv
# ---

# %% [markdown]
# # 00. 환경 점검
#
# ## 목적
# 실외 노트북(01~08, 99)을 실행하기 전에 GPU, 두 커널(`3dv`, `3dv-point`), 고정 커밋 저장소, 데이터, 결과 폴더를
# 한 번에 점검합니다. 여기서 문제가 없으면 나머지 노트북은 데이터와 체크포인트만 있으면 돌아가야 합니다.
#
# ## 비교 모델과 원리
# 이 노트북에는 비교 모델이 없습니다. 대신 각 노트북이 기대하는 실행 조건을 확인합니다.
# - GPU 아키텍처: RTX 3060은 Ampere(sm86)라서 bf16(VGGT)과 FlashAttention2(Pointcept)를 쓸 수 있습니다.
# - VRAM: 하드웨어 프로파일(`low`/`mid`/`high`)을 고르는 기준입니다.
# - 커널: 의존성 충돌 때문에 torch 2.1.2(`3dv`)와 torch 2.5.0(`3dv-point`) 두 env로 나눴습니다.
#
# ## 원 논문
# 해당 없음. 환경 설계 근거는 `docs/step0_research.md`, 설치 방법은 `envs/README.md`를 보세요.
#
# ## 커널
# `3dv`. `3dv-point`의 점검은 그 env의 python을 subprocess로 호출해 수행합니다.
#
# ## 예상 VRAM·시간 (추정)
# VRAM은 거의 쓰지 않습니다. 시간은 스모크 테스트 두 번에 1~3분 정도로 추정합니다.
#
# ## 이 노트북이 답하려는 질문
# - 이 머신에서 어떤 프로파일로 돌려야 하는가?
# - 두 커널이 등록돼 있고 필요한 패키지가 모두 import되는가?
# - nuScenes-mini와 lidarseg가 올바른 위치에 있는가?

# %%
FORCE_RERUN = False
NOTEBOOK = "00_env_check"

import json
import os
import shutil
import subprocess
import sys

import pandas as pd
from bench3d.config import load_config
from bench3d.io import result_dir
from bench3d.runtime import env_report, set_seed, suggest_profile

cfg = load_config("outdoor")
set_seed(cfg["seed"])
ENV = env_report(["torch", "numpy", "nuscenes-devkit", "mmdet3d", "nerfstudio", "gsplat"])
print(json.dumps(ENV, indent=1))
print("configured profile:", cfg["profile"])

# %% [markdown]
# ## 1. GPU와 프로파일
# 프로파일 기준(CLAUDE.md): VRAM 8GB 이하 `low`, 16GB 이하 `mid`, 그 이상 `high`.
# 설정된 프로파일과 권장 프로파일이 다르면 `outdoor/config.yaml`의 `profile:`을 고치거나
# 환경변수 `THREEDCHECK_PROFILE`을 지정하세요.

# %%
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
                          "--format=csv"], capture_output=True, text=True).stdout)
else:
    print("nvidia-smi not found: no NVIDIA driver visible to this kernel")

gpu_mb = ENV.get("gpu_total_mb")
recommended = suggest_profile(gpu_mb)
print(f"GPU total: {gpu_mb} MB -> recommended profile: {recommended} (configured: {cfg['profile']})")
if ENV.get("cuda_available"):
    import torch

    free, total = torch.cuda.mem_get_info()
    print(f"free now: {free / 2**20:.0f} / {total / 2**20:.0f} MB")
    cap = tuple(int(x) for x in ENV["gpu_capability"].split("."))
    print("bf16 (VGGT autocast):", ENV["bf16_supported"], "| FlashAttention2 needs sm80+:", cap >= (8, 0))
if recommended != cfg["profile"]:
    print(f"WARNING: configured profile '{cfg['profile']}' differs from recommended '{recommended}'")

# %% [markdown]
# ## 2. 커널과 import 스모크 테스트
# 등록된 Jupyter 커널에서 각 env의 python 경로를 찾아 `envs/smoke_test.py`를 실행합니다.
# 선택 확장(BEVFusion 연산자, curope, pointops)은 빌드하지 않았으면 SKIP으로 나옵니다.

# %%
ks = json.loads(subprocess.run([sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
                               capture_output=True, text=True).stdout or "{}").get("kernelspecs", {})
kernel_python = {name: spec["spec"]["argv"][0] for name, spec in ks.items() if name in ("3dv", "3dv-point")}
print("registered:", kernel_python)

smoke = {}
for name in ("3dv", "3dv-point"):
    py = kernel_python.get(name)
    if not py:
        print(f"[{name}] kernel not registered -> run envs/setup_{name}.sh")
        continue
    out_json = result_dir(cfg, NOTEBOOK) / f"smoke_{name}.json"
    r = subprocess.run([py, str(cfg.root / "envs" / "smoke_test.py"), name, "--json", str(out_json)],
                       capture_output=True, text=True, env={**os.environ, "THIRD_PARTY": str(cfg.third_party())})
    print(r.stdout[-3000:])
    if out_json.exists():
        smoke[name] = json.loads(out_json.read_text())

# %%
rows = [{"env": n, **c} for n, s in smoke.items() for c in s["checks"]]
pd.DataFrame(rows)[["env", "check", "ok", "version", "error"]] if rows else "no smoke results"

# %% [markdown]
# ## 3. 고정 커밋 저장소
# `envs/third_party.lock`의 저장소가 clone돼 있고, 지정한 커밋에 체크아웃돼 있는지 확인합니다.

# %%
lock = [line.split() for line in (cfg.root / "envs" / "third_party.lock").read_text().splitlines()
        if line.strip() and not line.startswith("#")]
tp_rows = []
for name, env_name, ref, _url in lock:
    d = cfg.third_party(name)
    head = tag = ""
    if (d / ".git").exists():
        head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        tag = subprocess.run(["git", "-C", str(d), "rev-parse", f"{ref}^{{commit}}"], capture_output=True,
                             text=True).stdout.strip()
    tp_rows.append({"repo": name, "env": env_name, "pinned": ref, "present": d.exists(),
                    "at_pinned_commit": bool(head) and head == tag})
pd.DataFrame(tp_rows)

# %% [markdown]
# ## 4. 데이터
# `outdoor/scripts/verify_nuscenes_mini.py`를 실행합니다. 데이터가 없으면
# `outdoor/scripts/place_nuscenes_mini.sh`의 안내를 따라 내려받은 뒤 다시 실행하세요.

# %%
r = subprocess.run([sys.executable, str(cfg.root / "outdoor" / "scripts" / "verify_nuscenes_mini.py")],
                   capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-2000:])
DATA_OK = r.returncode == 0
print("DATA_OK:", DATA_OK)

# %% [markdown]
# ## 5. 결과 폴더 현황
# 노트북별로 지금까지 저장된 `summary*.json` 개수를 봅니다. 99 종합 노트북은 이 파일들만 읽습니다.

# %%
res_root = cfg.path(cfg["results_root"])
nb_dirs = sorted(p for p in res_root.glob("[0-9][0-9]_*") if p.is_dir()) if res_root.exists() else []
pd.DataFrame([{"notebook": d.name, "summaries": len(list(d.glob("**/summary*.json")))} for d in nb_dirs]) \
    if nb_dirs else "no results yet"

# %%
report = {"env": ENV, "recommended_profile": recommended, "configured_profile": cfg["profile"],
          "kernels": kernel_python, "smoke": {n: sum(c["ok"] is False for c in s["checks"]) for n, s in smoke.items()},
          "third_party": tp_rows, "data_ok": DATA_OK}
(result_dir(cfg, NOTEBOOK) / "env_check.json").write_text(json.dumps(report, indent=2, default=str))
print("saved", result_dir(cfg, NOTEBOOK) / "env_check.json")

# %% [markdown]
# ## 해석
# > (실행 후 작성) 권장 프로파일, 실패한 스모크 항목과 조치, 데이터 상태를 적습니다.
