# ---
# jupyter:
#   kernelspec:
#     display_name: 3dv (torch 2.1.2, cu121)
#     language: python
#     name: 3dv
# ---

# %% [markdown]
# # 99. 실외 종합 비교: 카메라로 본 3D vs LiDAR로 본 3D
#
# ## 목적
# 02~08이 저장한 `results/outdoor/**/summary*.json`과 보조 파일만 읽어 비교 표와 그래프를 만듭니다.
# 모델을 다시 돌리지 않으며, 아직 결과가 없는 노트북의 절은 건너뜁니다.
#
# ## 비교 모델과 원리
# 각 노트북의 모델을 축별로 모읍니다.
# - 깊이(02): Depth Anything V2 metric(VKITTI), Depth Anything 3(METRIC / NESTED / MONO). 정답은 LiDAR 투영 희소 깊이입니다.
# - 다시점 기하(03): DUSt3R, MASt3R(최적화형 전역 정렬)와 VGGT, Depth Anything 3(순수 피드포워드). 자세와 점군을 LiDAR와 비교합니다.
# - 3D 탐지(05): PointPillars, CenterPoint(LiDAR), BEVFusion(LiDAR+카메라), FCOS3D(카메라 전용), 그리고 가짜 LiDAR 입력.
# - 포인트 이해(06): PointNet(직접 구현), Concerto 선형탐침, PTv3, Sonata(특징 시각화).
# - 장면 표현(07): Nerfacto, Instant-NGP, Splatfacto. 엣지(08): PointPillars ONNX/TensorRT.
#
# ## 원 논문
# 각 노트북 첫 셀에 있습니다. nuScenes: https://arxiv.org/abs/1903.11027
#
# ## 커널
# `3dv` (파일만 읽으므로 `3dv-point` 결과도 함께 읽습니다)
#
# ## 예상 VRAM·시간 (추정)
# GPU를 쓰지 않습니다. 1분 이내로 추정합니다.
#
# ## 이 노트북이 답하려는 질문
# - 거리에 따라 카메라 깊이 오차는 어떻게 커지며, 어디부터 LiDAR가 필요한가?
# - 피드포워드 기하 모델은 최적화형보다 얼마나 빠르고, 정확도는 얼마나 잃는가?
# - 카메라만으로 만든 3D(가짜 LiDAR, 카메라 전용 탐지기)는 LiDAR 탐지를 어디까지 대체하는가?
# - 결론: 카메라만으로 충분한 영역과 LiDAR가 필요한 영역은 어디인가?

# %%
NOTEBOOK = "99_summary"

import json
import re

import matplotlib.pyplot as plt
import pandas as pd
from bench3d.config import load_config
from bench3d.io import collect_summaries, result_dir
from bench3d.runtime import env_report
from bench3d.viz import PALETTE, error_vs_distance_plot, model_colors

cfg = load_config("outdoor")
ENV = env_report(["pandas", "matplotlib"])
RES = cfg.path(cfg["results_root"])
OUT = result_dir(cfg, NOTEBOOK)
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)

df = collect_summaries(RES)
print(f"{len(df)} metric rows from {df['file'].nunique() if len(df) else 0} summary files under {RES}")


def table(notebook: str, metrics: list[str] | None = None, index=("model", "variant")) -> pd.DataFrame:
    """Wide table (index x metric) for one notebook; empty frame if no results."""
    sub = df[df["notebook"] == notebook] if len(df) else df
    if metrics is not None and len(sub):
        sub = sub[sub["metric"].isin(metrics)]
    if not len(sub):
        return pd.DataFrame()
    t = sub.pivot_table(index=list(index), columns="metric", values="value", aggfunc="first")
    cols = [m for m in (metrics or t.columns) if m in t.columns]
    return t[cols]


def note(msg: str):
    print(f"[skip] {msg}")


# %% [markdown]
# ## 0. 결과 현황
# 노트북별로 저장된 요약 파일 수와 모델 목록입니다. 비어 있는 절은 해당 노트북을 먼저 실행하세요.

# %%
if len(df):
    cov = df.groupby("notebook").agg(files=("file", "nunique"), models=("model", lambda s: sorted(set(s))))
    display(cov)
else:
    note("no summaries yet")

# %% [markdown]
# ## 1. 단안 깊이(02): 거리별 오차
# - 표: 모델별 **primary** 결과(metric 모델은 원출력, 상대깊이 모델은 median scaling). 구간별 AbsRel은 GT 거리 기준입니다.
# - `median_scaled` 변형과 비교하면 오차 중 **스케일 오차**(전체 배율이 틀림)와 **형상 오차**(상대 구조가 틀림)를 나눠 볼 수 있습니다.
# - 그래프: 거리별 평균 절대오차의 log-log 곡선과 기울기 k. k≈2는 스테레오의 이론 법칙이고, 단안 신경망에는 그런 법칙이 없으므로
#   k는 관찰된 경향을 요약하는 값일 뿐입니다.

# %%
bins = ["0-10m", "10-20m", "20-30m", "30-50m", "50m+"]
depth_cols = ["abs_rel", "rmse", "delta1", "err_slope_k"] + [f"abs_rel@{b}" for b in bins]
t_depth = table("02_monocular_depth", depth_cols)
if len(t_depth):
    display(t_depth.round(3))
    curves = {}
    for f in sorted((RES / "02_monocular_depth").glob("*/bins_primary.json")):
        d = json.loads(f.read_text())
        curves[f"{d['model']} ({d.get('label', '')})"] = d["curve"]
    if curves:
        fig, ax = plt.subplots(figsize=(7, 5))
        error_vs_distance_plot(ax, curves)
        ax.set_title("monocular depth: mean |error| vs distance (primary)")
        fig.savefig(OUT / "depth_error_vs_distance.png", dpi=120, bbox_inches="tight")
    # scale vs shape error
    t_ms = table("02_monocular_depth", ["abs_rel", "delta1"])
    if len(t_ms):
        display(t_ms.unstack("variant").round(3))
else:
    note("02 has no results")

# %% [markdown]
# ## 2. 다시점 기하(03): 속도-정확도
# - `front`(CAM_FRONT 연속 프레임): 주 지표는 자세 AUC@30, ATE, Chamfer, F-score@0.5m입니다.
# - `surround`(같은 시각의 6개 카메라): 카메라 중심이 1~2m 안에 모여 있어 스케일과 Sim(3)가 불안정합니다.
#   그래서 상대 회전·방향 오차(RRA/RTA)와 AUC만 주 지표로 봅니다.
# - 그래프: x축 실행 시간(창당 초), y축 정확도. 왼쪽 위일수록 좋습니다.

# %%
mv_cols = ["auc@30", "auc@15", "rra_mean_deg", "rta_mean_deg", "ate_rmse", "chamfer", "fscore@0.5",
           "sim3_scale_median", "rot_center_frac"]
t_mv = table("03_multiview_geometry", mv_cols)
if len(t_mv):
    display(t_mv.round(3))
    rows = df[(df["notebook"] == "03_multiview_geometry") & (df["metric"].isin(["auc@30", "fscore@0.5"]))]
    rt = rows.groupby(["model", "variant"]).agg(runtime_s=("runtime_s", "first"))
    piv = rows.pivot_table(index=["model", "variant"], columns="metric", values="value", aggfunc="first").join(rt)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cols = model_colors(sorted(piv.index.get_level_values(0).unique()))
    for ax, met in zip(axes, ["auc@30", "fscore@0.5"]):
        for (m, v), r in piv.iterrows():
            if met in r and pd.notna(r[met]) and pd.notna(r["runtime_s"]):
                ax.scatter(r["runtime_s"], r[met], color=cols[m], marker="o" if "front" in str(v) else "s", s=60)
                ax.annotate(f"{m}\n{v}", (r["runtime_s"], r[met]), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("runtime per window [s] (log)")
        ax.set_ylabel(met)
        ax.grid(alpha=0.3)
    fig.suptitle("feed-forward vs optimisation: speed-accuracy (o: front, s: surround)")
    fig.savefig(OUT / "multiview_speed_accuracy.png", dpi=120, bbox_inches="tight")
else:
    note("03 has no results")

# %% [markdown]
# ## 3. 가짜 LiDAR(04): 점 밀도
# 같은 카메라 시야 안에서 거리 링별 점 밀도를 실제 LiDAR와 비교합니다.

# %%
t_pl = table("04_pseudo_lidar")
display(t_pl.T.round(3)) if len(t_pl) else note("04 has no results")

# %% [markdown]
# ## 4. 3D 탐지(05): LiDAR vs 가짜 LiDAR vs 융합 vs 카메라 전용
# - 주 지표 `mAP`/`NDS`는 **카메라 시야(FOV) 필터를 적용한 값**입니다. 가짜 LiDAR와 카메라 전용 탐지기는 시야 밖을 볼 수 없으므로,
#   같은 영역에서 비교해야 공정합니다. `nofov_*`는 360° 기준 참고값입니다.
# - mini의 4개 장면 결과라 공식 val 수치와 직접 비교할 수 없습니다.

# %%
det_cols = ["mAP", "NDS", "nofov_mAP", "nofov_NDS", "samples_per_s", "samples_per_s_model_only"]
t_det = table("05_lidar_detection", det_cols)
if len(t_det):
    display(t_det.round(3))
    sub = df[df["notebook"] == "05_lidar_detection"]
    pat = re.compile(r"^fov_d([\d.]+)-([\d.]+)_mAP$")
    dist = sub[sub["metric"].str.match(pat)].copy()
    if len(dist):
        dist["bin"] = dist["metric"].str.extract(pat).apply(lambda r: f"{float(r[0]):g}-{float(r[1]):g}m", axis=1)
        dist["run"] = dist["model"] + " / " + dist["variant"].astype(str)
        piv = dist.pivot_table(index="bin", columns="run", values="value", aggfunc="first")
        piv = piv.loc[sorted(piv.index, key=lambda s: float(s.split("-")[0]))]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, c in enumerate(piv.columns):
            ax.plot(piv.index, piv[c], "o-", color=PALETTE[i % len(PALETTE)], label=c)
        ax.set_ylabel("mAP (FOV-filtered)")
        ax.set_xlabel("GT ego distance bin")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.savefig(OUT / "detection_ap_vs_distance.png", dpi=120, bbox_inches="tight")
else:
    note("05 has no results")

# %% [markdown]
# ## 5. 포인트 분할(06)
# `miou`는 eval4에 나타난 클래스 평균, `miou_all16`은 16개 클래스 전체 평균입니다.
# 가짜 LiDAR 결과(`summary_pseudo_*`)는 실제 LiDAR 라벨을 가까운 점으로 옮긴 **대리(proxy) 지표**입니다.

# %%
t_seg = table("06_point_cloud_understanding", ["miou", "miou_all16", "overall_acc", "accuracy", "miou_pointcept_log"])
display(t_seg.round(3)) if len(t_seg) else note("06 has no results")

# %% [markdown]
# ## 6. 장면 표현(07)
# 평가 뷰가 장면당 4장 정도라 통계적 근거가 약합니다. 수치보다 실패 유형(전진 궤적, 동적 물체, 하늘·원거리)을 함께 봅니다.

# %%
t_nvs = table("07_scene_representation", ["psnr", "ssim", "lpips", "psnr_static", "psnr_top", "psnr_bottom",
                                          "train_time_s", "render_fps"])
display(t_nvs.round(3)) if len(t_nvs) else note("07 has no results")

# %% [markdown]
# ## 7. 엣지 배포(08): 단계별 지연시간(ms)

# %%
t_edge = table("08_edge_deployment")
if len(t_edge):
    lat = t_edge[[c for c in t_edge.columns if c.startswith("lat_")]].T
    display(lat.round(3))
else:
    note("08 has no results")

# %% [markdown]
# ## 8. 저장
# 표를 CSV로 저장합니다. 그림은 위 셀에서 같은 폴더에 저장했습니다.

# %%
for name, t in {"depth": t_depth, "multiview": t_mv, "pseudo_lidar": t_pl, "detection": t_det,
                "segmentation": t_seg, "nvs": t_nvs, "edge": t_edge}.items():
    if len(t):
        t.to_csv(OUT / f"table_{name}.csv")
(OUT / "summary_index.json").write_text(json.dumps(
    {"n_rows": len(df), "files": sorted(df["file"].unique().tolist()) if len(df) else []}, indent=1))
print("saved tables to", OUT)

# %% [markdown]
# ## 결론: 카메라만으로 충분한 영역과 LiDAR가 필요한 영역
# 아래 칸은 실행 결과를 보고 근거 수치와 함께 채웁니다. 숫자는 위 표의 셀을 인용합니다.
#
# **카메라만으로 충분한 영역**
# > (실행 후 작성) 예: 0–20m 깊이 AbsRel, 근거리 탐지 AP, 상대 자세(RRA/RTA)
#
# **LiDAR가 필요한 영역**
# > (실행 후 작성) 예: 30m 이상 깊이 오차 증가(기울기 k), 원거리 AP 하락, 야간·비 장면
#
# **비용(속도·메모리) 관점**
# > (실행 후 작성) 피드포워드 vs 최적화 실행 시간, 엣지 지연시간에서 voxelization 비중
#
# **한계**
# > mini 4개 장면, 평가 뷰 수, LiDAR 정답의 희소성과 시차, 사전학습 데이터 누수 미확인 항목(docs/step0_research.md)
