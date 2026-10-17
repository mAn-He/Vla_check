"""Unit tests for bench3d on synthetic data (no dataset, no GPU). Run: python -m pytest common/tests -q"""

import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bench3d import align
from bench3d.metrics import depth, pointcloud, pose, segmentation

RNG = np.random.default_rng(0)


def _rand_sim3():
    return 2.5, Rotation.random(random_state=1).as_matrix(), np.array([1.0, -2.0, 3.0])


def test_umeyama_recovers_sim3():
    s, R, t = _rand_sim3()
    src = RNG.normal(size=(50, 3))
    s2, R2, t2 = align.umeyama(src, align.apply_sim3(src, s, R, t))
    assert s2 == pytest.approx(s)
    np.testing.assert_allclose(R2, R, atol=1e-9)
    np.testing.assert_allclose(t2, t, atol=1e-9)


def _poses(centers, rots):
    T = np.tile(np.eye(4), (len(centers), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = rots, centers
    return T


def test_collinear_trajectory_uses_rotation_alignment():
    # forward-driving camera: centers on a line, small heading jitter
    n = 8
    centers = np.c_[np.zeros(n), np.zeros(n), np.linspace(0, 20, n)]
    rots = Rotation.from_euler("y", RNG.normal(0, 2, n), degrees=True).as_matrix()
    pred = _poses(centers, rots)
    s, R, t = _rand_sim3()
    gt = align.apply_sim3_to_poses(pred, s, R, t)
    assert align.centers_degenerate(gt[:, :3, 3])
    s2, R2, t2, used = align.align_poses_sim3(pred, gt)
    assert used == "rot+center"
    np.testing.assert_allclose(R2, R, atol=1e-6)
    assert s2 == pytest.approx(s, rel=1e-6)
    # points off the trajectory line must also land in place (this is what Chamfer depends on)
    off = RNG.normal(size=(20, 3)) * 5
    np.testing.assert_allclose(align.apply_sim3(off, s2, R2, t2), align.apply_sim3(off, s, R, t), atol=1e-6)


def test_median_scale_and_lstsq():
    gt = RNG.uniform(2, 50, size=1000)
    assert align.median_scale(gt / 3.0, gt) == pytest.approx(3.0)
    a, b = align.lstsq_scale_shift(0.5 * gt - 1.0, gt)
    assert a == pytest.approx(2.0) and b == pytest.approx(2.0)


def test_depth_metrics_values():
    gt = RNG.uniform(1, 80, size=(40, 50))
    m = depth.depth_metrics(gt, gt)
    assert m["abs_rel"] == pytest.approx(0) and m["delta1"] == 1.0 and m["rmse"] == pytest.approx(0)
    m = depth.depth_metrics(gt * 1.1, gt)
    assert m["abs_rel"] == pytest.approx(0.1) and m["delta1"] == 1.0
    m = depth.depth_metrics(gt * 1.3, gt)
    assert m["delta1"] == 0.0 and m["delta2"] == 1.0


def test_depth_invalid_pixels_ignored():
    gt = np.array([[0.0, 10.0], [np.nan, 20.0]])
    pred = np.array([[5.0, 10.0], [7.0, 20.0]])
    assert depth.depth_metrics(pred, gt)["n"] == 2


def test_binned_metrics_labels_and_counts():
    gt = np.array([5.0, 15.0, 25.0, 40.0, 70.0])
    b = depth.binned_depth_metrics(gt, gt)
    assert list(b) == ["0-10m", "10-20m", "20-30m", "30-50m", "50m+"]
    assert all(v["n"] == 1 for v in b.values())


def test_error_vs_distance_recovers_quadratic_law():
    gt = RNG.uniform(2, 80, size=200_000)
    pred = gt + 0.01 * gt**2 * RNG.choice([-1, 1], size=gt.size)
    r = depth.error_vs_distance(pred, gt)
    assert r["slope_k"] == pytest.approx(2.0, abs=0.05)


def test_pose_metrics_identity_and_auc():
    centers = np.c_[RNG.normal(size=(6, 2)), np.arange(6)]
    rots = Rotation.random(6, random_state=2).as_matrix()
    T = _poses(centers, rots)
    m = pose.pose_metrics(T, T)
    assert m["ate_rmse"] == pytest.approx(0, abs=1e-9) and m["auc@30"] == pytest.approx(1.0)
    rot_err = np.array([0.5, 10.0, 40.0])
    trans_err = np.array([0.5, 2.0, 1.0])
    # errors 0.5, 10, 40 -> acc(t) = 1/3 for t=1..10, 2/3 for t=11..30
    assert pose.pose_auc(rot_err, trans_err, 30) == pytest.approx((10 / 3 + 20 * 2 / 3) / 30)


def test_rpe_detects_relative_drift():
    T = _poses(np.c_[np.zeros(4), np.zeros(4), np.arange(4.0)], np.tile(np.eye(3), (4, 1, 1)))
    P = T.copy()
    P[:, 2, 3] *= 1.1
    assert pose.rpe(P, T)["rpe_trans_rmse"] == pytest.approx(0.1)


def test_reconstruction_metrics():
    gt = RNG.uniform(-10, 10, size=(5000, 3))
    m = pointcloud.reconstruction_metrics(gt, gt)
    assert m["chamfer"] == 0 and m["fscore@0.5"] == 1.0
    m = pointcloud.reconstruction_metrics(gt + [0.3, 0, 0], gt)
    assert m["accuracy_median"] == pytest.approx(0.3, abs=0.1)
    assert m["fscore@0.25"] < m["fscore@0.5"]


def test_frustum_mask():
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    cam = {"K": K, "w2c": np.eye(4), "width": 100, "height": 100}
    pts = np.array([[0, 0, 5.0], [0, 0, -5.0], [10, 0, 5.0]])
    assert pointcloud.frustum_mask(pts, [cam]).tolist() == [True, False, False]


def test_iou():
    gt = np.array([0, 0, 1, 1, 2])
    pr = np.array([0, 1, 1, 1, 2])
    r = segmentation.iou_from_confusion(segmentation.confusion_matrix(pr, gt, 3), ["a", "b", "c"])
    assert r["per_class_iou"] == {"a": 0.5, "b": pytest.approx(2 / 3), "c": 1.0}
    assert r["miou"] == pytest.approx((0.5 + 2 / 3 + 1) / 3)


def test_backproject_then_project_roundtrip():
    from bench3d.nusc import backproject, sparse_depth_map

    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    d = RNG.uniform(2, 60, size=(48, 64))
    pts, pix = backproject(d, K, stride=1)
    uvw = pts @ K.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    np.testing.assert_allclose(uv[:, 0], pix[:, 1], atol=1e-6)
    np.testing.assert_allclose(uv[:, 1], pix[:, 0], atol=1e-6)
    dm = sparse_depth_map(uv, pts[:, 2], 64, 48)
    np.testing.assert_allclose(dm, d.astype(np.float32), rtol=1e-6)


def test_sparse_depth_keeps_nearest():
    from bench3d.nusc import sparse_depth_map

    dm = sparse_depth_map(np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([30.0, 5.0]), 3, 3)
    assert dm[1, 1] == 5.0


def test_points_files_roundtrip(tmp_path):
    from bench3d.io import load_nuscenes_bin, load_ply, save_nuscenes_bin, save_ply

    xyz = RNG.normal(size=(100, 3)).astype(np.float32)
    p = save_nuscenes_bin(tmp_path / "a.bin", xyz)
    back = load_nuscenes_bin(p)
    assert back.shape == (100, 5) and np.all(back[:, 3:] == 0)
    np.testing.assert_array_equal(back[:, :3], xyz)
    q = load_ply(save_ply(tmp_path / "a.ply", xyz, rgb=np.full((100, 3), 0.5), scalars={"conf": np.ones(100)}))
    np.testing.assert_allclose(q["xyz"], xyz)
    assert q["rgb"].dtype == np.uint8 and "conf" in q


def test_config_and_summaries(tmp_path, monkeypatch):
    from bench3d.config import load_config
    from bench3d.io import collect_summaries, result_dir, save_summary

    (tmp_path / "envs").mkdir()
    (tmp_path / "common" / "bench3d").mkdir(parents=True)
    (tmp_path / "outdoor").mkdir()
    (tmp_path / "outdoor" / "config.yaml").write_text(
        "profile: mid\nresults_root: results/outdoor\nseed: 0\n"
        "profiles:\n  mid: {depth: {cams: [CAM_FRONT], stride: 2}}\n  low: {depth: {stride: 4}}\n")
    monkeypatch.setenv("THREEDCHECK_ROOT", str(tmp_path))
    cfg = load_config("outdoor", profile="low")
    assert cfg.p["depth"] == {"cams": ["CAM_FRONT"], "stride": 4}  # low inherits from mid
    d = result_dir(cfg, "02_monocular_depth", "m1")
    save_summary(d, part="outdoor", notebook="02_monocular_depth", profile=cfg["profile"], model="m1",
                 metrics={"abs_rel": np.float32(0.1), "rmse": math.nan}, runtime_s=1.0, peak_vram_mb=None,
                 checkpoint="x", packages={}, seed=0)
    rec = json.loads((d / "summary.json").read_text())
    assert rec["metrics"]["rmse"] is None and rec["schema"] == 1
    df = collect_summaries(cfg.path("results"))
    assert set(df["metric"]) == {"abs_rel", "rmse"}
    with pytest.raises(ValueError):
        save_summary(d, part="outdoor", notebook="x", profile="mid", model="m", metrics={"bins": {}},
                     runtime_s=0, peak_vram_mb=0, checkpoint="", packages={}, seed=0)


def test_write_custom_splits(tmp_path):
    from bench3d.metrics.detection import write_custom_splits

    (tmp_path / "v1.0-mini").mkdir()
    write_custom_splits(tmp_path, "v1.0-mini", {"eval4": ["scene-0103"]})
    write_custom_splits(tmp_path, "v1.0-mini", {"probe6": ["scene-0061"]})
    data = json.loads((tmp_path / "v1.0-mini" / "splits.json").read_text())
    assert data == {"eval4": ["scene-0103"], "probe6": ["scene-0061"]}
