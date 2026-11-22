"""回归：扫描线稿旧合成样例与既有数值验证套件必须运行到底且不退化。

阈值直接取自仓库基线 report.json 的历史结果并留出容差，而不是新写的
乐观默认值；这样任何让旧样例变差的改动都会被拦下。
"""
from __future__ import annotations

from inkline.model import Params
from inkline.pipeline import Project
from inkline.synthetic import SyntheticSpec, generate
from inkline.tests.run_tests import run_all


def test_legacy_synthetic_sample_runs_end_to_end(tmp_path):
    data = generate(SyntheticSpec(seed=7))
    proj = Project(image=data["image"], params=Params())
    result = proj.evaluate()
    views = proj.views()
    metrics = proj.metrics(gt_alpha=data["alpha_gt"][..., 0])

    # 必须产出全部关键视图与覆盖率，且原件未被改写
    expected_views = {"original", "paper_field", "ink_layer_gray", "alpha",
                      "composite", "composite_dark", "composite_light",
                      "recomp_orig", "residual_heat", "missing_heat"}
    assert expected_views.issubset(views.keys())
    assert result["alpha"].shape == data["image"].shape[:2]

    # 历史基线（output/tests/report.json）+ 容差
    assert metrics["alpha_mae"] < 0.006, metrics
    assert metrics["alpha_weak_recall"] > 0.85, metrics
    thin = (data["alpha_gt"][..., 0] > 0.2) & (data["alpha_gt"][..., 0] < 0.6)
    missed = float((result["alpha"][thin] < 0.05).mean())
    assert missed < 0.02, missed


def test_legacy_report_suite_completes(tmp_path):
    report = run_all(tmp_path)
    # 8 个编号用例全部产出
    expected_ids = {f"0{i}_" for i in range(1, 9)}
    ids = {k[:3] for k in report}
    assert expected_ids.issubset(ids), ids
    # 关键历史结论不退化
    assert report["01_standard"]["alpha_mae"] < 0.006
    assert report["03_rebase_missing_ink"]["strict_gate_missing_lift"] > 1.3
    assert report["04_fringe"]["halo_on_dark_fraction"] < 0.05
    assert report["05_multiply_baseline_drift"]["drift_ratio"] > 8.0
    assert report["06_binarization_damage"]["weak_pixels_destroyed_fraction"] > 0.95
    assert report["07_recompute"]["original_unchanged"] is True
    # 诊断 PNG 落盘，可从画布打开人工复核
    assert (tmp_path / "01_standard" / "alpha_est.png").exists()
    assert (tmp_path / "01_standard" / "residual_heat.png").exists()


def test_fringe_sample_runs_through(tmp_path):
    data = generate(SyntheticSpec(seed=11, fringe=True))
    proj = Project(image=data["image"], params=Params())
    views = proj.views()
    # 污染边场景同样要跑完整链路并产出深/浅底重铺
    assert views["composite_dark"].shape == data["image"].shape
    assert views["composite_light"].shape == data["image"].shape
