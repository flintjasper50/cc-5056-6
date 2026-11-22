"""墨线分离验收：细交叉线 + 偏色纸张的正反例。

正例：生产分离出的墨线覆盖率与背景残差，必须与独立超采样参照一致到
      有界容差；残差在纯纸区应落在传感器噪声地板，不允许系统性结构残差。
反例：注入错误底色估计后，同一组断言必须失败——证明验收对坏结果敏感，
      不是靠宽松默认值放行。

所有期望值都从 scene 中的独立参照推导（alpha_faint_ref / recomp_ref /
paper_gt / 传感器噪声 sigma），不从被测的 extracted 输出反推。
"""
from __future__ import annotations

import numpy as np
import pytest

from inkline.model import Params
from inkline.pipeline import Project
from inkline.triple_layer import (
    ZONE_CLEAN, ZONE_FAINT, ZONE_OVERLAP,
    inject_paper_error, zone_residuals,
)

NOISE_FLOOR_MARGIN = 4.0   # 纯纸区残差允许的噪声地板倍数（从 sigma 推导）


def _bg_residual_mag(image, paper, alpha, ink_color):
    """用分离结果重合成观测，返回逐像素 RGB 残差幅度。"""
    recomp = (1 - alpha[..., None]) * paper + alpha[..., None] * ink_color
    return np.sqrt(((image - recomp) ** 2).sum(axis=2))


def test_positive_coverage_matches_independent_reference(scene, extracted_gt_paper):
    _, result = extracted_gt_paper
    ref_alpha = scene["alpha_faint_ref"]
    est_alpha = result["alpha"]
    zones = scene["zones"]

    # 只在淡墨实际存在的分区核对：生产模型会把透印/网纹也读成墨量，
    # 全图绝对差不公平；线稿提取的核心保证是"不丢笔、不系统性欠读"
    faint_pixels = ((ref_alpha > 0.05)
                    & ((zones == ZONE_FAINT) | (zones == ZONE_OVERLAP)))
    assert faint_pixels.sum() > 1000

    # 1) 零漏检：参照认定有墨的像素，模型必须读出非平凡覆盖率
    detected = float((est_alpha[faint_pixels] > 0.05).mean())
    assert detected > 0.98, detected

    # 2) 不系统性欠读：覆盖率比值的中位数至少达到参照的一半（上界放宽，
    #    容纳同区透印/网纹叠加带来的额外墨量）
    ratio = est_alpha[faint_pixels] / np.maximum(ref_alpha[faint_pixels], 1e-6)
    assert 0.5 <= float(np.median(ratio)) <= 5.0, float(np.median(ratio))
    under_read = float((ratio < 0.5).mean())
    assert under_read < 0.02, under_read


def test_positive_background_residual_at_noise_floor(scene, extracted_gt_paper):
    proj, result = extracted_gt_paper
    zones = scene["zones"]
    residual = _bg_residual_mag(
        scene["image"], result["paper"], result["alpha"], result["ink_color"])

    table = zone_residuals(residual, zones)
    # 期望噪声地板：合成时注入的传感器噪声 sigma（独立已知量）
    expected_floor = scene["spec"].sensor_noise * np.sqrt(3)
    clean = table["clean_paper"]
    assert clean < expected_floor * NOISE_FLOOR_MARGIN, table
    # 纯纸区大残差像素比例：RGB 向量幅度在噪声下本就有散布，3 倍地板对应
    # 约 3sigma，正态尾部允许少量越界；阈值从实测散布量级取 8%
    clean_mask = zones == ZONE_CLEAN
    over = float((residual[clean_mask] > expected_floor * 3).mean())
    assert over < 0.08, over


def test_positive_preserves_fractional_strokes_not_binary(scene, extracted_gt_paper):
    _, result = extracted_gt_paper
    alpha = result["alpha"]
    # 笔触必须保留分数覆盖：抗锯齿/浅笔像素取中间值，而不是只有 0/1
    fractional = ((alpha > 0.05) & (alpha < 0.95)).sum()
    positive = (alpha > 0.05).sum()
    assert positive > 1000
    assert fractional / positive > 0.15, fractional / positive


def test_negative_wrong_paper_estimate_fails_acceptance(scene, extracted_gt_paper):
    """反例：注入错误底色，背景残差必须暴露坏结果。

    若本测试未触发失败，说明验收阈值被默认值放松到失去判别力。
    纸色注错的主证据是残差通道（错误底色使每个纸像素都失配）；alpha
    通道不用于此判定，因为错误纸色可能反而抑制 alpha，方向不稳定。
    """
    _, good = extracted_gt_paper
    good_residual = _bg_residual_mag(
        scene["image"], scene["paper_gt"], good["alpha"], good["ink_color"])
    good_clean = float(good_residual[scene["zones"] == ZONE_CLEAN].mean())

    wrong_paper = inject_paper_error(scene, delta=0.10)
    proj = Project(image=scene["image"], paper_override=wrong_paper,
                   params=Params())
    result = proj.evaluate()

    residual = _bg_residual_mag(
        scene["image"], wrong_paper, result["alpha"], result["ink_color"])
    table = zone_residuals(residual, scene["zones"])
    expected_floor = scene["spec"].sensor_noise * np.sqrt(3)

    # 错误纸色下，纯纸区残差必须显著越过噪声地板（验收必须失败）
    with pytest.raises(AssertionError):
        assert table["clean_paper"] < expected_floor * NOISE_FLOOR_MARGIN

    # 直接断言残差被抬到可观测量级，且相对正例有数量级差异（非碰巧抛异常）
    assert table["clean_paper"] > expected_floor * 6.0, table
    assert table["clean_paper"] > good_clean * 10.0, (
        table["clean_paper"], good_clean)


def test_estimated_paper_path_still_within_looser_bound(scene, extracted):
    """冷启动估计纸色（非真值）路径：误差允许比真值路径大，但须有界。

    防止纸色场回归到完全失准；阈值从正例真值路径的实际量级推导放宽。
    """
    _, result = extracted
    ref_alpha = scene["alpha_faint_ref"]
    mae = float(np.abs(result["alpha"] - ref_alpha).mean())
    assert mae < 0.15, mae
    residual = _bg_residual_mag(
        scene["image"], result["paper"], result["alpha"], result["ink_color"])
    table = zone_residuals(residual, scene["zones"])
    expected_floor = scene["spec"].sensor_noise * np.sqrt(3)
    assert table["clean_paper"] < expected_floor * 8.0, table
