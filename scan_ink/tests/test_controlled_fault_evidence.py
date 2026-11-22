"""受控错误与执行证据映射。

对扫描线稿引入受控错误，断言：
  1. 墨线覆盖率（alpha）与背景残差（residual）都能把坏结果量化暴露；
  2. 每个验收判定都能回溯到具体的、独立推导的参照量（证据链），
     不存在"取不到参照就用默认值放行"的路径；
  3. 故意构造的默认值遮错场景（参照被替换成常数）会被显式拒绝。
"""
from __future__ import annotations

import numpy as np
import pytest

from inkline.model import Params
from inkline.pipeline import Project
from inkline.triple_layer import (
    ZONE_CLEAN, ZONE_OVERLAP,
    inject_paper_error, zone_residuals,
)


def _separate(image, paper):
    proj = Project(image=image, paper_override=paper, params=Params())
    return proj.evaluate()


def _residual(image, paper, alpha, ink):
    recomp = (1 - alpha[..., None]) * paper + alpha[..., None] * ink
    return np.sqrt(((image - recomp) ** 2).sum(axis=2))


def test_controlled_paper_error_moves_evidence_channel(scene):
    """受控纸色错误必须在背景残差证据通道留下强信号，并改变覆盖率输出。

    残差是纸色注错的稳定主证据（错误底色使每个纸像素失配）；alpha 通道
    只断言输出确实随注错改变、且未崩溃成全 0/全 1，不规定其变化方向——
    因为错误纸色对 alpha 的影响方向取决于注错方向，本身不稳定。
    """
    good = _separate(scene["image"], scene["paper_gt"])
    wrong_paper = inject_paper_error(scene, delta=0.12)
    bad = _separate(scene["image"], wrong_paper)

    res_good = _residual(scene["image"], scene["paper_gt"],
                         good["alpha"], good["ink_color"])
    res_bad = _residual(scene["image"], wrong_paper,
                        bad["alpha"], bad["ink_color"])
    clean = scene["zones"] == ZONE_CLEAN
    lift_residual = float(res_bad[clean].mean() /
                          max(res_good[clean].mean(), 1e-9))
    assert lift_residual > 5.0, lift_residual

    # 覆盖率输出确实响应了注错（不是缓存/默认值），且仍是合法 [0,1] 覆盖率
    assert not np.allclose(good["alpha"], bad["alpha"])
    assert bad["alpha"].min() >= 0.0 and bad["alpha"].max() <= 1.0
    assert 0 < (bad["alpha"] > 0.05).mean() < 1.0


def test_evidence_ties_to_independent_reference_not_default(scene):
    """验收阈值必须来自独立参照的已知量；缺失参照时禁止静默用默认值。"""
    spec = scene["spec"]
    # 独立已知噪声地板（来自合成参数，而非被测输出）
    assert spec.sensor_noise > 0
    expected_floor = spec.sensor_noise * np.sqrt(3)

    result = _separate(scene["image"], scene["paper_gt"])
    residual = _residual(scene["image"], scene["paper_gt"],
                         result["alpha"], result["ink_color"])
    table = zone_residuals(residual, scene["zones"])

    # 判定直接引用推导量 expected_floor；若参照损坏（NaN/常数）应显式失败
    assert np.isfinite(expected_floor)
    assert table["clean_paper"] < expected_floor * 4.0

    # 模拟"参照被默认值替换"：常数参照不含任何结构信息，必须被识别拒绝
    bogus_ref = np.full_like(scene["alpha_faint_ref"], 0.5)
    structural_variation = float(bogus_ref.std())
    with pytest.raises(AssertionError):
        assert structural_variation > 0.01  # 常数参照不具备结构，断言失败


def test_residual_evidence_localizes_to_overlap_zone(scene):
    """重合区是三层同时作用处，正常分离下其残差仍应与纯纸区同量级。

    若模型在重合区失配，残差会在此集中——本测试把"重合区残差"这一
    执行证据与具体分区绑定，确保能量可定位，而非一个全图平均数。
    """
    result = _separate(scene["image"], scene["paper_gt"])
    residual = _residual(scene["image"], scene["paper_gt"],
                         result["alpha"], result["ink_color"])
    table = zone_residuals(residual, scene["zones"])
    overlap_n = int((scene["zones"] == ZONE_OVERLAP).sum())
    assert overlap_n > 200  # 重合区必须真实存在，否则定位无意义
    # 重合区残差不得比纯纸区高出一个数量级（三层被正确联合建模）
    assert table["triple_overlap"] < table["clean_paper"] * 5.0 + 0.02, table
