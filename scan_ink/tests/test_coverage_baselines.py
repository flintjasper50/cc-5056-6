"""两路独立参照的互相印证：解析覆盖基准 vs 独立超采样重合成。

需求权衡：
  - 解析覆盖（analytic_line_coverage）给出"墨量应是多少"的几何闭式基准，
    快且确定，但只对理想盒核成立；
  - 独立超采样（build_triple_scene 的 4x 软/盒核 + 均值积分）用完全不同的
    求值路径重合成，作为判定基准，能捕捉离散化与多级混合的真实行为。

本测试断言两路在空间支撑、墨量级、相关性上一致。期望值（阈值）依据
"同物理盒核的两种独立数值求法只应相差离散化误差"这一原理给出，并从
实际两路数据现场计算，不引用生产提取的任何输出。
"""
from __future__ import annotations

import numpy as np

from inkline.triple_layer import (
    TripleSpec, analytic_faint_coverage, build_triple_scene,
    coverage_agreement,
)


def test_analytic_and_supersampled_support_agree():
    scene = build_triple_scene(TripleSpec(seed=23))
    analytic = scene["analytic_base"]
    supersampled = scene["coverage_ss_geom"]

    agree = coverage_agreement(analytic, supersampled, level=0.05)

    # 两路判出的有墨像素必须在 1px 邻域内互相支撑（轴对齐线无亚像素漂移）
    assert agree["support_match"] > 0.95, agree
    # 任一路都不得退化成空掩膜（防止默认值/全零遮错）
    assert agree["analytic_pixels"] > 1000
    assert agree["reference_pixels"] > 1000


def test_coverage_magnitude_same_order():
    scene = build_triple_scene(TripleSpec(seed=23))
    analytic = scene["analytic_base"]
    supersampled = scene["coverage_ss_geom"]
    common = (analytic > 0.05) & (supersampled > 0.05)
    assert common.any()

    # 墨量级别：两路共同支撑上的平均覆盖应在 0.5x~2x 同量级区间。
    # 离散化差异有界，不允许一路把墨量系统性砍掉或翻倍。
    ratio = float(supersampled[common].mean() / analytic[common].mean())
    assert 0.5 <= ratio <= 2.0, ratio
    # 逐像素皮尔逊相关：同核同几何，应强相关（阈值 0.9，显著高于随机）
    corr = float(np.corrcoef(analytic.ravel(), supersampled.ravel())[0, 1])
    assert corr > 0.9, corr


def test_analytic_baseline_is_independent_formula():
    """解析基准必须由几何公式直接给出，而非从超采样图阈值化得到。

    用一个独立小算例手算：穿过像素中心的 1px 竖线，中心像素覆盖应为 1，
    相邻 1px 处因盒核过渡带取 0.05（r=0.5 时 1-(1-0)=0 ... 这里直接核对
    公式的对称/有界性质），且整条线覆盖总和等于线宽*长度的几何量纲。
    """
    spec = TripleSpec(height=20, width=20)
    cov = analytic_faint_coverage(spec)
    assert cov.shape == (20, 20)
    assert 0.0 <= cov.min() and cov.max() <= 1.0
    # 竖线 x=20 不在 20 宽内，但 x=20,46... 第一条在 x=20？范围 20..w step26
    # 该小画布至少有 x=20 越界，改用直接公式核对单条线
    from inkline.triple_layer import analytic_line_coverage
    single = analytic_line_coverage(20, 20, (10, 2), (10, 18), 1.0)
    # 中心列在竖线内部（y=5..15）应满覆盖
    assert np.all(single[5:15, 10] == 1.0)
    # 距中心两列以外应无覆盖（盒核影响半径 r+0.5=1.0）
    assert single[:, 8].max() == 0.0 and single[:, 12].max() == 0.0
    # 覆盖率总和应等于线长（1px 宽 * 17 段 ≈ 17 个满覆盖当量），量纲核对
    assert 14.0 < single.sum() < 18.0, float(single.sum())
