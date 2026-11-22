"""注错分层定位：透印/淡墨/网纹三重重合图上注入两类错误，
要求能从失败区域判别出错的是哪一层。

  A. 纸色估计注错（inject_paper_error）：纸色是所有混合的共同底色，错误
     会在*所有*分区抬高残差，包括不含任何墨的纯纸区——这是纸色层的签名。
  B. 混合次序注错（invert_blend_order）：只改变墨层之间的上下关系，纯纸区
     完全不受影响，残差集中在含淡墨的区域（淡墨区、三重重合区）——这是
     次序/墨层的签名。

期望值（残差地板、抬升倍数）由独立参照 recomp_ref 与已知噪声 sigma 推导。
"""
from __future__ import annotations

import numpy as np

from inkline.model import Params
from inkline.pipeline import Project
from inkline.triple_layer import (
    inject_paper_error, invert_blend_order, zone_residuals,
)


def _residual_table(image, rebuild, zones):
    mag = np.abs(image - rebuild).mean(axis=2)
    return zone_residuals(mag, zones), mag


def test_baseline_residual_is_flat_across_zones(scene):
    """未注错时，独立参照重合成在各分区残差都接近噪声地板（对照组）。"""
    table, _ = _residual_table(scene["image"], scene["recomp_ref"],
                               scene["zones"])
    floor = scene["spec"].sensor_noise
    values = [table[z] for z in
              ("clean_paper", "show_through_only", "halftone_only",
               "faint_ink_only", "triple_overlap")]
    assert max(values) < floor * 3.0, table


def test_paper_error_fails_everywhere_including_clean_paper(scene):
    wrong_paper = inject_paper_error(scene, delta=0.10)
    proj = Project(image=scene["image"], paper_override=wrong_paper,
                   params=Params())
    result = proj.evaluate()
    recomp = ((1 - result["alpha"][..., None]) * wrong_paper
              + result["alpha"][..., None] * result["ink_color"])
    table, _ = _residual_table(scene["image"], recomp, scene["zones"])
    floor = scene["spec"].sensor_noise

    # 纸色层签名：纯纸区（无墨）残差也被整体抬高
    assert table["clean_paper"] > floor * 6.0, table
    # 各分区普遍受影响，最小区也远超地板
    affected = [table["show_through_only"], table["halftone_only"],
                table["faint_ink_only"]]
    assert min(affected) > floor * 4.0, table


def test_order_error_spares_clean_paper(scene):
    wrong = invert_blend_order(scene)
    table, _ = _residual_table(scene["image"], wrong, scene["zones"])
    floor = scene["spec"].sensor_noise

    # 次序签名：纯纸区与基线无异（次序不碰纸面）
    assert table["clean_paper"] < floor * 3.0, table
    # 而含淡墨的重合区残差被显著抬高
    assert table["triple_overlap"] > floor * 6.0, table
    assert table["faint_ink_only"] > table["clean_paper"] * 3.0, table


def test_faults_are_distinguishable_by_zone_signature(scene):
    """两类注错的残差分区向量必须可分：用 argmax 区域特征判别出错层。"""
    zones = scene["zones"]
    floor = scene["spec"].sensor_noise

    wrong_paper = inject_paper_error(scene, delta=0.10)
    proj = Project(image=scene["image"], paper_override=wrong_paper,
                   params=Params())
    rp = proj.evaluate()
    recomp_p = ((1 - rp["alpha"][..., None]) * wrong_paper
                + rp["alpha"][..., None] * rp["ink_color"])
    paper_table, _ = _residual_table(scene["image"], recomp_p, zones)

    order_table, _ = _residual_table(
        scene["image"], invert_blend_order(scene), zones)

    # 判别量：纯纸区残差占总残差的比例。
    # 纸色错误：该比例高（纸面本身就错）；次序错误：该比例低（纸面无错）
    def clean_share(t):
        total = sum(t[z] for z in
                    ("clean_paper", "show_through_only", "halftone_only",
                     "faint_ink_only", "triple_overlap"))
        return t["clean_paper"] / max(total, 1e-9)

    share_paper = clean_share(paper_table)
    share_order = clean_share(order_table)
    assert share_paper > share_order * 1.5, (share_paper, share_order)
    # 次序错误的失败能量应集中在淡墨/重合区
    assert order_table["triple_overlap"] > order_table["halftone_only"] * 2
