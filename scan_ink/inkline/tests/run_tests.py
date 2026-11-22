"""数值验证：所有结论来自真实像素运算，输出可复核的 PNG 与指标。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..model import Params, composite
from ..pipeline import Project
from ..synthetic import SyntheticSpec, generate
from ..util import to_u8, gray_to_u8


def _weak_recall(alpha: np.ndarray, gt: np.ndarray) -> float:
    """弱笔触（真值 alpha 在 0.05~0.35 的像素）被找回的平均比例。"""
    sel = (gt > 0.05) & (gt <= 0.35)
    if not sel.any():
        return 0.0
    return float(np.clip(alpha[sel] / np.maximum(gt[sel], 1e-6), 0, 1).mean())


def _thin_line_stats(alpha: np.ndarray, gt: np.ndarray) -> dict:
    """极细线（真值 0.2~0.6 的边缘带）上的保留率与漏检比例。"""
    thin = (gt > 0.2) & (gt < 0.6)
    if not thin.any():
        return {}
    ratio = alpha[thin] / np.maximum(gt[thin], 1e-6)
    return {
        "thin_pixels": int(thin.sum()),
        "thin_mean_ratio": float(np.clip(ratio, 0, 1).mean()),
        "thin_missed_fraction": float((alpha[thin] < 0.05).mean()),
    }


def _fringe_halo(comp_light: np.ndarray, comp_dark: np.ndarray,
                 alpha: np.ndarray) -> dict:
    """白边检验：墨缘两侧合成结果应分别平滑趋近底色，不允许出现
    与底色无关的固定亮圈。比较浅/深两底色上的缘边残差方向。"""
    from ..util import box_blur
    edge = np.clip(alpha - box_blur(alpha, 2), 0, 1) > 0.05
    outside = (alpha < 0.02) & (box_blur((alpha > 0.1).astype(np.float32), 2) > 0)
    if edge.sum() < 8 or outside.sum() < 8:
        return {}
    # 外缘像素在浅底上不应系统性比底色更暗于墨量解释（白边=夹纸色的亮环，
    # 在深底上表现为异常亮）
    dark_base = 0.25
    bright_outside_dark = float((comp_dark[outside].mean(axis=1) >
                                 dark_base + 0.12).mean())
    return {
        "edge_pixels": int(edge.sum()),
        "halo_on_dark_fraction": bright_outside_dark,
    }


def run_all(out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # 1) 标准素材：已知 K、alpha、纸底
    out = out_root / "01_standard"
    data = generate(SyntheticSpec(seed=7))
    proj = Project(image=data["image"], params=Params())
    r = proj.evaluate()
    gt = data["alpha_gt"][..., 0]
    out.mkdir(parents=True, exist_ok=True)
    views = proj.views()
    for k in ("original", "paper_field", "ink_layer_gray", "composite",
              "composite_dark", "composite_light", "recomp_orig",
              "residual_heat", "missing_heat"):
        Image.fromarray(to_u8(views[k])).save(out / f"{k}.png")
    Image.fromarray(gray_to_u8(gt)).save(out / "alpha_gt.png")
    Image.fromarray(gray_to_u8(r["alpha"])).save(out / "alpha_est.png")

    m = proj.metrics(gt_alpha=gt)
    report["01_standard"] = {
        **m,
        "weak_stroke_recall": _weak_recall(r["alpha"], gt),
        **_thin_line_stats(r["alpha"], gt),
        "ink_color_max_err": float(np.abs(r["ink_color"] - data["ink_gt"]).max()),
    }

    # 2) 给定真值纸色，隔离纸色场误差
    proj_gtp = Project(image=data["image"], paper_override=data["paper_gt"])
    r2 = proj_gtp.evaluate()
    report["02_gt_paper"] = {
        **proj_gtp.metrics(gt_alpha=gt),
        "weak_stroke_recall": _weak_recall(r2["alpha"], gt),
    }

    # 3) 重铺两种底色后的失墨检查（深底/浅底）
    miss = r["missing"]
    # 故意提高纸纹门限剪掉浅笔触，失墨图必须能定位被压制的笔迹
    strict = Project(image=data["image"],
                     params=Params(grain_sigma_k=6.0)).evaluate()
    weak_zone = (gt > 0.05) & (gt <= 0.35)
    clean_zone = gt < 0.01
    lift = float(strict["missing"][weak_zone].mean()
                 / max(strict["missing"][clean_zone].mean(), 1e-9))
    report["03_rebase_missing_ink"] = {
        "missing_mean": float(miss.mean()),
        "missing_p99": float(np.quantile(miss, 0.99)),
        "missing_over_5pct_fraction": float((miss > 0.05).mean()),
        "strict_gate_weak_recall": float(
            np.clip(strict["alpha"][weak_zone] / gt[weak_zone], 0, 1).mean()),
        "strict_gate_missing_lift": lift,
    }
    Image.fromarray(to_u8(views["missing_dark"])).save(out / "missing_dark.png")
    Image.fromarray(to_u8(views["missing_light"])).save(out / "missing_light.png")

    # 4) 污染边素材：换底不得出现白边
    out4 = out_root / "04_fringe"
    out4.mkdir(parents=True, exist_ok=True)
    data_f = generate(SyntheticSpec(seed=11, fringe=True))
    proj_f = Project(image=data_f["image"], params=Params())
    vf = proj_f.views()
    for k in ("composite", "composite_dark", "composite_light", "original"):
        Image.fromarray(to_u8(vf[k])).save(out4 / f"{k}.png")
    report["04_fringe"] = {
        **proj_f.metrics(gt_alpha=data_f["alpha_gt"][..., 0]),
        **_fringe_halo(vf["composite_light"], vf["composite_dark"], vf["alpha"]),
    }

    # 5) 与乘法混合对照：把扫描图直接乘新底色时，原纸色 P 会作为缩放
    #    因子继续参与每个像素；本模型合成只让新底色 under 参与。
    h, w = data["image"].shape[:2]
    dark = np.full((h, w, 3), 0.25, np.float32)
    # 朴素乘法工作流：I / P_white * 新底色（P_white 取纸张基色估计）
    paper_ref = np.clip(r["paper"], 0.2, 1.0)
    mult = np.clip(data["image"] / paper_ref * dark, 0, 1)
    proper = composite(dark, r["ink_color"], r["alpha"])
    # 漂移度量：无墨像素在乘法结果中仍残留旧纸色的色相/明度差异
    clean = gt < 0.02
    mult_drift = float(np.abs(mult[clean] - dark[clean]).mean())
    model_drift = float(np.abs(proper[clean] - dark[clean]).mean())
    report["05_multiply_baseline_drift"] = {
        "multiply_clean_region_drift": mult_drift,
        "model_clean_region_drift": model_drift,
        "drift_ratio": mult_drift / max(model_drift, 1e-9),
    }
    Image.fromarray(to_u8(mult)).save(out / "multiply_dark.png")
    Image.fromarray(to_u8(proper)).save(out / "model_dark.png")

    # 6) 二值化对照：量化二值化对浅笔触的破坏
    binary = (r["alpha"] > 0.5).astype(np.float32)
    report["06_binarization_damage"] = {
        "weak_pixels_destroyed_fraction":
            float(((gt > 0.05) & (gt <= 0.35) & (binary < 0.05)).sum()
                  / max(((gt > 0.05) & (gt <= 0.35)).sum(), 1)),
        "alpha_mae_binary": float(np.abs(binary - gt).mean()),
        "alpha_mae_model": float(np.abs(r["alpha"] - gt).mean()),
    }

    # 7) 参数变化重新求值：原件不变，结果随参数可逆更新
    before = data["image"].copy()
    proj.params.grain_sigma_k = 1.5
    proj.invalidate()
    a1 = proj.evaluate()["alpha"].copy()
    proj.params.grain_sigma_k = 6.0
    proj.invalidate()
    a2 = proj.evaluate()["alpha"].copy()
    report["07_recompute"] = {
        "original_unchanged": bool(np.array_equal(before, proj.image)),
        "alpha_changes_with_params": bool(np.abs(a1 - a2).mean() > 1e-4),
        "alpha_mean_k1_5": float(a1.mean()),
        "alpha_mean_k6_0": float(a2.mean()),
    }

    # 8) 用户标记：可靠纸区补充 + 保护细节，弱笔触召回应不低于无标记
    proj_base = Project(image=data["image"], params=Params())
    base_recall = _weak_recall(proj_base.evaluate()["alpha"], gt)

    proj_mark = Project(image=data["image"], params=Params())
    # 补充可靠纸区只从自动判定为纸的像素中采样，避免把浅墨钳成纸色
    auto = proj_mark._auto_reliable()
    ys, xs = np.where(auto)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ys), size=min(400, len(ys)), replace=False)
    proj_mark.mark_reliable(list(zip(xs[sel], ys[sel])), radius=6)
    # 保护浅铅笔排线区域（合成素材右中部），强制该区不被当纸色观测
    proj_mark.mark_protect([(280, 180), (300, 220), (270, 250)], radius=28)
    rm = proj_mark.evaluate()
    weak_in_protect = (gt > 0.05) & (gt <= 0.35) & proj_mark.protect_mask
    report["08_user_marks"] = {
        "weak_stroke_recall": _weak_recall(rm["alpha"], gt),
        "baseline_weak_recall": base_recall,
        "protected_weak_pixels": int(weak_in_protect.sum()),
        "protected_weak_mean_alpha_ratio": float(
            np.clip(rm["alpha"][weak_in_protect] / gt[weak_in_protect], 0, 1).mean()
            if weak_in_protect.any() else 0.0),
        "protected_region_mean_alpha":
            float(rm["alpha"][proj_mark.protect_mask].mean()),
        "original_unchanged": bool(np.array_equal(data["image"], proj_mark.image)),
    }
    vm = proj_mark.views()
    out8 = out_root / "08_marks"
    out8.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_u8(vm["composite"])).save(out8 / "composite.png")
    Image.fromarray(gray_to_u8(rm["alpha"])).save(out8 / "alpha.png")
    Image.fromarray((proj_mark.protect_mask.astype(np.uint8) * 255)
                    ).save(out8 / "protect_mask.png")
    Image.fromarray((proj_mark.reliable_mask.astype(np.uint8) * 255)
                    ).save(out8 / "reliable_mask.png")

    (out_root / "report.json").write_text(json.dumps(report, indent=2,
                                                      ensure_ascii=False))
    return report
