"""命令行入口：合成真值素材、跑提取、导出全部诊断视图与误差报告。

用法：
  python -m inkline.cli demo --out output/demo
  python -m inkline.cli run  --input scan.png --out output/scan
  python -m inkline.cli test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .model import Params
from .pipeline import Project
from .regions import RegionLayer
from .synthetic import SyntheticSpec, generate
from .util import to_float, to_u8, gray_to_u8


def _save_views(proj: Project, out: Path, gt_alpha=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    views = proj.views()
    for name in ("original", "paper_field", "ink_layer_gray", "composite",
                 "composite_dark", "composite_light", "recomp_orig",
                 "region_layer", "residual_heat", "missing_heat",
                 "missing_dark", "missing_light"):
        Image.fromarray(to_u8(views[name])).save(out / f"{name}.png")
    for name in ("alpha", "coverage", "strength", "confidence"):
        Image.fromarray(gray_to_u8(views[name])).save(out / f"{name}.png")
    metrics = proj.metrics(gt_alpha)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def cmd_demo(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = SyntheticSpec(seed=args.seed, fringe=args.fringe)
    data = generate(spec)
    Image.fromarray(to_u8(data["image"])).save(out / "synthetic_input.png")
    Image.fromarray(to_u8(data["paper_gt"])).save(out / "synthetic_paper_gt.png")
    Image.fromarray(gray_to_u8(data["alpha_gt"][..., 0])).save(out / "synthetic_alpha_gt.png")

    proj = Project(image=data["image"], name="synthetic",
                   params=Params(paper_radius=args.radius, grain_sigma_k=args.grain_k))
    h, w = data["image"].shape[:2]
    rl = RegionLayer(w, h, default_color=(0.96, 0.95, 0.92))
    rl.add_polygon("sky", (0.62, 0.78, 0.92), [(0, 0), (w, 0), (w, h * 0.45), (0, h * 0.5)])
    rl.add_polygon("ground", (0.78, 0.88, 0.62),
                   [(0, h * 0.5), (w, h * 0.45), (w, h), (0, h)])
    rl.add_polygon("accent", (0.95, 0.72, 0.55),
                   [(w * 0.22, h * 0.35), (w * 0.45, h * 0.35),
                    (w * 0.45, h * 0.7), (w * 0.22, h * 0.7)])
    proj.regions = rl

    metrics = _save_views(proj, out, gt_alpha=data["alpha_gt"][..., 0])
    # 用真值纸色再跑一次，隔离纸色场误差的影响
    proj2 = Project(image=data["image"], name="synthetic-gt-paper",
                    paper_override=data["paper_gt"], params=Params(grain_sigma_k=args.grain_k))
    m2 = proj2.metrics(gt_alpha=data["alpha_gt"][..., 0])
    (out / "metrics_gt_paper.json").write_text(json.dumps(m2, indent=2))
    print(json.dumps({"estimated_paper": metrics, "gt_paper": m2}, indent=2))


def cmd_run(args) -> None:
    img = to_float(Image.open(args.input))
    proj = Project(image=img, name=Path(args.input).stem,
                   params=Params(paper_radius=args.radius, grain_sigma_k=args.grain_k))
    if args.project and Path(args.project).exists():
        saved = Project.load(args.project)
        proj.reliable_mask = saved.reliable_mask
        proj.protect_mask = saved.protect_mask
        proj.regions = saved.regions
        proj.params = saved.params
    print(json.dumps(_save_views(proj, Path(args.out)), indent=2))


def cmd_test(args) -> None:
    from .tests.run_tests import run_all
    run_all(Path(args.out))


def main() -> None:
    ap = argparse.ArgumentParser(description="扫描线稿着色合成器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default="output/run")
        p.add_argument("--radius", type=int, default=6)
        p.add_argument("--grain-k", type=float, default=2.0)

    d = sub.add_parser("demo", help="生成真值素材并完整提取")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--fringe", action="store_true")
    common(d)
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="处理真实/外部扫描图")
    r.add_argument("--input", required=True)
    r.add_argument("--project", default=None)
    common(r)
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("test", help="数值验证套件")
    t.add_argument("--out", default="output/tests")
    t.set_defaults(func=cmd_test)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
