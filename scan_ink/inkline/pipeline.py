"""扫描线稿着色合成器主编排。

Project 持有原件（永不被覆盖）、用户标记（可靠纸区/保护细节）与参数；
任何参数或标记变化都从原件重新求值，不做破坏性改写。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .model import (Params, estimate_paper_field, estimate_ink_color,
                    estimate_ink, render_ink_layer, composite, error_metrics)
from .regions import RegionLayer
from .util import to_float, to_u8, gray_to_u8, overlay_heat_u8


@dataclass
class Project:
    image: np.ndarray                 # 原件 RGB float
    name: str = "scan"
    params: Params = field(default_factory=Params)
    reliable_mask: np.ndarray | None = None    # 用户标记的可靠纸区
    protect_mask: np.ndarray | None = None     # 用户标记的需保留细节
    regions: RegionLayer | None = None
    result: dict | None = None
    paper_override: np.ndarray | None = None   # 外部给定纸色（测试用）

    # ---------- 标记 ----------
    def _ensure_masks(self) -> None:
        h, w = self.image.shape[:2]
        if self.reliable_mask is None:
            self.reliable_mask = np.zeros((h, w), bool)
        if self.protect_mask is None:
            self.protect_mask = np.zeros((h, w), bool)

    def mark_reliable(self, points: list[tuple], radius: int = 12,
                      additive: bool = True) -> None:
        self._stroke("reliable_mask", points, radius, additive)

    def mark_protect(self, points: list[tuple], radius: int = 8,
                     additive: bool = True) -> None:
        self._stroke("protect_mask", points, radius, additive)

    def _stroke(self, attr: str, points, radius, additive) -> None:
        self._ensure_masks()
        target = getattr(self, attr)
        h, w = self.image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        stamp = np.zeros((h, w), bool)
        for x, y in points:
            stamp |= (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        if additive:
            target |= stamp
        else:
            target &= ~stamp
        self.result = None  # 标记变化 -> 下次取值时重算

    def clear_marks(self) -> None:
        self._ensure_masks()
        self.reliable_mask.fill(False)
        self.protect_mask.fill(False)
        self.result = None

    # ---------- 核心求值 ----------
    def _auto_reliable(self) -> np.ndarray:
        """无用户标记时的冷启动可靠纸区：取局部亮度峰值附近的像素。

        不用全局固定分位——那样阴影下的纸面会被整体排除、形成大片低
        可信填充区。局部最大值滤波给出每个位置的"纸色亮度上界"，仅
        相对该上界显著变暗处才可能是墨，其余纸面（含阴影区）均可作为
        纸色观测；保护细节区永远排除。
        """
        from .util import percentile_filter
        lum = np.dot(self.image, (0.299, 0.587, 0.114))
        local_peak = percentile_filter(lum[..., None], 15, 0.98)[..., 0]
        rel = lum >= local_peak - 0.06
        if self.protect_mask is not None:
            rel &= ~self.protect_mask
        return rel

    def evaluate(self) -> dict:
        if self.result is not None:
            return self.result
        self._ensure_masks()
        p = self.params
        # 用户标记是自动纸区的补充而非替代：冷启动先取局部峰值纸区，
        # 用户再把遗漏的可信纸面补进来；保护细节始终排除在纸区观测外
        reliable = self._auto_reliable()
        if self.reliable_mask.any():
            reliable = reliable | self.reliable_mask
        reliable = reliable & ~self.protect_mask

        if self.paper_override is not None:
            paper = self.paper_override
            confidence = np.ones(self.image.shape[:2], np.float32)
        else:
            paper, confidence = estimate_paper_field(self.image, reliable, p)
        ink_color = estimate_ink_color(self.image, paper, reliable,
                                       p.ink_dark_fraction)
        ink = estimate_ink(self.image, paper, confidence, ink_color, reliable, p)

        # 保护细节区：残差越过局部纸纹门限时强制保留弱墨（只抬不压），
        # 门限之内的纸纹噪声不抬，避免保护区凭空产生假笔迹
        if self.protect_mask.any():
            from .model import _local_residual_scale
            scale = _local_residual_scale(
                self.image, paper, reliable, max(3, p.strength_radius))
            darkening = np.clip(
                np.dot(paper - self.image, (0.299, 0.587, 0.114)), 0, None)
            gate = np.clip((darkening - p.grain_sigma_k * scale) /
                           np.maximum(scale, 1e-6), 0, 1)
            boost = gate * self.protect_mask
            a2 = np.maximum(ink["alpha"], boost * 0.95)
            ink["alpha"] = a2.astype(np.float32)
            ink["coverage"] = np.clip(a2 / ink["strength"], 0, 1).astype(np.float32)
            ink["recomp_orig"] = ((1 - a2[..., None]) * paper
                                  + a2[..., None] * ink_color)

        self.result = {
            "paper": paper,
            "confidence": confidence,
            "ink_color": ink_color,
            "reliable": reliable,
            **ink,
        }
        return self.result

    def invalidate(self) -> None:
        self.result = None

    # ---------- 视图 ----------
    def views(self, base_dark: np.ndarray | None = None,
              base_light: np.ndarray | None = None) -> dict:
        r = self.evaluate()
        h, w = self.image.shape[:2]
        if base_dark is None:
            base_dark = np.tile(np.array([0.25, 0.27, 0.32], np.float32), (h, w, 1))
        if base_light is None:
            base_light = np.tile(np.array([0.97, 0.96, 0.93], np.float32), (h, w, 1))
        region_layer = self.regions.render() if self.regions is not None else base_light

        ink_gray = render_ink_layer(r["ink_color"], r["alpha"],
                                    bg=np.full((h, w, 3), 0.5, np.float32))
        comp_final = composite(region_layer, r["ink_color"], r["alpha"])
        comp_dark = composite(base_dark, r["ink_color"], r["alpha"])
        comp_light = composite(base_light, r["ink_color"], r["alpha"])

        # 失墨信号来自模型对原件的单向暗化残差，与新底色无关；
        # 深/浅两视图只是把同一信号叠在不同底色上，便于对照定位
        miss = r["missing"]
        gray = np.dot(self.image, (0.299, 0.587, 0.114))
        return {
            "original": self.image,
            "paper_field": r["paper"],
            "confidence": r["confidence"],
            "ink_layer_gray": ink_gray,
            "alpha": r["alpha"],
            "coverage": r["coverage"],
            "strength": r["strength"],
            "composite": comp_final,
            "composite_dark": comp_dark,
            "composite_light": comp_light,
            "recomp_orig": r["recomp_orig"],
            "residual_heat": overlay_heat_u8(gray, r["residual_mag"], 0.12) / 255.0,
            "missing_heat": overlay_heat_u8(gray, miss, 0.08) / 255.0,
            "missing_dark": overlay_heat_u8(
                np.dot(comp_dark, (0.299, 0.587, 0.114)), miss, 0.08) / 255.0,
            "missing_light": overlay_heat_u8(
                np.dot(comp_light, (0.299, 0.587, 0.114)), miss, 0.08) / 255.0,
            "region_layer": region_layer,
        }

    def metrics(self, gt_alpha: np.ndarray | None = None) -> dict:
        r = self.evaluate()
        return error_metrics(self.image, r["recomp_orig"], r["alpha"], gt_alpha)

    # ---------- 存取 ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        Image.fromarray(to_u8(self.image)).save(path / "original.png")
        self._save_gray(self.reliable_mask, path / "reliable.png")
        self._save_gray(self.protect_mask, path / "protect.png")
        payload = {"name": self.name, "params": self.params.to_dict()}
        if self.regions is not None:
            payload["regions"] = [
                {"name": x.name, "color": list(x.color), "feather": x.feather}
                for x in self.regions.regions]
            payload["default_color"] = list(self.regions.default_color)
            for i, reg in enumerate(self.regions.regions):
                if reg.mask is not None:
                    Image.fromarray(to_u8(reg.mask)).save(path / f"region_{i}.png")
        (path / "project.json").write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        image = to_float(Image.open(path / "original.png"))
        data = json.loads((path / "project.json").read_text())
        proj = cls(image=image, name=data.get("name", "scan"),
                   params=Params.from_dict(data.get("params", {})))
        if (path / "reliable.png").exists():
            proj.reliable_mask = np.asarray(Image.open(path / "reliable.png")) > 127
        if (path / "protect.png").exists():
            proj.protect_mask = np.asarray(Image.open(path / "protect.png")) > 127
        if "regions" in data:
            h, w = image.shape[:2]
            rl = RegionLayer(w, h, tuple(data.get("default_color",
                                                    (0.96, 0.95, 0.92))))
            for i, rd in enumerate(data["regions"]):
                mp = path / f"region_{i}.png"
                if mp.exists():
                    mask = np.asarray(Image.open(mp), np.float32) / 255.0
                    rl.add_mask(rd["name"], tuple(rd["color"]), mask,
                                feather=0)
            proj.regions = rl
        return proj

    @staticmethod
    def _save_gray(mask, path: Path) -> None:
        if mask is None:
            return
        Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
