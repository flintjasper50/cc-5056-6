"""区域颜色层：位于墨层之下，可编辑且与纸色场完全解耦。"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .util import to_u8


@dataclass
class Region:
    name: str
    color: tuple                     # 0-1 RGB
    mask: np.ndarray | None = None   # bool/0-1 (H,W)
    feather: float = 1.2             # 羽化半径 px，避免区域边界硬接缝


@dataclass
class RegionLayer:
    width: int
    height: int
    default_color: tuple = (0.96, 0.95, 0.92)
    regions: list[Region] = field(default_factory=list)

    def add_polygon(self, name: str, color: tuple, points: list[tuple],
                    feather: float = 1.2) -> None:
        m = Image.new("L", (self.width, self.height), 0)
        ImageDraw.Draw(m).polygon([(int(x), int(y)) for x, y in points], fill=255)
        if feather > 0:
            m = m.filter(ImageFilter.GaussianBlur(feather))
        arr = np.asarray(m, np.float32) / 255.0
        self.regions.append(Region(name, tuple(color), arr, feather))

    def add_mask(self, name: str, color: tuple, mask: np.ndarray,
                 feather: float = 1.2) -> None:
        m = Image.fromarray(to_u8(np.clip(mask, 0, 1)))
        if feather > 0:
            m = m.filter(ImageFilter.GaussianBlur(feather))
        self.regions.append(
            Region(name, tuple(color), np.asarray(m, np.float32) / 255.0, feather))

    def render(self) -> np.ndarray:
        layer = np.tile(np.array(self.default_color, np.float32),
                        (self.height, self.width, 1))
        for r in self.regions:
            if r.mask is None:
                continue
            w = r.mask[..., None]
            layer = layer * (1 - w) + np.array(r.color, np.float32) * w
        return np.clip(layer, 0, 1).astype(np.float32)
