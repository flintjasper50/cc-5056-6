"""生成带真值的扫描素材：已知墨层 (K, alpha) 与有色纸底 P。

流程（输出全部记录，供提取误差核对）：
  1. 2x 超采样画布绘制抗锯齿墨线（粗细不一 + 浅灰铅笔线 + 45° 细线）
  2. 与彩色纸纹 P 按 I=(1-a)P+aK 合成
  3. 叠加光照阴影与传感器噪声，降采样到目标分辨率（产生分数 alpha 边缘）
另含边缘纸色污染（fringe）场景：墨缘混入旧纸色，用于检查换底白边。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .util import box_blur


@dataclass
class SyntheticSpec:
    width: int = 512
    height: int = 384
    paper_color: tuple = (0.93, 0.90, 0.80)   # 暖米色纸
    ink_color: tuple = (0.10, 0.09, 0.12)
    grain_sigma: float = 0.018                 # 纸纹强度
    shadow_strength: float = 0.12              # 光照阴影
    sensor_noise: float = 0.006
    seed: int = 7
    fringe: bool = False                       # 墨缘纸色污染


def _make_paper(spec: SyntheticSpec, ss: int) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    h, w = spec.height * ss, spec.width * ss
    base = np.tile(np.array(spec.paper_color, np.float32), (h, w, 1))
    # 大尺度纸色不均（云状）
    cloud = rng.normal(size=(h // 16 + 1, w // 16 + 1, 1)).astype(np.float32)
    cloud = np.asarray(Image.fromarray((cloud[..., 0] * 127 + 128).astype(np.uint8))
                       .resize((w, h), Image.BICUBIC), np.float32)[..., None] / 255 - 0.5
    # 细纸纹
    fine = rng.normal(scale=spec.grain_sigma, size=(h, w, 1)).astype(np.float32)
    # 纤维状条纹
    fibers = rng.normal(size=(h, 1, 1)).astype(np.float32) * 0.01
    paper = base + cloud * 0.05 + fine + fibers
    return np.clip(paper, 0, 1)


def _draw_ink(spec: SyntheticSpec, ss: int) -> np.ndarray:
    """返回 2x 画布上的覆盖率 alpha（H,W,1）。"""
    h, w = spec.height * ss, spec.width * ss
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)

    def line(p0, p1, width_px, alpha=255):
        d.line([p0[0] * ss, p0[1] * ss, p1[0] * ss, p1[1] * ss],
               fill=alpha, width=max(1, int(round(width_px * ss))),
               joint="curve")

    def ellipse(c, rxy, width_px, alpha=255):
        x0, y0 = (c[0] - rxy[0]) * ss, (c[1] - rxy[1]) * ss
        x1, y1 = (c[0] + rxy[0]) * ss, (c[1] + rxy[1]) * ss
        d.ellipse([x0, y0, x1, y1], outline=alpha,
                  width=max(1, int(round(width_px * ss))))

    # 粗主轮廓
    line((30, 300), (30, 90), 4.0)
    line((30, 90), (180, 40), 4.0)
    line((180, 40), (330, 90), 4.0)
    line((330, 90), (330, 300), 4.0)
    line((30, 300), (330, 300), 4.0)
    # 中等内线
    ellipse((180, 190), (90, 60), 2.0)
    line((90, 190), (270, 190), 1.4)
    # 细笔触（1px -> 降采样后 ~0.5 px，含分数 alpha）
    for i, y in enumerate(range(120, 290, 18)):
        line((46, y), (80 + (i % 3) * 6, y), 1.0)
    # 浅灰铅笔排线（低强度，不能丢）
    for i in range(9):
        line((250, 120 + i * 16), (315, 105 + i * 16), 1.2, alpha=70)
    # 45° 细交叉线（抗锯齿极限测试）
    for k in range(-6, 7):
        line((120 + k * 22, 310), (200 + k * 22, 230), 1.0, alpha=160)
    # 极淡点触
    for i in range(40):
        rng = np.random.default_rng(100 + i)
        x, y = rng.uniform(360, 490), rng.uniform(40, 330)
        d.ellipse([x * ss - 1, y * ss - 1, x * ss + 1, y * ss + 1], fill=55)

    return np.asarray(layer, np.float32)[..., None] / 255.0


def _add_shadow(paper: np.ndarray, ss: int, strength: float) -> np.ndarray:
    h, w = paper.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    # 单侧渐变 + 角落暗斑，模拟纸张弯曲/光照不均
    grad = 1 - strength * (xx / w) * 0.7
    spot = 1 - strength * 0.6 * np.exp(
        -(((xx - 0.8 * w) / (0.25 * w)) ** 2 + ((yy - 0.2 * h) / (0.2 * h)) ** 2))
    shade = (grad * spot)[..., None]
    return np.clip(paper * shade, 0, 1)


def generate(spec: SyntheticSpec | None = None) -> dict:
    spec = spec or SyntheticSpec()
    ss = 2
    paper_hi = _make_paper(spec, ss)
    alpha_hi = _draw_ink(spec, ss)
    ink = np.array(spec.ink_color, np.float32)

    if spec.fringe:
        # 墨缘污染：在墨线外圈把纸色朝旧纸基色推，模拟扫描时墨边夹带纸色
        edge = alpha_hi - box_blur(alpha_hi, 2)
        edge = np.clip(edge, 0, 1)
        old_tint = np.array(spec.paper_color, np.float32)
        paper_hi = paper_hi * (1 - edge * 0.6) + old_tint * (edge * 0.6)

    comp_hi = (1 - alpha_hi) * paper_hi + alpha_hi * ink
    paper_sh = _add_shadow(paper_hi, ss, spec.shadow_strength)
    # 阴影只作用于纸面（被墨覆盖处仍由墨主导），加传感器噪声
    img_hi = (1 - alpha_hi) * paper_sh + alpha_hi * ink
    rng = np.random.default_rng(spec.seed + 1)
    img_hi = img_hi + rng.normal(scale=spec.sensor_noise, size=img_hi.shape)
    img_hi = np.clip(img_hi, 0, 1)

    def down(a: np.ndarray) -> np.ndarray:
        single = a.ndim == 2 or a.shape[-1] == 1
        flat = a[..., 0] if single else a
        im = Image.fromarray((np.clip(flat, 0, 1) * 255 + 0.5).astype(np.uint8))
        im = im.resize((spec.width, spec.height), Image.LANCZOS)
        out = np.asarray(im, np.float32) / 255.0
        return out[..., None] if single else out

    return {
        "image": down(img_hi),
        "paper_gt": down(paper_sh),
        "alpha_gt": down(alpha_hi),
        "ink_gt": ink,
        "spec": spec,
    }
