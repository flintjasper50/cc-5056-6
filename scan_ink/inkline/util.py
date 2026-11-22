"""像素级处理原语：全部基于 numpy/PIL 真实运算，无二值化捷径。"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _integral(arr: np.ndarray) -> np.ndarray:
    """带零边的积分图，arr 可为 (H,W) 或 (H,W,C)。"""
    if arr.ndim == 2:
        pad = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), np.float64)
        pad[1:, 1:] = arr
        return pad.cumsum(0).cumsum(1)
    pad = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1, arr.shape[2]), np.float64)
    pad[1:, 1:] = arr
    return pad.cumsum(0).cumsum(1)


def box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    """半径 r 的盒式均值（直径 2r+1），边界按实际像素数归一。"""
    if radius <= 0:
        return arr.astype(np.float32, copy=False)
    h, w = arr.shape[:2]
    ii = _integral(arr.astype(np.float64, copy=False))
    ys = np.clip(np.arange(h) - radius, 0, h)
    ye = np.clip(np.arange(h) + radius + 1, 0, h)
    xs = np.clip(np.arange(w) - radius, 0, w)
    xe = np.clip(np.arange(w) + radius + 1, 0, w)
    total = (ii[np.ix_(ye, xe)] - ii[np.ix_(ys, xe)]
             - ii[np.ix_(ye, xs)] + ii[np.ix_(ys, xs)])
    count = (ye - ys)[:, None] * (xe - xs)[None, :]
    if total.ndim == 3:
        count = count[..., None]
    return (total / count).astype(np.float32)


def box_blur_masked(vals: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """仅对 mask>0 的像素求局部加权平均；空洞位置由邻居扩散填充。"""
    m = mask.astype(np.float32)
    wsum = box_blur(m, radius)
    wsum = np.maximum(wsum, 1e-6)
    if vals.ndim == 3:
        num = box_blur(vals * m[..., None], radius)
        return num / wsum[..., None]
    return box_blur(vals * m, radius) / wsum


def percentile_filter(arr: np.ndarray, size: int, quantile: float) -> np.ndarray:
    """逐通道二维分位滤波（用于可靠区内抑制细小墨点、估计局部 t 上限）。"""
    size = int(size)
    rank = int(round(np.clip(quantile, 0, 1) * (size * size - 1)))

    def _one(plane: np.ndarray) -> np.ndarray:
        im = Image.fromarray(np.clip(plane * 255, 0, 255).astype(np.uint8))
        out = im.filter(ImageFilter.RankFilter(size=size, rank=rank))
        return np.asarray(out, np.float32) / 255.0

    if arr.ndim == 2:
        return _one(arr)
    return np.stack([_one(arr[..., c]) for c in range(arr.shape[2])], axis=-1)


def to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), np.float32) / 255.0


def to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8)


def gray_to_u8(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.full(arr.shape, 128, np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255 + 0.5, 0, 255).astype(np.uint8)


def heatmap_u8(arr: np.ndarray, fixed_max: float | None = None) -> np.ndarray:
    """残差/失墨热图：蓝->青->黄->红。

    fixed_max 给定时按固定物理尺度归一（跨参数/跨视图可比），否则退化
    为 99 分位自适应。返回 (H,W,3) uint8。
    """
    lo = float(np.min(arr))
    hi = fixed_max if fixed_max is not None else float(np.quantile(arr, 0.99))
    x = np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1)
    r = np.clip(x * 3 - 1, 0, 1)
    g = np.clip(2 - np.abs(x * 3 - 1.5), 0, 1)
    b = np.clip(1 - x * 3, 0, 1)
    return (np.stack([r, g, b], -1) * 255 + 0.5).astype(np.uint8)


def overlay_heat_u8(gray_bg: np.ndarray, signal: np.ndarray,
                    fixed_max: float) -> np.ndarray:
    """把非负信号以红-黄热度叠加在灰度背景上，便于定位失墨/残差。"""
    g = np.clip(gray_bg, 0, 1)[..., None]
    base = np.repeat(g, 3, axis=2) * 0.75
    x = np.clip(signal / fixed_max, 0, 1)[..., None]
    hot = np.concatenate([
        np.clip(0.5 + x, 0, 1),
        np.clip(1.0 - x * 0.7, 0, 1),
        np.clip(0.25 - x * 0.25, 0, 1)], axis=2)
    out = base * (1 - x) + hot * x
    return (out * 255 + 0.5).astype(np.uint8)
