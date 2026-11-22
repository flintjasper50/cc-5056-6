"""显式扫描成像模型与线稿提取。

成像模型（逐通道 c）：
    I = (1 - b) * P + b * K
其中
    P : 局部纸色场（3 通道，含纸纹），由可靠纸区扩散/平滑估计
    K : 全局墨色（深灰或带色墨），由暗像素稳健估计
    b : 有效墨覆盖率 alpha，抗锯齿边缘取分数值；浅笔触表现为小的 b，
        而不是被截断成 0 或被二值化

若进一步区分覆盖率 a 与笔触强度 t（笔压），b = a*t：
    E = (1-t)P + tK 为局部有效墨色，I = (1-a)P + aE
本模块同时输出 b（合成实际使用）、a、t 三张图，灰度笔触的去向可复核。

所有估计都保留残差图与可信度图，不做"亮像素直接透明"的硬截断。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from .util import box_blur, box_blur_masked, percentile_filter


@dataclass
class Params:
    # 纸色场
    paper_radius: int = 6          # 扩散填充平滑半径
    paper_blur_radius: int = 2     # 最终接缝平滑
    grain_quantile: float = 0.85   # 可靠纸区取亮侧分位，避免墨点拉低纸色
    # 墨色
    ink_dark_fraction: float = 0.02  # 最暗的多少比例像素参与墨色估计
    # 墨 alpha
    strength_radius: int = 5       # 局部分母 t 的最大值窗口
    grain_sigma_k: float = 2.0     # 纸纹保护：低于 k*局部尺度 的差异视为纸纹
    b_floor: float = 0.0           # 不设硬底，浅笔触保留
    b_ceiling: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def estimate_ink_color(img: np.ndarray, paper: np.ndarray,
                       reliable: np.ndarray,
                       dark_fraction: float = 0.02) -> np.ndarray:
    """从全图最暗的一簇像素稳健估计全局墨色 K（逐通道中位数）。

    墨核在扫描图中必然落在亮度最低的 dark_fraction 分位里，取低百分位
    可避免抗锯齿边缘（中间灰）把墨色拉亮。同时要求候选显著暗于局部
    纸色，防止深色纸底被误判成墨。
    """
    lum_i = np.dot(img, (0.299, 0.587, 0.114))
    lum_p = np.dot(paper, (0.299, 0.587, 0.114))
    cutoff = np.quantile(lum_i, dark_fraction)
    dark = lum_i <= cutoff
    # 深色纸底保护：候选必须比纸色明显暗，否则只保留相对变暗最强的像素
    if np.median(lum_p[dark]) - np.median(lum_i[dark]) < 0.15:
        darkening = lum_p - lum_i
        dark = darkening >= np.quantile(darkening, 1 - dark_fraction)
    pts = img[dark]
    return np.median(pts, axis=0)


def estimate_paper_field(img: np.ndarray, reliable: np.ndarray,
                         p: Params) -> tuple[np.ndarray, np.ndarray]:
    """估计纸色场 P 与逐像素可信度 c。

    纸色场分两层构造：
      1. 低频基色：reliable 像素以软锚定（有限权重拉向平滑观测）参与
         扩散迭代，墨迹/阴影/保护细节等空洞完全由邻域调和填充。软锚定
         而非硬钳制，即使用户把浅墨误标成纸也不会把笔迹吸进纸色场；
      2. 纸纹包络：可靠区内保留观测相对基色的残差（经分位滤波去除细小
         墨点），空洞处由扩散填充。这样浅纸纹不会在后续被误判成浅墨，
         灰墨与纸纹相交处的判决也有明确的局部噪声尺度可依。
    可信度 c=1 为直接观测，远离观测区按扩散距离指数下降。
    """
    mean_col = img[reliable].mean(axis=0) if reliable.any() else img.mean(axis=(0, 1))
    # 可靠像素的纸色观测先做亮侧分位滤波：去除纸纹低谷与混入的细小
    # 暗点，得到平滑的纸面观测；用户标记点同样经过此处理，避免标记
    # 处纸纹尖峰把附近浅墨拉入纸色场
    img_smooth = percentile_filter(img, p.paper_radius * 2 + 1, p.grain_quantile)
    obs = np.where(reliable[..., None], img_smooth, mean_col.astype(np.float32))
    base = obs.astype(np.float32)

    # 1) 低频基色：软锚定扩散。可靠像素不做硬钳制（否则用户误标到浅
    #    墨上会把笔迹强制吸进纸色场、永久丢墨），而是以有限权重 w 拉向
    #    平滑观测，与邻域扩散项联合迭代；空洞完全由邻居调和填充
    w = (reliable.astype(np.float32) * 0.25)[..., None]
    for r, iters in ((p.paper_radius * 2, 60),
                     (p.paper_radius, 80),
                     (max(2, p.paper_radius // 2), 60)):
        for _ in range(iters):
            neigh = box_blur(base, r)
            base = (neigh + w * obs) / (1.0 + w)
    base = box_blur(base, p.paper_blur_radius)

    # 2) 纸纹包络：可靠区内观测残差（基于平滑观测，抑制细墨点），
    #    空洞处按掩码加权扩散；可靠处同样软锚定，避免误标吸墨
    resid_q = img_smooth - base
    env = np.where(reliable[..., None], resid_q, np.zeros_like(base))
    env_mask = reliable.astype(np.float32)
    for r, iters in ((p.paper_radius, 80), (max(2, p.paper_radius // 2), 60)):
        for _ in range(iters):
            wsum = np.maximum(box_blur(env_mask, r), 1e-6)
            filled = box_blur(env * env_mask[..., None], r) / wsum[..., None]
            env = (filled + w * resid_q) / (1.0 + w)
    paper = np.clip(base + env, 0, 1)

    # 可信度：观测掩码向外扩散，按距最近观测的扩散层数指数衰减
    reach = reliable.astype(np.float32)
    distance = np.zeros(reliable.shape, np.float32)
    step = max(1, p.paper_radius // 2)
    for layer in range(1, 200):
        grown = (box_blur(reach, step) > 0).astype(np.float32)
        newly = (reach <= 0) & (grown > 0)
        if not newly.any():
            break
        distance[newly] = layer
        reach = grown
    confidence = np.clip(0.97 ** distance, 0, 1).astype(np.float32)
    return paper, confidence


def _local_grain_sigma(img: np.ndarray, paper: np.ndarray,
                       reliable: np.ndarray) -> float:
    """可靠纸区内观测值相对纸色场的残差尺度（全局 MAD）。"""
    if reliable.sum() < 16:
        return 2 / 255.0
    resid = img - paper
    sig = np.sqrt((resid ** 2).sum(axis=2))[reliable]
    return float(np.median(sig) * 1.4826 + 1e-6)


def _local_residual_scale(img: np.ndarray, paper: np.ndarray,
                          reliable: np.ndarray, radius: int) -> np.ndarray:
    """逐像素的纸纹残差尺度（与纸色场同平滑尺度，向量距离量纲）。

    纸色场经平滑后不含细纸纹，残差的典型幅度由纸纹+填充误差共同决定。
    墨迹产生的大幅暗化残差不能计入噪声地板，否则门限被抬高、浅笔触被
    剪掉：先按暗化投影排除显著墨像素，再在剩余纸面像素上用局部
    中位 + 1*MAD 估计尺度；墨洞处由邻居扩散填补。
    """
    mag = np.sqrt(np.maximum(((img - paper) ** 2).sum(axis=2), 0))
    darkening = np.clip(np.dot(paper - img, (0.299, 0.587, 0.114)), 0, None)
    paper_pixels = reliable & (darkening <= np.quantile(darkening[reliable], 0.8)
                               if reliable.any() else True)
    pm = paper_pixels.astype(np.float32)
    local_med = box_blur_masked(mag, pm, radius)
    dev = np.abs(mag - local_med)
    local_mad = box_blur_masked(dev, pm, radius)
    scale = local_med + 1.0 * local_mad
    global_scale = float(np.median(mag[paper_pixels])) if paper_pixels.any() else 0.02
    scale = np.where(paper_pixels, scale, global_scale)
    scale = box_blur_masked(scale, np.ones_like(reliable, np.float32), radius)
    return np.maximum(scale.astype(np.float32), 1e-3)


def estimate_ink(img: np.ndarray, paper: np.ndarray, confidence: np.ndarray,
                 ink_color: np.ndarray, reliable: np.ndarray,
                 p: Params) -> dict:
    """求解有效墨 alpha b，并拆分覆盖率 a 与笔触强度 t。

    b* = <P-I, P-K> / |P-K|^2，按物理边界裁剪到 [0,1]：
      - I 比 P 更亮（纸纹高光/噪声）时 b* 为负 -> 归 0，纸纹不被当墨
      - I 比 K 更暗时 b* >1 -> 归 1
    纸纹保护带内（|I-P| <= k*sigma）按可信度软判决，避免灰墨与浅纸纹
    相交处产生虚假笔迹或丢笔。
    """
    pk = paper - ink_color
    pi = paper - img
    denom = np.maximum((pk ** 2).sum(axis=2), 1e-8)
    b_star = (pk * pi).sum(axis=2) / denom

    sigma = _local_grain_sigma(img, paper, reliable)
    local_scale = _local_residual_scale(
        img, paper, reliable, max(3, p.strength_radius))
    dist = np.sqrt(np.maximum((pi ** 2).sum(axis=2), 0))

    # 纸纹门限：残差幅度低于 k*局部纸纹尺度 视为纸纹/填充误差，alpha
    # 归零（基于模型残差的统计门限，而非按亮度硬切亮像素）；门限之上
    # 线性放开，低可信度区额外收缩，避免填充区制造假笔迹。
    gate = np.clip((dist - p.grain_sigma_k * local_scale) /
                   np.maximum(local_scale, 1e-6), 0, 1)
    b = np.clip(b_star, p.b_floor, p.b_ceiling)
    b = b * gate * np.clip(0.25 + 0.75 * confidence, 0, 1)
    b = np.clip(b, 0, 1).astype(np.float32)

    # 拆分 a / t：假设笔触核芯处 a≈1，局部 b 最大值估计该处笔压 t
    t_hat = percentile_filter(b[..., None], p.strength_radius * 2 + 1, 0.95)[..., 0]
    t_hat = np.maximum(t_hat, 0.05).astype(np.float32)
    a = np.clip(b / t_hat, 0, 1).astype(np.float32)

    # 用模型重算观测并留残差（失墨/模型失配的可复核证据）
    recomp = (1 - b[..., None]) * paper + b[..., None] * ink_color
    residual = img - recomp
    residual_mag = np.sqrt((residual ** 2).sum(axis=2)).astype(np.float32)
    # 单向"失墨"：原图比重建更暗，说明有墨未被 alpha 捕获。做小半径
    # 空间聚合，孤立传感器噪声被平均掉，沿笔触分布的结构性失墨被保留
    missing_raw = np.clip((recomp - img).mean(axis=2), 0, None)
    missing = box_blur(missing_raw.astype(np.float32), 1).astype(np.float32)

    return {
        "alpha": b,            # 有效覆盖率 b = a*t，合成用这张
        "coverage": a,         # 几何覆盖率
        "strength": t_hat,     # 笔触强度/笔压
        "residual": residual.astype(np.float32),
        "residual_mag": residual_mag,
        "missing": missing,
        "grain_sigma": sigma,
        "recomp_orig": recomp.astype(np.float32),
    }


def render_ink_layer(ink_color: np.ndarray, alpha: np.ndarray,
                     bg: np.ndarray | None = None) -> np.ndarray:
    """把墨层渲染到指定底色上（默认中灰，便于观察浅笔触）。"""
    if bg is None:
        bg = np.full((*alpha.shape, 3), 0.5, np.float32)
    return ((1 - alpha[..., None]) * bg + alpha[..., None] * ink_color).astype(np.float32)


def composite(paper_base: np.ndarray, ink_color: np.ndarray,
              alpha: np.ndarray, region_layer: np.ndarray | None = None) -> np.ndarray:
    """在新底色上合成：区域色在下，墨层在上。

    墨层公式 out=(1-b)*under + b*K，under 永远是新底色，
    原纸色 P 不参与混合，因此边缘不存在夹带旧纸色的白边。
    """
    under = paper_base if region_layer is None else region_layer
    out = (1 - alpha[..., None]) * under + alpha[..., None] * ink_color
    return np.clip(out, 0, 1).astype(np.float32)


def error_metrics(orig: np.ndarray, recomp: np.ndarray,
                  alpha: np.ndarray, gt_alpha: np.ndarray | None = None) -> dict:
    """提取误差指标；有合成真值时额外比较 alpha。"""
    err = np.abs(orig - recomp).mean(axis=2)
    out = {
        "mae_rgb": float(err.mean()),
        "p99_rgb_err": float(np.quantile(err, 0.99)),
        "max_rgb_err": float(err.max()),
        "alpha_coverage": float((alpha > 0.02).mean()),
    }
    if gt_alpha is not None:
        out["alpha_mae"] = float(np.abs(alpha - gt_alpha).mean())
        weak = (gt_alpha > 0.05) & (gt_alpha <= 0.35)
        if weak.any():
            out["alpha_weak_recall"] = float(
                np.clip(alpha[weak] / np.maximum(gt_alpha[weak], 1e-6), 0, 1).mean())
            out["alpha_weak_detected"] = float((alpha[weak] > 0.02).mean())
    return out
