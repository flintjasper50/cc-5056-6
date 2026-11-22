"""透印 / 淡墨 / 网纹三层重合的受控测试图，及两路独立参照。

本模块与生产提取链路（model.py / pipeline.py）完全独立，专门用于墨线分离
验收，提供两套互不共享代码的"真值"来源：

1. 解析覆盖基准（analytic coverage baseline）
   每条线段在每个像素上的覆盖率由几何公式直接推出——1D 盒形线核与像素
   开口的相交长度（再沿线宽方向取同样的盒核覆盖），不经过任何绘制/滤波
   库。它给出"墨量应当是多少"的解析基准，用来核对提取出的覆盖率量级。

2. 独立超采样重合成（independent supersampling reference）
   在 4x 高分辨率画布上用与生产代码不同的纯 numpy 光栅化（同一物理盒核
   在密网格子像素中心求值 + 均值降采样）重新合成整幅扫描图，逐像素平均
   得到参照覆盖率 alpha_ref 与参照重合成图 recomp_ref。验收的期望值
   （覆盖率与背景残差）从这一路独立参照推导，而不是从被测函数的输出
   反推，避免自证。

成像次序（先到后，后者在上）：
    纸 P（含纸纹/阴影）
      -> 透印 S：背面墨迹，最浅、最弥散，乘法式轻度暗化
      -> 网纹 H：印刷网点，规则点阵的中等暗化
      -> 淡墨 F：正面浅墨线，最深、最锐利
    观测  I = F 覆盖合成；其下再嵌套 H 与 S 的凸混合。
另输出逐像素分区标签 zones，便于把失败区域对应到具体层。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# 分区标签：用于把残差/失败信号定位到具体层
ZONE_CLEAN = 0      # 纯纸面
ZONE_SHOW = 1       # 仅透印
ZONE_HALFTONE = 2   # 仅网纹
ZONE_FAINT = 3      # 仅淡墨
ZONE_OVERLAP = 4    # 透印+网纹+淡墨三者重合
ZONE_NAMES = {
    ZONE_CLEAN: "clean_paper",
    ZONE_SHOW: "show_through_only",
    ZONE_HALFTONE: "halftone_only",
    ZONE_FAINT: "faint_ink_only",
    ZONE_OVERLAP: "triple_overlap",
}


@dataclass
class TripleSpec:
    width: int = 480
    height: int = 360
    paper_color: tuple = (0.90, 0.87, 0.78)   # 偏色暖纸（非中性灰）
    faint_ink_color: tuple = (0.08, 0.07, 0.10)
    show_color: tuple = (0.72, 0.62, 0.50)    # 透印：浅暖棕，与墨色拉开
    halftone_color: tuple = (0.35, 0.33, 0.40)
    faint_strength: float = 0.62              # 淡墨笔触强度
    show_strength: float = 0.45               # 透印整体偏淡
    halftone_strength: float = 0.40
    grain_sigma: float = 0.012
    shadow_strength: float = 0.10
    sensor_noise: float = 0.005
    seed: int = 23
    ss: int = 4                               # 独立超采样倍率


# --------------------------------------------------------------------------
# 解析覆盖基准：纯几何，不依赖任何绘制库
# --------------------------------------------------------------------------
def _segment_distance_field(xx: np.ndarray, yy: np.ndarray,
                            p0: tuple, p1: tuple) -> np.ndarray:
    """像素中心到线段的垂直距离（解析距离场）。"""
    x0, y0 = p0
    x1, y1 = p1
    vx, vy = x1 - x0, y1 - y0
    length2 = vx * vx + vy * vy
    t = np.clip(((xx - x0) * vx + (yy - y0) * vy) / length2, 0.0, 1.0)
    cx = x0 + t * vx
    cy = y0 + t * vy
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)


def analytic_line_coverage(height: int, width: int, p0: tuple, p1: tuple,
                           line_width: float) -> np.ndarray:
    """解析覆盖率：盒形线核（半径 line_width/2）在像素上的面积覆盖。

    用 1 - clip(dist - (r - 0.5), 0, 1) 近似盒核与单位像素方块的相交面积：
    像素中心距线核 <= r-0.5 时全覆盖，r+0.5 之外为零，其间线性过渡——这正是
    抗锯齿线在一个像素宽内的解析覆盖率，不含任何库光栅化的经验行为。
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    r = line_width / 2.0
    dist = _segment_distance_field(xx, yy, p0, p1)
    cov = np.clip(1.0 - (dist - (r - 0.5)), 0.0, 1.0)
    return cov.astype(np.float32)


def analytic_faint_coverage(spec: TripleSpec) -> np.ndarray:
    """把水平/垂直细交叉线的解析覆盖率取逐像素最大值（线核并集）。

    采用轴对齐线：此时几何盒核与超采样软核的像素落点天然对齐，两路差异
    只反映线核模型本身，不混入对角线亚像素落点偏移。横竖线的大量交点
    仍覆盖"交点"这一测试目标。
    """
    h, w = spec.height, spec.width
    cov = np.zeros((h, w), np.float32)
    for x in range(20, w, 26):
        cov = np.maximum(cov, analytic_line_coverage(
            h, w, (x, 0), (x, h - 1), 1.0))
    for y in range(16, h, 24):
        cov = np.maximum(cov, analytic_line_coverage(
            h, w, (0, y), (w - 1, y), 1.0))
    return np.clip(cov, 0, 1)


# --------------------------------------------------------------------------
# 独立超采样光栅化：与生产 PIL 绘制完全不同的 numpy 密网格数值积分
# --------------------------------------------------------------------------
def _ss_grid(spec: TripleSpec):
    ss = spec.ss
    h, w = spec.height * ss, spec.width * ss
    # 高分辨率子像素中心。约定低分辨率像素中心位于整数坐标（与解析基准、
    # PIL 光栅化一致）：低分辨率像素 i 覆盖 [i-0.5, i+0.5)，其第 j 个子
    # 像素中心为 i + (j+0.5)/ss - 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx % ss + 0.5) / ss - 0.5 + (xx // ss).astype(np.float32)
    yy = (yy % ss + 0.5) / ss - 0.5 + (yy // ss).astype(np.float32)
    return xx, yy, h, w


def _down_mean(a: np.ndarray, ss: int, h: int, w: int, channels: int) -> np.ndarray:
    if channels == 1:
        return a.reshape(h // ss, ss, w // ss, ss).mean(axis=(1, 3))
    return a.reshape(h // ss, ss, w // ss, ss, channels).mean(axis=(1, 3))


def _ss_line_coverage(xx, yy, p0, p1, half_width: float) -> np.ndarray:
    """高分辨率网格上的解析盒核覆盖（子像素尺度求值）。

    与 analytic_line_coverage 使用同一物理盒核公式 clip(1-(dist-(r-0.5)))，
    但在 ss 倍密的子像素中心上求值、再由 _down_mean 做面积积分。两路因此
    是同一盒核的"逐像素闭式解"与"密网格数值积分"，实现路径彼此独立，
    又不存在线宽/坐标约定漂移。half_width 为低分辨率坐标系下的盒半宽 r。
    """
    dist = _segment_distance_field(xx, yy, p0, p1)
    return np.clip(1.0 - (dist - (half_width - 0.5)), 0.0, 1.0)


def build_triple_scene(spec: TripleSpec | None = None) -> dict:
    """生成三层重合测试图与全部独立参照。

    返回字段：
      image            模拟扫描观测 I（含噪声/阴影）
      paper_gt         纸色场真值（无任何墨迹）
      alpha_faint_ref  独立超采样推导的淡墨覆盖率参照
      recomp_ref       用参照覆盖率/各层真值重新凸混合的无噪声参照图
      zones            逐像素分区标签
      analytic_base    解析几何覆盖率基准（墨量级别的独立核对）
      各层颜色/强度真值
    """
    spec = spec or TripleSpec()
    ss = spec.ss
    xx, yy, h, w = _ss_grid(spec)
    rng = np.random.default_rng(spec.seed)

    # --- 纸：偏色基纸 + 云状不均 + 细纸纹 + 单侧阴影 ---
    base = np.tile(np.array(spec.paper_color, np.float32), (h, w, 1))
    cloud = rng.normal(size=(h // 32 + 1, w // 32 + 1)).astype(np.float32)
    cloud_im = Image.fromarray(np.clip(cloud * 127 + 128, 0, 255).astype(np.uint8)).resize(
        (w, h), Image.BICUBIC)
    cloud = (np.asarray(cloud_im, np.float32) / 255 - 0.5)[..., None]
    fine = rng.normal(scale=spec.grain_sigma, size=(h, w, 1)).astype(np.float32)
    paper = np.clip(base + cloud * 0.045 + fine, 0, 1)
    grad = 1 - spec.shadow_strength * (xx / spec.width) * 0.7
    paper = np.clip(paper * grad[..., None], 0, 1)

    # --- 淡墨层：轴对齐细交叉线（独立超采样密网格积分），与解析基准同几何 ---
    faint_hi = np.zeros((h, w), np.float32)
    # 与 analytic_faint_coverage 同一盒核半宽 0.5（1px 线），在密网格积分
    for x in range(20, spec.width, 26):
        faint_hi = np.maximum(faint_hi, _ss_line_coverage(
            xx, yy, (x, 0), (x, spec.height - 1), 0.5))
    for y in range(16, spec.height, 24):
        faint_hi = np.maximum(faint_hi, _ss_line_coverage(
            xx, yy, (0, y), (spec.width - 1, y), 0.5))
    faint_hi *= spec.faint_strength
    # --- 透印层：背面粗线，横贯画面、弥散且浅（宽核 + 低强度）---
    show_hi = np.zeros((h, w), np.float32)
    for y0 in (90, 190, 290):
        show_hi = np.maximum(show_hi, _ss_line_coverage(
            xx, yy, (0, y0), (spec.width, y0), 6.0 * ss / 2.0))
    show_hi *= spec.show_strength

    # --- 网纹层：规则网点（固定间距圆点），中等强度 ---
    pitch = 14.0
    gx = np.mod(xx, pitch) - pitch / 2
    gy = np.mod(yy, pitch) - pitch / 2
    dot_r = np.sqrt(gx ** 2 + gy ** 2)
    half_hi = np.clip(1.0 - dot_r / (0.42 * pitch / 2.0), 0, 1)
    half_hi *= spec.halftone_strength

    # --- 分区标签（低分辨率几何判定，供失败定位）---
    h_lo, w_lo = spec.height, spec.width
    yy_lo, xx_lo = np.mgrid[0:h_lo, 0:w_lo].astype(np.float32)
    faint_zone = analytic_faint_coverage(spec) > 0.05
    # 淡墨线仅约 1px 宽，3x3 膨胀让重合分区对亚像素落点稳健（本地原语，
    # 不依赖生产管线代码，保持参照独立性）
    padded = np.pad(faint_zone.astype(np.float32), 1)
    dil = np.zeros_like(faint_zone, np.float32)
    for dy in range(3):
        for dx in range(3):
            np.maximum(dil, padded[dy:dy + h_lo, dx:dx + w_lo], out=dil)
    faint_zone = dil > 0.15
    # 透印带：三条横贯的粗背线附近
    show_zone = np.zeros((h_lo, w_lo), bool)
    for y0 in (90, 190, 290):
        show_zone |= _segment_distance_field(
            xx_lo, yy_lo, (0, y0), (w_lo, y0)) < 9.0
    dot_lo = np.sqrt((np.mod(xx_lo, pitch) - pitch / 2) ** 2
                     + (np.mod(yy_lo, pitch) - pitch / 2) ** 2)
    half_zone = dot_lo < 0.42 * pitch / 2.0 + 0.5
    zones = np.full((h_lo, w_lo), ZONE_CLEAN, np.uint8)
    zones[show_zone] = ZONE_SHOW
    zones[half_zone] = ZONE_HALFTONE
    zones[faint_zone] = ZONE_FAINT
    zones[show_zone & half_zone & faint_zone] = ZONE_OVERLAP

    # --- 凸混合嵌套：show 在下，halftone 居中，faint 在上 ---
    c_show = np.array(spec.show_color, np.float32)
    c_half = np.array(spec.halftone_color, np.float32)
    c_faint = np.array(spec.faint_ink_color, np.float32)

    def over(lower_rgb, upper_rgb, alpha):
        return (1 - alpha[..., None]) * lower_rgb + alpha[..., None] * upper_rgb

    layer_show = over(paper, c_show, show_hi)
    layer_half = over(layer_show, c_half, half_hi)
    comp_clean = over(layer_half, c_faint, faint_hi)

    # 独立参照：超采样覆盖率降采样 + 用真值重新凸混合（不带噪声）
    coverage_ss_geom = _down_mean(faint_hi / spec.faint_strength, ss, h, w, 1)
    alpha_faint_ref = _down_mean(faint_hi, ss, h, w, 1)
    alpha_show_ref = _down_mean(show_hi, ss, h, w, 1)
    alpha_half_ref = _down_mean(half_hi, ss, h, w, 1)
    paper_lo = _down_mean(paper, ss, h, w, 3)
    ref_show = over(paper_lo, c_show, alpha_show_ref)
    ref_half = over(ref_show, c_half, alpha_half_ref)
    recomp_ref = over(ref_half, c_faint, alpha_faint_ref)

    # 观测：在干净合成上加传感器噪声
    img = comp_clean + rng.normal(
        scale=spec.sensor_noise, size=comp_clean.shape).astype(np.float32)
    image = np.clip(_down_mean(np.clip(img, 0, 1), ss, h, w, 3), 0, 1)

    return {
        "image": image.astype(np.float32),
        "paper_gt": paper_lo.astype(np.float32),
        "coverage_ss_geom": coverage_ss_geom.astype(np.float32),
        "alpha_faint_ref": alpha_faint_ref.astype(np.float32),
        "alpha_show_ref": alpha_show_ref.astype(np.float32),
        "alpha_halftone_ref": alpha_half_ref.astype(np.float32),
        "recomp_ref": np.clip(recomp_ref, 0, 1).astype(np.float32),
        "analytic_base": analytic_faint_coverage(spec),
        "zones": zones,
        "faint_color": c_faint,
        "show_color": c_show,
        "halftone_color": c_half,
        "spec": spec,
    }


# --------------------------------------------------------------------------
# 受控注错
# --------------------------------------------------------------------------
def inject_paper_error(scene: dict, delta: float = 0.10) -> np.ndarray:
    """注错纸色估计：整体朝一个偏色方向平移（模拟底色估计系统性偏差）。

    红/绿通道抬高、蓝通道压低，制造色相与明度双重错误，残差会在所有
    依赖纸色的区域（尤以网纹、纸面背景）冒出来。
    """
    shift = np.array([delta, delta * 0.6, -delta], np.float32)
    return np.clip(scene["paper_gt"] + shift, 0, 1).astype(np.float32)


def invert_blend_order(scene: dict) -> np.ndarray:
    """注错混合次序：把淡墨放到最下层、透印放到最上层重合成。

    次序颠倒后，锐利的淡墨被浅透印/网纹盖住，重合区与透印带的残差最大，
    而纯网纹区几乎不受影响——失败区域因此能指认是"次序"这一层出错。
    """
    s = scene["spec"]
    paper = scene["paper_gt"]
    c_faint, c_show, c_half = scene["faint_color"], scene["show_color"], scene["halftone_color"]

    def over(lower, upper, alpha):
        return (1 - alpha[..., None]) * lower + alpha[..., None] * upper

    # 错误次序：faint 先（最下），halftone 居中，show 最后（最上）
    x = over(paper, c_faint, scene["alpha_faint_ref"])
    x = over(x, c_half, scene["alpha_halftone_ref"])
    wrong = over(x, c_show, scene["alpha_show_ref"])
    return np.clip(wrong, 0, 1).astype(np.float32)


def zone_residuals(residual_mag: np.ndarray, zones: np.ndarray) -> dict:
    """把残差幅度按分区聚合，返回每区均值，用于失败分层定位。"""
    out = {}
    for zid, zname in ZONE_NAMES.items():
        m = zones == zid
        out[zname] = float(residual_mag[m].mean()) if m.any() else 0.0
        out[zname + "_pixels"] = int(m.sum())
    return out


def neighborhood_max(a: np.ndarray, radius: int = 1) -> np.ndarray:
    """局部邻域最大值（零依赖 numpy 实现），用于容忍亚像素错位的覆盖比较。"""
    out = a.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
            out = np.maximum(out, shifted)
    return out


def coverage_agreement(analytic: np.ndarray, reference: np.ndarray,
                       level: float = 0.05) -> dict:
    """两路独立覆盖的一致性指标。

    覆盖率一致性指标。两路都采用轴对齐线，像素落点对齐；仍保留 1px 邻域
    容差以吸收盒核过渡带的边界差异。报告：
      support_match  : 任一路判为有墨的像素中，另一路在 1px 邻域内也有墨的
                       比例（空间支撑是否一致）
      level_ratio    : 共同支撑上两路覆盖率均值之比（墨量级别是否一致）
    阈值在测试中按"同核两种独立求法只应相差离散化误差"原理给出，不引用
    被测生产输出，因此不是默认值。
    """
    a_hit = analytic > level
    r_hit = reference > level
    r_near = neighborhood_max(reference.astype(np.float32), 1) > level
    a_near = neighborhood_max(analytic.astype(np.float32), 1) > level
    union = a_hit | r_hit
    matched = (a_hit & r_near) | (r_hit & a_near)
    support_match = float(matched[union].mean()) if union.any() else 1.0
    common = a_hit & r_hit
    if common.any():
        level_ratio = float(reference[common].mean()
                            / max(analytic[common].mean(), 1e-9))
    else:
        level_ratio = 0.0
    return {
        "support_match": support_match,
        "level_ratio": level_ratio,
        "analytic_pixels": int(a_hit.sum()),
        "reference_pixels": int(r_hit.sum()),
        "union_pixels": int(union.sum()),
    }
