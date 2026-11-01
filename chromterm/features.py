# -*- coding: utf-8 -*-
"""features.py —— 特征块。每个块单独返回，便于做消融。

为什么要分块而不是"一把梭给网络"
----------------------------
961 个样本、6 类、跨 5 个成像批次。在这个量级上，端到端 CNN 要么靠强增强硬记，
要么退化成"看染色深浅"。按块的经典特征 + 线性/核方法反而是稳的基线；
每个块还能单独报告"它带来了多少"，这正是审稿人会问的。

三个"物理来源"不同的块
---------------------
  pyr   : 规范化窗口在 OD 空间的多尺度池化（形态 + 带纹的二维信息）
  band  : 沿臂方向的一维带纹剖面（文档 2.3 的"把管变成序列"；去趋势后是带序）
  width : 沿臂方向的宽度剖面（膨大端、染色体中心附近变粗，是分 X 与 3R 的线索）
  geo   : 第 5～9 步量出来的标量（长度、臂宽、弯曲度、带数、端部朝向…）
  rgb   : 原始 RGB 池化（**只给基线用**，用来证明"直接把颜色喂进去"不如 OD）

注意：所有块都在**同一个规范化窗口**上算，块之间不共享信息，拼接后统一标准化。
"""

from __future__ import annotations

import numpy as np

from .common import EPS, gaussian_blur, resize_bilinear


# ------------------------------------------------------------------ 工具
def pool2d(x: np.ndarray, gw: int, gh: int) -> np.ndarray:
    """区域平均池化到 gw×gh（对 HxW 或 HxWxC 都适用）。

    实现上先用 σ≈格边长的可分离高斯预平滑、再双线性缩放，等价于面积平均但快得多。
    """
    h, w = x.shape[:2]
    sy = max(h / (2.0 * gh), 0.5)
    sx = max(w / (2.0 * gw), 0.5)
    if x.ndim == 2:
        b = gaussian_blur(x.astype(np.float32), sx, sy)
    else:
        b = np.stack([gaussian_blur(x[..., c].astype(np.float32), sx, sy)
                      for c in range(x.shape[2])], axis=-1)
    return resize_bilinear(b, gw, gh)


def row_profile(s_n: np.ndarray, q: float = 0.6) -> np.ndarray:
    """沿臂方向的一维染色剖面：每一列的"中心带"强度（取列内高分位，抗边界）。"""
    h, w = s_n.shape
    # 只取中间 60% 的行，避开上下边缘的空白与另一条臂
    r0, r1 = int(0.2 * h), int(0.8 * h)
    sub = s_n[r0:r1]
    return np.percentile(sub, q * 100, axis=0).astype(np.float32)


def width_profile(s_n: np.ndarray, thr: float = 0.35) -> np.ndarray:
    """沿臂方向的"粗细剖面"：每列超过阈值的像素数（相对中位数归一）。"""
    prof = (s_n > thr).sum(0).astype(np.float32)
    med = float(np.median(prof[prof > 0])) if (prof > 0).any() else 1.0
    return prof / max(med, 1.0)


def resample_1d(x: np.ndarray, n: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros(n, np.float32)
    t = np.linspace(0, 1, x.size)
    tn = np.linspace(0, 1, n)
    return np.interp(tn, t, x).astype(np.float32)


# ------------------------------------------------------------------ 特征块
def feat_pyr(s_n: np.ndarray) -> np.ndarray:
    """多尺度池化：32×10 + 16×5 = 400 维。

    分辨率是扫出来的（见 docs 里的消融表）：再细（48×16）反而降点——961 个样本
    撑不起那么细的二维描述，多出来的维数只是噪声。
    """
    a = pool2d(s_n, 32, 10).ravel()
    b = pool2d(s_n, 16, 5).ravel()
    return np.concatenate([a, b]).astype(np.float32)


def feat_band(s_n: np.ndarray, n: int = 64) -> np.ndarray:
    """带纹剖面：原始 + 去趋势（高通）+ 一阶导数，各 n 维。"""
    p = row_profile(s_n, q=0.6)
    low = gaussian_blur(p[None, :], max(2.0, p.size / 24.0))[0]
    hp = p - low                                  # 去趋势 = 保留横纹、去掉整体亮度梯度
    d = np.gradient(hp)
    return np.concatenate([resample_1d(p, n), resample_1d(hp, n), resample_1d(d, n)]).astype(np.float32)


def feat_width(s_n: np.ndarray, n: int = 32) -> np.ndarray:
    """粗细剖面（归一化后重采样）。"""
    return resample_1d(width_profile(s_n), n).astype(np.float32)


def feat_geo(geo: np.ndarray, names) -> np.ndarray:
    """把几何标量变成可比的特征：长度/宽度取 log，角度拆成 sin/cos，其余截断。"""
    g = dict(zip([str(x) for x in names], np.asarray(geo, np.float64)))
    out = []
    for k in ["length_px", "width_px", "curv", "area_frac", "fg_frac", "coh",
              "n_bands", "band_contrast", "tip_ang_sin", "tip_ang_cos"]:
        v = g.get(k, np.nan)
        if k in ("length_px", "width_px"):
            v = np.log1p(max(0.0, v)) if np.isfinite(v) else 0.0
        if k == "curv":
            v = min(max(v, 0.0), 400.0) if np.isfinite(v) else 0.0
        if not np.isfinite(v):
            v = 0.0
        out.append(float(v))
    # 宽长比（尺度无关的形态量）
    L, W = g.get("length_px", np.nan), g.get("width_px", np.nan)
    out.append(float(W / L) if np.isfinite(L) and np.isfinite(W) and L > 0 else 0.0)
    return np.asarray(out, np.float32)


def feat_rgb(rgb: np.ndarray, gw: int = 24, gh: int = 8) -> np.ndarray:
    """RGB 池化（基线用）：24×8×3 = 576 维。"""
    return pool2d(rgb.astype(np.float32), gw, gh).ravel().astype(np.float32)


def feat_hog(s_n: np.ndarray, cells=(6, 3), bins: int = 9) -> np.ndarray:
    """梯度方向直方图（HOG 的简化版）。

    为什么加它：横纹的**间距/走向/锐利度**才是类别线索，而"池化后的亮度"会被
    染色深浅与条带宽度混在一起。梯度方向直方图对绝对亮度不敏感，正好补这一块。
    """
    b = gaussian_blur(s_n.astype(np.float32), 1.0)
    gy, gx = np.gradient(b)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi           # 0..1，方向无符号
    cx, cy = cells
    h, w = s_n.shape
    ys = np.linspace(0, h, cy + 1).astype(int)
    xs = np.linspace(0, w, cx + 1).astype(int)
    out = np.zeros((cy, cx, bins), np.float32)
    for i in range(cy):
        for j in range(cx):
            m = mag[ys[i]:max(ys[i] + 1, ys[i + 1]), xs[j]:max(xs[j] + 1, xs[j + 1])]
            a = ang[ys[i]:max(ys[i] + 1, ys[i + 1]), xs[j]:max(xs[j] + 1, xs[j + 1])]
            wgt = np.zeros(bins, np.float32)
            if m.size:
                idx = np.clip((a.ravel() * bins).astype(int), 0, bins - 1)
                np.add.at(wgt, idx, m.ravel())
            n = np.linalg.norm(wgt) + EPS
            out[i, j] = wgt / n
    return out.ravel().astype(np.float32)


def feat_auto(s_n: np.ndarray, max_lag: int = 40) -> np.ndarray:
    """带纹剖面的自相关（lag 2..max_lag）。

    横纹是准周期的：自相关第一个峰的位置 ≈ 带间距，峰的深度 ≈ 带的规律程度。
    这两个量比"某一列的亮度"稳定得多，是判 2R/3L/3R 这类"只差几条带"的类的常用依据。
    """
    p = row_profile(s_n, q=0.6)
    p = (p - p.mean()) / (p.std() + EPS)
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    ac = ac / (ac[0] + EPS)
    lags = ac[2:max_lag + 1]
    if lags.size < max_lag - 1:
        lags = np.pad(lags, (0, max_lag - 1 - lags.size))
    return lags.astype(np.float32)


# ------------------------------------------------------------------ 组装
BLOCKS = ("rgb", "pyr", "band", "width", "geo", "hog", "auto")


def build_matrix(feat: dict, mode: str, blocks=("pyr",)):
    """按 (归一化模式, 块集合) 组装特征矩阵 N×D。"""
    parts = []
    for b in blocks:
        if b == "rgb":
            key = "rgb_raw" if mode == "raw" else f"can_{mode}_rgb"
            parts.append(np.stack([feat_rgb(x) for x in feat[key]]))
        elif b == "pyr":
            parts.append(np.stack([feat_pyr(np.asarray(x, np.float32))
                                   for x in feat[f"can_{mode}"]]))
        elif b == "band":
            parts.append(np.stack([feat_band(np.asarray(x, np.float32))
                                   for x in feat[f"can_{mode}"]]))
        elif b == "width":
            parts.append(np.stack([feat_width(np.asarray(x, np.float32))
                                   for x in feat[f"can_{mode}"]]))
        elif b == "geo":
            parts.append(np.stack([feat_geo(g, feat["geo_names"]) for g in feat["geo"]]))
        elif b == "hog":
            parts.append(np.stack([feat_hog(np.asarray(x, np.float32))
                                   for x in feat[f"can_{mode}"]]))
        elif b == "auto":
            parts.append(np.stack([feat_auto(np.asarray(x, np.float32))
                                   for x in feat[f"can_{mode}"]]))
        else:
            raise ValueError(b)
    return np.concatenate(parts, axis=1).astype(np.float32)


FEATURE_SETS = {
    # 名称: (归一化模式, 块)  —— 这组就是"最终消融表"的六行
    "A_RGB基线": ("raw", ("rgb",)),
    "B_OD多尺度": ("raw", ("pyr",)),
    "C_OD全特征": ("raw", ("pyr", "band", "auto", "hog", "width", "geo")),
    "D_OD全特征_转正": ("deskew", ("pyr", "band", "auto", "hog", "width", "geo")),
    "E_OD全特征_端部锚定": ("tip", ("pyr", "band", "auto", "hog", "width", "geo")),
    "F_OD全特征_去几何": ("raw", ("pyr", "band", "auto", "hog", "width")),
}

# 生产用配置（与消融表胜出者一致）：OD 归一化 + 全特征块
DEFAULT_SET = "C_OD全特征"
