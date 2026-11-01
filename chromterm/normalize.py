# -*- coding: utf-8 -*-
"""normalize.py —— 把端部小图变成几何规范化的 192×64 窗口。

问题
----
data_share/level1 的 246×112 窗口**大体**已规范，但实测（200 张抽样）：
  * 32.5% 的窗口臂轴偏离水平 >15°，13.5% >30°；
  * 端部在画面里的位置随标注框漂移；
  * 40× 与 100× 物镜下臂粗差 2–3 倍。
这些变量与类别无关，却要模型用容量去忽略——961 个样本下这部分容量非常昂贵。

三种归一化强度（脚本里用 --norm 切换，实测结果见 runs/metrics.csv）
------------------------------------------------------------------
  raw     : 直接把 246×112 缩放到 192×64（基线，等价"把图丢给网络"）
  deskew  : 用结构张量估臂轴角 φ，把图转正后再取中心 3:1 窗口
  tip     : 在 deskew 的基础上，用第 5～9 步量出的端部 T、向内方向 u、臂宽 W，
            把 T 硬钉在 (6%, 50%)、沿 u 覆盖 7 个臂宽（最激进，也最依赖几何成功率）

设计要点
--------
* 旋转用 rotate_expand，保证四个角不被裁掉；
* tip 模式在几何失败（掩膜过小/骨架过短/方向不一致）时自动退回 deskew；
* 所有模式都输出同样的 192×64，下游特征提取对模式无感。
"""

from __future__ import annotations

import numpy as np

from .common import EPS, _bilinear_sample, resize_bilinear, rotate_expand
from .geometry import repair_mask, segment_stain


# ------------------------------------------------------------------ 工具
def crop_with_pad(img: np.ndarray, x0: int, y0: int, x1: int, y1: int):
    """整数框裁剪，越界按边缘复制补齐。"""
    h, w = img.shape[:2]
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x1 - w), max(0, y1 - h)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return None
    sub = img[y0c:y1c, x0c:x1c]
    pad = [(py0, py1), (px0, px1)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(sub, pad, mode="edge")


def _unit(v):
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else np.array([1.0, 0.0])).astype(np.float32)


def estimate_width(geom: dict | None, shape) -> float:
    """臂宽：优先半高全宽中位数；退化用 面积÷骨架长；再退化成经验值。"""
    h, w = shape[:2]
    hi = 0.6 * min(h, w)
    if geom:
        meas = geom.get("measure") or {}
        wd = meas.get("width_px")
        if wd is not None and np.isfinite(wd) and 3.0 <= wd <= hi:
            return float(wd)
        mask, L = geom.get("mask"), meas.get("length_px", 0) or 0
        if mask is not None and L > 5:
            wd2 = float(np.asarray(mask).sum()) / float(L)
            if np.isfinite(wd2) and 2.0 < wd2 <= hi:
                return float(wd2)
    return float(max(4.0, min(0.30 * min(h, w), hi)))


def canonical_crop(img: np.ndarray, tip_xy, inward_uv, arm_width: float,
                   out_w: int = 192, out_h: int = 64, tip_frac: float = 0.06,
                   height_frac: float = 2.2, keep_inside: bool = True) -> np.ndarray:
    """按"端部在左、臂水平、宽度归一"的斜采样（tip 模式的核心）。"""
    h, w = img.shape[:2]
    if not np.isfinite(arm_width) or arm_width < 1.0:
        arm_width = max(2.0, 0.3 * min(h, w))
    u = _unit(np.asarray(inward_uv, np.float32))
    v = np.array([-u[1], u[0]], np.float32)

    h_phys = height_frac * arm_width
    w_phys = height_frac * out_w / out_h * arm_width
    w_phys = min(w_phys, 1.05 * max(w, h))
    tip = np.asarray(tip_xy, np.float32).copy()

    if keep_inside:                     # 采样框出界就整体平移，避免补边条纹
        ca = np.array([-tip_frac, 1.0 - tip_frac], np.float32)
        cb = np.array([-0.5, 0.5], np.float32)
        C = np.array([tip + a_ * w_phys * u + b_ * h_phys * v for a_ in ca for b_ in cb],
                     np.float32)
        dx = 0.0
        if C[:, 0].min() < 0:
            dx = -C[:, 0].min()
        if C[:, 0].max() + dx > w - 1:
            dx = (w - 1) - C[:, 0].max()
        dy = 0.0
        if C[:, 1].min() < 0:
            dy = -C[:, 1].min()
        if C[:, 1].max() + dy > h - 1:
            dy = (h - 1) - C[:, 1].max()
        tip = tip + np.array([dx, dy], np.float32)

    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    du = (xx / (out_w - 1) - tip_frac) * w_phys
    dv = (yy / (out_h - 1) - 0.5) * h_phys
    sx = tip[0] + du * u[0] + dv * v[0]
    sy = tip[1] + du * u[1] + dv * v[1]
    return _bilinear_sample(img, sx, sy)


def center_crop_deskewed(img: np.ndarray, out_w: int = 192, out_h: int = 64) -> np.ndarray:
    """从转正后的画布上取居中的 3:1 窗口并缩放（deskew 模式）。"""
    h, w = img.shape[:2]
    cw = w
    ch = int(round(cw * out_h / out_w))
    if ch > h:
        ch = h
        cw = int(round(ch * out_w / out_h))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    sub = crop_with_pad(img, x0, y0, x0 + cw, y0 + ch)
    return resize_bilinear(sub, out_w, out_h)


# ------------------------------------------------------------------ 主入口
def normalize(img: np.ndarray, mode: str = "tip", geom: dict | None = None,
              axis_deg: float | None = None, out_w: int = 192, out_h: int = 64) -> dict:
    """把一张图按指定模式规范化。

    img     : HxW 或 HxWxC（可以是 RGB，也可以是 0..1 的染色量场）
    mode    : raw / deskew / tip
    geom    : tip 模式需要的几何结果（geometry.analyse_window 的返回）
    axis_deg: 臂轴角；None 时内部用结构张量估计（img 需为染色量场）

    返回 dict(img, mode, used, axis_deg, width, tip_xy, quality)
    """
    from .geometry import axis_from_tensor

    h, w = img.shape[:2]
    if mode == "raw":
        return dict(img=resize_bilinear(img, out_w, out_h), mode=mode, used="raw",
                    axis_deg=np.nan, width=np.nan, tip_xy=(np.nan, np.nan), quality=1.0)

    if axis_deg is None or not np.isfinite(axis_deg):
        s_n = img if img.ndim == 2 else img[..., 0]
        axis_deg, _ = axis_from_tensor(s_n, sigma_tensor=max(4.0, min(h, w) / 12))

    rot = rotate_expand(img, -axis_deg)

    if mode == "deskew":
        out = center_crop_deskewed(rot, out_w, out_h)
        return dict(img=out, mode=mode, used="deskew", axis_deg=float(axis_deg),
                    width=np.nan, tip_xy=(np.nan, np.nan), quality=1.0)

    # ---- tip 模式 ----
    if geom and geom.get("ok") and geom.get("path") and geom.get("tip_xy"):
        tip = np.array(geom["tip_xy"], np.float32)
        path = np.asarray(geom["path"], np.float32)
        d0 = np.linalg.norm(path[0][::-1] - tip)
        d1 = np.linalg.norm(path[-1][::-1] - tip)
        inner = path[-1][::-1] if d0 <= d1 else path[0][::-1]
        inward = _unit(inner - tip)
        g = np.array([np.cos(np.deg2rad(axis_deg)), np.sin(np.deg2rad(axis_deg))],
                     np.float32)
        used = "tip"
        if abs(float(np.dot(inward, g))) < 0.65:       # 骨架走向与全局方向不一致
            inward = g if np.dot(g, inner - tip) >= 0 else -g
            used = "tip+axis"
        width = estimate_width(geom, img.shape)
        out = canonical_crop(img, tip, inward, width, out_w, out_h)
        return dict(img=out, mode=mode, used=used, axis_deg=float(axis_deg),
                    width=float(width), tip_xy=(float(tip[0]), float(tip[1])), quality=1.0)

    out = center_crop_deskewed(rot, out_w, out_h)
    return dict(img=out, mode=mode, used="deskew_fallback", axis_deg=float(axis_deg),
                width=np.nan, tip_xy=(np.nan, np.nan), quality=0.0)


def normalize_pair(rgb: np.ndarray, s_n: np.ndarray, mode: str, geom: dict | None,
                   axis_deg: float | None = None, out_w: int = 192,
                   out_h: int = 64) -> dict:
    """同时规范化 RGB 与染色量两路（保持同一采样几何）。"""
    a = normalize(rgb, mode, geom, axis_deg, out_w, out_h)
    b = normalize(s_n, mode, geom, axis_deg, out_w, out_h)
    a["s_n"] = b["img"]
    return a
