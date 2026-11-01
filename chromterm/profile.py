# -*- coding: utf-8 -*-
"""profile.py —— 沿染色体中线采样（对齐《中线采样说明.docx》的口径）

这份模块把组里《中线采样说明》那套"从一条中线到一张表"的做法，落到本项目
（numpy + Pillow，无 cv2 / scipy）的实现里。它是**识别环节的上游**：把一张
端部窗口从"一块像素"变成"一条带横纹的曲线 + 一张条带表"，后面才谈得上比对。

对照说明文档的每一条约定，逐条实现：

  3.1 距离从端部起算      —— 先把中线方向统一，第 0 个采样点一定是端粒端
  3.2 按 1 像素弧长重采样 —— 先算累计弧长，再在 0,1,2,3… 处线性插值取点
  3.3 横向平均 3 个像素   —— 沿"垂直于中线"的方向取 3 点（双线性）求平均
  4.1 灰度用标准亮度      —— 0.299R + 0.587G + 0.114B，与旧数据同一把尺子
  4.2 横向宽度默认掩膜量  —— 宽度 = 2 × 该点到掩膜边界的欧氏距离（精确 EDT）
  4.3 梯度 = |dGray/ds|   —— 亮→暗的转折处出尖峰，供"找边界"用
  5.x 条带宽度            —— σ=1.5 平滑 → 曲线取反找峰 → prominence ≥ 3×噪声
                             （噪声 = 残差 MAD）→ 量半高宽（FWHM）

为什么这套东西对"识别"重要（也是此前 50% 准确率的病根所在）：
  以前的特征是把 246×112 的窗口按网格池化（金字塔 / HOG / 自相关），
  于是"带在臂上的先后顺序"和"带与带之间的间距"这两条最有鉴别力的信息，
  被平均掉了。沿中线采样恰好把它们还原成一维序列：横纹的顺序、绝对位置、
  宽度、深浅都在里面，而且臂的朝向、弯曲、粗细都不再干扰。

对外只有三个函数：
  build_profile(rgb, S, S_n, info)  -> dict   一张图 → 一条曲线 + 条带表
  profile_table(prof)               -> (rows, band_rows)  与旧数据同格式
  profile_features(prof)            -> dict   识别用的定长特征
"""

from __future__ import annotations

import os

import numpy as np

from .common import EPS, conv1d_reflect, gaussian_blur, load_rgb
from .geometry import analyse_window
from .prep_od import stain_map

# 《中线采样说明》表 1 的列序（前两列与 data_code_figure/data 逐字相同）
TABLE_COLUMNS = ["Distance_(pixels)", "Gray_Value", "Width_px", "Grad"]
BAND_COLUMNS = ["band_id", "s_center_px", "s_start_px", "s_end_px",
                "width_px", "gray_min", "prominence"]

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


# ============================================================================
# 1. 精确欧氏距离变换（掩膜宽度要用；不引 scipy）
# ============================================================================
_INF = 1e20


def _edt_1d_sq(f: np.ndarray) -> np.ndarray:
    """一维平方欧氏距离变换（Felzenszwalb & Huttenlocher 1987，抛物线包络）。

    f 里 0 表示前景（掩膜内），_INF 表示背景。返回每个位置到最近前景的平方距离。
    这是"两趟"做法里的一趟：先按行做，再按列做，就得到二维精确 EDT。
    """
    n = f.size
    v = np.zeros(n, np.int32)
    z = np.zeros(n + 1, np.float64)
    k = 0
    v[0] = 0
    z[0] = -_INF
    z[1] = _INF
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = _INF
    d = np.empty(n, np.float64)
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        dq = q - v[k]
        d[q] = dq * dq + f[v[k]]
    return d


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """掩膜内每点到最近背景点的距离（像素）。掩膜外为 0。"""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros(m.shape, np.float32)
    h, w = m.shape
    # 背景代价 _INF，前景 0：得到"到最近前景的距离"→ 取反即"到最近背景的距离"
    f = np.where(m, _INF, 0.0)
    # 先按列（axis=0）再按行：两次一维变换
    tmp = np.empty((h, w), np.float64)
    for x in range(w):
        tmp[:, x] = _edt_1d_sq(f[:, x])
    out = np.empty((h, w), np.float64)
    for y in range(h):
        out[y, :] = _edt_1d_sq(tmp[y, :])
    return np.sqrt(out).astype(np.float32)


# ============================================================================
# 2. 一维找峰 / 半高宽（等价 scipy.signal.find_peaks + peak_widths）
# ============================================================================
def _peak_prominence(x: np.ndarray, p: int) -> tuple[float, int, int]:
    """单个峰 p 的 (prominence, 左鞍部下标, 右鞍部下标)。"""
    n = x.size
    i = p
    lmin_i = p
    while i > 0 and x[i - 1] <= x[p]:
        i -= 1
        if x[i] < x[lmin_i]:
            lmin_i = i
    i = p
    rmin_i = p
    while i < n - 1 and x[i + 1] <= x[p]:
        i += 1
        if x[i] < x[rmin_i]:
            rmin_i = i
    prom = float(x[p] - max(x[lmin_i], x[rmin_i]))
    return prom, lmin_i, rmin_i


def find_peaks(x: np.ndarray, prominence: float = 0.0, distance: int = 1):
    """返回 (峰下标, 每个峰的 prominence)。只认"比两侧都高"的局部极大。

    等价于 scipy.signal.find_peaks(x, prominence=..., distance=...) 的常用子集。
    """
    x = np.asarray(x, np.float64)
    n = x.size
    empty = (np.zeros(0, np.int64), np.zeros(0))
    if n < 3:
        return empty
    cand = np.nonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    if cand.size == 0:
        return empty

    prom = np.zeros(cand.size)
    for j, p in enumerate(cand):
        prom[j] = _peak_prominence(x, int(p))[0]

    keep = prom >= prominence
    cand, prom = cand[keep], prom[keep]
    if cand.size == 0:
        return empty

    # 最小间距：按 prominence 从大到小贪心保留
    chosen = []
    for j in np.argsort(-prom):
        if all(abs(int(cand[j]) - int(cand[c])) >= distance for c in chosen):
            chosen.append(int(j))
    chosen = np.array(sorted(chosen), np.int64)
    return cand[chosen], prom[chosen]


def peak_widths(x: np.ndarray, peaks, rel_height: float = 0.5):
    """按 rel_height 那条水平线量峰宽，返回 (widths, left, right)。

    rel_height=0.5 即"半高宽"；水平线 = 鞍部 + (1 - rel_height) × prominence。
    """
    x = np.asarray(x, np.float64)
    n = x.size
    peaks = np.asarray(peaks, np.int64)
    widths = np.zeros(peaks.size)
    lefts = np.zeros(peaks.size)
    rights = np.zeros(peaks.size)
    for j, p in enumerate(peaks):
        prom, lb, rb = _peak_prominence(x, int(p))
        base = max(x[lb], x[rb])
        h = base + (1.0 - rel_height) * prom
        i = int(p)
        while i > lb and x[i - 1] > h:
            i -= 1
        lefts[j] = i
        i = int(p)
        while i < rb and x[i + 1] > h:
            i += 1
        rights[j] = i
        widths[j] = rights[j] - lefts[j]
    return widths, lefts, rights


def detect_bands(gray: np.ndarray, min_prom: float = 3.0, min_dist: int = 4,
                 sigma: float = 1.5):
    """把灰度曲线切成一条一条的"带"（说明文档 5.1 的四步）。

    1) 轻微平滑 σ=1.5；2) 曲线取反（灰=染色深=带 → 变成峰）；
    3) 找峰并按 prominence ≥ min_prom × 噪声 筛（噪声 = 残差 MAD）；
    4) 量半高宽（FWHM）。
    """
    g = np.asarray(gray, np.float64)
    if g.size < 10:
        return []
    sm = gaussian_blur(g[None, :].astype(np.float32), sigma)[0].astype(np.float64)
    resid = g - gaussian_blur(g[None, :].astype(np.float32), 8.0)[0].astype(np.float64)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    noise = 1.4826 * mad + 1e-6
    inv = -sm
    peaks, prom = find_peaks(inv, prominence=min_prom * noise, distance=min_dist)
    if peaks.size == 0:
        return []
    widths, lefts, rights = peak_widths(inv, peaks, rel_height=0.5)
    bands = []
    for k, pk in enumerate(peaks):
        bands.append(dict(
            band_id=k + 1,
            s_center=float(pk),
            s_start=float(lefts[k]),
            s_end=float(rights[k]),
            width_px=float(widths[k]),
            gray_min=float(g[int(pk)]),
            prominence=float(prom[k]),
        ))
    return bands


# ============================================================================
# 3. 沿中线的采样（说明文档 3.1 / 3.2 / 3.3）
# ============================================================================
def _resample_by_arc(path, step: float = 1.0):
    """按弧长等间距重采样，返回 (samples (n,2)[y,x], tangents (n,2))。

    骨架走斜线时相邻点距 √2≈1.41，直接按点列当横轴会把斜段压缩 30%。
    所以先累加弧长 s，再在 0, step, 2·step … 上线性插值。
    """
    p = np.asarray(path, np.float64)
    if len(p) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2)), 0.0
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 2.0:
        return np.zeros((0, 2)), np.zeros((0, 2)), total
    t = np.arange(0.0, total + 1e-9, step)
    ys = np.interp(t, cum, p[:, 0])
    xs = np.interp(t, cum, p[:, 1])
    samples = np.stack([ys, xs], 1)

    tan = np.zeros_like(samples)
    if len(samples) >= 3:
        tan[1:-1] = samples[2:] - samples[:-2]
        tan[0] = samples[1] - samples[0]
        tan[-1] = samples[-1] - samples[-2]
    else:
        tan[:] = samples[-1] - samples[0]
    nrm = np.linalg.norm(tan, axis=1) + EPS
    return samples, tan / nrm[:, None], total


def _bilinear(field: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """双线性采样（边界外按复制处理，不产生 NaN）。"""
    h, w = field.shape
    ys = np.clip(np.asarray(ys, np.float64), 0, h - 1.000001)
    xs = np.clip(np.asarray(xs, np.float64), 0, w - 1.000001)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = ys - y0
    fx = xs - x0
    top = field[y0, x0] * (1 - fx) + field[y0, x1] * fx
    bot = field[y1, x0] * (1 - fx) + field[y1, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float64)


def sample_along(field: np.ndarray, samples, tangents, width: int = 3) -> np.ndarray:
    """在每个采样点上取值；width>1 时沿法线（切向转 90°）取 width 点求平均。"""
    if len(samples) == 0:
        return np.zeros(0, np.float64)
    ny = -tangents[:, 1]
    nx = tangents[:, 0]
    offs = np.arange(width) - (width - 1) / 2.0
    acc = np.zeros(len(samples), np.float64)
    for off in offs:
        acc += _bilinear(field, samples[:, 0] + off * ny, samples[:, 1] + off * nx)
    return acc / max(width, 1)


def mask_width_along(dist: np.ndarray, samples) -> np.ndarray:
    """掩膜宽度 = 2 × 该点到掩膜边界的距离（说明文档 4.2 的默认量法）。"""
    if len(samples) == 0:
        return np.zeros(0, np.float64)
    w = 2.0 * _bilinear(dist, samples[:, 0], samples[:, 1])
    # 去掉 ±1px 的量化台阶，不改变"哪段粗哪段细"
    k = np.ones(3) / 3.0
    return conv1d_reflect(w.astype(np.float32)[None, :], k, axis=1)[0].astype(np.float64)


def fwhm_width_along(od: np.ndarray, samples, tangents, half: int = 26,
                     frac: float = 0.5) -> np.ndarray:
    """另一种宽度量法：OD 剖面半高宽（量"染色最浓的核心"，与掩膜宽度不同口径）。"""
    n = len(samples)
    out = np.full(n, np.nan, np.float64)
    if n == 0:
        return out
    ks = np.arange(-half, half + 1)
    for i in range(n):
        ny = -tangents[i, 1]
        nx = tangents[i, 0]
        prof = _bilinear(od, samples[i, 0] + ks * ny, samples[i, 1] + ks * nx)
        bg = float(np.median(np.r_[prof[:6], prof[-6:]]))
        c = len(ks) // 2
        peak = float(np.max(prof[max(0, c - 3):c + 4]))
        if peak - bg < 1e-6:
            continue
        thr = bg + frac * (peak - bg)
        above = prof > thr
        lo = hi = c
        while lo > 0 and above[lo - 1]:
            lo -= 1
        while hi < len(ks) - 1 and above[hi + 1]:
            hi += 1
        if lo == 0 or hi == len(ks) - 1:
            continue
        out[i] = float(hi - lo + 1)
    return out


# ============================================================================
# 4. 一张图 → 一条曲线
# ============================================================================
def build_profile(rgb: np.ndarray, S: np.ndarray, S_n: np.ndarray, info: dict,
                  width_avg: int = 3, width_mode: str = "mask", step: float = 1.0,
                  band_min_prom: float = 3.0, flip: bool = False) -> dict:
    """把 analyse_window 的结果变成《中线采样说明》那样的曲线表。

    rgb     : 原图（RGB float32）
    S, S_n  : stain_map 给的染色量与归一化染色量
    info    : analyse_window 的输出（含 path / tip_xy / mask）
    flip    : 调试用，强制反向（正常情况下方向由端部自动定）
    """
    path = info.get("path")
    if path is None or len(path) < 4 or not info.get("ok"):
        return {"ok": False, "reason": "无可用中线"}

    arr = np.asarray(path, np.float64)
    # ---- 3.1 方向统一：第 0 个采样点必须是端部 ----
    tip = np.asarray(info["tip_xy"][::-1], np.float64)      # (y, x)
    tip_is_start = float(np.linalg.norm(arr[0] - tip)) <= float(np.linalg.norm(arr[-1] - tip))
    if flip:
        tip_is_start = not tip_is_start
    if not tip_is_start:
        arr = arr[::-1]

    samples, tangents, total = _resample_by_arc(arr, step=step)
    if len(samples) < 8:
        return {"ok": False, "reason": f"中线过短（{total:.0f}px）"}

    gray_field = (rgb @ LUMA).astype(np.float32)
    gray = sample_along(gray_field, samples, tangents, width=width_avg)
    od = sample_along(S.astype(np.float32), samples, tangents, width=width_avg)
    odn = sample_along(S_n.astype(np.float32), samples, tangents, width=width_avg)

    dist = distance_transform(info["mask"])
    if width_mode == "fwhm":
        width = fwhm_width_along(S.astype(np.float32), samples, tangents)
        width = np.where(np.isfinite(width), width, np.nan)
    else:
        width = mask_width_along(dist, samples)

    grad = np.abs(np.gradient(gray))
    grad = conv1d_reflect(grad.astype(np.float32)[None, :], np.ones(3) / 3.0, axis=1)[0]

    bands = detect_bands(gray, min_prom=band_min_prom)

    return {
        "ok": True,
        "s": np.arange(len(gray), dtype=np.float64) * step,   # Distance_(pixels)
        "gray": gray,
        "od": od,
        "odn": odn,
        "width": width,
        "grad": grad.astype(np.float64),
        "bands": bands,
        "length_px": float(len(gray) - 1) * step,
        "n_points": int(len(gray)),
        "path_xy": samples[:, ::-1].copy(),
        "tip_xy": tuple(float(v) for v in info["tip_xy"]),
        "width_mode": width_mode,
        "width_avg": width_avg,
    }


def profile_of_window(rgb: np.ndarray, *, width_avg: int = 3, width_mode: str = "mask",
                      step: float = 1.0, band_min_prom: float = 3.0,
                      extend_frac: int = 30) -> dict:
    """一步到位：RGB 窗口 → 曲线（内部跑 OD + 几何 + 采样）。"""
    rgb = np.asarray(rgb, np.float32)
    m = stain_map(rgb)
    info = analyse_window(m["S"], m["S_n"], thr_frac=0.8,
                          sigma_along=max(12.0, min(rgb.shape[:2]) / 8),
                          core_grow=None, extend_frac=extend_frac)
    prof = build_profile(rgb, m["S"], m["S_n"], info, width_avg=width_avg,
                         width_mode=width_mode, step=step, band_min_prom=band_min_prom)
    prof["info"] = info
    prof["stain"] = m
    return prof


def profile_of_file(path: str, **kw) -> dict:
    return profile_of_window(load_rgb(path), **kw)


# ============================================================================
# 4b. 把臂"拉直"成一条规范化长条（中线坐标系上的二维采样）
# ============================================================================
def rectify_strip(rgb: np.ndarray, S: np.ndarray, S_n: np.ndarray, info: dict,
                  out_w: int = 96, out_h: int = 48, span_w: float = 4.0,
                  half_w: float = 1.6, width_frac: float = 1.0) -> dict:
    """在中线坐标系里重采样出一张"拉直的长条"。

    为什么需要它（这是此前几个版本都缺的一环）：
      * 端部窗口里臂是斜的、弯的、位置随图漂移的，网络必须自己学会忽略这些；
      * 只取一维剖面（profile）又会把**横向**的信息（带的胖瘦、边缘锐利度）丢掉。
    把中线当坐标系：横轴 = 沿轴的弧长（单位：臂宽 W），纵轴 = 垂直中线（单位 W），
    就同时解决了"朝向/平移/尺度"三个变量，同时保留了横向结构。

    采样范围按《标注方案_优化建议》2.2 的规范化裁剪口径：沿轴 4 个臂宽、
    横向 ±1.6 个臂宽（正好把整条臂连同两侧背景一起包住）。

    返回 (strip dict)，其中 *_od / *_gray 是 (out_h, out_w) 的 float32。
    """
    path = info.get("path")
    if path is None or len(path) < 4 or not info.get("ok"):
        return {"ok": False, "reason": "无可用中线"}

    arr = np.asarray(path, np.float64)
    tip = np.asarray(info["tip_xy"][::-1], np.float64)
    tip_is_start = float(np.linalg.norm(arr[0] - tip)) <= float(np.linalg.norm(arr[-1] - tip))
    if not tip_is_start:
        arr = arr[::-1]

    samples, tangents, total = _resample_by_arc(arr, step=1.0)
    if len(samples) < 8:
        return {"ok": False, "reason": "中线过短"}

    dist = distance_transform(info["mask"])
    wv = 2.0 * _bilinear(dist, samples[:, 0], samples[:, 1])
    wv = conv1d_reflect(wv.astype(np.float32)[None, :], np.ones(3) / 3.0, axis=1)[0]
    good = np.isfinite(wv) & (wv > 1.0)
    W = float(np.median(wv[good])) if good.any() else float(total) / 4.0
    W = max(W, 4.0)

    U = min(span_w * W * width_frac, float(total))     # 沿轴取 4 个臂宽（不够长就取到底）
    # 沿轴坐标（像素）→ 直接线性插值到 samples 上
    u = np.linspace(0.0, U, out_w)
    py = np.interp(u, np.arange(len(samples)), samples[:, 0])
    px = np.interp(u, np.arange(len(samples)), samples[:, 1])
    ty = np.interp(u, np.arange(len(samples)), tangents[:, 0])
    tx = np.interp(u, np.arange(len(samples)), tangents[:, 1])
    nrm = np.hypot(ty, tx) + EPS
    ny, nx = -tx / nrm, ty / nrm                      # 法向

    v = (np.linspace(-half_w, half_w, out_h))[:, None] * W   # (out_h, 1) 像素
    ys = py[None, :] + v * ny[None, :]
    xs = px[None, :] + v * nx[None, :]

    # 画幅之外的取样点：按 0（=背景）填，而不是让双线性"复制"边缘像素。
    # 否则横向取到 ±1.6 个臂宽时，超出画面的行会被同一个边缘像素填成竖条纹，
    # 在拉直图里看起来像"条纹结构"，其实是采样伪影。
    h, w = rgb.shape[:2]
    inside = (ys >= 0) & (ys <= h - 1) & (xs >= 0) & (xs <= w - 1)

    def _samp(field):
        out = _bilinear(field, ys, xs)
        return np.where(inside, out, 0.0).astype(np.float32)

    gray_field = (rgb @ LUMA).astype(np.float32)
    strip = dict(
        ok=True,
        strip_od=_samp(S_n),
        strip_odabs=_samp(S),
        strip_gray=_samp(gray_field),
        strip_mask=_samp(dist),
        strip_inside=inside.astype(np.float32),
        W=W, U=U, out_w=out_w, out_h=out_h, span_w=span_w, half_w=half_w,
        path_xy=samples[:, ::-1].copy(),
        n_points=int(len(samples)), length_px=float(total),
    )
    return strip


# ============================================================================
# 5. 输出成表（与 data_code_figure/data 同格式）
# ============================================================================
def profile_table(prof: dict):
    """返回 (逐点表 rows, 条带表 rows)；前两列与旧数据逐字相同。"""
    if not prof.get("ok"):
        return [], []
    rows = []
    for i in range(prof["n_points"]):
        wv = prof["width"][i]
        rows.append([int(prof["s"][i]), f"{prof['gray'][i]:.4f}",
                     f"{float(wv):.1f}" if np.isfinite(wv) else "",
                     f"{prof['grad'][i]:.3f}"])
    brows = []
    for b in prof["bands"]:
        brows.append([b["band_id"], f"{b['s_center']:.1f}", f"{b['s_start']:.1f}",
                      f"{b['s_end']:.1f}", f"{b['width_px']:.1f}",
                      f"{b['gray_min']:.1f}", f"{b['prominence']:.1f}"])
    return rows, brows


def write_profile_csv(prof: dict, out_csv: str, bands_csv: str | None = None) -> None:
    import csv
    rows, brows = profile_table(prof)
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(TABLE_COLUMNS)
        wr.writerows(rows)
    if bands_csv:
        with open(bands_csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(BAND_COLUMNS)
            wr.writerows(brows)


# ============================================================================
# 6. 识别用的定长特征
# ============================================================================
def _resample_1d(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return np.zeros(n)
    if x.size == n:
        return x.copy()
    t = np.linspace(0.0, 1.0, x.size)
    u = np.linspace(0.0, 1.0, n)
    return np.interp(u, t, x)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    s = float(np.std(x))
    if s < 1e-9:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / s


def profile_features(prof: dict, n_len: int = 48, n_wide: int = 24,
                     n_band: int = 6) -> dict:
    """由一条曲线算出识别用的特征块（每块都是定长向量）。

    prof_len   : 灰度曲线按"全长归一"重采样后 z-score（形状，与绝对长度无关）
    prof_abs   : 灰度曲线按"绝对弧长"分箱（0–160 px，每 8 px 一格）——保留
                 "第几像素处有条带"这种绝对信息，这是端部识别真正的判别点
    od_len     : 同一套，但用染色量（OD），对比更强
    wide_len   : 横向宽度曲线（全长归一 + z-score）——与灰度近乎独立的一路信息
    band_sig   : 条带序列签名：前 6 条带的（相对位置、相对宽度、归一 prominence）
    band_stat  : 条带统计：条数、带宽中位/离散、prominence 中位、占空比
    """
    if not prof.get("ok"):
        return dict(ok=False)

    gray = prof["gray"]
    od = prof["od"]
    L = max(1.0, float(prof["length_px"]))

    # --- 形状块（全长归一） ---
    prof_len = _zscore(_resample_1d(gray, n_len))
    od_len = _zscore(_resample_1d(od, n_len))

    wv = np.asarray(prof["width"], np.float64)
    good = np.isfinite(wv) & (wv > 0)
    if good.sum() >= 8:
        w_fill = np.interp(np.arange(wv.size), np.nonzero(good)[0], wv[good])
        # 宽度只用相对形状（绝对粗细受放大倍数影响，不可比）
        wide_len = _zscore(_resample_1d(w_fill, n_wide))
        w_med = float(np.median(w_fill))
    else:
        wide_len = np.zeros(n_wide)
        w_med = float("nan")

    # --- 绝对弧长分箱（0–160 px） ---
    edges = np.arange(0, 161, 8, dtype=np.float64)          # 21 个边界 = 20 格
    prof_abs = np.zeros(len(edges) - 1)
    od_abs = np.zeros(len(edges) - 1)
    width_abs = np.zeros(len(edges) - 1)
    s = prof["s"]
    for i in range(len(edges) - 1):
        sel = (s >= edges[i]) & (s < edges[i + 1])
        if sel.any():
            prof_abs[i] = float(np.mean(gray[sel]))
            od_abs[i] = float(np.mean(od[sel]))
            width_abs[i] = float(np.nanmean(wv[sel])) if np.isfinite(wv[sel]).any() else 0.0
    # 用整条曲线的均值/标准差归一，避免"这张片子深/淡"直接变成特征
    prof_abs = _zscore(prof_abs) if np.std(prof_abs) > 1e-9 else prof_abs
    od_abs = _zscore(od_abs) if np.std(od_abs) > 1e-9 else od_abs
    width_abs = _zscore(width_abs) if np.std(width_abs) > 1e-9 else width_abs

    # --- 条带签名 ---
    bands = prof["bands"] or []
    bands = sorted(bands, key=lambda b: b["s_center"])
    band_sig = np.zeros(n_band * 3, np.float64)
    band_mask = np.zeros(n_band, np.float64)
    for k, b in enumerate(bands[:n_band]):
        band_sig[k * 3 + 0] = b["s_center"] / L
        band_sig[k * 3 + 1] = b["width_px"] / L
        band_sig[k * 3 + 2] = b["prominence"] / (float(np.ptp(gray)) + EPS)
        band_mask[k] = 1.0

    if bands:
        bw = np.array([b["width_px"] for b in bands], np.float64)
        bp = np.array([b["prominence"] for b in bands], np.float64)
        band_stat = np.array([
            len(bands),
            float(np.median(bw)) / L,
            float(np.std(bw)) / L if len(bw) > 1 else 0.0,
            float(np.median(bp)) / (float(np.ptp(gray)) + EPS),
            float(np.max(bp)) / (float(np.ptp(gray)) + EPS),
            float(bw.sum()) / L,                     # 占空比（带宽之和 / 臂长）
            float(np.median(np.diff([b["s_center"] for b in bands]))) / L
            if len(bands) > 1 else 0.0,
        ], np.float64)
    else:
        band_stat = np.zeros(7)

    geo = np.array([
        L,
        w_med,
        float(np.ptp(gray)),
        float(np.std(gray)),
        float(np.std(wv[np.isfinite(wv)])) if np.isfinite(wv).any() else 0.0,
        float(len(bands)),
        L / max(w_med, EPS) if np.isfinite(w_med) and w_med > 0 else 0.0,
    ], np.float64)

    return dict(ok=True, prof_len=prof_len, od_len=od_len, prof_abs=prof_abs,
                od_abs=od_abs, wide_len=wide_len, width_abs=width_abs,
                band_sig=band_sig, band_mask=band_mask, band_stat=band_stat, geo=geo)


FEATURE_BLOCKS = ("prof_len", "od_len", "prof_abs", "od_abs", "wide_len",
                  "width_abs", "band_sig", "band_mask", "band_stat", "geo")
BLOCK_DIMS = dict(prof_len=48, od_len=48, prof_abs=20, od_abs=20, wide_len=24,
                  width_abs=20, band_sig=18, band_mask=6, band_stat=7, geo=7)


def stack_features(feats: list) -> tuple:
    """把一列 profile_features 结果拼成 (X, 块名 -> 列区间)。失败的样本按零向量补。"""
    blocks = {}
    xs = []
    at = 0
    for name in FEATURE_BLOCKS:
        width = BLOCK_DIMS[name]
        A = np.zeros((len(feats), width), np.float32)
        for i, f in enumerate(feats):
            if f.get("ok"):
                v = np.atleast_1d(np.asarray(f[name], np.float64))
                if v.size == width:
                    A[i] = v
        blocks[name] = (at, at + width)
        xs.append(A)
        at += width
    X = np.concatenate(xs, axis=1) if xs else np.zeros((len(feats), 0), np.float32)
    return X, blocks
