# -*- coding: utf-8 -*-
"""pfeat.py —— 由"中线曲线"算出识别特征（定长、可跨放大倍数比较）。

为什么要单独一层：`profile.py` 只管"把臂采成一条曲线"，识别好不好用取决于
**把曲线变成什么特征**。这一层把《中线采样说明》里的几个量组织成特征块，
并解决一个此前被忽略的关键问题——**尺度**。

尺度的病根
    端部窗口都是 246×112，但来自不同放大倍数（40X / 100X）与不同裁剪，
    同一条臂在窗口里可能是 90 像素长，也可能是 390 像素。所以"第 106 像素处
    有一条带"这样的特征在跨批次时毫无意义。

解法（与组里《标注方案_优化建议》2.2 节"规范化裁剪"同一个思路）
    把沿轴距离换算成**臂宽 W 的倍数**：s/W。臂被压扁后 W 也会跟着变，
    所以 s/W 在"同一放大倍数内部"保持不变、在不同放大倍数之间可比。
    实测这批窗口 W 中位约 32–40 px，条带沿轴宽度中位约 17 px ≈ 0.45 W，
    与说明书第 6 节的统计（40.5 px / 16.9 px）一致，说明这条换算立得住。

特征块（每块定长，缺样本补零）：
    shape_gray / shape_od / shape_w   全长归一的形状（与绝对长度无关）
    absW_gray / absW_od / absW_w       以 W 为单位的"绝对弧长分箱"（0–4W）
    band_sig / band_sigW / band_mask   条带序列签名（位置/宽度/深度，W 单位）
    band_stat                         条带统计（条数、占空比、间距、深浅）
    geo                               标量几何（长度/W、对比度、宽度变异…）
    dtw_proto                         到 5 个类别原型的 DTW 距离（见 ProtoDTW）
"""

from __future__ import annotations

import numpy as np

from .common import EPS

N_SHAPE = 48
N_SHAPE_W = 24
N_ABS = 20
ABS_MAX_W = 4.0          # 绝对弧长分箱覆盖 0 – 4 个臂宽
N_BAND = 6
BAND_SIG_DIM = N_BAND * 3
BAND_STAT_DIM = 8
GEO_DIM = 6

PROFILE_BLOCKS = ("shape_gray", "shape_od", "shape_w", "absW_gray", "absW_od",
                  "absW_w", "band_sig", "band_sigW", "band_mask", "band_stat", "geo")
BLOCK_DIMS = dict(shape_gray=N_SHAPE, shape_od=N_SHAPE, shape_w=N_SHAPE_W,
                  absW_gray=N_ABS, absW_od=N_ABS, absW_w=N_ABS,
                  band_sig=BAND_SIG_DIM, band_sigW=BAND_SIG_DIM,
                  band_mask=N_BAND, band_stat=BAND_STAT_DIM, geo=GEO_DIM)


# ------------------------------------------------------------------ 小工具
def _resample(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return np.zeros(n)
    if x.size == n:
        return x.copy()
    t = np.linspace(0.0, 1.0, x.size)
    return np.interp(np.linspace(0.0, 1.0, n), t, x)


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    s = float(np.std(x))
    return (x - float(np.mean(x))) / s if s > 1e-9 else np.zeros_like(x)


def _fill_nan(w: np.ndarray) -> np.ndarray:
    ok = np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return np.zeros_like(w)
    idx = np.arange(w.size)
    return np.interp(idx, idx[ok], w[ok])


def _bin_means(s: np.ndarray, v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.zeros(len(edges) - 1)
    ok = np.isfinite(v)
    if not ok.any():
        return out
    s, v = s[ok], v[ok]
    for i in range(len(edges) - 1):
        sel = (s >= edges[i]) & (s < edges[i + 1])
        if sel.any():
            out[i] = float(v[sel].mean())
    # 空箱用最近的有值箱补（而不是 0，避免制造假的"突然空洞"）
    filled = out != 0.0
    if filled.any() and not filled.all():
        idx = np.arange(len(out))
        out = np.interp(idx, idx[filled], out[filled])
    return out


# ------------------------------------------------------------------ 单窗特征
def window_features(g: np.ndarray, od: np.ndarray, w: np.ndarray,
                    bands: list, length: int, abs_mode: str = "W") -> dict:
    """一条曲线 → 一组定长特征块。

    g, od, w : 灰度 / 染色量 / 横向宽度（长度 = length，w 可含 NaN）
    bands    : 条带表（profile.detect_bands 的输出）
    """
    g = np.asarray(g, np.float64)[:length]
    od = np.asarray(od, np.float64)[:length]
    w = np.asarray(w, np.float64)[:length]
    if g.size < 8:
        return dict(ok=False)

    wf = _fill_nan(w)
    W = float(np.median(wf[wf > 0])) if (wf > 0).any() else float(length) / 4.0
    W = max(W, 4.0)
    L = float(max(1, g.size - 1))
    s = np.arange(g.size, dtype=np.float64)

    # ---- 形状（全长归一） ----
    shape_gray = _z(_resample(g, N_SHAPE))
    shape_od = _z(_resample(od, N_SHAPE))
    shape_w = _z(_resample(wf, N_SHAPE_W))

    # ---- 绝对弧长分箱 ----
    # abs_mode = "W" 时横轴是弧长 / 臂宽（跨放大倍数可比）；"px" 时是原始像素（对照用）
    if abs_mode == "px":
        scale, hi = 1.0, 160.0
    else:
        scale, hi = 1.0 / W, ABS_MAX_W
    edges = np.linspace(0.0, hi, N_ABS + 1)
    absW_gray = _z(_bin_means(s * scale, g, edges))
    absW_od = _z(_bin_means(s * scale, od, edges))
    absW_w = _z(_bin_means(s * scale, wf / W, edges))

    # ---- 条带签名 ----
    bands = sorted(bands or [], key=lambda b: b["s_center"])
    rng = float(np.ptp(g)) + EPS
    sig_px = np.zeros(BAND_SIG_DIM)
    sig_w = np.zeros(BAND_SIG_DIM)
    mask = np.zeros(N_BAND)
    for k, b in enumerate(bands[:N_BAND]):
        sig_px[k * 3 + 0] = b["s_center"] / L
        sig_px[k * 3 + 1] = b["width_px"] / L
        sig_px[k * 3 + 2] = b["prominence"] / rng
        sig_w[k * 3 + 0] = b["s_center"] / W
        sig_w[k * 3 + 1] = b["width_px"] / W
        sig_w[k * 3 + 2] = b["prominence"] / rng
        mask[k] = 1.0

    # ---- 条带统计 ----
    if bands:
        bw = np.array([b["width_px"] for b in bands], np.float64)
        bp = np.array([b["prominence"] for b in bands], np.float64)
        gap = np.diff([b["s_center"] for b in bands]) if len(bands) > 1 else np.zeros(1)
        band_stat = np.array([
            len(bands) / max(L / W, EPS),          # 每个臂宽里有几条带（尺度无关）
            float(np.median(bw)) / W,
            float(np.std(bw)) / W if len(bw) > 1 else 0.0,
            float(np.median(gap)) / W if gap.size else 0.0,
            float(bw.sum()) / L,                   # 占空比
            float(np.median(bp)) / rng,
            float(np.max(bp)) / rng,
            float(bands[0]["s_center"]) / W,       # 第一条带离端部多远（W 单位）
        ], np.float64)
    else:
        band_stat = np.zeros(BAND_STAT_DIM)

    # ---- 标量几何（全部尺度无关，只保留"形态"） ----
    geo = np.array([
        L / W,                                   # 窗口里装了几个臂宽
        rng / 255.0,                             # 灰度对比度
        float(np.std(g)) / 255.0,
        float(np.std(wf)) / (float(np.mean(wf)) + EPS),   # 宽度变异系数
        float(np.min(wf)) / W,
        float(np.max(wf)) / W,
    ], np.float64)

    return dict(ok=True, shape_gray=shape_gray, shape_od=shape_od, shape_w=shape_w,
                absW_gray=absW_gray, absW_od=absW_od, absW_w=absW_w,
                band_sig=sig_px, band_sigW=sig_w, band_mask=mask,
                band_stat=band_stat, geo=geo)


# ------------------------------------------------------------------ 批量
def build_matrix(prof: dict, abs_mode: str = "W", bands_by_name: dict | None = None):
    """由 `_cache/profiles.npz` 的内容拼出特征矩阵。

    返回 (X (N, D), blocks{name: (i0, i1)}, ok (N,))
    """
    names = [str(n) for n in prof["names"]]
    length = prof["length"].astype(int)
    G, O, W = prof["gray"], prof["od"], prof["width"]
    n = len(names)
    feats = []
    for i in range(n):
        L = int(length[i])
        if L < 8:
            feats.append(dict(ok=False))
            continue
        bands = (bands_by_name or {}).get(names[i], [])
        feats.append(window_features(G[i, :L], O[i, :L], W[i, :L], bands, L,
                                     abs_mode=abs_mode))
    X, blocks = stack(feats, PROFILE_BLOCKS)
    ok = np.array([bool(f.get("ok")) for f in feats], bool)
    return X, blocks, ok


def stack(feats: list, order=PROFILE_BLOCKS):
    blocks = {}
    xs = []
    at = 0
    for name in order:
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
    return np.concatenate(xs, axis=1), blocks


def select(X: np.ndarray, blocks: dict, names) -> np.ndarray:
    """按块名取列子集（保持 names 的顺序）。"""
    cols = []
    for nm in names:
        i0, i1 = blocks[nm]
        cols.append(X[:, i0:i1])
    return np.concatenate(cols, axis=1)


# ------------------------------------------------------------------ 一核约束求解
def hungarian(cost: np.ndarray) -> np.ndarray:
    """匈牙利算法（O(n^3)），返回每行指派到的列下标。自实现，不依赖 scipy。

    cost: (n, m) 且 n <= m。返回 (n,) 的列下标数组，使总代价最小。
    实现用的是最短增广路（Jonker-Volgenant 的经典形式），对这里的规模（n<=30）足够。
    """
    cost = np.asarray(cost, np.float64)
    n, m = cost.shape
    assert n <= m, "行数必须 <= 列数"
    INF = 1e18
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, np.int64)      # p[j] = 与列 j 匹配的行
    way = np.zeros(m + 1, np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = -np.ones(n, np.int64)
    for j in range(1, m + 1):
        if p[j] > 0:
            out[p[j] - 1] = j - 1
    return out


def constrained_labels(P: np.ndarray, cap: int = 2, n_other_slot: int | None = None,
                       n_arm: int = 5) -> np.ndarray:
    """在一张照片内做"每类最多 cap 个"的全局指派。

    P: (k, 6) 该图 k 个窗口的类别后验。返回 (k,) 的类别下标。

    生物学依据：二倍体唾腺细胞里每条臂只有两条同源染色体，所以一张片子里
    同一类出现 3 次以上几乎一定是错的。`other`（下标 5）不设上限。
    代价用 -log P（即"指派后总似然最大"），用匈牙利算法求最优。
    """
    P = np.asarray(P, np.float64)
    k = P.shape[0]
    slots = []
    for c in range(n_arm):
        slots += [c] * cap
    n_other = n_other_slot if n_other_slot is not None else k
    slots += [n_arm] * n_other
    slots = np.asarray(slots, np.int64)
    cost = -np.log(np.clip(P, 1e-9, 1.0))[:, slots]      # (k, m)，且 m >= k
    r = hungarian(cost)
    return slots[r]


# ------------------------------------------------------------------ DTW
def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """经典 DTW（无窗约束），代价 = 平方差。a、b 长度都是 ~48，够快。"""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    n, m = a.size, b.size
    prev = np.full(m, np.inf)
    prev[0] = (a[0] - b[0]) ** 2
    for j in range(1, m):
        prev[j] = prev[j - 1] + (a[0] - b[j]) ** 2
    cur = np.empty(m)
    for i in range(1, n):
        cur[0] = prev[0] + (a[i] - b[0]) ** 2
        # 列循环用 numpy 前缀最小：DTW 递推不满足前缀可分离，只能逐步
        for j in range(1, m):
            cur[j] = (a[i] - b[j]) ** 2 + min(prev[j], cur[j - 1], prev[j - 1])
        prev, cur = cur, prev
    return float(np.sqrt(prev[m - 1] / (n + m)))


class ProtoDTW:
    """每类一条"原型曲线"，把样本到原型的 DTW 距离当特征。

    原型 = 该类训练样本在"全长归一并 z-score"后的逐点中位数（比 DBA 便宜、
    在小样本上更稳）。距离最后除以 √2 并做 z-score，便于与其它特征一起进线性模型。

    **必须在每个 CV 折内重新 fit**，否则原型里已经包含验证样本，指标会虚高。
    """

    def __init__(self, n_proto_dim: int = 5):
        self.n_proto_dim = n_proto_dim
        self.proto = None

    def fit(self, shapes: np.ndarray, y: np.ndarray, n_cls: int = 6):
        self.proto = np.zeros((n_cls, shapes.shape[1]))
        self.present = np.zeros(n_cls, bool)
        for c in range(n_cls):
            sel = y == c
            if sel.sum() == 0:
                continue
            self.present[c] = True
            P = shapes[sel]
            med = np.median(P, axis=0)
            self.proto[c] = _z(med)
        return self

    def transform(self, shapes: np.ndarray) -> np.ndarray:
        n = shapes.shape[0]
        D = np.zeros((n, self.proto.shape[0]))
        for i in range(n):
            for c in range(self.proto.shape[0]):
                if self.present[c]:
                    D[i, c] = dtw_distance(shapes[i], self.proto[c])
                else:
                    D[i, c] = np.nan
        med = np.nanmedian(D, axis=1, keepdims=True)
        D = np.where(np.isfinite(D), D, med)
        # 类内相对：减去该样本到所有原型距离的中位数，再按列 z-score
        D = D - D.mean(axis=1, keepdims=True)
        sd = D.std(axis=0) + 1e-9
        return (D / sd).astype(np.float32)
