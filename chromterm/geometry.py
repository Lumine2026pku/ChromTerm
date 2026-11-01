# -*- coding: utf-8 -*-
"""geometry.py —— 流程说明第 5～9 步：方向估计、分割修补、淡端外推、骨架与量测。

对应关系（文档章节 -> 本文件函数）：
  3.5 沿长轴平滑      -> anisotropic_smooth()
  3.6 二值化与修补    -> segment_stain() / repair_mask()
  3.7 淡端外推        -> extend_terminal()
  3.8 骨架化与剪枝    -> skeletonize() / prune_skeleton()
  3.9 定两端、量数据  -> longest_path() / measure_path()

全部用 numpy 重写（运行环境没有 cv2 / skimage）。相对原流程的两处改进：
  * 方向估计不用"全局单方向"硬编码，而是**结构张量 + 一致性权重**：
    对平直臂得到与全局一致的方向；对 U 形臂，在局部窗口上取各向异性最强的方向，
    并用 coherence 判断"这个方向可不可信"，不可信时退回全局方向（避免原流程 4.2 节
    说的"分段把好图改坏"）。
  * 淡端外推改成**扇形搜索**：不只看切线正前方一个圆盘，而是在 ±25° 内扫若干方向，
    取"加权信号最大"的那条继续走，能跟上端部的轻微弯曲（U/V 形常见）。
"""

from __future__ import annotations

import numpy as np

from .common import (EPS, conv1d_reflect, dilate_disc, fill_holes, gaussian_blur,
                    label_components, largest_component, mad_sigma, morph_open,
                    otsu_threshold, resize_bilinear)


# ------------------------------------------------------------------ 方向
def structure_tensor(s: np.ndarray, sigma_grad: float = 1.0, sigma_tensor: float = 6.0):
    """结构张量。返回 (Jxx, Jyy, Jxy)。

    s 为"染色量"图（越亮=染色越浓）。梯度方向垂直于带纹 = 沿臂方向，
    所以 J 的主特征向量就是臂轴方向。
    """
    b = gaussian_blur(s, sigma_grad)
    gy, gx = np.gradient(b)
    jxx = gaussian_blur(gx * gx, sigma_tensor)
    jyy = gaussian_blur(gy * gy, sigma_tensor)
    jxy = gaussian_blur(gx * gy, sigma_tensor)
    return jxx, jyy, jxy


def axis_from_tensor(s: np.ndarray, sigma_tensor: float = 6.0, mask=None):
    """返回 (沿臂方向角度°, 一致性 coherence 0..1)。角度约定：0°=沿 +x（图像向右）。

    一致性 = (λ1-λ2)/(λ1+λ2)，越大说明"这个方向"越可信。
    """
    jxx, jyy, jxy = structure_tensor(s, 1.0, sigma_tensor)
    if mask is not None and mask.sum() > 20:
        w = mask.astype(np.float32)
        jxx, jyy, jxy = jxx * w, jyy * w, jxy * w
    trace = jxx + jyy
    diff = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2 + 1e-9)
    coherence = float(np.mean(diff / (trace + 1e-6))) if trace.size else 0.0
    # 主特征向量方向（梯度方向）
    theta = 0.5 * np.arctan2(2 * np.mean(jxy), np.mean(jxx - jyy)) if trace.size else 0.0
    # 转成"沿臂方向"：梯度方向 = 臂轴，方向 ±90° 等价，取到 (-90, 90]
    ang = np.rad2deg(theta)
    while ang > 90:
        ang -= 180
    while ang <= -90:
        ang += 180
    return float(ang), float(np.clip(coherence, 0, 1))


def anisotropic_smooth(s: np.ndarray, angle_deg: float, sigma_along: float = 30.0,
                       sigma_across: float = 2.0) -> np.ndarray:
    """沿臂方向强平滑、垂直方向弱平滑（文档 3.5 的核心技巧）。

    angle_deg 为沿臂方向；内部先把图转到"臂水平"，平滑后再转回。
    """
    from .common import rotate_about
    h, w = s.shape
    rot = rotate_about(s, -angle_deg, out_shape=(h, w))
    sm = gaussian_blur(rot, sigma_along, sigma_across)
    return rotate_about(sm, angle_deg, out_shape=(h, w))


# ------------------------------------------------------------------ 分割
def background_level(x: np.ndarray, q: float = 20.0) -> tuple[float, float]:
    """估计"背景水平"和"噪声σ"。

    原流程用"背景中位数 + 1.5σ"作阈值下限；这在整片图上成立（染色体只占几个百分点），
    但在 246×112 的端部小图上，臂往往占 30%–60% 的面积，**中位数已经不是背景**了
    （实测会把下限抬到 0.45，掩膜被切成碎片）。
    这里改用低分位（默认 p20）作背景水平，噪声取"低于中位数那一半"的 MAD，
    两者都不受臂所占面积比例的影响。
    """
    v = np.asarray(x, dtype=np.float32).ravel()
    bg = float(np.percentile(v, q))
    lo = v[v <= np.percentile(v, 50)]
    noise = mad_sigma(lo) if lo.size > 20 else mad_sigma(v)
    noise = max(float(noise), 1e-3)
    return bg, noise


def segment_stain(s: np.ndarray, angle_deg: float = 0.0, thr_frac: float = 0.8,
                  sigma_along: float = 30.0, sigma_across: float = 2.0,
                  use_smooth: bool = True, noise: float | None = None,
                  bg_q: float = 20.0):
    """阈值 + 修补，得到"染色体在哪"的二值掩膜。

    s       : 归一化染色量 (0..~1)，建议用 preprocess 给出的 S/S95
    thr_frac: 阈值 = Otsu × thr_frac（<1 偏松，宁可把淡端包进来）
    返回 dict(mask, smooth, thr, otsu, bg)
    """
    sm = anisotropic_smooth(s, angle_deg, sigma_along, sigma_across) if use_smooth else s
    otsu = otsu_threshold(sm)
    bg, noise_bg = background_level(sm, bg_q)
    if noise is not None:
        noise_bg = max(noise_bg, float(noise))
    thr = max(otsu * thr_frac, bg + 1.5 * noise_bg)
    mask = sm > thr
    return dict(mask=mask, smooth=sm, thr=float(thr), otsu=float(otsu), bg=bg)


def repair_mask(mask: np.ndarray, s: np.ndarray, open_r: int = 1,
                core_grow: int | None = 35, core_q: float | None = None) -> np.ndarray:
    """填孔 + 开运算去毛刺 + "离横纹核心太远"的区域裁掉 + 只留最大连通块。

    「核心」的取法很关键：原流程用"亮度前 15%"，这在整片照片上会失效——端粒/横纹
    只占画面的几个百分点，第 85 百分位其实还是背景，于是薄雾被当成核心、掩膜连带
    背景一起长出来（实测 800 px 图上掩膜面积占到 24%，端点全落在背景里）。
    这里改成「不低于最亮值的 45%，且明显高于背景噪声」，薄雾立刻被排除。
    """
    m = fill_holes(mask)
    if open_r > 0:
        m = morph_open(m.astype(np.float32), open_r) > 0.5
    if core_grow is not None:
        hi = float(np.percentile(s, 99.5))
        bg, noise = background_level(s, 20.0)
        thr_core = max(0.45 * hi, bg + 5.0 * noise)
        core = s >= thr_core
        if core.sum() < 20:                      # 极端淡图：退回分位数判据
            core = s >= float(np.percentile(s, 99.0))
        allowed = dilate_disc(core.astype(np.float32), int(core_grow)) > 0.5
        m = m & allowed
    m, _ = largest_component(m, 8)
    return m


def mask_props(mask: np.ndarray):
    """掩膜的质心、外接框、面积、主轴长度。"""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return dict(area=0, cx=np.nan, cy=np.nan, bbox=None, extent=0.0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    return dict(area=int(ys.size), cx=float(xs.mean()), cy=float(ys.mean()),
                bbox=(int(x0), int(y0), int(x1), int(y1)),
                extent=float(max(y1 - y0 + 1, x1 - x0 + 1)))


def arm_width_estimate(mask: np.ndarray, angle_deg: float) -> float:
    """用面积 / 轴向长度估一次平均臂宽（作为外推圆盘半径的初值）。"""
    p = mask_props(mask)
    if p["area"] == 0:
        return 0.0
    from .common import rotate_about
    rot = rotate_about(mask.astype(np.float32), -angle_deg, out_shape=mask.shape)
    col = rot.sum(0)                     # 每列（= 垂直于臂的一刀）上的臂像素数
    col = col[col > 0]
    if col.size == 0:
        return 1.0
    return max(1.0, float(np.median(col)))


def extend_terminal(mask: np.ndarray, s: np.ndarray, end_xy, direction, width: float,
                    max_steps: int = 60, fan_deg: float = 25.0, noise: float | None = None,
                    arm_mean: float | None = None) -> tuple[np.ndarray, int]:
    """淡端外推：从端点沿切线走，把"还比背景亮"的圆盘并进掩膜（文档 3.7）。

    改进点：在 ±fan_deg 内扫 5 个方向，取"圆盘内均值最大"的方向继续走，
    因此轻微弯曲的端部也能跟上；同时若前进方向的信号明显低于门槛就停止。

    实现注意：只在候选圆盘的小邻域切片上算均值（而不是整幅 mgrid），
    否则 1600 px 的图上这一步会比其余所有步骤加起来还慢。
    """
    m = mask.copy()
    h, w = m.shape
    if noise is None:
        noise = mad_sigma(s)
    floor = max(noise * 0.8, 0.05)
    if arm_mean is None:
        arm_mean = float(np.mean(s[mask])) if mask.any() else float(np.percentile(s, 90))
    y, x = float(end_xy[1]), float(end_xy[0])
    a = float(np.arctan2(direction[1], direction[0]))
    r = max(2.0, 0.5 * width)
    rad = max(1, int(round(0.55 * width)))
    steps = 0
    for _ in range(max_steps):
        best = None
        for da in np.deg2rad(np.linspace(-fan_deg, fan_deg, 5)):
            ang = a + da
            ny, nx = y + r * np.sin(ang), x + r * np.cos(ang)
            iy, ix = int(round(ny)), int(round(nx))
            y0, y1 = max(0, iy - rad), min(h, iy + rad + 1)
            x0, x1 = max(0, ix - rad), min(w, ix + rad + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            sy, sx = np.mgrid[y0:y1, x0:x1]
            disc = (sy - ny) ** 2 + (sx - nx) ** 2 <= rad ** 2
            if disc.sum() == 0:
                continue
            val = float(s[y0:y1, x0:x1][disc].mean())
            if best is None or val > best[0]:
                best = (val, ang, ny, nx, (y0, y1, x0, x1, disc))
        if best is None or best[0] < max(floor, 0.25 * arm_mean):
            break
        val, ang, ny, nx, (y0, y1, x0, x1, disc) = best
        a = ang
        y, x = ny, nx
        block = m[y0:y1, x0:x1]
        block |= disc
        m[y0:y1, x0:x1] = block
        steps += 1
    return m, steps


# ------------------------------------------------------------------ 骨架
def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 细化，输出 1 像素宽、居中的骨架（等价 skimage.morphology.skeletonize）。"""
    img = mask.astype(np.uint8).copy()
    if img.sum() == 0:
        return img.astype(bool)
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1, mode="constant")
            P2 = p[0:-2, 1:-1]
            P3 = p[0:-2, 2:]
            P4 = p[1:-1, 2:]
            P5 = p[2:, 2:]
            P6 = p[2:, 1:-1]
            P7 = p[2:, 0:-2]
            P8 = p[1:-1, 0:-2]
            P9 = p[0:-2, 0:-2]
            nb = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(nb)
            A = sum(((1 - nb[i]) * nb[(i + 1) % 8]) for i in range(8))
            if step == 0:
                c1 = P2 * P4 * P6 == 0
                c2 = P4 * P6 * P8 == 0
            else:
                c1 = P2 * P4 * P8 == 0
                c2 = P2 * P6 * P8 == 0
            rem = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if rem.any():
                img[rem] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skel_neighbors(skel: np.ndarray):
    """返回每个骨架像素的邻居数（8 邻域）。"""
    p = np.pad(skel.astype(np.int16), 1)
    h, w = skel.shape
    B = np.zeros((h, w), np.int16)
    for dy, dx in _NB8:
        B += p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return np.where(skel, B, 0)


def skel_endpoints(skel: np.ndarray) -> np.ndarray:
    n = _skel_neighbors(skel)
    return skel & (n == 1)


def skeleton_graph(skel: np.ndarray):
    """骨架像素 -> 邻接表（局部索引）。"""
    ys, xs = np.nonzero(skel)
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys.tolist(), xs.tolist()))}
    adj = [[] for _ in range(len(ys))]
    for (y, x), i in idx.items():
        for dy, dx in _NB8:
            j = idx.get((y + dy, x + dx))
            if j is not None:
                adj[i].append(j)
    coords = list(zip(ys.tolist(), xs.tolist()))
    return coords, adj, idx


def _bfs_far(adj, start):
    """从 start 出发的单源最短路（8 邻接无权重 BFS），返回 (dist, prev)。"""
    n = len(adj)
    dist = np.full(n, -1, np.int32)
    prev = np.full(n, -1, np.int32)
    dist[start] = 0
    q = [start]
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                prev[v] = u
                q.append(v)
    return dist, prev


def longest_path(skel: np.ndarray, start_idx: int | None = None):
    """骨架上的"最长测地路径"（树的直径，两趟 BFS）。

    返回 (path_coords[(y,x)...], length_px)。
    """
    coords, adj, idx = skeleton_graph(skel)
    if not coords:
        return [], 0.0
    # 从任一点出发的最远点，再取该点出发的最远点
    s0 = 0 if start_idx is None else int(start_idx)
    dist, _ = _bfs_far(adj, s0)
    dist = np.where(dist < 0, -1, dist)
    a = int(np.argmax(dist))
    d2, prev = _bfs_far(adj, a)
    b = int(np.argmax(d2))
    path = []
    cur = b
    while cur != -1:
        path.append(coords[cur])
        if cur == a:
            break
        cur = int(prev[cur])
    path.reverse()
    if len(path) < 2:
        return path, 0.0
    arr = np.asarray(path, np.float32)
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    return path, float(seg.sum())


def prune_skeleton(skel: np.ndarray, min_frac: float = 0.25) -> tuple[np.ndarray, int]:
    """反复剪掉"从端点出发、短于主径 min_frac 的分支"。返回 (剪枝后骨架, 剪掉的枝数)。"""
    s = skel.copy()
    pruned = 0
    for _ in range(20):
        path, total = longest_path(s)
        if total < 3:
            break
        cutoff = max(4.0, total * min_frac)
        coords, adj, idx = skeleton_graph(s)
        if not adj:
            break
        ends = [i for i in range(len(adj)) if len(adj[i]) == 1]
        if len(ends) <= 2:
            break
        # 主干像素集合（用于识别"从端点走到主干"的距离）
        main = set(path)
        removed_any = False
        for e in ends:
            # 沿骨架从该端点走到第一个"属于主干"或"分叉"的点
            prev, cur, walked = -1, e, 0
            chain = [e]
            while True:
                nxt = [v for v in adj[cur] if v != prev]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
                walked += 1
                chain.append(cur)
                if coords[cur] in main:
                    break
                if walked > cutoff + 5:
                    break
            if walked <= cutoff and coords[cur] in main:
                for k in chain[:-1]:
                    y, x = coords[k]
                    s[y, x] = False
                pruned += 1
                removed_any = True
        if not removed_any:
            break
    return s, pruned


# ------------------------------------------------------------------ 量测
def _profile_at(s: np.ndarray, y: float, x: float, dy: float, dx: float,
                half: int = 40) -> np.ndarray:
    """沿 (dy,dx) 方向取一条 ±half 的双线性剖面。"""
    h, w = s.shape
    t = np.arange(-half, half + 1, dtype=np.float32)
    ys = y + t * dy
    xs = x + t * dx
    ok = (ys >= 0) & (ys <= h - 1) & (xs >= 0) & (xs <= w - 1)
    vals = np.full(t.size, np.nan, np.float32)
    if ok.sum() > 4:
        y0 = np.floor(ys[ok]).astype(np.int32)
        x0 = np.floor(xs[ok]).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        fx = xs[ok] - x0
        fy = ys[ok] - y0
        top = s[y0, x0] * (1 - fx) + s[y0, x1] * fx
        bot = s[y1, x0] * (1 - fx) + s[y1, x1] * fx
        vals[ok] = top * (1 - fy) + bot * fy
    return vals


def fwhm_width(vals: np.ndarray) -> float:
    """半高全宽：先减基线（剖面低分位），再找峰，量"降到峰值一半"两点间距离。"""
    v = vals[~np.isnan(vals)]
    if v.size < 5:
        return np.nan
    base = float(np.percentile(v, 15))
    vv = v - base
    if vv.max() <= 1e-6:
        return np.nan
    half = 0.5 * vv.max()
    above = np.nonzero(vv >= half)[0]
    if above.size == 0:
        return np.nan
    # 用线性插值找两侧半高处
    i0, i1 = above[0], above[-1]
    left = i0 if i0 == 0 else i0 - 1 + (half - vv[i0 - 1]) / (vv[i0] - vv[i0 - 1] + EPS)
    right = i1 if i1 == v.size - 1 else i1 + (half - vv[i1]) / (vv[i1 + 1] - vv[i1] + EPS)
    return float(right - left)


def resample_path(path, step: float = 3.0):
    """把像素级路径按弧长等间距重采样，顺便得到累计弧长。

    为什么必须重采样：8 邻接路径的点距在 1~1.41 px 之间跳变，直接求切线会得到
    锯齿状的假转角（曲率虚高一个量级）。按 ~3 px 等距重采样后，切线才反映真实走向。
    返回 (resampled (n,2) [y,x], cum, total)。
    """
    arr = np.asarray(path, np.float32)
    if len(arr) < 2:
        return arr, np.zeros(len(arr), np.float32), 0.0
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    total = float(cum[-1])
    if total < 1e-3:
        return arr[:1], np.zeros(1, np.float32), 0.0
    n = int(max(4, min(256, round(total / max(step, 0.5)) + 1)))
    t = np.linspace(0, total, n)
    y = np.interp(t, cum, arr[:, 0])
    x = np.interp(t, cum, arr[:, 1])
    return np.stack([y, x], 1).astype(np.float32), cum, total


def measure_path(path, s: np.ndarray, tip_is_start: bool = True) -> dict:
    """沿骨架量：长度、宽度（远端一半中位数）、端部朝向、弯曲度。"""
    if len(path) < 3:
        return dict(length_px=0.0, width_px=np.nan, tip_angle_deg=np.nan,
                    curvature_deg_per_100px=np.nan, width_profile=[])
    pts, cum_old, total = resample_path(path, step=3.0)
    if total < 3 or len(pts) < 4:
        return dict(length_px=float(total), width_px=np.nan, tip_angle_deg=np.nan,
                    curvature_deg_per_100px=np.nan, width_profile=[])
    if not tip_is_start:
        pts = pts[::-1]

    d = np.gradient(pts, axis=0)
    norm = np.linalg.norm(d, axis=1) + EPS
    tang = d / norm[:, None]

    widths = []
    for i in range(len(pts)):
        ty, tx = tang[i]
        wy, wx = -tx, ty                       # 法向
        vals = _profile_at(s, pts[i, 0], pts[i, 1], wy, wx, half=30)
        widths.append(((i / (len(pts) - 1)) * total, fwhm_width(vals)))
    wd_valid = [w for _, w in widths if np.isfinite(w)]
    # 远端一半（离端部最近的那一半）的代表宽度：不被染色中心的粗区拉高
    far_half = [w for c, w in widths if c <= 0.5 * total and np.isfinite(w)]
    width_px = float(np.median(far_half)) if far_half else (
        float(np.median(wd_valid)) if wd_valid else np.nan)

    # 端部朝向：末端约 40 px 做 PCA 拟合直线
    near_tip = np.linalg.norm(pts - pts[0], axis=1) <= 40.0
    tail = pts[near_tip] if near_tip.sum() >= 4 else pts[:min(len(pts), 8)]
    M = tail - tail.mean(0)
    w_, v_ = np.linalg.eigh(M.T @ M)
    v = v_[:, int(np.argmax(w_))]
    inner = pts[-1] - pts[0]
    if np.dot(v, inner) < 0:
        v = -v
    tip_angle = float(np.rad2deg(np.arctan2(v[1], v[0])))

    # 弯曲度：重采样后的切线累计转角 / 长度（度/100px）
    ang = np.arctan2(tang[:, 1], tang[:, 0])
    dang = np.abs(np.diff(np.unwrap(ang)))
    curv = float(np.rad2deg(dang.sum()) / max(total, 1.0) * 100.0)

    return dict(length_px=total, width_px=width_px, tip_angle_deg=tip_angle,
                curvature_deg_per_100px=curv,
                width_profile=widths, path_xy=pts[:, ::-1].copy())


def band_profile(path, s: np.ndarray, n_bins: int = 64, half: int = 16):
    """沿中线取"法向平均"的染色剖面（文档 2.3：把管变成序列）。"""
    if len(path) < 4:
        return None
    pts, _, total = resample_path(path, step=3.0)
    if len(pts) < 4 or total < 4:
        return None
    d = np.gradient(pts, axis=0)
    norm = np.linalg.norm(d, axis=1) + EPS
    tang = d / norm[:, None]
    # 按弧长均匀取 n_bins 个采样点（重采样已是等距，直接线性重索引即可）
    idx = np.linspace(0, len(pts) - 1, n_bins)
    prof = np.zeros(n_bins, np.float32)
    for k, i in enumerate(idx):
        i0 = int(np.floor(i)); i1 = min(i0 + 1, len(pts) - 1)
        f = i - i0
        p = pts[i0] * (1 - f) + pts[i1] * f
        tg = tang[i0] * (1 - f) + tang[i1] * f
        tg = tg / (np.linalg.norm(tg) + EPS)
        wy, wx = -tg[1], tg[0]
        vals = _profile_at(s, p[0], p[1], wy, wx, half=half)
        v = vals[~np.isnan(vals)]
        prof[k] = float(np.median(v) if v.size else 0.0)
    return prof


def band_features(prof: np.ndarray | None) -> dict:
    """从一维剖面里提"带纹"统计量：对比度、带的个数、间距、周期性。"""
    if prof is None or len(prof) < 8:
        return dict(band_contrast=0.0, n_bands=0.0, band_spacing=0.0, band_periodicity=0.0,
                    prof_smooth=None)
    p = prof.astype(np.float32)
    p = (p - p.mean()) / (p.std() + EPS)
    sm = gaussian_blur(p[None, :], 1.5)[0]
    thr = float(np.percentile(sm, 70))
    above = sm > thr
    # 连通段数 = 深带（横纹）个数
    n_bands = 0
    i = 0
    while i < len(above):
        if above[i]:
            n_bands += 1
            while i < len(above) and above[i]:
                i += 1
        else:
            i += 1
    peaks = np.nonzero(above)[0]
    spacing = float(np.median(np.diff(peaks))) if peaks.size > 2 else 0.0
    # 周期性：自相关第一峰
    ac = np.correlate(sm, sm, mode="full")[len(sm) - 1:]
    ac = ac / (ac[0] + EPS)
    per = 0.0
    for lag in range(4, len(ac) // 2):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.15:
            per = float(ac[lag])
            break
    return dict(band_contrast=float(np.percentile(sm, 90) - np.percentile(sm, 10)),
                n_bands=float(n_bands), band_spacing=spacing, band_periodicity=per,
                prof_smooth=sm)


# ------------------------------------------------------------------ 一站式
def analyse_window(s: np.ndarray, s_n: np.ndarray | None = None, thr_frac: float = 0.8,
                   sigma_along: float = 25.0, core_grow: int | None = None,
                   extend_frac: int = 40) -> dict:
    """对一张"端部小图"跑完整的第 5～9 步，返回掩膜、骨架、中线、量测。

    s   : 原始染色量（用于量宽度，保留绝对尺度）
    s_n : 归一化染色量（用于阈值与骨架，0..1）。缺省时用 s。
    core_grow=None：小图上不做"核心约束"（整张图就是端部，没有远端杂物可裁）
    """
    if s_n is None:
        s_n = s
    ang, coh = axis_from_tensor(s_n, sigma_tensor=max(4.0, min(s_n.shape) / 12))
    seg = segment_stain(s_n, ang, thr_frac=thr_frac, sigma_along=sigma_along,
                        sigma_across=2.0)
    mask = repair_mask(seg["mask"], seg["smooth"], open_r=1, core_grow=core_grow)
    info = dict(axis_deg=ang, coherence=coh, thr=seg["thr"], otsu=seg["otsu"],
                mask_area=int(mask.sum()))

    if mask.sum() < 25:
        info.update(ok=False, reason="掩膜过小")
        return info

    # 骨架
    skel = skeletonize(mask)
    if skel.sum() < 4:
        info.update(ok=False, reason="骨架过小", mask=mask, skel=skel)
        return info
    skel2, n_pruned = prune_skeleton(skel, min_frac=0.25)
    path, total = longest_path(skel2)
    info.update(ok=True, n_pruned=n_pruned, len_before_prune=float(total))

    if len(path) < 4:
        info.update(ok=False, reason="骨架路径过短")
        return info

    # ---- 定两端：离画面边框更远的一端 = 端部（端粒端，文档 3.9） ----
    # 注意：只能在**两个端点**之间比较，不能对整条路径取 argmax
    # （路径中点离四周都远，会被误选成"端部"）。
    arr = np.asarray(path, np.float32)          # (y, x)
    h, w = mask.shape

    def border_dist(p):
        return float(min(p[1], w - 1 - p[1], p[0], h - 1 - p[0]))

    b0, b1 = border_dist(arr[0]), border_dist(arr[-1])
    if abs(b0 - b1) >= 2.0:
        tip_is_start = b0 > b1
    else:
        # 两端都不贴边或都贴边：取"末端更细"的一端（端部收窄）
        yy, xx = np.mgrid[0:h, 0:w]
        cnt = []
        for p in (arr[0], arr[-1]):
            near = (yy - p[0]) ** 2 + (xx - p[1]) ** 2 <= 12 ** 2
            cnt.append(float((mask & near).sum()))
        tip_is_start = cnt[0] <= cnt[1]
    tip = arr[0] if tip_is_start else arr[-1]
    tip_border = border_dist(tip)
    prox_border = border_dist(arr[-1] if tip_is_start else arr[0])

    # ---- 淡端外推（只朝 tip 一侧） ----
    if extend_frac and extend_frac > 0:
        d_tip = arr[1] - arr[0] if tip_is_start else arr[-2] - arr[-1]
        d_tip = d_tip / (np.linalg.norm(d_tip) + EPS)
        width0 = arm_width_estimate(mask, ang)
        before = mask.sum()
        mask, n_ext = extend_terminal(mask, s, (tip[1], tip[0]), (d_tip[1], d_tip[0]),
                                      width0, max_steps=int(extend_frac))
        info["extend_steps"] = n_ext
        info["extend_px"] = int(mask.sum() - before)
        # 外推后再走一遍骨架
        skel = skeletonize(mask)
        skel2, n_pruned = prune_skeleton(skel, min_frac=0.25)
        path2, total2 = longest_path(skel2)
        if len(path2) >= 4:
            arr2 = np.asarray(path2, np.float32)

            def bd(p):
                return float(min(p[1], w - 1 - p[1], p[0], h - 1 - p[0]))

            b2a, b2b = bd(arr2[0]), bd(arr2[-1])
            tip_is_start = (b2a > b2b) if abs(b2a - b2b) >= 2.0 else tip_is_start
            path, arr = path2, arr2
            tip = arr[0] if tip_is_start else arr[-1]
            tip_border = bd(tip)
            total = total2

    meas = measure_path(path, s, tip_is_start=tip_is_start)
    prof = band_profile(path, s, n_bins=64, half=18)
    bf = band_features(prof)

    # ---- 质量标记 ----
    flags = []
    if tip_border < 3:
        flags.append("端部贴边(可能被裁)")
    if prox_border >= 1.5:
        flags.append("近端不在画面内(长度不完整)")
    if n_pruned > 2:
        flags.append(f"剪枝{n_pruned}条(可能弯折/粘连)")
    if info.get("len_before_prune", 0) - meas["length_px"] > 30:
        flags.append("剪枝缩短>30px")
    area_frac = float(mask.sum()) / (h * w)
    if area_frac < 0.05:
        flags.append("掩膜面积偏小(可能过淡)")
    if area_frac > 0.75:
        flags.append("掩膜面积偏大(可能选错对象)")
    if coh < 0.2:
        flags.append("方向一致性低")

    info.update(mask=mask, skel=skel2, path=path, measure=meas, bands=bf, band_prof=prof,
                tip_xy=(float(tip[1]), float(tip[0])), tip_border_px=tip_border,
                prox_in_frame=bool(prox_border < 1.5), flags=flags,
                area_frac=area_frac, fg_frac=float((s_n > 0.35).mean()))
    return info
