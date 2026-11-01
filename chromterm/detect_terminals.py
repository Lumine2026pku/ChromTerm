# -*- coding: utf-8 -*-
"""detect_terminals.py —— 从一张图里找出"末端"（可多个），供网页与批处理使用。

输入：一张染色体照片（整片），或一张端部小图（246×112 那种）
输出：0..N 个末端的列表，每个末端带位置、向内方向、臂宽、评分、质检标记

为什么单独写一个模块
--------------------
训练好的分类器只回答"这个端部像哪条臂"，它不回答"端部在哪"。
`geometry.analyse_window` 只追**最长的一条**骨架（适合单臂小图，整片图上会横穿多条臂）。
这里换成"骨架图 + 全部端点"的做法：

  1. 染色量 + 分割 + 修掩膜（与第 5–6 步同一套参数）
  2. 骨架化 + 剪短枝 → 得到"臂的骨架图"
  3. 图上所有**度为 1** 的像素就是候选端点（真实端粒端、被画面切断的近端、毛刺都会出现）
  4. 对每个端点沿骨架往里走 20–30 px，得到"向内方向"与"局部臂宽"
  5. 打分：往里走时变粗（端部收窄）加分、离画面边框远加分、宽度可测加分；
     贴边的端点不丢掉，只降权并标记"可能被裁"（它仍可能是用户想看的末端）
  6. 去重（同一臂两端/同一位置多个端点只留一个）

整片里"染色中心"那一端是度为 3 以上的分叉点，不是度为 1 的端点，所以不会被当成末端——
这正好符合"端粒端才是末端"的标注口径。
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .common import EPS, mad_sigma, resize_bilinear
from .geometry import (analyse_window, arm_width_estimate, fwhm_width, longest_path,
                      mask_props, prune_skeleton, repair_mask, segment_stain,
                      skeleton_graph, skeletonize, _profile_at)


# ------------------------------------------------------------------ 几何小工具
def _endpoints(skel: np.ndarray):
    """骨架里所有度为 1 的像素坐标 [(y, x), ...]。"""
    coords, adj, _ = skeleton_graph(skel)
    return [coords[i] for i in range(len(adj)) if len(adj[i]) == 1]


def _walk_inward(coords, adj, idx, start_i, max_steps: int = 30):
    """从端点沿骨架往里走，返回路径坐标列表（含起点）。"""
    path = [coords[start_i]]
    prev, cur = -1, start_i
    for _ in range(max_steps):
        nxt = [v for v in adj[cur] if v != prev]
        if len(nxt) != 1:                 # 走到分叉或尽头就停
            break
        prev, cur = cur, nxt[0]
        path.append(coords[cur])
    return path


def _width_at(mask: np.ndarray, s: np.ndarray, y: float, x: float,
              dy: float, dx: float) -> float:
    """在 (y,x) 处用法向剖面量宽度；量不出时退回"掩膜在这条法向上的跨度"。"""
    vals = _profile_at(s, y, x, dy, dx, half=24)
    w = fwhm_width(vals)
    if not np.isfinite(w) or w <= 0:
        h, wd = mask.shape
        t = np.arange(-24, 25)
        ys = np.clip((y + t * dy).astype(int), 0, h - 1)
        xs = np.clip((x + t * dx).astype(int), 0, wd - 1)
        on = mask[ys, xs]
        w = float(on.sum())
    return float(w)


# ------------------------------------------------------------------ 主过程
def find_terminals(rgb: np.ndarray, max_side: int = 800, max_terminals: int = 6,
                   min_width: float = 4.0, verbose: bool = False) -> dict:
    """在整张图上找候选末端。

    返回 dict(ok, scale, img_shape, mask, s_n, terminals=[...], flags=[...])
    每个 terminal: tip(x,y) / inward(ux,uy) / width / score / border_px / taper / flags

    两个关键点（第一版就栽在这里）：
      * **核心约束必须开**（core_grow）。整片照片常有弥散的背景薄雾，纯阈值会把薄雾
        连成一大片掩膜，骨架于是穿过背景、端点是假的（实测一张图 8 个候选全在背景里）。
        只保留"离最浓像素不超过 core_grow"的区域，薄雾立刻消失。
      * **阈值要自动回退**：不同片子的染色深浅差很多，固定 0.8×Otsu 在淡片上会切太松。
        按 0.8→1.0→1.2→1.5 逐步收紧，取第一个"掩膜面积占比落在 2%–30%"的结果。
    """
    h0, w0 = rgb.shape[:2]
    scale = min(1.0, float(max_side) / max(h0, w0))
    img = resize_bilinear(rgb, int(round(w0 * scale)), int(round(h0 * scale))) if scale < 1 else rgb
    h, w = img.shape[:2]

    from .prep_od import stain_map
    m = stain_map(img)
    s, s_n = m["S"], m["S_n"]

    core_grow = max(12, int(round(30 * scale)))
    mask, seg, area_frac, tried = None, None, None, []
    for tf in (0.8, 1.0, 1.2, 1.5):
        seg = segment_stain(s_n, 0.0, thr_frac=tf,
                            sigma_along=max(16.0, min(h, w) / 14.0), sigma_across=2.5)
        mk = repair_mask(seg["mask"], seg["smooth"], open_r=2, core_grow=core_grow)
        af = float(mk.sum()) / (h * w)
        tried.append((tf, round(af, 3)))
        if 0.02 <= af <= 0.30:
            mask, area_frac = mk, af
            break
        if mask is None or af < area_frac:
            mask, area_frac = mk, af
    if verbose:
        print(f"[detect] {w0}x{h0} -> {w}x{h}, thr_frac={seg['thr']:.3f}, area_frac={area_frac:.3f}, "
              f"tried={tried}")
    if mask is None or mask.sum() < 0.005 * h * w:
        return dict(ok=False, reason="掩膜过小（画面里几乎没有染色体）", mask=mask,
                    s_n=s_n, scale=scale, img_shape=(h0, w0), terminals=[])

    skel = skeletonize(mask)
    skel, n_pruned = prune_skeleton(skel, min_frac=0.18)
    coords, adj, idx = skeleton_graph(skel)
    if not coords:
        return dict(ok=False, reason="骨架为空", mask=mask, s_n=s_n, scale=scale,
                    img_shape=(h0, w0), terminals=[])

    end_i = [i for i in range(len(adj)) if len(adj[i]) == 1]
    cands = []
    for i in end_i:
        # 一路走到分叉/尽头，得到"这条臂"的长度；再沿前 40 px 量若干次宽度
        walk = _walk_inward(coords, adj, idx, i, max_steps=240)
        if len(walk) < 8:
            continue
        y0, x0 = coords[i]
        k = min(len(walk) - 1, 6)
        ux, uy = walk[k][1] - x0, walk[k][0] - y0
        n = math.hypot(ux, uy)
        if n < 1e-6:
            continue
        ux, uy = ux / n, uy / n
        dy, dx = -ux, uy                     # 法向

        # 宽度：沿前 40 px 取多点，取 25% 分位（合并处会把宽度拉大，取小的更接近真实臂宽）
        ws = []
        for t in (4, 8, 12, 16, 20, 26, 32, 40):
            if t >= len(walk):
                break
            yy, xx = walk[t]
            ws.append(_width_at(mask, s, yy, xx, dy, dx))
        ws = [v for v in ws if np.isfinite(v) and v > 0]
        if not ws:
            continue
        width = float(np.percentile(ws, 25))
        w_tip = _width_at(mask, s, y0 + 3 * uy, x0 + 3 * ux, dy, dx)
        if not np.isfinite(width) or width < min_width:
            continue

        # 这条臂有多长（以臂宽为单位）：真实端粒端后面总跟着一段像样的臂
        branch = []
        prev, cur = -1, i
        for _ in range(240):
            nxt = [v for v in adj[cur] if v != prev]
            if len(nxt) != 1:
                break
            prev, cur = cur, nxt[0]
            branch.append(coords[cur])
        branch_len = 0.0
        if branch:
            b = np.asarray(branch, np.float32)
            branch_len = float(np.linalg.norm(np.diff(b, axis=0), axis=1).sum())
        arm_lengths = branch_len / max(width, EPS)
        taper = (width - w_tip) / max(width, EPS) if np.isfinite(w_tip) else 0.0

        border = float(min(x0, w - 1 - x0, y0, h - 1 - y0))
        cut = border < max(2.0, 0.012 * min(h, w))
        flags = []
        if cut:
            flags.append("端部贴边(可能被裁)")
        if arm_lengths < 2.5:
            flags.append("后方臂太短(可能是毛刺/粘连)")
        if taper < -0.35:
            flags.append("端部突然变粗(可能不是端粒端)")

        # 评分：臂够长 > 端部收窄 > 不贴边
        s_len = float(np.clip(arm_lengths / 6.0, 0.0, 1.0))
        s_taper = float(np.clip(0.5 + taper, 0.0, 1.0))
        score = 0.55 * s_len + 0.30 * s_taper + (0.15 if not cut else 0.0)
        score *= 0.6 if cut else 1.0
        cands.append(dict(tip=(float(x0), float(y0)), inward=(ux, uy),
                          width=float(width), w_tip=float(w_tip), taper=float(taper),
                          arm_lengths=float(arm_lengths),
                          border_px=border, cut=bool(cut), score=float(score),
                          flags=flags, walk_len=len(walk)))

    # ---- 过滤 + 去重：真实端粒端后面总跟着一段像样的臂 ----
    strong = [c for c in cands if c["arm_lengths"] >= 2.0]
    pool = strong if strong else cands
    pool.sort(key=lambda c: -c["score"])
    keep = []
    for c in pool:
        x, y = c["tip"]
        r = max(8.0, 1.2 * c["width"])
        if any((x - k["tip"][0]) ** 2 + (y - k["tip"][1]) ** 2 < r ** 2 for k in keep):
            continue
        keep.append(c)
        if len(keep) >= max_terminals:
            break

    for c in keep:
        c["tip_full"] = (c["tip"][0] / scale, c["tip"][1] / scale)
        c["width_full"] = c["width"] / scale
    if verbose:
        print(f"[detect] 端点 {len(end_i)} 个 -> 候选 {len(cands)} -> 保留 {len(keep)} 个，"
              f"剪掉短枝 {n_pruned}")
    return dict(ok=bool(keep), mask=mask, s_n=s_n, scale=scale, img_shape=(h0, w0),
                area_frac=area_frac, n_endpoints=len(end_i), n_pruned=n_pruned,
                terminals=keep)


# ------------------------------------------------------------------ 取窗口
def extract_window(rgb: np.ndarray, tip, inward, width: float,
                   out_w: int = 246, out_h: int = 112, height_frac: float = 4.4,
                   tip_frac: float = 0.10):
    """按 data_share 的窗口口径取一张"端部小图"，并返回采样四角（画框用）。

    训练数据的窗口是"末端在左、臂水平、高约 4.5 个臂宽、长约 10 个臂宽"，这里就按这个比例取。
    """
    from .normalize import canonical_crop

    win = canonical_crop(rgb, tip, inward, width, out_w=out_w, out_h=out_h,
                         tip_frac=tip_frac, height_frac=height_frac)
    # 反算采样框在源图上的四角（与 canonical_crop 内部一致的平移逻辑）
    h_phys = height_frac * width
    w_phys = height_frac * out_w / out_h * width
    u = np.asarray(inward, np.float32)
    u /= (np.linalg.norm(u) + EPS)
    v = np.array([-u[1], u[0]], np.float32)
    t = np.asarray(tip, np.float32)
    H, W = rgb.shape[:2]
    ca = np.array([-tip_frac, 1.0 - tip_frac], np.float32)
    cb = np.array([-0.5, 0.5], np.float32)
    C = np.array([t + a * w_phys * u + b * h_phys * v for a in ca for b in cb], np.float32)
    dx = 0.0
    if C[:, 0].min() < 0:
        dx = -C[:, 0].min()
    if C[:, 0].max() + dx > W - 1:
        dx = (W - 1) - C[:, 0].max()
    dy = 0.0
    if C[:, 1].min() < 0:
        dy = -C[:, 1].min()
    if C[:, 1].max() + dy > H - 1:
        dy = (H - 1) - C[:, 1].max()
    t = t + np.array([dx, dy], np.float32)
    quad = [tuple((t + a * w_phys * u + b * h_phys * v).tolist()) for a in ca for b in cb]
    quad = [quad[0], quad[1], quad[3], quad[2]]      # 左上-右上-右下-左下
    return win, quad


# ------------------------------------------------------------------ 单张图 -> 结论
VERDICT_MIN_CONF = 0.45


def analyse_image(rgb: np.ndarray, predictor, max_terminals: int = 6,
                  prefer_single: bool | None = None, mode: str | None = None,
                  verbose: bool = False) -> dict:
    """给一张任意图片，返回"网页要用"的完整结果（JSON 友好）。

    模式：
      * mode="single"（网页默认，精度最高）：整张当作一个末端，居中裁 2.2:1 后判读
      * mode="multi"（本地服务，实验性）：先找多个候选末端，再逐个判读
      * mode=None：按尺寸/长宽比自动猜（大图 -> multi）
    单末端模式下，返回的 (x, y) 与 box 都换算回**原图坐标**，
    否则前端在大图上按窗口坐标画点会画到图外（这也是"点不显示"的一个原因）。
    """
    import time
    from .common import resize_bilinear as _rz

    t0 = time.time()
    h, w = rgb.shape[:2]
    aspect = w / max(h, 1)
    # mode 显式给定时直接用；否则按尺寸/长宽比猜（大图默认多末端）
    if mode in ("single", "multi"):
        prefer_single = (mode == "single")
    elif prefer_single is None:
        prefer_single = (max(h, w) <= 420) or (1.5 <= aspect <= 3.6 and max(h, w) <= 720)
    mode = "single" if prefer_single else "multi"
    out = dict(ok=True, mode=mode, image=dict(w=w, h=h),
               terminals=[], flags=[], scale=1.0)

    # ---- 整片：多端点 ----
    if mode == "multi":
        det = find_terminals(rgb, max_terminals=max_terminals, verbose=verbose)
        out["scale"] = det.get("scale", 1.0)
        out["area_frac"] = det.get("area_frac")
        if det.get("terminals"):
            for c in det["terminals"]:
                tip = c["tip_full"]
                width = c["width_full"]
                win, quad = extract_window(rgb, tip, c["inward"], width)
                pred = predictor.predict(win)
                g = pred["geom"]
                out["terminals"].append(dict(
                    x=float(tip[0]), y=float(tip[1]), quad=quad,
                    cls=pred["cls"], conf=pred["conf"], probs=pred["probs"],
                    second=pred["cls2"], conf2=pred["conf2"],
                    margin=float(pred["conf"] - pred["conf2"]),
                    length_px=_num(pred["length_px"]), width_px=_num(pred["width_px"]),
                    tip_angle_deg=_num(pred["tip_angle_deg"]),
                    curvature=_num(pred["curvature"]),
                    n_bands=_num(pred["n_bands"]),
                    axis_deg=_num(pred["axis_deg"]),
                    coherence=_num(pred["coherence"]),
                    prox_in_frame=bool(pred["prox_in_frame"]),
                    window_flags=_flags(pred),
                    detect_flags=c["flags"],
                    detect_score=round(float(c["score"]), 3),
                    tapered=round(float(c["taper"]), 3),
                    quality=_quality(pred, c),
                    crop=win))
            mode = "multi"
        else:
            out["flags"].append(det.get("reason") or "未找到可用的末端")

    # ---- 端部小图 / 回退：整张当一个末端 ----
    if not out["terminals"]:
        # 与训练口径一致：先居中裁成 2.2:1，再缩放到 246×112
        target = 246 / 112
        if w / h > target:
            cw = int(h * target); ch = h; x0 = (w - cw) // 2; y0 = 0
        else:
            ch = int(w / target); cw = w; x0 = 0; y0 = (h - ch) // 2
        crop = rgb[y0:y0 + ch, x0:x0 + cw]
        win = crop if (cw, ch) == (246, 112) else _rz(crop, 246, 112)
        pred = predictor.predict(win)
        g = pred["geom"]
        if not _has_signal(g, pred):
            out.update(ok=False, mode="none", verdict=None, n_terminals=0,
                       flags=["画面里没有可识别的染色体末端（对比度太低或掩膜过小）"])
            out["elapsed_s"] = round(time.time() - t0, 2)
            return out
        # 窗口坐标 -> 原图坐标
        tip = g.get("tip_xy") or (0.1 * 246, 0.5 * 112)
        px = x0 + (float(tip[0]) / 246.0) * cw
        py = y0 + (float(tip[1]) / 112.0) * ch
        out["terminals"].append(dict(
            x=px, y=py, quad=None,
            box=dict(x0=float(x0), y0=float(y0), cw=float(cw), ch=float(ch)),
            cls=pred["cls"], conf=pred["conf"], probs=pred["probs"],
            second=pred["cls2"], conf2=pred["conf2"],
            margin=float(pred["conf"] - pred["conf2"]),
            length_px=_num(pred["length_px"]), width_px=_num(pred["width_px"]),
            tip_angle_deg=_num(pred["tip_angle_deg"]),
            curvature=_num(pred["curvature"]), n_bands=_num(pred["n_bands"]),
            axis_deg=_num(pred["axis_deg"]), coherence=_num(pred["coherence"]),
            prox_in_frame=bool(pred["prox_in_frame"]), window_flags=_flags(pred),
            detect_flags=[], detect_score=1.0, tapered=0.0,
            quality=_quality(pred, dict(flags=[], taper=0.0)),
            crop=win))
        out["mode"] = "single"

    # ---- 结论 ----
    if not out["terminals"]:
        out.update(ok=False, mode="none", verdict=None, n_terminals=0)
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    # 排序：置信度高、以及非 other 优先
    def rank(t):
        s = t["conf"] - (0.10 if t["cls"] == "other" else 0.0)
        if "端部贴边(可能被裁)" in (t.get("detect_flags") or []):
            s -= 0.12
        return -s

    out["terminals"].sort(key=rank)
    for i, t in enumerate(out["terminals"], 1):
        t["id"] = i
    best = out["terminals"][0]
    verdict = dict(cls=best["cls"], conf=round(best["conf"], 4),
                   second=best["second"], margin=round(best["margin"], 3),
                   terminal_id=best["id"],
                   reliable=bool(best["conf"] >= VERDICT_MIN_CONF),
                   text=_verdict_text(best, len(out["terminals"])))
    out["verdict"] = verdict
    out["n_terminals"] = len(out["terminals"])
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


# ------------------------------------------------------------------ 小工具
def _flags(pred) -> list:
    """把 Predictor 返回的 'a;b' 字符串拆成列表（前端按数组渲染）。"""
    s = pred.get("flags") or ""
    return [x for x in str(s).split(";") if x]


def _num(v):
    try:
        v = float(v)
    except Exception:
        return None
    return round(v, 1) if np.isfinite(v) else None


def _quality(pred, c) -> str:
    """粗分四档，供前端上色。"""
    conf = float(pred["conf"])
    flags = list(pred["flags"] or []) + list(c.get("flags") or [])
    hard = any(("掩膜" in f) or ("分割" in f) or ("剪枝缩短" in f) for f in flags)
    if c.get("cut"):
        return "cut"
    if conf >= 0.55 and not hard:
        return "good"
    if conf >= 0.35 and not hard:
        return "fair"
    return "poor"


def _has_signal(geom, pred) -> bool:
    if not geom.get("ok"):
        return False
    af = float(geom.get("area_frac", 0) or 0)
    coh = float(geom.get("coherence", 0) or 0)
    return af >= 0.02 and coh >= 0.08


def _verdict_text(best, n: int) -> str:
    conf = float(best["conf"])
    if best["cls"] == "other":
        return (f"最可能不是五条主臂之一（other，{conf:.0%}），"
                f"建议按「断裂端/着丝粒端/模糊」处理")
    if conf >= VERDICT_MIN_CONF:
        if n > 1:
            return f"共找到 {n} 个候选末端，这是其中最确定的一个"
        return "整张图按一个末端判读"
    return (f"倾向 {best['cls']}，但置信度只有 {conf:.0%}；"
            f"第二名 {best['second']}，建议人工确认")
