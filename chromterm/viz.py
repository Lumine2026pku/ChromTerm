# -*- coding: utf-8 -*-
"""viz.py —— 用 Pillow 画的几张图（运行环境里没有 matplotlib）。

  confusion_png   混淆矩阵热图
  bars_png        指标条形图（消融对比）
  curve_png       折线（风险-覆盖率曲线、置信度分布）
  grid_png        多张小图拼图（带标题）
  overlay_png     在窗口图上叠：绿=掩膜边界，蓝=中线，红圈=端部，橙圈=近端
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .common import draw_text_safe, to_pil

_KNOTS = np.array([[12, 12, 24], [70, 40, 120], [190, 70, 70], [245, 170, 50],
                   [255, 252, 235]], np.float32) / 255.0
_POS = np.linspace(0, 1, len(_KNOTS))


def heat_color(v: float):
    v = float(np.clip(v, 0, 1))
    return tuple(int(round(255 * np.interp(v, _POS, _KNOTS[:, c]))) for c in range(3))


def confusion_png(cm: np.ndarray, classes, path: str, title: str = "",
                  cell: int = 46, pad_top: int = 56, pad_left: int = 78) -> None:
    cm = np.asarray(cm, np.float64)
    n = cm.shape[0]
    rown = np.maximum(cm.sum(1, keepdims=True), 1)
    frac = cm / rown
    W, H = pad_left + cell * n + 14, pad_top + cell * n + 44
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    draw_text_safe(d, (8, 8), title or "confusion matrix", (0, 0, 0), 15)
    for i in range(n):
        for j in range(n):
            c = heat_color(frac[i, j])
            x0, y0 = pad_left + j * cell, pad_top + i * cell
            d.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=c)
            tc = (0, 0, 0) if frac[i, j] > 0.55 else (245, 245, 245)
            draw_text_safe(d, (x0 + 8, y0 + 15), f"{cm[i, j]:.0f}", tc, 13)
    for j in range(n):
        draw_text_safe(d, (pad_left + j * cell + 8, pad_top - 24), str(classes[j]), (0, 0, 0), 13)
    for i in range(n):
        draw_text_safe(d, (12, pad_top + i * cell + 15), str(classes[i]), (0, 0, 0), 13)
    draw_text_safe(d, (10, H - 26), "rows = true, cols = predicted; numbers = counts",
                   (95, 95, 95), 11)
    img.save(path)


def bars_png(labels, values, path: str, title: str = "", vmin: float = 0.0,
             vmax: float = 1.0, highlight: int | None = None,
             color=(70, 110, 190)) -> None:
    n = len(labels)
    rowh, left, top, right = 26, 200, 46, 80
    H = top + n * rowh + 34
    W = left + 320 + right
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    draw_text_safe(d, (10, 10), title, (0, 0, 0), 15)
    span = max(vmax - vmin, 1e-6)
    for i, (lab, v) in enumerate(zip(labels, values)):
        y = top + i * rowh
        draw_text_safe(d, (8, y + 6), str(lab), (30, 30, 30), 12)
        w = max(0.0, (float(v) - vmin) / span) * 320
        col = (220, 120, 60) if (highlight is not None and i == highlight) else color
        d.rectangle([left, y + 3, left + w, y + rowh - 7], fill=col)
        draw_text_safe(d, (left + w + 5, y + 6), f"{float(v):.3f}", (40, 40, 40), 12)
    for k in range(6):
        x = left + 320 * k / 5
        d.line([x, top - 4, x, H - 24], fill=(228, 228, 228))
        draw_text_safe(d, (x - 8, H - 22), f"{vmin + span * k / 5:.2f}", (120, 120, 120), 10)
    img.save(path)


def curve_png(curves, path: str, title: str = "", xlabel: str = "coverage",
              ylabel: str = "accuracy") -> None:
    """curves: list of (label, xs, ys, color tuple)"""
    W, H = 540, 400
    left, top, right, bottom = 62, 46, 20, 48
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    draw_text_safe(d, (10, 10), title, (0, 0, 0), 15)
    pw, ph = W - left - right, H - top - bottom
    for k in range(6):
        x = left + pw * k / 5
        y = top + ph * k / 5
        d.line([x, top, x, top + ph], fill=(236, 236, 236))
        d.line([left, y, left + pw, y], fill=(236, 236, 236))
        draw_text_safe(d, (x - 10, top + ph + 8), f"{k / 5:.1f}", (110, 110, 110), 11)
        draw_text_safe(d, (left - 32, top + ph - ph * k / 5 - 7), f"{k / 5:.1f}", (110, 110, 110), 11)
    d.rectangle([left, top, left + pw, top + ph], outline=(170, 170, 170))
    for lab, xs, ys, col in curves:
        pts = [(left + pw * float(x), top + ph * (1.0 - float(y))) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
    draw_text_safe(d, (left, H - 20), xlabel, (80, 80, 80), 12)
    draw_text_safe(d, (left - 40, 22), ylabel, (80, 80, 80), 12)
    ly = top + 6
    for lab, xs, ys, col in curves:
        d.rectangle([left + pw - 160, ly, left + pw - 148, ly + 10], fill=col)
        draw_text_safe(d, (left + pw - 142, ly - 3), str(lab), (40, 40, 40), 11)
        ly += 16
    img.save(path)


def grid_png(items, path: str, cols: int = 6, title: str = "", cell_w: int = 192,
             cell_h: int = 64, label_h: int = 15, pad: int = 5) -> None:
    """items: list of (np array HxW or HxWxC, 文本)"""
    n = len(items)
    rows = (n + cols - 1) // cols
    W = cols * (cell_w + pad) + pad
    H = 32 + rows * (cell_h + label_h + pad) + pad
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    draw_text_safe(d, (8, 8), title, (0, 0, 0), 14)
    for i, (arr, txt) in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = 32 + r * (cell_h + label_h + pad)
        im = to_pil(arr).resize((cell_w, cell_h), Image.BILINEAR)
        img.paste(im, (x, y + label_h))
        draw_text_safe(d, (x + 2, y + 1), str(txt)[:30], (20, 20, 20), 11)
    img.save(path)


def overlay_png(rgb: np.ndarray, mask=None, path: str = None, path_xy=None, tip=None,
                prox=None, title: str = "", scale: int = 2) -> Image.Image:
    """在窗口图上叠加几何量测结果。"""
    base = to_pil(np.clip(rgb, 0, 255)).resize(
        (rgb.shape[1] * scale, rgb.shape[0] * scale), Image.BILINEAR)
    d = ImageDraw.Draw(base)
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.any():
            er = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
            ys, xs = np.nonzero(er)
            for y, x in zip(ys.tolist(), xs.tolist()):
                yy, xx = y * scale, x * scale
                d.rectangle([xx, yy, xx + 1, yy + 1], fill=(0, 220, 90))
    if path_xy is not None:
        pts = [(float(x) * scale, float(y) * scale) for x, y in np.asarray(path_xy)]
        if len(pts) > 1:
            d.line(pts, fill=(60, 150, 255), width=2)
    for pt, col, rr in ((tip, (255, 60, 60), 6), (prox, (255, 170, 40), 6)):
        if pt is not None and np.isfinite(pt).all():
            x, y = float(pt[0]) * scale, float(pt[1]) * scale
            d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=col, width=2)
    if title:
        draw_text_safe(d, (4, 2), title, (255, 255, 255), 12)
    if path:
        base.save(path)
    return base
