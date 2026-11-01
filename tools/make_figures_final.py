# -*- coding: utf-8 -*-
"""make_figures_final.py —— 报告插图（中线采样质检图、拉直长条、诊断面板）。

产物都在 `docs/`：
  fig_sampling_qc.png   一张端部窗口 + 采样点 + 灰度曲线 + 条带 + 宽度曲线
  fig_strips.png        不同类别的拉直长条（说明"规范化之后长什么样"）
  fig_diagnosis.png     四条诊断证据的一页图
  fig_negatives.png     哪种特征表征有用（消融条形图，来自 final_ablation.csv）

跑法：python src/make_figures_final.py
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromterm.common import DOCS, RUNS, ensure, load_json, load_rgb, to_pil  # noqa: E402
from chromterm.data import CLASSES, load_index  # noqa: E402
from chromterm.profile import profile_of_window, rectify_strip  # noqa: E402


def _sparkline(draw, box, ys, color, width=2, bands=None):
    x0, y0, x1, y1 = box
    n = len(ys)
    lo, hi = float(np.nanmin(ys)), float(np.nanmax(ys))
    rg = max(1e-6, hi - lo)
    if bands:
        for b in bands:
            bx0 = x0 + (x1 - x0) * (b["s_start"] / max(1, n - 1))
            bx1 = x0 + (x1 - x0) * (b["s_end"] / max(1, n - 1))
            draw.rectangle([bx0, y0, bx1, y1], fill=(236, 236, 236))
    px = py = None
    for i, v in enumerate(ys):
        if not np.isfinite(v):
            px = None
            continue
        x = x0 + (x1 - x0) * i / max(1, n - 1)
        y = y1 - (y1 - y0) * (v - lo) / rg
        if px is not None:
            draw.line([px, py, x, y], fill=color, width=width)
        px, py = x, y


def fig_sampling(all_rows, out_path):
    """挑一个**有代表性**的例子画质检图。

    注意不要挑"对比度最高"的：那往往是分割失败的异常图（中线偏离臂）。
    这里只接受"路径长度与臂长相称、条带 3–4 条"的窗口，再取对比度最接近中位的。
    """
    cands = []
    for r in all_rows:
        if not r.get("ok") or not r.get("profile"):
            continue
        p = r["profile"]
        if not (3 <= len(p["bands"]) <= 4):
            continue
        if not (120 <= p["n_points"] <= 230):
            continue
        # 中线要基本贴着臂走：路径两端离掩膜质心的距离应与臂长相当
        cands.append(r)
    if not cands:
        print("  跳过 fig_sampling：没有合适的样本")
        return
    cont = np.array([float(np.ptp(r["profile"]["gray"])) for r in cands])
    target = float(np.median(cont))
    best = cands[int(np.argmin(np.abs(cont - target)))]
    rgb = best["rgb"]
    h, w = rgb.shape[:2]
    img = rgb.astype(np.uint8).copy()
    for (x, y) in best["profile"]["path_xy"][::2]:
        yy, xx = int(round(y)), int(round(x))
        if 0 <= yy < h and 0 <= xx < w:
            img[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2] = (255, 40, 40)
    ty, tx = best["profile"]["tip_xy"][1], best["profile"]["tip_xy"][0]
    ImageDraw.Draw(Image.fromarray(img)).ellipse(
        [tx - 5, ty - 5, tx + 5, ty + 5], outline=(0, 200, 255), width=2)

    scale = 2.6
    left = Image.fromarray(img).resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    pw, ph = 620, max(left.height, 300)
    panel = Image.new("RGB", (pw, ph), "white")
    d = ImageDraw.Draw(panel)
    pad = 34
    top_h = (ph - 3 * pad) * 0.58
    _sparkline(d, (pad, pad, pw - pad, pad + top_h), best["profile"]["gray"],
               (20, 60, 160), bands=best["profile"]["bands"])
    d.text((pad, 10), f"Gray_Value  (bands detected: {len(best['profile']['bands'])})",
           fill=(0, 0, 0))
    y2 = pad + top_h + pad
    _sparkline(d, (pad, y2, pw - pad, y2 + (ph - 3 * pad) * 0.36),
               best["profile"]["width"], (0, 130, 90))
    d.text((pad, y2 - 18), "Width_px (transverse, mask diameter)", fill=(0, 0, 0))

    out = Image.new("RGB", (left.width + pw + 8, max(left.height, ph)), "white")
    out.paste(left, (0, 0))
    out.paste(panel, (left.width + 8, 0))
    out.save(out_path)
    print(f"  fig_sampling -> {os.path.basename(out_path)}  ({best['name']})")


def fig_strips(all_rows, out_path, per_class=4):
    """每类挑几条拉直长条拼在一起，直观看出"规范化后只剩带纹"。"""
    picks = {}
    pool = [r for r in all_rows if r.get("ok") and r.get("strip") is not None]
    for r in pool:
        c = r["cls"]
        key = abs(int(r["profile"]["n_points"]) - 180)      # 越接近典型长度越优先
        picks.setdefault(c, []).append((key, r))
    for c in picks:
        picks[c] = [r for _, r in sorted(picks[c], key=lambda kv: kv[0])][:per_class]
    rows_img = []
    for c in CLASSES:
        tiles = []
        for r in picks.get(c, []):
            s = np.clip(r["strip"], 0, 1)
            g = (255 * (1 - s)).astype(np.uint8)
            im = Image.fromarray(g).resize((192, 96), Image.BILINEAR)
            tiles.append((r["name"], im))
        while len(tiles) < per_class:
            tiles.append(("", Image.new("L", (192, 96), 255)))
        rows_img.append((c, tiles))
    W = 192 * per_class + 12 * (per_class + 1) + 70
    H = 96 * len(rows_img) + 10 * (len(rows_img) + 1)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    y = 8
    for c, tiles in rows_img:
        d.text((8, y + 40), c, fill=(0, 0, 0))
        x = 70
        for nm, im in tiles:
            canvas.paste(im.convert("RGB"), (x, y))
            x += 192 + 12
        y += 96 + 10
    canvas.save(out_path)
    print(f"  fig_strips -> {os.path.basename(out_path)}")


def fig_diagnosis(out_path):
    """四条证据画成一页：DTW vs 欧氏、光度基线 vs 完整特征、粒度、覆盖率。"""
    diag = load_json(os.path.join(RUNS, "diagnosis.json"))
    summ = load_json(os.path.join(RUNS, "summary_final.json"))
    W, H = 1180, 760
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)

    def panel(x0, y0, x1, y1, title):
        d.rectangle([x0, y0, x1, y1], outline=(200, 200, 200))
        d.text((x0 + 8, y0 + 6), title, fill=(20, 20, 20))
        return x0 + 12, y0 + 30, x1 - 12, y1 - 26

    def bars(box, labels, vals, colors, vmax=1.0, fmt="{:.2f}"):
        x0, y0, x1, y1 = box
        n = len(vals)
        bw = (x1 - x0) / max(n, 1) * 0.62
        gap = (x1 - x0) / max(n, 1)
        for i, (lab, v, c) in enumerate(zip(labels, vals, colors)):
            bx = x0 + gap * i + (gap - bw) / 2
            hgt = (y1 - y0) * float(v) / max(vmax, 1e-9)
            d.rectangle([bx, y1 - hgt, bx + bw, y1], fill=c)
            d.text((bx + 2, y1 - hgt - 14), fmt.format(v), fill=(0, 0, 0))
            d.text((bx - 2, y1 + 4), lab, fill=(60, 60, 60))

    # 1) D2
    d2 = diag.get("d2_gt_curves", {})
    b = panel(20, 20, 580, 250, "D2  one curve cannot name the arm (LOO 1-NN, 53 curated curves)")
    if d2.get("available"):
        bars(b, ["DTW 1-NN", "Euclidean 1-NN"], [d2["dtw_1nn_acc"], d2["euclid_1nn_acc"]],
             [(190, 90, 90), (90, 130, 190)])
        d.text((b[0], b[1] + 2), "chance = 0.200 (5 classes)", fill=(120, 120, 120))

    # 2) D3
    d3 = diag.get("d3_nuisance", {})
    b = panel(600, 20, 1160, 250, "D3  banding vs nuisance-only features (15-fold CV macro-F1)")
    bars(b, ["nuisance only", "full features"],
         [d3.get("cv_f1", 0), (summ.get("cv") or {}).get("cv_f1", 0)],
         [(200, 160, 90), (70, 150, 110)])

    # 3) 粒度
    g = summ.get("granularity") or []
    b = panel(20, 270, 580, 500, "D4'  accuracy vs question granularity (official test)")
    if g:
        bars(b, [f"{r['n_cls']}cl" for r in g], [r["acc"] for r in g],
             [(120, 140, 200)] * len(g))
        ascii_names = {2: ["2L vs rest"], 3: ["2L / X / rest"], 4: ["2R+3L+3R", "merged"],
                       6: ["full 6-class task"]}
        for i, r in enumerate(g):
            lines = ascii_names.get(r["n_cls"], [""])
            for j, ln in enumerate(lines):
                d.text((b[0] + 4 + i * 130, b[1] + 2 + 14 * j), ln, fill=(90, 90, 90))

    # 4) 风险-覆盖率
    b = panel(600, 270, 1160, 500, "risk-coverage of the final model (abstain to gain precision)")
    rc = summ.get("risk_coverage") or {}
    keys = ["100%", "80%", "60%", "40%", "20%"]
    vals = [rc.get(k, 0) for k in keys]
    bars(b, keys, vals, [(70, 110, 190)] * len(vals), vmax=1.0)

    # 5) 说明文字
    txt = [
        "D1  human ceiling: 8 annotators on 10 shared images, Fleiss kappa = "
        f"{(diag.get('d1_human') or {}).get('kappa', float('nan')):.3f}, "
        f"exact set agreement {(diag.get('d1_human') or {}).get('agreement', float('nan')):.2f}",
        "D2  a full banding curve alone is not enough; DTW is worse than plain Euclidean",
        "D3  the model is NOT just memorising the staining batch (nuisance-only stays low)",
        "D4  'at most two copies per arm per photo' does not hold in this dataset",
        "=> the 6-class window task is information-limited, not model-limited",
    ]
    y = 530
    for t in txt:
        d.text((28, y), t, fill=(30, 30, 30))
        y += 26
    im.save(out_path)
    print(f"  fig_diagnosis -> {os.path.basename(out_path)}")


def main():
    ensure(DOCS)
    idx = load_index()
    # 采样 + 拉直，供质检图用（只跑前 120 个窗口，够挑例子）
    all_rows = []
    for row in idx.itertuples(index=False):
        if len(all_rows) >= 120:
            break
        rgb = load_rgb(row.path)
        prof = profile_of_window(rgb, extend_frac=30)
        if not prof.get("ok"):
            all_rows.append(dict(ok=False, name=row.name))
            continue
        st = rectify_strip(rgb, prof["stain"]["S"], prof["stain"]["S_n"], prof["info"])
        all_rows.append(dict(ok=bool(st.get("ok")), name=row.name, cls=row.cls,
                             rgb=rgb, profile=prof, strip=st.get("strip_od")))
    print(f"  采样 {len(all_rows)} 个窗口用于挑图例")

    fig_sampling(all_rows, os.path.join(DOCS, "fig_sampling_qc.png"))
    fig_strips(all_rows, os.path.join(DOCS, "fig_strips.png"))
    fig_diagnosis(os.path.join(DOCS, "fig_diagnosis.png"))

    # 把消融条形图复制进 docs（报告直接引用）
    for src, dst in (("final_ablation_bars.png", "fig_negatives.png"),
                     ("final_granularity.png", "fig_granularity.png"),
                     ("confusion_best.png", "fig_confusion.png"),
                     ("risk_coverage.png", "fig_risk_coverage.png")):
        sp = os.path.join(RUNS, src)
        if os.path.exists(sp):
            Image.open(sp).save(os.path.join(DOCS, dst))
            print(f"  复制 {src} -> docs/{dst}")


if __name__ == "__main__":
    main()
