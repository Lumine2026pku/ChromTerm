# -*- coding: utf-8 -*-
"""profile_stats.py —— 复现《中线采样说明》第六节的统计，验证采样实现是否同口径。

说明书表 6 给了 148 条平直曲线的实测统计。我们跑的是全部 961 个端部窗口，
把同样的指标重算一遍，逐条对照。若数字对得上，说明 `profile.py` 与组里的
`profile.py` 是同一把尺子；对不上就说明某条口径（平滑宽度、噪声门槛、
宽度量法）不同，必须先修。

跑法：python src/profile_stats.py
产物：runs/profiles_stats.json
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromterm.common import CACHE, RUNS, ensure, save_json, setup_console

# 说明书表 6 的参考值（148 条平直曲线）
DOC = dict(n_curves=148, n_points=167, n_bands=3, band_width=16.9, arm_width=40.5,
           width_in_band=41.7, width_between=37.2, frac_thicker=0.84,
           corr_gray_width=-0.26, frac_abs_corr_gt07=0.02)


def main():
    setup_console()
    ensure(RUNS)
    z = dict(np.load(os.path.join(CACHE, "profiles.npz"), allow_pickle=True))
    names = [str(n) for n in z["names"]]
    L = z["length"].astype(int)
    G, Wd = z["gray"], z["width"]
    bands = pd.read_csv(os.path.join(CACHE, "profiles_bands.csv"))

    by_name = {n: g for n, g in bands.groupby("name")}
    n_pts, n_band, bw, wmed, w_in, w_bet, corr = [], [], [], [], [], [], []
    for i, nm in enumerate(names):
        l = int(L[i])
        if l < 10:
            continue
        n_pts.append(l)
        g = np.asarray(G[i, :l], np.float64)
        w = np.asarray(Wd[i, :l], np.float64)
        ok = np.isfinite(w) & (w > 0)
        if ok.sum() > 5:
            wmed.append(float(np.median(w[ok])))
            if np.std(g) > 1e-6 and np.std(w[ok]) > 1e-6:
                corr.append(float(np.corrcoef(g[ok], w[ok])[0, 1]))
        b = by_name.get(nm)
        if b is None or len(b) == 0:
            n_band.append(0)
            continue
        n_band.append(len(b))
        bw.append(float(np.median(b["width_px"])))
        inside = np.zeros(l, bool)
        for _, r in b.iterrows():
            inside[int(max(0, r.s_start)):int(min(l - 1, r.s_end)) + 1] = True
        if ok.any() and inside.any() and (~inside & ok).any():
            w_in.append(float(np.median(w[inside & ok])))
            w_bet.append(float(np.median(w[~inside & ok])))

    ours = dict(
        n_curves=int(len(n_pts)),
        n_points=float(np.median(n_pts)),
        n_bands=float(np.median(n_band)),
        band_width=float(np.median(bw)) if bw else float("nan"),
        arm_width=float(np.median(wmed)),
        width_in_band=float(np.median(w_in)) if w_in else float("nan"),
        width_between=float(np.median(w_bet)) if w_bet else float("nan"),
        frac_thicker=float(np.mean(np.asarray(w_in) > np.asarray(w_bet))) if w_in else float("nan"),
        corr_gray_width=float(np.median(corr)) if corr else float("nan"),
        frac_abs_corr_gt07=float(np.mean(np.abs(corr) > 0.7)) if corr else float("nan"),
    )
    rows = []
    labels = [("n_curves", "曲线条数", 0), ("n_points", "每条曲线采样点数（中位）", 0),
              ("n_bands", "每条曲线条带数（中位）", 1),
              ("band_width", "条带沿轴宽度 px（中位）", 1),
              ("arm_width", "臂的横向宽度 px（中位）", 1),
              ("width_in_band", "条带处横向宽度 px", 1),
              ("width_between", "条带间横向宽度 px", 1),
              ("frac_thicker", "带处更粗的曲线占比", 3),
              ("corr_gray_width", "灰度–宽度相关系数（中位）", 3),
              ("frac_abs_corr_gt07", "|r|>0.7 的曲线占比", 3)]
    print("指标                              本实现        说明书表6")
    for k, lab, nd in labels:
        o, d = ours[k], DOC[k]
        fmt = f"{{:.{nd}f}}"
        print(f"  {lab:28s} {fmt.format(o):>10s}    {fmt.format(d):>10s}")
        rows.append(dict(key=k, label=lab, ours=round(o, 4), doc=round(d, 4)))

    save_json(dict(ours=ours, doc=DOC, rows=rows,
                   note="本实现跑的是全部 961 个端部窗口；说明书跑的是 148 条平直曲线。"),
              os.path.join(RUNS, "profiles_stats.json"))
    print("\n写入 runs/profiles_stats.json")


if __name__ == "__main__":
    main()
