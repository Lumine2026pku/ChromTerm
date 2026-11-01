# -*- coding: utf-8 -*-
"""demo_photo.py —— 演示：输入"一张照片"时到底能得到什么。

对比两件事：
  A) 整片模式（pipeline.py full）：在整张照片上分割 -> 骨架 -> 只给出最长一条臂的两个端点
  B) 端部窗口模式（pipeline.py windows）：对这张照片里**每一个**已裁好的端部窗口做 6 分类
结论一目了然：照片 -> 窗口这一步（检测/定位）目前还没有训练好的模型，
现在能可靠输出的信息，是「给定一个端部窗口 -> 类别 + 置信度 + 几何量测 + 质检」。

用法：python src/demo_photo.py
"""

import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromterm.common import (DATA, DOCS, RUNS, ensure, hstack_panels, load_rgb, normalise_for_display,
                    resize_bilinear, save_json, setup_console, to_pil, vstack_panels)
from chromterm.geometry import analyse_window
from chromterm.pipeline import Predictor
from chromterm.prep_od import stain_map
from chromterm.viz import overlay_png


def main():
    setup_console()
    ensure(DOCS)
    pred = Predictor()

    # ---- 选一张有 5 个端部框的 1600px 测试图 ----
    photo = os.path.join(DATA, "dataset", "det", "images", "test", "zlg40__WT-40X-9.jpg")
    rgb = load_rgb(photo)
    h, w = rgb.shape[:2]

    # ---- A) 整片模式 ----
    scale = min(1.0, 900.0 / max(h, w))
    img = resize_bilinear(rgb, int(w * scale), int(h * scale)) if scale < 1 else rgb
    m = stain_map(img)
    g = analyse_window(m["S"], m["S_n"], thr_frac=0.8,
                       sigma_along=max(20.0, min(img.shape[:2]) / 40),
                       core_grow=60, extend_frac=20)
    print("整片模式：ok =", g.get("ok"), "掩膜面积比 =", round(g.get("area_frac", 0), 3))
    if g.get("ok"):
        ov = overlay_png(img, g.get("mask"), None, g.get("path"), g.get("tip_xy"), None,
                         title="整片模式：只追上最长的一条臂", scale=1)
        ov = ov.resize((ov.width // 2, ov.height // 2), Image.BILINEAR)
    else:
        ov = to_pil(img).resize((img.shape[1] // 2, img.shape[0] // 2))

    # ---- B) 这张图里所有端部窗口 ----
    lab = pd.read_csv(os.path.join(DATA, "dataset", "level1", "labels.csv"))
    sub = lab[lab["image"] == "zlg40__WT-40X-9"].reset_index(drop=True)
    rows, panels = [], []
    for r in sub.itertuples(index=False):
        p = os.path.join(DATA, "dataset", "level1", "images", r.split, r.name + ".jpg")
        win = np.asarray(Image.open(p).convert("RGB"), np.float32)
        q = pred.predict(win, bubbles=False)
        rows.append(dict(window=r.name, true=r.cls, pred=q["cls"], conf=round(q["conf"], 3),
                         second=q["cls2"], conf2=round(q["conf2"], 3),
                         width_px=round(q["width_px"], 1), n_bands=q["n_bands"],
                         length_px=round(q["length_px"], 1),
                         flags=q["flags"] or "-"))
        panels.append(overlay_png(win, q["geom"].get("mask"), None, q["geom"].get("path"),
                                  q["geom"].get("tip_xy"), None,
                                  title=f'{q["cls"]} {q["conf"]:.2f} / true {r.cls}', scale=2))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RUNS, "demo_photo_windows.csv"), index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

    # ---- 拼图 ----
    grid = vstack_panels(panels)
    sheet = hstack_panels([ov, grid])
    sheet.save(os.path.join(DOCS, "fig5_photo_demo.png"))
    print("saved", os.path.join(DOCS, "fig5_photo_demo.png"), sheet.size)

    # ---- 统计：这些窗口的置信度分布 ----
    acc = float((df["pred"] == df["true"]).mean())
    hi = df["conf"] > 0.5
    print(f"这 {len(df)} 个端部：准确率 {acc:.2f}；置信度 >0.5 的 {int(hi.sum())} 个，"
          f"其中正确 {int((df.loc[hi, 'pred'] == df.loc[hi, 'true']).sum())} 个")


if __name__ == "__main__":
    main()
