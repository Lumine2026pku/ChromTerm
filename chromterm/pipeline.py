# -*- coding: utf-8 -*-
"""pipeline.py —— 端到端推理：一张图 -> 类别 + 置信度 + 几何量测 + 质检标记。

两个入口
--------
  windows : 处理"端部小图"（data_share/level1 那种 246×112 窗口，或你自己裁的图）
            输出 CSV（每张一行）+ 叠加图（掩膜/中线/端部）
  full    : 处理整张显微照片（实验性）：分割 -> 骨架 -> 找端点 -> 每个端点取窗口
            -> 分类。整片里几条臂在染色中心互相粘连，这一步只做到"候选"，需人工复核；
            正式路线应该用检测器（见 src/detect_stage.py）。

用法
----
  python src/pipeline.py windows --split test --limit 60
  python src/pipeline.py windows --dir D:\\crops\\new
  python src/pipeline.py full --image D:\\photo\\WT-40X-12.jpg

输出：runs/predict_*.csv、runs/pred_grid_*.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .common import (DATA, MODEL_FILE, RUNS, ensure, load_rgb, resize_bilinear,
                     setup_console, to_pil, vstack_panels)
from .data import CLASSES
from .features import (feat_auto, feat_band, feat_geo, feat_hog, feat_pyr, feat_rgb,
                      feat_width)
from .geometry import analyse_window
from .model import Linear, NCP, PCAReducer, RFFRidge, Standardizer
from .normalize import canonical_crop, normalize
from .prep_od import find_bubbles, stain_map, suppress_bubbles
from .viz import overlay_png

GEO_NAMES = ["s_h", "s_w", "area_frac", "fg_frac", "coh", "length_px", "width_px",
             "curv", "tip_ang_sin", "tip_ang_cos", "n_bands", "band_contrast"]


# ------------------------------------------------------------------ 模型
class Predictor:
    """加载 models/model_best.npz，对外只暴露 predict(rgb)。"""

    def __init__(self, path: str | None = None):
        path = path or MODEL_FILE
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到模型 {path}；训练脚本见 tools/train_classifier.py")
        z = np.load(path, allow_pickle=True)
        self.feature_set = str(z["feature_set"][0])
        self.mode = str(z["mode"][0])
        self.blocks = [str(b) for b in z["blocks"][0]]
        self.classes = [str(c) for c in z["classes"]]
        self.model = str(z["model"][0])

        st = Standardizer()
        st.mu = z["mu"].astype(np.float32)
        st.sd = z["sd"].astype(np.float32)
        self.st = st
        self.pc = None
        if "pca_comp" in z.files:
            pc = PCAReducer(n_components=int(z["pca_k"]))
            pc.mu = z["pca_mu"].astype(np.float32)
            pc.components_ = z["pca_comp"].astype(np.float32)
            self.pc = pc
        if self.model == "linear":
            clf = Linear()
            clf.W, clf.b = z["W"].astype(np.float32), z["b"].astype(np.float32)
        elif self.model == "rff":
            clf = RFFRidge(gamma=float(z["gamma"]))
            clf.Wr, clf.br = z["Wr"].astype(np.float32), z["br"].astype(np.float32)
            clf.Wo = z["Wo"].astype(np.float32)
        else:
            clf = NCP()
            clf.C = z["C"].astype(np.float32)
        clf.classes_ = np.arange(len(self.classes))
        self.clf = clf

    # ---- 内部：OD + 几何 + 规范化，一次算全 ----
    def prepare(self, rgb: np.ndarray, bubbles: bool = False):
        h, w = rgb.shape[:2]
        m = stain_map(rgb)
        if bubbles:
            bub = find_bubbles(m["S_n"])
            if bub:
                m["S_n"], touched = suppress_bubbles(m["S_n"], bub)
                m["bubbles"] = bub
        geom = analyse_window(m["S"], m["S_n"], thr_frac=0.8,
                              sigma_along=max(12.0, min(h, w) / 8),
                              core_grow=None, extend_frac=30)
        ang = float(geom.get("axis_deg", 0.0))
        if self.mode == "raw":            # raw 模式：训练时就是"原图缩放"
            can = resize_bilinear(m["S_n"], 192, 64)
            can_rgb = resize_bilinear(rgb, 192, 64)
        else:
            can = normalize(m["S_n"], self.mode, geom, axis_deg=ang, out_w=192, out_h=64)["img"]
            can_rgb = normalize(rgb, self.mode, geom, axis_deg=ang, out_w=192, out_h=64)["img"]
        return m, geom, can, can_rgb

    def _vector(self, rgb, m, geom, can, can_rgb):
        parts = []
        for b in self.blocks:
            if b == "rgb":
                parts.append(feat_rgb(can_rgb))
            elif b == "pyr":
                parts.append(feat_pyr(can))
            elif b == "band":
                parts.append(feat_band(can))
            elif b == "auto":
                parts.append(feat_auto(can))
            elif b == "hog":
                parts.append(feat_hog(can))
            elif b == "width":
                parts.append(feat_width(can))
            elif b == "geo":
                meas = geom.get("measure", {})
                a = meas.get("tip_angle_deg", np.nan)
                g = np.array([rgb.shape[0], rgb.shape[1], geom.get("area_frac", 0.0),
                              geom.get("fg_frac", 0.0), geom.get("coherence", 0.0),
                              meas.get("length_px", 0.0), meas.get("width_px", np.nan),
                              meas.get("curvature_deg_per_100px", np.nan),
                              np.sin(np.deg2rad(a)) if np.isfinite(a) else 0.0,
                              np.cos(np.deg2rad(a)) if np.isfinite(a) else 0.0,
                              geom.get("bands", {}).get("n_bands", 0.0),
                              geom.get("bands", {}).get("band_contrast", 0.0)], np.float32)
                parts.append(feat_geo(g, GEO_NAMES))
            else:
                raise ValueError(b)
        return np.concatenate(parts).astype(np.float32)

    def predict(self, rgb: np.ndarray, bubbles: bool = False) -> dict:
        m, geom, can, can_rgb = self.prepare(rgb, bubbles)
        x = self._vector(rgb, m, geom, can, can_rgb)[None, :]
        z = self.st.transform(x)
        if self.pc is not None:
            z = self.pc.transform(z)
        prob = self.clf.predict_proba(z)[0]
        order = np.argsort(prob)[::-1]
        meas = geom.get("measure", {})
        r = dict(
            cls=self.classes[order[0]], conf=float(prob[order[0]]),
            cls2=self.classes[order[1]], conf2=float(prob[order[1]]),
            probs={self.classes[i]: float(prob[i]) for i in range(len(prob))},
            length_px=float(meas.get("length_px", np.nan)),
            width_px=float(meas.get("width_px", np.nan)),
            tip_angle_deg=float(meas.get("tip_angle_deg", np.nan)),
            curvature=float(meas.get("curvature_deg_per_100px", np.nan)),
            n_bands=float(geom.get("bands", {}).get("n_bands", np.nan)),
            prox_in_frame=bool(geom.get("prox_in_frame", False)),
            axis_deg=float(geom.get("axis_deg", np.nan)),
            coherence=float(geom.get("coherence", np.nan)),
            flags=";".join(geom.get("flags", [])),
            geom=geom, can=can,
        )
        return r


# ------------------------------------------------------------------ windows 模式
ROW_KEYS = ["file", "true", "pred", "conf", "pred2", "conf2", "margin", "length_px",
            "width_px", "tip_angle_deg", "n_bands", "prox_in_frame", "axis_deg",
            "coherence", "flags"]


def _row(name, true, r):
    def num(v, nd=1):
        return round(float(v), nd) if v is not None and np.isfinite(v) else ""

    return dict(file=name, true=true, pred=r["cls"], conf=round(r["conf"], 4),
                pred2=r["cls2"], conf2=round(r["conf2"], 4),
                margin=round(r["conf"] - r["conf2"], 4),
                length_px=num(r["length_px"]), width_px=num(r["width_px"]),
                tip_angle_deg=num(r["tip_angle_deg"]), n_bands=num(r["n_bands"], 0),
                prox_in_frame=int(r["prox_in_frame"]), axis_deg=num(r["axis_deg"]),
                coherence=num(r["coherence"], 2), flags=r["flags"])


def run_windows(args) -> pd.DataFrame:
    pred = Predictor(args.model)
    if args.dir:
        paths = sorted(os.path.join(args.dir, f) for f in os.listdir(args.dir)
                       if f.lower().endswith((".jpg", ".png", ".jpeg", ".tif", ".tiff")))
        labels = None
        tag = "dir"
    else:
        lab = pd.read_csv(os.path.join(DATA, "dataset", "level1", "labels.csv"))
        if args.split != "all":
            lab = lab[lab.split == args.split]
        if args.limit:
            lab = lab.head(args.limit)
        lab = lab.reset_index(drop=True)
        paths = [os.path.join(DATA, "dataset", "level1", "images", s, n + ".jpg")
                 for s, n in zip(lab["split"], lab["name"])]
        labels = lab
        tag = args.split

    rows, panels = [], []
    for i, p in enumerate(paths):
        rgb = load_rgb(p)
        r = pred.predict(rgb, bubbles=args.bubbles)
        name = os.path.basename(p)
        true = str(labels.iloc[i]["cls"]) if labels is not None else ""
        rows.append(_row(name, true, r))
        if len(panels) < args.n_fig:
            g = r["geom"]
            hit = "" if not true else ("  ok" if true == r["cls"] else f"  x({true})")
            panels.append(overlay_png(rgb, g.get("mask"), None, g.get("path"),
                                      g.get("tip_xy"), None,
                                      title=f'{r["cls"]} {r["conf"]:.2f}{hit}', scale=2))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RUNS, f"predict_{tag}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"写入 {out_csv}（{len(df)} 行）")
    if labels is not None:
        acc = float((df["pred"] == df["true"]).mean())
        keep = df["conf"] > 0.5
        print(f"本批准确率 {acc:.3f}（{int((df['pred'] == df['true']).sum())}/{len(df)}）")
        if keep.sum() > 5:
            print(f"置信度>0.5 的窗口占 {keep.mean()*100:.0f}%，其中准确率 "
                  f"{float((df.loc[keep, 'pred'] == df.loc[keep, 'true']).mean()):.3f}")
    if panels:
        fp = os.path.join(RUNS, f"pred_grid_{tag}.png")
        vstack_panels(panels).save(fp)
        print(f"写入 {fp}")
    return df


# ------------------------------------------------------------------ full 模式
def run_full(args) -> pd.DataFrame | None:
    pred = Predictor(args.model)
    rgb = load_rgb(args.image)
    h0, w0 = rgb.shape[:2]
    scale = min(1.0, 900.0 / max(h0, w0))
    img = resize_bilinear(rgb, int(w0 * scale), int(h0 * scale)) if scale < 1 else rgb
    m = stain_map(img)
    g = analyse_window(m["S"], m["S_n"], thr_frac=0.8,
                       sigma_along=max(20.0, min(img.shape[:2]) / 40),
                       core_grow=60, extend_frac=20)
    if not g.get("ok"):
        print("整片分割失败：", g.get("reason"))
        return None
    path = g["path"]
    W = g["measure"].get("width_px") or 20.0
    rows, panels = [], []
    for tag, pt, inner in (("end0", path[0], path[-1]), ("end1", path[-1], path[0])):
        tip = np.array([pt[1], pt[0]], np.float32)
        u = np.array([inner[1] - pt[1], inner[0] - pt[0]], np.float32)
        u /= (np.linalg.norm(u) + 1e-6)
        win = canonical_crop(img, tip, u, float(W), 192, 64)
        r = pred.predict(win)
        rows.append(dict(end=tag, x=round(float(tip[0] / scale), 1),
                         y=round(float(tip[1] / scale), 1), pred=r["cls"],
                         conf=round(r["conf"], 3), width_px=round(float(W), 1),
                         flags=r["flags"]))
        panels.append(to_pil(win))
    df = pd.DataFrame(rows)
    out = os.path.join(RUNS, "predict_full.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"写入 {out}（整片模式只取最长那条臂的两个端点，需人工复核）")
    return df


# ------------------------------------------------------------------ CLI
def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["windows", "full"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dir", default=None, help="自定义图片文件夹")
    ap.add_argument("--image", default=None, help="整片模式：单张图片路径")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-fig", type=int, default=12, help="拼图最多放几张")
    ap.add_argument("--bubbles", action="store_true", help="开启气泡抑制（较慢）")
    args = ap.parse_args()
    ensure(RUNS)
    if args.cmd == "windows":
        run_windows(args)
    else:
        if not args.image:
            ap.error("full 模式需要 --image")
        run_full(args)


if __name__ == "__main__":
    main()
