# -*- coding: utf-8 -*-
"""data.py —— 数据索引、切分与特征缓存。

数据来自 `data_share/`（组织者整理好的可训练版本）：
  dataset/level1/labels.csv   每个 246×112 端部窗口一行：name/image/split/cls/...
  dataset/level1/images/...   窗口图（RGB，246×112）
  dataset/det/                整图检测数据集（YOLO 格式，6 类）
  dataset/meta.csv            每张原图的来源、尺寸类别、同源组、各类框数
  annotations/p01..p08,own    X-AnyLabeling 原始标注（旋转框 + 端点）
  control/                    10 张"8 人都标过"的对照图（只做一致性分析）

本文件只做三件事：
  1) 把上述索引读成 DataFrame，做一致性校验（数量、类别、切分）；
  2) 定义评估用的切分：官方 train/val/test，以及"按来源批次"的 LOSO（留一批次）；
  3) 建/读特征缓存（每窗的 OD、几何量测、三种归一化窗口），避免重复跑重活。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image

from .common import CACHE, DATA, ensure

CLASSES = ["2L", "2R", "3L", "3R", "X", "other"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
DS = os.path.join(DATA, "dataset")
CACHE_FILE = os.path.join(CACHE, "level1_features.npz")
CACHE_CSV = os.path.join(CACHE, "level1_features.csv")
MODES = ["raw", "deskew", "tip"]


# ------------------------------------------------------------------ 索引
def load_index() -> pd.DataFrame:
    """level1 窗口索引（961 行），带类别编号与所属原图信息。"""
    lab = pd.read_csv(os.path.join(DS, "level1", "labels.csv"))
    meta = pd.read_csv(os.path.join(DS, "meta.csv"))
    m = meta.set_index("name")
    lab["cls_idx"] = lab["cls"].map(CLS2IDX).astype(int)
    lab["path"] = [os.path.join(DS, "level1", "images", s, n + ".jpg")
                   for s, n in zip(lab["split"], lab["name"])]
    lab["source"] = lab["image"].map(m["source"])
    lab["size_class"] = lab["image"].map(m["size_class"])
    lab["img_split"] = lab["image"].map(m["split"])
    missing = [p for p in lab["path"] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} 个窗口图缺失，例如 {missing[0]}")
    return lab


def load_det_index() -> pd.DataFrame:
    """整图检测数据集索引：每张图 + 若干 YOLO 框。"""
    rows = []
    for split in ("train", "val", "test"):
        img_dir = os.path.join(DS, "det", "images", split)
        lbl_dir = os.path.join(DS, "det", "labels", split)
        for f in sorted(os.listdir(lbl_dir)):
            name = os.path.splitext(f)[0]
            boxes = []
            with open(os.path.join(lbl_dir, f), encoding="utf-8") as fh:
                for line in fh:
                    p = line.split()
                    if len(p) >= 5:
                        boxes.append((int(p[0]), *map(float, p[1:5])))
            rows.append(dict(name=name, split=split,
                             image=os.path.join(img_dir, name + ".jpg"), boxes=boxes))
    return pd.DataFrame(rows)


def sanity_report(idx: pd.DataFrame) -> dict:
    """数据体检：数量、类别分布、各切分是否有漏类、有无重复窗口。"""
    rep = dict(n=len(idx), n_images=idx["image"].nunique())
    rep["cls_counts"] = idx["cls"].value_counts().to_dict()
    rep["split_counts"] = idx["split"].value_counts().to_dict()
    per = pd.crosstab(idx["cls"], idx["split"])
    rep["cls_per_split"] = per.to_dict()
    rep["empty_cells"] = int((per == 0).sum().sum())
    rep["dup_names"] = int(idx["name"].duplicated().sum())
    rep["n_sources"] = int(idx["source"].nunique())
    rep["sources"] = idx["source"].value_counts().to_dict()
    return rep


# ------------------------------------------------------------------ 切分
def official_split(idx: pd.DataFrame):
    tr = np.nonzero((idx["split"] == "train").to_numpy())[0]
    va = np.nonzero((idx["split"] == "val").to_numpy())[0]
    te = np.nonzero((idx["split"] == "test").to_numpy())[0]
    return tr, va, te


def loso_folds(idx: pd.DataFrame):
    """留一来源批次（leave-one-source-out）：跨染色/物镜/批次的泛化检验。

    data_share 的 source 有 5 个值（org_full / org_crop / xgw100 / xgw40 / zlg40），
    正好对应不同的成像条件。这比随机切分严格得多——真实使用场景就是"换一批片子"。
    """
    folds = []
    src = idx["source"].to_numpy()
    for s in sorted(pd.unique(src[~pd.isna(src)])):
        te = np.nonzero(src == s)[0]
        tr = np.nonzero(src != s)[0]
        folds.append((str(s), tr, te))
    return folds


# ------------------------------------------------------------------ 缓存
def build_cache(force: bool = False, verbose: bool = True, limit: int | None = None):
    """对每个窗口跑一遍 OD + 几何 + 三种归一化，落盘为 npz。

    返回 (idx, feat) 其中 feat 是 dict：
      names         (N,) 窗口名
      cls           (N,) 类别编号
      split         (N,) train/val/test
      source        (N,) 来源批次
      s_n_raw       (N,H,W) float16  原始窗口的归一化染色量
      rgb_raw       (N,H,W,3) uint8  原始窗口 RGB（基线用）
      can_<mode>    (N,64,192) float16  三种归一化的染色量窗口
      can_<mode>_rgb(N,64,192,3) uint8  三种归一化的 RGB 窗口（几何特征/可视化用）
      geo           (N,12) float32  几何量测标量
      geo_names     (12,) str
    """
    from .geometry import analyse_window
    from .normalize import normalize_pair
    from .prep_od import stain_map

    idx = load_index()
    if limit:
        idx = idx.iloc[:limit].reset_index(drop=True)
    s_n_list, rgb_list, geo_list = [], [], []
    can = {m: [] for m in MODES}
    can_rgb = {m: [] for m in MODES}
    used = []
    geo_names = ["s_h", "s_w", "area_frac", "fg_frac", "coh", "length_px", "width_px",
                 "curv", "tip_ang_sin", "tip_ang_cos", "n_bands", "band_contrast"]
    hw = None
    for k, row in enumerate(idx.itertuples(index=False)):
        rgb = np.asarray(Image.open(row.path).convert("RGB"), np.float32)
        m = stain_map(rgb)
        geom = analyse_window(m["S"], m["S_n"], thr_frac=0.8,
                              sigma_along=max(12.0, min(rgb.shape[:2]) / 8),
                              core_grow=None, extend_frac=30)
        hw = m["S_n"].shape
        s_n_list.append(m["S_n"].astype(np.float16))
        rgb_list.append(rgb.astype(np.uint8))
        meas = geom.get("measure", {})
        ang = meas.get("tip_angle_deg", np.nan)
        geo_list.append([
            float(rgb.shape[0]), float(rgb.shape[1]),
            float(geom.get("area_frac", 0.0)), float(geom.get("fg_frac", 0.0)),
            float(geom.get("coherence", 0.0)),
            float(meas.get("length_px", 0.0)), float(meas.get("width_px", np.nan)),
            float(meas.get("curvature_deg_per_100px", np.nan)),
            float(np.sin(np.deg2rad(ang))) if np.isfinite(ang) else 0.0,
            float(np.cos(np.deg2rad(ang))) if np.isfinite(ang) else 0.0,
            float(geom.get("bands", {}).get("n_bands", 0.0)),
            float(geom.get("bands", {}).get("band_contrast", 0.0)),
        ])
        for mo in MODES:
            a = normalize_pair(rgb, m["S_n"], mo, geom,
                               axis_deg=geom.get("axis_deg", 0.0))
            can[mo].append(a["s_n"].astype(np.float16))
            can_rgb[mo].append(np.clip(a["img"], 0, 255).astype(np.uint8))
            used.append(a["used"])
        if verbose and (k + 1) % 100 == 0:
            print(f"  缓存 {k+1}/{len(idx)} ...", flush=True)

    feat = dict(
        names=idx["name"].to_numpy().astype(object),
        cls=idx["cls_idx"].to_numpy().astype(np.int32),
        split=idx["split"].to_numpy().astype(object),
        source=idx["source"].to_numpy().astype(object),
        image=idx["image"].to_numpy().astype(object),
        s_n_raw=np.stack(s_n_list),
        rgb_raw=np.stack(rgb_list),
        geo=np.asarray(geo_list, np.float32),
        geo_names=np.asarray(geo_names, dtype=object),
        used=np.asarray(used, dtype=object),
    )
    for mo in MODES:
        feat[f"can_{mo}"] = np.stack(can[mo])
        feat[f"can_{mo}_rgb"] = np.stack(can_rgb[mo])

    ensure(os.path.dirname(CACHE_FILE))
    np.savez_compressed(CACHE_FILE, **feat)
    # 每窗三种归一化的实际生效路径（tip / deskew_fallback ...），便于统计成功率
    used = np.asarray(feat["used"], dtype=object).reshape(len(MODES), len(idx)).T
    for j, mo in enumerate(MODES):
        idx[f"used_{mo}"] = used[:, j]
    idx.to_csv(CACHE_CSV, index=False, encoding="utf-8-sig")
    if verbose:
        print(f"缓存写入 {CACHE_FILE}")
    return idx, feat


def load_cache():
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"找不到缓存 {CACHE_FILE}，先跑 python src/build_cache.py")
    return dict(np.load(CACHE_FILE, allow_pickle=True))
