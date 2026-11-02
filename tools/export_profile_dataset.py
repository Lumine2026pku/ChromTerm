# -*- coding: utf-8 -*-
"""export_profile_dataset.py —— 把中线采样得到的"两个宽度通道"导出成数据集表格。

组里给 level1 加的两个通道是
    ① 横向宽度（Width_px）：这一点处臂有多粗（掩膜内切直径）
    ② 沿臂轴宽度（band width）：一条带沿长轴占多远（半高宽）
`profile.py` 已经把它们算出来了，这个脚本负责**落成表格**，让任何下游脚本
（DTW、聚类的旧代码，或别的同学的 ResNet 融合）都能直接 merge。

产物（runs/）：
    level1_profiles.csv        每窗一行：臂宽、带数、带宽统计、尺度归一的带序列
    level1_bands.csv           每条带一行（长表，列名与《中线采样说明》一致）
    labels_with_profiles.csv   labels.csv + 上述窗口级列（可直接替换使用）

跑法：
    python src/export_profile_dataset.py            # 生成上面三个表
    python src/export_profile_dataset.py --merge-only
"""
from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromterm.common import CACHE, DATA, RUNS, ensure, save_json, setup_console

DS = os.path.join(DATA, "dataset")


def _series_from_bands(bands: pd.DataFrame, L: float, W: float, rng: float) -> dict:
    """把一条曲线的条带表压成窗口级标量（尺度统一用臂宽 W）。"""
    if bands is None or len(bands) == 0:
        return dict(n_bands=0, band_w_med=np.nan, band_w_med_W=np.nan,
                    band_gap_med_W=np.nan, band_duty_cycle=np.nan,
                    band_prom_med_rel=np.nan, band_first_s_W=np.nan,
                    band_last_s_W=np.nan, band_span_W=np.nan,
                    sig1=np.nan, sig2=np.nan, sig3=np.nan,
                    sig1_w=np.nan, sig2_w=np.nan, sig3_w=np.nan)
    b = bands.sort_values("s_center")
    bw = b["width_px"].to_numpy(float)
    sc = b["s_center"].to_numpy(float)
    pr = b["prominence"].to_numpy(float)
    gap = np.diff(sc) if len(sc) > 1 else np.array([np.nan])
    out = dict(
        n_bands=int(len(b)),
        band_w_med=float(np.median(bw)),
        band_w_med_W=float(np.median(bw) / W),
        band_gap_med_W=float(np.nanmedian(gap) / W) if np.isfinite(gap).any() else np.nan,
        band_duty_cycle=float(bw.sum() / max(L, 1e-9)),
        band_prom_med_rel=float(np.median(pr) / rng) if rng > 0 else np.nan,
        band_first_s_W=float(sc[0] / W),
        band_last_s_W=float(sc[-1] / W),
        band_span_W=float((sc[-1] - sc[0]) / W) if len(sc) > 1 else np.nan,
    )
    # 前三条带的"签名"：位置（归一化到 0–1 与 0–若干臂宽两种写法）
    for k in range(3):
        if k < len(b):
            out[f"sig{k+1}"] = float(sc[k] / max(L, 1e-9))
            out[f"sig{k+1}_w"] = float(sc[k] / W)
        else:
            out[f"sig{k+1}"] = np.nan
            out[f"sig{k+1}_w"] = np.nan
    return out


def build_tables() -> tuple:
    z = dict(np.load(os.path.join(CACHE, "profiles.npz"), allow_pickle=True))
    names = [str(n) for n in z["names"]]
    L = z["length"].astype(int)
    G, Wd = z["gray"], z["width"]
    bands = pd.read_csv(os.path.join(CACHE, "profiles_bands.csv"))
    by_name = {n: g for n, g in bands.groupby("name")}

    rows = []
    for i, nm in enumerate(names):
        l = int(L[i])
        w = np.asarray(Wd[i, :l], np.float64)
        g = np.asarray(G[i, :l], np.float64)
        ok = np.isfinite(w) & (w > 0)
        W = float(np.median(w[ok])) if ok.any() else np.nan
        rng = float(np.ptp(g)) if l > 1 else 0.0
        rec = dict(name=nm, n_points=l, length_px=float(max(l - 1, 0)),
                   arm_width_px=W,
                   arm_width_sd=float(np.std(w[ok])) if ok.any() else np.nan,
                   width_cv=float(np.std(w[ok]) / (W + 1e-9)) if ok.any() else np.nan,
                   length_over_width=float((l - 1) / W) if W and np.isfinite(W) else np.nan,
                   gray_contrast=rng,
                   gray_std=float(np.std(g)) if l else np.nan)
        rec.update(_series_from_bands(by_name.get(nm), max(l - 1, 1), W, rng))
        rows.append(rec)
    prof = pd.DataFrame(rows)

    # 条带长表：补上 W 归一化列，方便跨放大倍数比较
    bands = bands.copy()
    wmap = dict(zip(prof["name"], prof["arm_width_px"]))
    bands["arm_width_px"] = bands["name"].map(wmap)
    bands["s_center_W"] = bands["s_center"] / bands["arm_width_px"]
    bands["width_W"] = bands["width_px"] / bands["arm_width_px"]
    return prof, bands


def merge_with_labels(prof: pd.DataFrame) -> pd.DataFrame:
    lab = pd.read_csv(os.path.join(DS, "level1", "labels.csv"))
    out = lab.merge(prof, on="name", how="left", validate="one_to_one")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="导出中线采样的宽度通道表格")
    ap.add_argument("--merge-only", action="store_true", help="只用已有缓存重新合并")
    args = ap.parse_args(argv)
    setup_console()
    ensure(RUNS)
    prof, bands = build_tables()
    merged = merge_with_labels(prof)

    p1 = os.path.join(RUNS, "level1_profiles.csv")
    p2 = os.path.join(RUNS, "level1_bands.csv")
    p3 = os.path.join(RUNS, "labels_with_profiles.csv")
    prof.to_csv(p1, index=False, encoding="utf-8-sig")
    bands.to_csv(p2, index=False, encoding="utf-8-sig")
    merged.to_csv(p3, index=False, encoding="utf-8-sig")

    hit = int(merged["arm_width_px"].notna().sum())
    stat = dict(
        n_windows=int(len(prof)), n_with_profile=hit,
        arm_width_median_px=float(prof["arm_width_px"].median()),
        n_bands_median=float(prof["n_bands"].median()),
        band_width_median_px=float(prof["band_w_med"].median()),
        band_width_median_W=float(prof["band_w_med_W"].median()),
        duty_cycle_median=float(prof["band_duty_cycle"].median()),
        length_over_width_median=float(prof["length_over_width"].median()),
    )
    save_json(stat, os.path.join(RUNS, "profile_dataset_summary.json"))
    print(f"窗口级表   {p1}  ({len(prof)} 行 × {prof.shape[1]} 列)")
    print(f"条带长表   {p2}  ({len(bands)} 行)")
    print(f"合并 labels {p3}  ({len(merged)} 行，{hit} 个窗口有采样结果)")
    print("\n数据集概况：")
    for k, v in stat.items():
        print(f"  {k:26s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
