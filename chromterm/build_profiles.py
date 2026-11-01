# -*- coding: utf-8 -*-
"""build_profiles.py —— 对 961 个端部窗口跑一遍"中线采样"，落盘成缓存。

产物（`_cache/`）：
  profiles.npz        ragged 曲线（用 NaN 补齐）+ 长度 + 条带表（CSV 另存）
  profiles_bands.csv  所有窗口的条带表（长表：name, band_id, s_*, ...）
  profiles.csv        每窗一行的摘要（点数、长度、灰度范围、条数、宽度中位）

有了缓存，后面调特征、换分类器、试 DTW 都不必再重跑采样（采样约 0.2 s/窗）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from .common import CACHE, ensure, setup_console
from .data import load_index
from .profile import profile_of_file

PROFILE_FILE = os.path.join(CACHE, "profiles.npz")
BANDS_CSV = os.path.join(CACHE, "profiles_bands.csv")
SUMMARY_CSV = os.path.join(CACHE, "profiles.csv")


def build(limit: int | None = None, verbose: bool = True) -> dict:
    idx = load_index()
    if limit:
        idx = idx.iloc[:limit].reset_index(drop=True)
    n = len(idx)

    grays, ods, widths, grads, lengths, oks = [], [], [], [], [], []
    band_rows = []
    rows = []
    t0 = time.time()
    for k, row in enumerate(idx.itertuples(index=False)):
        prof = profile_of_file(row.path, extend_frac=30)
        ok = bool(prof.get("ok"))
        oks.append(ok)
        if ok:
            grays.append(prof["gray"])
            ods.append(prof["od"])
            widths.append(prof["width"])
            grads.append(prof["grad"])
            lengths.append(prof["n_points"])
        else:
            grays.append(np.zeros(0))
            ods.append(np.zeros(0))
            widths.append(np.zeros(0))
            grads.append(np.zeros(0))
            lengths.append(0)
        for b in (prof.get("bands") or []):
            band_rows.append(dict(name=row.name, **{kk: b[kk] for kk in
                                                    ("band_id", "s_center", "s_start",
                                                     "s_end", "width_px", "gray_min",
                                                     "prominence")}))
        wv = prof["width"][np.isfinite(prof["width"])] if ok else np.zeros(0)
        rows.append(dict(
            name=row.name, cls=row.cls, split=row.split, source=row.source,
            ok=int(ok), n_points=int(lengths[-1]),
            gray_min=float(np.min(grays[-1])) if ok else np.nan,
            gray_mean=float(np.mean(grays[-1])) if ok else np.nan,
            gray_max=float(np.max(grays[-1])) if ok else np.nan,
            n_bands=len(prof.get("bands") or []),
            band_w_med=float(np.median([b["width_px"] for b in prof.get("bands") or []]))
            if prof.get("bands") else np.nan,
            width_med=float(np.median(wv)) if wv.size else np.nan,
            length_px=float(prof.get("length_px")) if ok else np.nan,
        ))
        if verbose and (k + 1) % 100 == 0:
            print(f"  采样 {k+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    lmax = max(lengths) if lengths else 0
    lmax = max(lmax, 1)

    def pad(seqs, fill):
        A = np.full((len(seqs), lmax), fill, np.float32)
        for i, s in enumerate(seqs):
            if s.size:
                A[i, :s.size] = s.astype(np.float32)
        return A

    feat = dict(
        names=idx["name"].to_numpy().astype(object),
        cls=idx["cls_idx"].to_numpy().astype(np.int32),
        split=idx["split"].to_numpy().astype(object),
        source=idx["source"].to_numpy().astype(object),
        image=idx["image"].to_numpy().astype(object),
        length=np.asarray(lengths, np.int32),
        ok=np.asarray(oks, bool),
        gray=pad(grays, np.nan),
        od=pad(ods, np.nan),
        width=pad(widths, np.nan),
        grad=pad(grads, np.nan),
    )
    ensure(CACHE)
    np.savez_compressed(PROFILE_FILE, **feat)
    pd.DataFrame(band_rows).to_csv(BANDS_CSV, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    if verbose:
        print(f"\n成功采样 {int(np.sum(oks))}/{n} 个窗口；最长曲线 {lmax} 点")
        print(f"点数中位 {np.median([l for l in lengths if l]):.0f}；"
              f"条带数中位 {summary['n_bands'].median():.0f}；"
              f"横向宽度中位 {summary['width_med'].median():.1f} px")
        print(f"写入 {PROFILE_FILE}\n     {BANDS_CSV}\n     {SUMMARY_CSV}")
    return dict(feat=feat, summary=summary, bands=pd.DataFrame(band_rows))


def load_profiles() -> dict:
    if not os.path.exists(PROFILE_FILE):
        raise FileNotFoundError(f"找不到 {PROFILE_FILE}，先跑 python src/build_profiles.py")
    return dict(np.load(PROFILE_FILE, allow_pickle=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description="对端部窗口做中线采样并缓存")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    setup_console()
    build(limit=args.limit or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
