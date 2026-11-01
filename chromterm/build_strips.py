# -*- coding: utf-8 -*-
"""build_strips.py —— 把每个端部窗口"拉直"成规范化长条并缓存。

产物：`_cache/strips.npz`
    strip_od     (N,48,96) float16  归一化染色量，横轴 0–4 个臂宽、纵轴 ±1.6 个臂宽
    strip_odabs  (N,48,96) float16  未归一化的染色量（保留绝对强度）
    strip_gray   (N,48,96) float16  标准亮度（与旧数据同口径）
    strip_mask   (N,48,96) float16  到掩膜边界的距离（"这条臂在这里有多厚"）
    W / U / length / ok

为什么值得单独存：拉直后同一张图既保留了"带纹序列"，又保留了"横向胖瘦"，
而且朝向/平移/尺度三个变量被消掉——这正是此前 ResNet 版本要用 360° 旋转增强
去硬学的东西。
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .common import CACHE, ensure, load_rgb, setup_console
from .data import load_index
from .profile import profile_of_window, rectify_strip

STRIP_FILE = os.path.join(CACHE, "strips.npz")


def build(limit: int | None = None, out_w: int = 96, out_h: int = 48,
          verbose: bool = True) -> dict:
    idx = load_index()
    if limit:
        idx = idx.iloc[:limit].reset_index(drop=True)
    n = len(idx)
    od, oda, gry, msk, ins = [], [], [], [], []
    Ws, Us, lens, oks = [], [], [], []
    t0 = time.time()
    for k, row in enumerate(idx.itertuples(index=False)):
        rgb = load_rgb(row.path)
        prof = profile_of_window(rgb, extend_frac=30)
        if not prof.get("ok"):
            st = {"ok": False}
        else:
            st = rectify_strip(rgb, prof["stain"]["S"], prof["stain"]["S_n"],
                               prof["info"], out_w=out_w, out_h=out_h)
        ok = bool(st.get("ok"))
        oks.append(ok)
        if ok:
            od.append(st["strip_od"])
            oda.append(st["strip_odabs"])
            gry.append(st["strip_gray"])
            msk.append(st["strip_mask"])
            ins.append(st["strip_inside"])
            Ws.append(st["W"])
            Us.append(st["U"])
            lens.append(prof.get("n_points", 0))
        else:
            od.append(np.zeros((out_h, out_w), np.float32))
            oda.append(np.zeros((out_h, out_w), np.float32))
            gry.append(np.zeros((out_h, out_w), np.float32))
            msk.append(np.zeros((out_h, out_w), np.float32))
            ins.append(np.zeros((out_h, out_w), np.float32))
            Ws.append(np.nan)
            Us.append(np.nan)
            lens.append(0)
        if verbose and (k + 1) % 100 == 0:
            print(f"  拉直 {k+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    feat = dict(
        names=idx["name"].to_numpy().astype(object),
        cls=idx["cls_idx"].to_numpy().astype(np.int32),
        split=idx["split"].to_numpy().astype(object),
        source=idx["source"].to_numpy().astype(object),
        image=idx["image"].to_numpy().astype(object),
        ok=np.asarray(oks, bool),
        W=np.asarray(Ws, np.float32), U=np.asarray(Us, np.float32),
        length=np.asarray(lens, np.int32),
        out_w=out_w, out_h=out_h,
        strip_od=np.stack(od).astype(np.float16),
        strip_odabs=np.stack(oda).astype(np.float16),
        strip_gray=np.stack(gry).astype(np.float16),
        strip_mask=np.stack(msk).astype(np.float16),
        strip_inside=np.stack(ins).astype(np.float16),
    )
    ensure(CACHE)
    np.savez_compressed(STRIP_FILE, **feat)
    if verbose:
        print(f"\n成功拉直 {int(np.sum(oks))}/{n}；臂宽 W 中位 "
              f"{np.nanmedian(Ws):.1f} px；沿轴长度 U 中位 {np.nanmedian(Us):.0f} px")
        print(f"写入 {STRIP_FILE}")
    return feat


def load_strips() -> dict:
    if not os.path.exists(STRIP_FILE):
        raise FileNotFoundError(f"找不到 {STRIP_FILE}，先跑 python src/build_strips.py")
    return dict(np.load(STRIP_FILE, allow_pickle=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description="把端部窗口拉直成规范化长条并缓存")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    setup_console()
    build(limit=args.limit or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
