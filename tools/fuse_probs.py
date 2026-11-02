# -*- coding: utf-8 -*-
"""fuse_probs.py —— 把两个模型的"每窗 6 类概率"融合，权重只用验证集选。

为什么需要它
------------
组里另一条线的 ResNet-18 在 test 上准确率 54.0%（macro-F1 0.531），
它自己与"带型曲线"做概率融合后到 58.7%（macro-F1 0.559）——这是目前已知
最大的一次单项提升。但那次融合依赖两件事：两张模型对同一批窗口都输出概率、
且融合权重只在 val 上选。本项目此前没法复现，因为拿不到对方的逐窗概率。

把对方导出的概率 CSV 放进一个目录，本脚本就能做同样的事（并防止在 test 上调参）。

输入格式（两份 CSV，都按窗口名对齐）
------------------------------------
    name,2L,2R,3L,3R,X,other
    org__2L-1__0,0.02,0.77,0.05,0.06,0.03,0.07

* 列名大小写不敏感；也接受 `file`/`image`/`window` 作为名字列；
* 概率不要求和为 1（脚本会归一化）；
* 只有 top-1 没有概率的 CSV 无法融合——会明确报错而不是瞎猜。

用法
----
    # 1) 先把本项目的概率导出（若还没有）
    python src/fuse_probs.py export --out runs/probs_ours.csv

    # 2) 与对方的概率融合（--labels 提供 y 与 split，用于选权重与报数）
    python src/fuse_probs.py fuse --mine runs/probs_ours.csv \
        --other /path/to/resnet_probs.csv \
        --labels data_share/dataset/level1/labels.csv \
        --out runs/fused.csv

    # 3) 自检（合成一个"外部模型"，验证融合机制本身）
    python src/fuse_probs.py self-test
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

from chromterm.common import DATA, RUNS, ensure, macro_f1, setup_console
from chromterm.data import CLASSES

NAME_COLS = ("name", "file", "image", "window", "id")


# ------------------------------------------------------------------ 读写
def _norm_name(s: str) -> str:
    s = str(s).strip()
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
    return s


def load_probs(path: str) -> pd.DataFrame:
    """读一份概率 CSV -> 索引为窗口名的 6 列 DataFrame（已归一化）。"""
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}
    name_col = next((lower[k] for k in NAME_COLS if k in lower), None)
    if name_col is None:
        raise SystemExit(f"{path}: 找不到名字列（支持 {NAME_COLS}）")
    cls_col = {}
    for c in CLASSES:
        for cand in (c, c.lower(), c.upper()):
            if cand in df.columns:
                cls_col[c] = cand
                break
            if cand in lower:
                cls_col[c] = lower[cand]
                break
    missing = [c for c in CLASSES if c not in cls_col]
    if missing:
        raise SystemExit(f"{path}: 缺少类别列 {missing}；只有 top-1 的 CSV 无法融合")
    out = pd.DataFrame({c: df[cls_col[c]].astype(float).to_numpy() for c in CLASSES})
    out.index = [_norm_name(v) for v in df[name_col]]
    s = out.sum(axis=1).to_numpy()
    bad = ~np.isfinite(s) | (s <= 0)
    if bad.any():
        raise SystemExit(f"{path}: {int(bad.sum())} 行的概率全为 0 或非数值")
    out = out.div(s, axis=0)
    out = out[~out.index.duplicated(keep="first")]
    return out


def load_labels(path: str) -> pd.DataFrame:
    lab = pd.read_csv(path)
    if "name" not in lab.columns:
        raise SystemExit(f"{path}: 需要 name 列")
    lab = lab.copy()
    lab["name"] = [_norm_name(v) for v in lab["name"]]
    return lab.set_index("name")


# ------------------------------------------------------------------ 融合
def fuse(P1: pd.DataFrame, P2: pd.DataFrame, w: float) -> pd.DataFrame:
    idx = P1.index.intersection(P2.index)
    return w * P1.loc[idx] + (1.0 - w) * P2.loc[idx]


def pick_w(P1, P2, y, w_grid=None) -> tuple:
    """只在给定子集（应为 val）上按 macro-F1 选权重。"""
    if w_grid is None:
        w_grid = np.arange(0.0, 1.0001, 0.05)
    idx = P1.index.intersection(P2.index)
    yy = np.array([CLASSES.index(v) for v in y.loc[idx, "cls"]])
    best = None
    rows = []
    for w in w_grid:
        Pf = fuse(P1, P2, float(w))
        pred = Pf.to_numpy().argmax(1)
        f1 = macro_f1(yy, pred, 6)
        acc = float((pred == yy).mean())
        rows.append(dict(w=float(w), f1=f1, acc=acc))
        if best is None or f1 > best[1]:
            best = (float(w), f1, acc)
    return best[0], best[1], best[2], pd.DataFrame(rows)


def _report(tag: str, P: pd.DataFrame, y: pd.DataFrame) -> dict:
    idx = P.index.intersection(y.index)
    yy = np.array([CLASSES.index(v) for v in y.loc[idx, "cls"]])
    A = P.loc[idx].to_numpy()
    pred = A.argmax(1)
    order = np.argsort(-A, axis=1)[:, :2]
    top2 = float(np.mean([yy[i] in order[i] for i in range(len(yy))]))
    r = dict(tag=tag, n=int(len(idx)), acc=float((pred == yy).mean()),
             f1=macro_f1(yy, pred, 6), top2=top2)
    print(f"  {tag:26s} n={r['n']:3d}  acc {r['acc']:.3f}  F1 {r['f1']:.3f}  "
          f"top2 {r['top2']:.3f}")
    return r


# ------------------------------------------------------------------ 命令
def cmd_export(args):
    from chromterm.data import load_cache, official_split
    from chromterm.features import FEATURE_SETS, build_matrix
    from chromterm.model import ShrinkLDA, Standardizer

    feat = load_cache()
    y = feat["cls"].astype(int)
    split = feat["split"].astype(str)
    X = build_matrix(feat, *FEATURE_SETS["C_OD全特征"])
    trva = np.nonzero(split != "test")[0]
    st = Standardizer().fit(X[trva])
    clf = ShrinkLDA(alpha=0.6).fit(st.transform(X[trva]), y[trva])
    P = np.zeros((len(X), 6))
    P[:, clf.classes_.astype(int)] = clf.predict_proba(st.transform(X))
    out = pd.DataFrame(P, columns=CLASSES)
    out.insert(0, "name", [str(n) for n in feat["names"]])
    out.insert(1, "split", split)
    ensure(os.path.dirname(os.path.abspath(args.out)))
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"已导出 {len(out)} 行 -> {args.out}")
    return 0


def cmd_fuse(args):
    P1 = load_probs(args.mine)
    P2 = load_probs(args.other)
    common = P1.index.intersection(P2.index)
    print(f"两份概率：{len(P1)} / {len(P2)} 行，按窗口名对齐后 {len(common)} 行")
    if len(common) < 20:
        raise SystemExit("可对齐的窗口太少，检查两份 CSV 的名字列格式")
    y = load_labels(args.labels) if args.labels else None
    if y is None:
        Pf = fuse(P1, P2, 0.5)
        Pf.to_csv(args.out, float_format="%.4f")
        print(f"没有给 --labels，按 w=0.5 直接融合并写出 {args.out}")
        return 0

    y = y.loc[y.index.intersection(common)]
    val = y.index[y["split"] == "val"] if "split" in y.columns else y.index[: len(y) // 4]
    te = y.index[y["split"] == "test"] if "split" in y.columns else y.index[len(y) // 4:]
    if len(val) < 10 or len(te) < 10:
        raise SystemExit(f"val/test 太小（{len(val)}/{len(te)}），检查 labels.csv 的 split 列")

    print("\n== 单模型（同一批窗口）==")
    r1 = _report("本项目模型", P1.loc[te], y.loc[te])
    r2 = _report("外部模型", P2.loc[te], y.loc[te])

    print(f"\n== 在 val（{len(val)} 窗）上选融合权重 ==")
    w, f1v, accv, grid = pick_w(P1.loc[val], P2.loc[val], y.loc[val])
    print(f"  最优 w={w:.2f}（val acc {accv:.3f}、F1 {f1v:.3f}）")

    Pf = fuse(P1, P2, w)
    print("\n== test 结果 ==")
    r3 = _report(f"融合 w={w:.2f}", Pf.loc[te], y.loc[te])
    print(f"\n  相对本项目模型：acc {r3['acc']-r1['acc']:+.3f}  F1 {r3['f1']-r1['f1']:+.3f}")
    print(f"  相对外部模型：  acc {r3['acc']-r2['acc']:+.3f}  F1 {r3['f1']-r2['f1']:+.3f}")
    se = np.sqrt(max(r3["acc"] * (1 - r3["acc"]), 1e-9) / max(r3["n"], 1))
    print(f"  提醒：test {r3['n']} 窗，准确率标准误 ±{se:.3f}；小于 ±5% 的差别可能是抽样波动")

    out = Pf.copy()
    out.insert(0, "name", out.index)
    out.to_csv(args.out, index=False, encoding="utf-8-sig", float_format="%.4f")
    grid.to_csv(os.path.splitext(args.out)[0] + "_w_sweep.csv", index=False)
    print(f"\n融合概率已写出 {args.out}（权重扫描表同名 _w_sweep.csv）")
    return 0


def cmd_self_test(args):
    """合成一个"外部模型"，验证融合机制：w 应被选到 0 与 1 之间且 test 变好。"""
    rng = np.random.default_rng(0)
    from chromterm.data import load_cache, official_split
    from chromterm.features import FEATURE_SETS, build_matrix
    from chromterm.model import ShrinkLDA, Standardizer

    feat = load_cache()
    y = feat["cls"].astype(int)
    split = feat["split"].astype(str)
    X = build_matrix(feat, *FEATURE_SETS["C_OD全特征"])
    tr, va = np.nonzero(split == "train")[0], np.nonzero(split == "val")[0]
    te = np.nonzero(split == "test")[0]
    st = Standardizer().fit(X[np.r_[tr, va]])
    clf = ShrinkLDA(alpha=0.6).fit(st.transform(X[np.r_[tr, va]]), y[np.r_[tr, va]])
    P = np.zeros((len(X), 6))
    P[:, clf.classes_.astype(int)] = clf.predict_proba(st.transform(X))
    P1 = pd.DataFrame(P, columns=CLASSES)
    P1.index = [str(n) for n in feat["names"]]

    # 合成"外部模型"：70% 的窗口给出正确的尖锐概率，30% 随机 —— 与主干互补但更弱
    P2 = P.copy() * 0.4 + rng.dirichlet(np.ones(6), size=len(P)) * 0.6
    for i in rng.choice(len(P2), size=int(0.3 * len(P2)), replace=False):
        P2[i] = rng.dirichlet(np.ones(6))
    P2 = P2 / P2.sum(1, keepdims=True)
    P2 = pd.DataFrame(P2, columns=CLASSES, index=P1.index)

    ydf = pd.DataFrame({"cls": [CLASSES[i] for i in y], "split": split}, index=P1.index)
    print("== 自检：合成的外部模型（更弱但互补）==")
    r1 = _report("本项目模型", P1.loc[P1.index[te]], ydf.loc[P1.index[te]])
    r2 = _report("合成外部模型", P2.loc[P1.index[te]], ydf.loc[P1.index[te]])
    val = P1.index[va]
    w, f1v, accv, _ = pick_w(P1.loc[val], P2.loc[val], ydf.loc[val])
    print(f"\n  val 上选出 w={w:.2f}（val acc {accv:.3f}）")
    Pf = fuse(P1, P2, w)
    r3 = _report(f"融合 w={w:.2f}", Pf.loc[P1.index[te]], ydf.loc[P1.index[te]])
    ok = (0.0 < w < 1.0) and (r3["acc"] >= max(r1["acc"], r2["acc"]) - 1e-9)
    print(f"\n  机制自检：{'通过' if ok else '未通过'}"
          f"（w 在 0-1 之间，且融合不差于两个单模型）")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="两模型概率融合（权重只用 val 选）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("export", help="导出本项目的逐窗概率")
    a.add_argument("--out", default=os.path.join(RUNS, "probs_ours.csv"))
    a.set_defaults(func=cmd_export)

    b = sub.add_parser("fuse", help="与外部模型的概率融合")
    b.add_argument("--mine", required=True)
    b.add_argument("--other", required=True)
    b.add_argument("--labels", default=os.path.join(DATA, "dataset", "level1", "labels.csv"))
    b.add_argument("--out", default=os.path.join(RUNS, "fused.csv"))
    b.set_defaults(func=cmd_fuse)

    c = sub.add_parser("self-test", help="用合成外部模型验证融合机制")
    c.set_defaults(func=cmd_self_test)

    args = ap.parse_args(argv)
    setup_console()
    ensure(RUNS)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
