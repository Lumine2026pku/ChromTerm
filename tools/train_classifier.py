# -*- coding: utf-8 -*-
"""train_classifier.py —— 端部类别分类器：消融、评估、保存模型（纯 CPU / numpy）。

跑法：
    python src/train_classifier.py

评估协议（三条并行报告，缺一不可）：
  1) 分组交叉验证（group-CV，按"原图"分组，5 折）：**用来选配置**。
     按原图分组是因为同一张片子的多个窗口高度相关，窗口级随机 CV 会高估。
  2) 官方 test（150 个窗口）：用来报告最终数字，和组织者给的切分保持一致。
  3) LOSO（留一来源批次）：换一批染色/物镜的片子还能不能用，最接近真实场景。

为什么选配置只认 group-CV：官方 val 只有 151 个窗口，单次划分的噪声 ±3–4 个点，
用它挑配置等于挑噪声（实测会把 test 从 0.49 挑到 0.36）。

输出（runs/）：
    ablation.csv / ablation.json   消融表（6 个特征集）
    model_compare.csv              分类器对比（linear / ncp / rff）
    confusion_best.png             混淆矩阵
    risk_coverage.png              风险-覆盖率曲线
    ablation_bars.png              消融条形图
    model_best.npz                 最佳模型（pipeline.py 直接加载）
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# 单线程 BLAS：多线程下浮点求和顺序不同，Adam 的更新会在末位产生差异，600 步累积后
# 会有 1–2 个测试样本翻转（实测 test_acc 在 0.480–0.507 之间摆动）。钉成 1 线程后
# 同一份代码重复跑结果逐位相同，报告里的数字才站得住。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromterm.common import (RUNS, bootstrap_ci, confusion_matrix, ensure, macro_f1,
                    per_class_report, save_json, setup_console)
from chromterm.data import CLASSES, load_cache, official_split
from chromterm.features import DEFAULT_SET, FEATURE_SETS, build_matrix
from chromterm.model import (Linear, NCP, PCAReducer, RFFRidge, Standardizer, ece,
                   risk_coverage, top2_accuracy)
from chromterm.viz import bars_png, confusion_png, curve_png

L2 = 1.0          # 强 L2：961 个样本、800+ 维特征，弱正则必然过拟合（扫出来的）
PCA = 0           # 不降维：PCA 白化在扫参里全面劣于直接强正则


# ------------------------------------------------------------------ 工具
def group_folds(groups, k: int = 5, seed: int = 0):
    """按"原图"分组的 K 折：同一原图的多个窗口一定落在同一折里。"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    idx = np.array([fold_of[g] for g in groups])
    return [(np.nonzero(idx != f)[0], np.nonzero(idx == f)[0]) for f in range(k)]


def repeated_folds(groups, k: int = 5, repeats: int = 3):
    """重复 K 折（不同洗牌种子）。

    单次 5 折的折间标准差约 ±0.03–0.04，而 C/F 两个配置的真实差距只有 0.007：
    用一次 5 折去选，等于在噪声里挑。重复 3 次（15 折）后选择才稳。
    """
    out = []
    for s in range(repeats):
        out += group_folds(groups, k, seed=s)
    return out


def make_clf(model: str, seed: int = 0):
    if model == "linear":
        return Linear(l2=L2, seed=seed)
    if model == "ncp":
        return NCP()
    if model == "rff":
        return RFFRidge(alpha=1.0, seed=seed)
    raise ValueError(model)


def fit_proba(Xtr, ytr, Xte, model="linear", pca=PCA, seed=0):
    st = Standardizer().fit(Xtr)
    A, B = st.transform(Xtr), st.transform(Xte)
    pc = None
    if pca:
        pc = PCAReducer(n_components=pca, whiten=True, seed=seed).fit(A)
        A, B = pc.transform(A), pc.transform(B)
    clf = make_clf(model, seed).fit(A, ytr)
    return clf.predict_proba(B), (st, pc, clf)


def sweep(X, y, folds, model="linear", pca=PCA, seed=0):
    f1, acc = [], []
    for a, b in folds:
        P, _ = fit_proba(X[a], y[a], X[b], model, pca, seed)
        f1.append(macro_f1(y[b], P.argmax(1), len(CLASSES)))
        acc.append(float((P.argmax(1) == y[b]).mean()))
    return float(np.mean(f1)), float(np.std(f1)), float(np.mean(acc)), f1


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="多种子重复（减少切分噪声）")
    args = ap.parse_args()
    ensure(RUNS)

    feat = load_cache()
    y = feat["cls"].astype(int)
    split = feat["split"].astype(str)
    groups = feat["image"].astype(str)
    tr, va, te = official_split(pd.DataFrame({"split": split}))
    folds = repeated_folds(groups, 5, 3)

    print("=" * 84)
    print(f"窗口 {len(y)}   类别 {dict(zip(CLASSES, np.bincount(y, minlength=6)))}")
    print(f"切分 train/val/test = {len(tr)}/{len(va)}/{len(te)}   原图 {len(np.unique(groups))}")
    used = pd.Series(feat["used"].astype(str))
    for i, mo in enumerate(("raw", "deskew", "tip")):
        col = used.iloc[i::3]
        print(f"  归一化 {mo:7s} " + ", ".join(f"{k}={v}" for k, v in col.value_counts().items()))

    # ---------------- 1) 消融 ----------------
    print("\n" + "-" * 84)
    print(f"消融（linear, l2=1.0, 无 PCA）；CV = 重复 5 折 ×3 次（共 {len(folds)} 折，按原图分组）")
    print(f"{'特征集':26s} {'维度':>5s} {'CV-F1':>16s} {'CV-Acc':>7s} {'test-Acc':>8s} {'test-F1':>8s}")
    rows = []
    Xs = {}
    for name, (mode, blocks) in FEATURE_SETS.items():
        X = build_matrix(feat, mode, blocks)
        Xs[name] = X
        cvf1, cvsd, cvacc, _ = sweep(X, y, folds)
        P_te, _ = fit_proba(X[np.r_[tr, va]], y[np.r_[tr, va]], X[te])
        acc = float((P_te.argmax(1) == y[te]).mean())
        lo, hi = bootstrap_ci(y[te], P_te.argmax(1),
                              lambda a, b: float((a == b).mean()))[1:]
        rows.append(dict(feature_set=name, mode=mode, blocks="+".join(blocks),
                         dim=int(X.shape[1]), cv_f1=cvf1, cv_f1_sd=cvsd, cv_acc=cvacc,
                         test_acc=acc, test_acc_lo=lo, test_acc_hi=hi,
                         test_f1=macro_f1(y[te], P_te.argmax(1), len(CLASSES)),
                         test_top2=top2_accuracy(P_te, y[te])))
        print(f"{name:26s} {X.shape[1]:5d} {cvf1:.3f}+-{cvsd:.3f}".ljust(64)
              + f" {cvacc:7.3f} {acc:8.3f} {rows[-1]['test_f1']:8.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RUNS, "ablation.csv"), index=False, encoding="utf-8-sig")
    save_json(rows, os.path.join(RUNS, "ablation.json"))

    # ---------------- 2) 分类器对比 ----------------
    best_row = df.iloc[int(np.argmax(df["cv_f1"]))]
    best_set = str(best_row["feature_set"])
    print(f"\n按 CV-F1 选出的特征集：{best_set}（CV-F1={best_row['cv_f1']:.3f}）")
    X = Xs[best_set]
    crow = []
    for m in ("linear", "ncp", "rff"):
        cvf1, cvsd, cvacc, _ = sweep(X, y, folds, m)
        P_te, _ = fit_proba(X[np.r_[tr, va]], y[np.r_[tr, va]], X[te], m)
        crow.append(dict(model=m, cv_f1=cvf1, cv_f1_sd=cvsd, cv_acc=cvacc,
                         test_acc=float((P_te.argmax(1) == y[te]).mean()),
                         test_f1=macro_f1(y[te], P_te.argmax(1), len(CLASSES))))
        print(f"  {m:7s} CV-F1={cvf1:.3f}±{cvsd:.3f}  CV-Acc={cvacc:.3f}  "
              f"test-Acc={crow[-1]['test_acc']:.3f}  test-F1={crow[-1]['test_f1']:.3f}")
    cdf = pd.DataFrame(crow)
    cdf.to_csv(os.path.join(RUNS, "model_compare.csv"), index=False, encoding="utf-8-sig")

    best_model = str(cdf.iloc[int(np.argmax(cdf["cv_f1"]))]["model"])
    print(f"按 CV-F1 选出的分类器：{best_model}")

    # ---------------- 3) 最终模型 + 详细报告 ----------------
    P_te, (st, pc, clf) = fit_proba(X[np.r_[tr, va]], y[np.r_[tr, va]], X[te], best_model)
    yte = y[te]
    pred = P_te.argmax(1)
    cm = confusion_matrix(yte, pred, len(CLASSES))
    confusion_png(cm, CLASSES, os.path.join(RUNS, "confusion_best.png"),
                  title=f"test confusion  ({best_set} / {best_model})")
    rep = per_class_report(yte, pred, CLASSES)
    cov, acc, cov90 = risk_coverage(P_te, yte)
    print("\n每类指标（test）：")
    for r in rep:
        print(f"  {r['cls']:6s} n={r['support']:3d}  P={r['precision']:.3f}  "
              f"R={r['recall']:.3f}  F1={r['f1']:.3f}")
    print(f"\n整体：acc={float((pred == yte).mean()):.3f}  macro-F1="
          f"{macro_f1(yte, pred, len(CLASSES)):.3f}  top2={top2_accuracy(P_te, yte):.3f}  "
          f"ECE={ece(P_te, yte):.3f}")
    print(f"拒识：守住 90% 精度需要放弃 {100 * (1 - (cov90 or 0)):.0f}% 的窗口"
          f"（覆盖率 {cov90:.2f}）" if cov90 else "拒识：任何覆盖率都到不了 90% 精度")

    # 风险-覆盖率曲线上的几个点
    print("覆盖率 -> 精度：" + "  ".join(
        f"{c:.0%}:{a:.2f}" for c, a in zip([1.0, 0.8, 0.6, 0.4, 0.2],
                                            [acc[min(len(acc) - 1, int(c * len(acc)) - 1)]
                                             for c in (1.0, 0.8, 0.6, 0.4, 0.2)])))
    lo = np.linspace(0.02, 1.0, 60)
    idx = np.clip((lo * (len(cov) - 1)).astype(int), 0, len(cov) - 1)
    curve_png([("test", lo, acc[idx], (70, 110, 190))],
              os.path.join(RUNS, "risk_coverage.png"),
              title=f"risk-coverage  ({best_set} / {best_model})",
              xlabel="coverage (accept highest-confidence first)", ylabel="accuracy")

    # 消融条形图
    bars_png(df["feature_set"].tolist(), df["cv_f1"].tolist(),
             os.path.join(RUNS, "ablation_bars.png"),
             title="ablation: macro-F1 (repeated 5-fold group CV)", vmin=0.20, vmax=0.55,
             highlight=int(np.argmax(df["cv_f1"].values)))

    # 质量闸门：几何量测不可信的窗口直接拒识
    geo = feat["geo"]
    gi = {str(n): i for i, n in enumerate(feat["geo_names"])}
    af, coh, wid = geo[:, gi["area_frac"]], geo[:, gi["coh"]], geo[:, gi["width_px"]]
    good = (af > 0.05) & (af < 0.75) & (coh > 0.2) & np.isfinite(wid) & (wid > 3)
    m = good[te]
    gate = None
    if m.sum() > 20:
        gate = dict(coverage=float(m.mean()),
                    acc_kept=float((pred[m] == yte[m]).mean()),
                    acc_all=float((pred == yte).mean()))
        print(f"\n质量闸门（几何标记不可信则拒识）：保留 {gate['coverage']*100:.0f}% 窗口，"
              f"精度 {gate['acc_all']:.3f} -> {gate['acc_kept']:.3f}")

    # ---- 留一来源批次（LOSO）：换一批片子的泛化 ----
    src = feat["source"].astype(str)
    loso = []
    for s in sorted(np.unique(src)):
        a = np.nonzero(src != s)[0]
        b = np.nonzero(src == s)[0]
        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
            continue
        P, _ = fit_proba(X[a], y[a], X[b], best_model)
        loso.append(dict(source=s, n=int(len(b)),
                         acc=float((P.argmax(1) == y[b]).mean()),
                         f1=macro_f1(y[b], P.argmax(1), len(CLASSES))))
    loso_acc = float(np.mean([r["acc"] for r in loso])) if loso else float("nan")
    print(f"\nLOSO（留一来源批次）平均准确率 {loso_acc:.3f}：" +
          "  ".join(f"{r['source']}={r['acc']:.2f}" for r in loso))

    # 风险-覆盖率曲线上的几个点（写进 summary，供报告/网页直接引用）
    rc_table = {}
    for c in (1.0, 0.8, 0.6, 0.4, 0.2):
        i = min(len(acc) - 1, max(0, int(c * len(acc)) - 1))
        rc_table[f"{int(round(c * 100))}%"] = round(float(acc[i]), 3)
    print("覆盖率 -> 精度：" + "  ".join(f"{k}:{v:.2f}" for k, v in rc_table.items()))

    summary = dict(feature_set=best_set, model=best_model, l2=L2, pca=PCA,
                   train_plus_val=int(len(tr) + len(va)), n_test=int(len(te)),
                   test_acc=float((pred == yte).mean()),
                   test_macro_f1=macro_f1(yte, pred, len(CLASSES)),
                   test_top2=top2_accuracy(P_te, yte), test_ece=ece(P_te, yte),
                   coverage_at_90=float(cov90) if cov90 else None,
                   risk_coverage=rc_table,
                   quality_gate=gate, per_class=rep, loso=loso, loso_acc=loso_acc,
                   cv_f1=float(sweep(X, y, folds, best_model)[0]),
                   cv_folds=int(len(folds)))
    save_json(summary, os.path.join(RUNS, "summary_best.json"))

    # 保存模型
    payload = dict(feature_set=np.array([best_set], dtype=object),
                   mode=np.array([FEATURE_SETS[best_set][0]], dtype=object),
                   blocks=np.array([list(FEATURE_SETS[best_set][1])], dtype=object),
                   classes=np.asarray(CLASSES, dtype=object),
                   model=np.array([best_model], dtype=object),
                   mu=st.mu, sd=st.sd)
    if pc is not None:
        payload.update(pca_mu=pc.mu, pca_comp=pc.components_, pca_k=pc.k)
    if isinstance(clf, Linear):
        payload.update(W=clf.W, b=clf.b)
    elif isinstance(clf, RFFRidge):
        payload.update(Wr=clf.Wr, br=clf.br, Wo=clf.Wo, gamma=clf.gamma)
    else:
        payload.update(C=clf.C)
    np.savez(os.path.join(RUNS, "model_best.npz"), **payload)
    print("\n已写入 runs/：ablation.csv / model_compare.csv / summary_best.json / "
          "model_best.npz / *.png")


if __name__ == "__main__":
    main()
