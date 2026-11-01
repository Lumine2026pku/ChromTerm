# -*- coding: utf-8 -*-
"""train_final.py —— 最终版训练/评估：把"中线采样"这条路线完整纳入消融，
并把真正有效的改动（收缩 LDA + 温度校准 + 拒识）落成可交付的模型。

与旧版 train_classifier.py 的关系
--------------------------------
* 评估协议完全沿用旧版（重复 5 折 ×3 按原图分组 CV 选配置、官方 test 报数、
  LOSO 看跨批次），所以两组数字可以直接对比。
* 新增三件事：
  1) **中线采样特征**（`profile.py` 的一维曲线 + `build_strips.py` 的拉直长条）
     作为独立的消融行，用同一套协议量化它到底有没有用；
  2) **收缩 LDA** 作为新的分类器候选（961 样本 / 836 维下比逻辑回归更稳）；
  3) **温度校准**：把 T 折进 W、b，所以导出的仍是一个线性模型，
     pipeline / 网页 / JS 引擎都不用改，但置信度从此可以当概率读。

跑法：python src/train_final.py
产物：runs/final_*.csv|json|png、runs/model_best.npz（覆盖旧模型）
"""

from __future__ import annotations

import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromterm.common import (RUNS, bootstrap_ci, confusion_matrix, ensure, macro_f1,
                    per_class_report, save_json, setup_console)
from chromterm.data import CLASSES, load_cache, official_split
from chromterm.features import FEATURE_SETS, build_matrix as build_window_features
from chromterm.model import (Calibrated, Linear, NCP, ShrinkLDA, Standardizer, ece,
                   fit_temperature, risk_coverage, softmax, top2_accuracy)
from chromterm import pfeat
from chromterm.viz import bars_png, confusion_png, curve_png

L2 = 1.0
LDA_ALPHA = 0.6


# ------------------------------------------------------------------ 工具
def group_folds(groups, k: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    idx = np.array([fold_of[g] for g in groups])
    return [(np.nonzero(idx != f)[0], np.nonzero(idx == f)[0]) for f in range(k)]


def repeated_folds(groups, k: int = 5, repeats: int = 3):
    out = []
    for s in range(repeats):
        out += group_folds(groups, k, seed=s)
    return out


def make_clf(model: str, seed: int = 0):
    if model == "linear":
        return Linear(l2=L2, seed=seed)
    if model == "ncp":
        return NCP()
    if model == "lda":
        return ShrinkLDA(alpha=LDA_ALPHA)
    raise ValueError(model)


def fit_proba(Xtr, ytr, Xte, model: str = "lda", n_cls: int = 6, seed: int = 0):
    st = Standardizer().fit(Xtr)
    clf = make_clf(model, seed).fit(st.transform(Xtr), ytr)
    P = clf.predict_proba(st.transform(Xte))
    out = np.zeros((P.shape[0], n_cls))
    out[:, clf.classes_.astype(int)] = P
    return out


def cv_scores(X, y, folds, model="lda"):
    f1, acc, t2 = [], [], []
    for a, b in folds:
        P = fit_proba(X[a], y[a], X[b], model)
        f1.append(macro_f1(y[b], P.argmax(1), 6))
        acc.append(float((P.argmax(1) == y[b]).mean()))
        t2.append(top2_accuracy(P, y[b]))
    return dict(cv_f1=float(np.mean(f1)), cv_f1_sd=float(np.std(f1)),
                cv_acc=float(np.mean(acc)), cv_top2=float(np.mean(t2)))


# ------------------------------------------------------------------ 主流程
def main():
    setup_console()
    ensure(RUNS)
    t0 = time.time()

    feat = load_cache()
    y = feat["cls"].astype(int)
    split = feat["split"].astype(str)
    groups = feat["image"].astype(str)
    src = feat["source"].astype(str)
    tr, va, te = official_split(pd.DataFrame({"split": split}))
    trva = np.r_[tr, va]
    folds = repeated_folds(groups, 5, 3)
    yte = y[te]

    print("=" * 92)
    print(f"窗口 {len(y)}（train/val/test = {len(tr)}/{len(va)}/{len(te)}），"
          f"原图 {len(np.unique(groups))} 张，来源批次 {len(np.unique(src))} 个")
    print("=" * 92)

    # ---------------- 0) 特征矩阵 ----------------
    Xwin = build_window_features(feat, *FEATURE_SETS["C_OD全特征"])
    Xrgb = build_window_features(feat, *FEATURE_SETS["A_RGB基线"])
    Xpyr = build_window_features(feat, *FEATURE_SETS["B_OD多尺度"])
    Xpro, Xstrip, Xfuse_extra = None, None, None
    try:
        from chromterm import build_profiles
        from chromterm import build_strips
        from chromterm.common import CACHE
        prof = build_profiles.load_profiles()
        bands = pd.read_csv(os.path.join(CACHE, "profiles_bands.csv"))
        bmap = {str(n): [dict(r) for _, r in g.iterrows()] for n, g in bands.groupby("name")}
        Xpro, _, _ = pfeat.build_matrix(prof, abs_mode="W", bands_by_name=bmap)
    except FileNotFoundError as e:
        print(f"  [跳过] 中线曲线缓存缺失（{e}）——先跑 python src/build_profiles.py")
    try:
        strips = build_strips.load_strips()
        from chromterm.features import (feat_auto, feat_band, feat_hog, feat_pyr,
                                        feat_width, pool2d)
        S = np.clip(strips["strip_od"].astype(np.float32), 0, 1)
        M = strips["strip_mask"].astype(np.float32)
        Xstrip = np.concatenate([
            np.stack([feat_pyr(s) for s in S]),
            np.stack([feat_band(s) for s in S]),
            np.stack([feat_hog(s) for s in S]),
            np.stack([feat_width(s) for s in S]),
            np.stack([feat_auto(s) for s in S]),
            np.stack([pool2d(m, 8, 6).ravel() for m in M]),
        ], axis=1).astype(np.float32)
    except FileNotFoundError as e:
        print(f"  [跳过] 拉直长条缓存缺失（{e}）——先跑 python src/build_strips.py")

    # ---------------- 1) 消融 ----------------
    print("\n" + "-" * 92)
    print("消融：每一行只用一组特征，协议相同（按原图分组的重复 5 折 ×3；test 只报数）")
    print(f"{'特征集':34s} {'维度':>5s} {'CV-F1':>16s} {'CV-Acc':>7s} {'CV-top2':>8s} "
          f"{'test-Acc':>9s} {'test-F1':>8s} {'test-top2':>9s}")
    rows = []
    feature_sets = [
        ("A_RGB基线（直接缩 RGB）", Xrgb),
        ("B_OD多尺度池化", Xpyr),
        ("C_OD全特征（此前最好）", Xwin),
    ]
    if Xpro is not None:
        feature_sets.append(("G_中线曲线1D（新）", Xpro))
    if Xstrip is not None:
        feature_sets.append(("H_拉直长条2D（新）", Xstrip))
    if Xpro is not None:
        feature_sets.append(("I_C+中线曲线1D", np.hstack([Xwin, Xpro])))
    if Xstrip is not None:
        feature_sets.append(("J_C+拉直长条2D", np.hstack([Xwin, Xstrip])))
    if Xpro is not None and Xstrip is not None:
        feature_sets.append(("K_C+中线+长条", np.hstack([Xwin, Xpro, Xstrip])))

    Xs = {}
    for name, X in feature_sets:
        Xs[name] = X
        cv = cv_scores(X, y, folds, "lda")
        P = fit_proba(X[trva], y[trva], X[te], "lda")
        acc = float((P.argmax(1) == yte).mean())
        rows.append(dict(feature_set=name, dim=int(X.shape[1]), **cv,
                         test_acc=acc,
                         test_f1=macro_f1(yte, P.argmax(1), 6),
                         test_top2=top2_accuracy(P, yte)))
        r = rows[-1]
        print(f"{name:34s} {X.shape[1]:5d} {r['cv_f1']:.3f}+-{r['cv_f1_sd']:.3f}".ljust(62)
              + f" {r['cv_acc']:7.3f} {r['cv_top2']:8.3f} {acc:9.3f} "
                f"{r['test_f1']:8.3f} {r['test_top2']:9.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RUNS, "final_ablation.csv"), index=False, encoding="utf-8-sig")
    save_json(rows, os.path.join(RUNS, "final_ablation.json"))

    # ---------------- 2) 分类器对比（固定 C_OD全特征） ----------------
    print("\n" + "-" * 92)
    print("分类器对比（特征固定为 C_OD全特征）：收缩 LDA 是这次唯一稳定为正的改动")
    print(f"{'分类器':14s} {'CV-F1':>16s} {'CV-Acc':>7s} {'CV-top2':>8s} "
          f"{'test-Acc':>9s} {'test-F1':>8s} {'test-top2':>9s} {'ECE':>6s}")
    crow = []
    for m, tag in (("linear", "多项逻辑回归"), ("ncp", "最近类原型"), ("lda", "收缩 LDA")):
        cv = cv_scores(Xwin, y, folds, m)
        P = fit_proba(Xwin[trva], y[trva], Xwin[te], m)
        crow.append(dict(model=m, tag=tag, **cv,
                         test_acc=float((P.argmax(1) == yte).mean()),
                         test_f1=macro_f1(yte, P.argmax(1), 6),
                         test_top2=top2_accuracy(P, yte), ece=ece(P, yte)))
        r = crow[-1]
        print(f"{tag:14s} {r['cv_f1']:.3f}+-{r['cv_f1_sd']:.3f}".ljust(32)
              + f" {r['cv_acc']:7.3f} {r['cv_top2']:8.3f} {r['test_acc']:9.3f} "
                f"{r['test_f1']:8.3f} {r['test_top2']:9.3f} {r['ece']:6.3f}")
    pd.DataFrame(crow).to_csv(os.path.join(RUNS, "final_models.csv"), index=False,
                              encoding="utf-8-sig")

    best = max(crow, key=lambda r: r["cv_f1"])
    best_model = best["model"]
    print(f"\n按 CV-F1 选出：{best['tag']}（CV-F1={best['cv_f1']:.3f}）")

    # ---------------- 3) 温度校准 ----------------
    # val 上的概率必须来自"没见过 val"的模型：用 train 单独训练一次
    Pva_raw = fit_proba(Xwin[tr], y[tr], Xwin[va], best_model)
    T = fit_temperature(Pva_raw, y[va])
    print(f"温度校准：在 val（{len(va)} 个窗口）上拟合 T = {T:.2f}"
          f"（T>1 = 原模型过度自信，T<1 = 过于保守）")

    st = Standardizer().fit(Xwin[trva])
    # 校准前的同一模型（train+val 训练、不折温度），用于对照 ECE
    clf_raw = make_clf(best_model).fit(st.transform(Xwin[trva]), y[trva])
    P_te_raw = np.zeros((len(te), 6))
    P_te_raw[:, clf_raw.classes_.astype(int)] = clf_raw.predict_proba(st.transform(Xwin[te]))

    clf = Calibrated(make_clf(best_model), T=T).fit(st.transform(Xwin[trva]), y[trva])
    lin = clf.fold()                       # 把 T 折进 W、b：下游无需改动
    P_te = lin.predict_proba(st.transform(Xwin[te]))

    # 校准前后指标
    cm = confusion_matrix(yte, P_te.argmax(1), 6)
    confusion_png(cm, CLASSES, os.path.join(RUNS, "confusion_best.png"),
                  title=f"official test (150 windows)  {best['tag']} + T={T:.2f}")
    rep = per_class_report(yte, P_te.argmax(1), CLASSES)
    acc = float((P_te.argmax(1) == yte).mean())
    lo, hi = bootstrap_ci(yte, P_te.argmax(1), lambda a, b: float((a == b).mean()))[1:]

    print("\n" + "-" * 92)
    print("最终模型（train+val 训练、温度已折进 W/b）")
    print(f"  test 准确率 {acc:.3f}  95%CI [{lo:.3f}, {hi:.3f}]   "
          f"macro-F1 {macro_f1(yte, P_te.argmax(1), 6):.3f}   "
          f"top-2 {top2_accuracy(P_te, yte):.3f}")
    print(f"  校准：ECE {ece(P_te_raw, yte):.3f} -> {ece(P_te, yte):.3f}"
          f"（argmax 与准确率不变，只是概率变得可读）")
    print("\n  每类：")
    for r in rep:
        print(f"    {r['cls']:6s} n={r['support']:3d}  P={r['precision']:.3f}  "
              f"R={r['recall']:.3f}  F1={r['f1']:.3f}")

    # ---------------- 4) 拒识：把"不确定"交给人 ----------------
    cov, accs, _ = risk_coverage(P_te, yte)
    conf = P_te.max(1)
    order = np.argsort(conf)[::-1]
    correct = (P_te.argmax(1) == yte)[order]
    rc_table = {}
    print("\n风险-覆盖率（按置信度从高到低接受）：")
    for c in (1.0, 0.8, 0.6, 0.4, 0.2):
        n = max(1, int(round(c * len(yte))))
        rc_table[f"{int(c*100)}%"] = round(float(correct[:n].mean()), 3)
        print(f"  接受 {c:4.0%} 的窗口 -> 精度 {rc_table[f'{int(c*100)}%']:.3f}")

    # 同一套切分上，把"第一版逻辑回归"的同样指标也算一遍，
    # 这样对照表里的每个数字都是同一批数据、同一时刻算出来的（不跨版本抄数）。
    lin_cmp = None
    if best_model != "linear":
        P_lin = fit_proba(Xwin[trva], y[trva], Xwin[te], "linear")
        conf_l = P_lin.max(1)
        order_l = np.argsort(conf_l)[::-1]
        corr_l = (P_lin.argmax(1) == yte)[order_l]
        rc_l = {}
        for c in (1.0, 0.8, 0.6, 0.4, 0.2):
            n = max(1, int(round(c * len(yte))))
            rc_l[f"{int(c*100)}%"] = round(float(corr_l[:n].mean()), 3)
        lin_cmp = dict(test_acc=float((P_lin.argmax(1) == yte).mean()),
                       test_f1=macro_f1(yte, P_lin.argmax(1), 6),
                       test_top2=top2_accuracy(P_lin, yte),
                       test_ece=ece(P_lin, yte), risk_coverage=rc_l)
        print("  对照（第一版逻辑回归，同一批数据）："
              f"acc={lin_cmp['test_acc']:.3f}  top2={lin_cmp['test_top2']:.3f}  "
              f"ECE={lin_cmp['test_ece']:.3f}  "
              + "  ".join(f"{k}:{v:.3f}" for k, v in rc_l.items()))
    lo_grid = np.linspace(0.02, 1.0, 60)
    idx = np.clip((lo_grid * (len(cov) - 1)).astype(int), 0, len(cov) - 1)
    curve_png([("test", lo_grid, accs[idx], (70, 110, 190))],
              os.path.join(RUNS, "risk_coverage.png"),
              title=f"risk-coverage  {best['tag']} + T={T:.2f}",
              xlabel="coverage (accept highest-confidence first)", ylabel="accuracy")

    # ---------------- 5) 任务粒度：这份数据到底能回答多细的问题 ----------------
    print("\n" + "-" * 92)
    print("任务粒度（同一套特征/分类器，只改变问题问得多细）")
    schemes = {
        "2类 2L vs 非2L": {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        "3类 2L / X / 其余": {0: 0, 1: 2, 2: 2, 3: 2, 4: 1, 5: 2},
        "4类 2R+3L+3R 合并": {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3},
        "6类 原任务": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    }
    grows = []
    for nm, mp in schemes.items():
        ym = np.array([mp[int(v)] for v in y], int)
        k = int(ym.max()) + 1
        P = fit_proba(Xwin[trva], ym[trva], Xwin[te], best_model, n_cls=k)
        a = float((P.argmax(1) == ym[te]).mean())
        l2_, h2_ = bootstrap_ci(ym[te], P.argmax(1),
                                lambda x_, z_: float((x_ == z_).mean()))[1:]
        grows.append(dict(task=nm, n_cls=k, acc=a, lo=l2_, hi=h2_,
                          f1=macro_f1(ym[te], P.argmax(1), k), chance=1.0 / k))
        print(f"  {nm:20s} 类数 {k}  准确率 {a:.3f} [{l2_:.3f},{h2_:.3f}]  "
              f"F1 {grows[-1]['f1']:.3f}  随机 {1.0/k:.3f}")
    pd.DataFrame(grows).to_csv(os.path.join(RUNS, "final_granularity.csv"), index=False,
                               encoding="utf-8-sig")

    # ---------------- 6) LOSO ----------------
    print("\n" + "-" * 102)
    loso = []
    for s in sorted(np.unique(src)):
        a = np.nonzero(src != s)[0]
        b = np.nonzero(src == s)[0]
        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
            continue
        P = fit_proba(Xwin[a], y[a], Xwin[b], best_model)
        loso.append(dict(source=s, n=int(len(b)),
                         acc=float((P.argmax(1) == y[b]).mean()),
                         f1=macro_f1(y[b], P.argmax(1), 6),
                         top2=top2_accuracy(P, y[b])))
        print(f"  留出 {s:10s} n={len(b):3d}  acc={loso[-1]['acc']:.3f}  "
              f"F1={loso[-1]['f1']:.3f}  top2={loso[-1]['top2']:.3f}")
    loso_acc = float(np.mean([r["acc"] for r in loso])) if loso else float("nan")
    print(f"  LOSO 平均准确率 {loso_acc:.3f}")

    # ---------------- 7) 保存 ----------------
    payload = dict(
        feature_set=np.array([FEATURE_SETS["C_OD全特征"][0]], dtype=object),
        mode=np.array(["raw"], dtype=object),
        blocks=np.array([list(FEATURE_SETS["C_OD全特征"][1])], dtype=object),
        classes=np.asarray(CLASSES, dtype=object),
        model=np.array(["linear"], dtype=object),      # 折完温度后仍是线性
        base_model=np.array([best_model], dtype=object),
        temperature=np.array([T], np.float32),
        lda_alpha=np.array([LDA_ALPHA], np.float32),
        mu=st.mu, sd=st.sd, W=lin.W, b=lin.b,
    )
    np.savez(os.path.join(RUNS, "model_best.npz"), **payload)

    summary = dict(
        feature_set="C_OD全特征", base_model=best_model, lda_alpha=LDA_ALPHA,
        temperature=T, linear_after_calibration=True,
        train_plus_val=int(len(trva)), n_test=int(len(te)),
        test_acc=acc, test_acc_lo=float(lo), test_acc_hi=float(hi),
        test_macro_f1=macro_f1(yte, P_te.argmax(1), 6),
        test_top2=top2_accuracy(P_te, yte),
        test_ece=ece(P_te, yte), test_ece_uncalibrated=ece(P_te_raw, yte),
        risk_coverage=rc_table,
        per_class=rep,
        cv=best, cv_folds=int(len(folds)),
        ablations=rows, classifiers=crow, granularity=grows,
        loso=loso, loso_acc=loso_acc,
        comparison_linear=lin_cmp,
        protocol=dict(cv="按原图分组的重复 5 折 ×3（共 15 折），配置选择只看 CV",
                      test="官方 train+val 训练、test 报数",
                      loso="留一来源批次"),
    )
    save_json(summary, os.path.join(RUNS, "summary_best.json"))
    save_json(summary, os.path.join(RUNS, "summary_final.json"))

    bars_png([r["feature_set"] for r in rows], [r["cv_f1"] for r in rows],
             os.path.join(RUNS, "final_ablation_bars.png"),
             title="ablation: macro-F1 (15-fold grouped CV)", vmin=0.15, vmax=0.55,
             highlight=int(np.argmax([r["cv_f1"] for r in rows])))
    bars_png([r["task"] for r in grows], [r["acc"] for r in grows],
             os.path.join(RUNS, "final_granularity.png"),
             title="accuracy vs question granularity (official test)",
             vmin=0.0, vmax=1.0, highlight=len(grows) - 1)

    print(f"\n已写入 runs/：final_ablation.csv、final_models.csv、final_granularity.csv、"
          f"summary_best.json、model_best.npz、*.png   （{time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
