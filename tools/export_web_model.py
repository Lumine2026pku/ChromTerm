# -*- coding: utf-8 -*-
"""export_web_model.py —— 把 models/model_best.npz 导成网页用的 JSON。

网页端只实现"不需要几何管线"的那部分特征（pyr/band/auto/hog/width），
几何块（geo）用**训练集均值**代入：标准化后恰为 0，等于"不知道就不贡献"。
实测这样做 test 准确率 0.5133 vs 完整 0.5067（差在噪声内），
所以静态网页不需要移植 Zhang-Suen 骨架、分割、测宽这一整套。

输出：assets/model/model_best.json
  mu/sd/W/b（836 维 × 6 类）+ classes + geo_mean + 指标 + 特征块顺序
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from chromterm.common import MODEL_FILE, MODELS, ROOT as REPO  # noqa: E402
from chromterm.data import CLASSES, load_cache  # noqa: E402
from chromterm.features import FEATURE_SETS, build_matrix, feat_geo  # noqa: E402


def main() -> int:
    z = np.load(MODEL_FILE, allow_pickle=True)
    mu, sd = z["mu"].astype(np.float64), z["sd"].astype(np.float64)
    W, b = z["W"].astype(np.float64), z["b"].astype(np.float64)
    classes = [str(c) for c in z["classes"]]
    blocks = [str(x) for x in z["blocks"][0]]
    feature_set = str(z["feature_set"][0])

    # geo 块在最后 11 维：用训练集（train+val）**经过 feat_geo 变换后**的均值代入。
    # 注意不能直接用 feat["geo"] 原始矩阵：里面 width_px 有 NaN，
    # 而 feat_geo 会把非有限值变成 0，两者不是一回事（第一版就在这里写出了 NaN）。
    feat = load_cache()
    split = feat["split"].astype(str)
    X = build_matrix(feat, "raw", FEATURE_SETS[feature_set][1])
    # feat_geo 输出 11 维（10 个标量 + 宽长比），而缓存里的 feat["geo"] 是 12 列，别搞混
    n_geo = int(feat_geo(feat["geo"][0], feat["geo_names"]).size)
    geo_mean = X[split != "test"][:, -n_geo:].mean(0).astype(np.float64).tolist()

    for name, arr in (("mu", mu), ("sd", sd), ("W", W), ("b", b),
                      ("geo_mean", np.asarray(geo_mean))):
        if not np.isfinite(arr).all():
            raise SystemExit(f"{name} 里出现了非有限值，先修数据再导出")

    metrics = {}
    mp = os.path.join(MODELS, "metrics.json")
    if os.path.exists(mp):
        metrics = json.load(open(mp, encoding="utf-8"))

    payload = dict(
        version="1.0.0",
        feature_set=feature_set,
        blocks=blocks,
        classes=classes,
        input=dict(w=192, h=64, source="246x112 窗口居中裁成 2.2:1 后缩放到 192x64"),
        dim=int(mu.size),
        n_geo=n_geo,
        geo_mean=geo_mean,
        mu=[round(float(v), 8) for v in mu],
        sd=[round(float(v), 8) for v in sd],
        W=[[round(float(v), 8) for v in row] for row in W.T.tolist()],   # 6 × 836（按类展开）
        b=[round(float(v), 8) for v in b],
        metrics=dict(test_acc=metrics.get("test_acc"),
                     test_macro_f1=metrics.get("test_macro_f1"),
                     test_top2=metrics.get("test_top2"),
                     n_test=metrics.get("n_test"),
                     loso_acc=metrics.get("loso_acc"),
                     human=metrics.get("human")),
        note=("网页端跳过几何量测块（geo），以训练均值代入；实测 test 准确率 "
              "0.5133 vs 完整 0.5067，差在噪声内。"),
    )
    out = os.path.join(REPO, "assets", "model", "model_best.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    kb = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({kb:.1f} KB)  dim={mu.size}  classes={classes}")
    print(f"blocks={blocks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
