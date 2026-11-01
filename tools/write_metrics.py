# -*- coding: utf-8 -*-
"""从 runs/summary_final.json 生成 ChromTerm/models/metrics.json。

保持旧文件的字段名（网页与 webapp 读的是这些），并追加本次新增的
temperature / granularity / diagnosis 摘要。字段只增不减、不改名。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "runs", "summary_final.json")
OUT = os.path.join(ROOT, "models", "metrics.json")
OLD = os.path.join(ROOT, "models", "metrics.json")

sm = json.load(open(SM, encoding="utf-8"))
old = {}
if os.path.exists(OLD):
    try:
        old = json.load(open(OLD, encoding="utf-8"))
    except Exception:
        old = {}

human_old = old.get("human") or {}

out = {
    "feature_set": sm.get("feature_set", "C_OD全特征"),
    "model": sm.get("base_model", "lda"),
    "linear_export": True,
    "temperature": round(float(sm.get("temperature", 1.0)), 4),
    "lda_alpha": sm.get("lda_alpha"),
    "pca": 0,
    "train_plus_val": sm.get("train_plus_val"),
    "n_test": sm.get("n_test"),
    "test_acc": sm.get("test_acc"),
    "test_macro_f1": sm.get("test_macro_f1"),
    "test_top2": sm.get("test_top2"),
    "test_ece": sm.get("test_ece"),
    "test_ece_uncalibrated": sm.get("test_ece_uncalibrated"),
    "risk_coverage": sm.get("risk_coverage"),
    # 本模型的精度在任何覆盖率上都到不了 90%（最高 70% @ 20% 覆盖），
    # 所以这一项必须是 null；旧文件里的 0.08 是上一版模型残留，不能带过来。
    "coverage_at_90": None,
    "per_class": sm.get("per_class"),
    "cv_f1": (sm.get("cv") or {}).get("cv_f1"),
    "cv_folds": sm.get("cv_folds"),
    "loso": sm.get("loso"),
    "loso_acc": sm.get("loso_acc"),
    "granularity": [
        {"task": g["task"], "n_cls": g["n_cls"], "acc": round(g["acc"], 4)}
        for g in (sm.get("granularity") or [])
    ],
    "human": human_old,
    "note": "第二版：收缩 LDA + 温度校准；温度已折进 W/b，导出的仍是线性模型。",
}

# 旧文件里可能还有 human 之外的有用字段，原样保留缺失的键
for k, v in old.items():
    out.setdefault(k, v)

json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"写入 {OUT}")
for k in ("test_acc", "test_macro_f1", "test_top2", "test_ece", "temperature",
          "cv_f1", "loso_acc"):
    print(f"  {k}: {out[k]}")
print("  risk_coverage:", out["risk_coverage"])
