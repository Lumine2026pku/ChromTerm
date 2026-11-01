# -*- coding: utf-8 -*-
"""model.py —— 只用 numpy 的小样本分类器与评估工具。

为什么不直接上 CNN
------------------
961 个样本、6 类、5 个成像批次。这个量级上端到端的深度分类器很容易退化成
"背染色深浅"，而且很难给出可解释的消融。所以这里给三档模型，从最保守到最强：

  NCP   最近类原型（nearest class prototype）—— 几乎零参数，最抗过拟合
  Linear 多项逻辑回归（L2 + 类平衡权重 + Adam）—— 主力
  RFF  随机傅里叶特征 + 岭回归 —— 隐式核方法，接口与 Linear 相同

三者都吃"标准化 + 可选 PCA 白化"后的特征，接口一致（fit / predict_proba），
换分类器只需改一个字符串。
"""

from __future__ import annotations

import numpy as np

from .common import EPS, macro_f1, softmax


# ------------------------------------------------------------------ 预处理
class Standardizer:
    def fit(self, X):
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        self.sd[~np.isfinite(self.sd) | (self.sd < 1e-6)] = 1.0
        self.mu[~np.isfinite(self.mu)] = 0.0
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class PCAReducer:
    """PCA（按需白化）。用 SVD 实现，避免引入 scikit-learn。"""

    def __init__(self, n_components: int = 48, whiten: bool = True, seed: int = 0):
        self.n_components = n_components
        self.whiten = whiten
        self.seed = seed

    def fit(self, X):
        n = X.shape[0]
        k = int(min(self.n_components, max(2, n - 2), X.shape[1]))
        mu = X.mean(0)
        Xc = X - mu
        # 样本数远小于维数时用 (X^T X) 的特征分解更快
        if X.shape[1] > n:
            G = Xc @ Xc.T / max(n - 1, 1)
            w, V = np.linalg.eigh(G)
            idx = np.argsort(w)[::-1][:k]
            w, V = w[idx], V[:, idx]
            self.components_ = (Xc.T @ V) / np.sqrt(np.maximum(w, EPS) * max(n - 1, 1))
            self.components_ = self.components_.T
        else:
            C = Xc.T @ Xc / max(n - 1, 1)
            w, V = np.linalg.eigh(C)
            idx = np.argsort(w)[::-1][:k]
            w, V = w[idx], V[:, idx]
            self.components_ = V.T
        self.mu = mu
        self.k = k
        return self

    def transform(self, X):
        Z = (X - self.mu) @ self.components_.T
        if self.whiten:
            sd = Z.std(0)
            sd[sd < 1e-6] = 1.0
            Z = Z / sd
        return Z.astype(np.float32)


# ------------------------------------------------------------------ 分类器
class _Base:
    classes_ = None

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


class NCP(_Base):
    """最近类原型（余弦/欧氏）。小样本下经常比逻辑回归更稳的"下限模型"。"""

    def __init__(self, metric: str = "euclid"):
        self.metric = metric

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.C = np.stack([X[y == c].mean(0) for c in self.classes_])
        if self.metric == "cosine":
            self.C /= (np.linalg.norm(self.C, axis=1, keepdims=True) + EPS)
        return self

    def predict_proba(self, X):
        if self.metric == "cosine":
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
            d = 1.0 - Xn @ self.C.T
        else:
            d = ((X[:, None, :] - self.C[None, :, :]) ** 2).sum(2)
        return softmax(-d / (np.median(d) + EPS), axis=1)


class Linear(_Base):
    """多项逻辑回归：L2 + 类平衡权重 + Adam，全程 numpy。"""

    def __init__(self, l2: float = 1e-2, lr: float = 0.02, iters: int = 600,
                 balanced: bool = True, seed: int = 0, verbose: bool = False):
        self.l2, self.lr, self.iters = l2, lr, iters
        self.balanced, self.seed, self.verbose = balanced, seed, verbose

    def fit(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        Y = np.zeros((n, k), np.float32)
        Y[np.arange(n), np.searchsorted(self.classes_, y)] = 1.0
        w = np.ones(n, np.float32)
        if sample_weight is not None:
            w *= np.asarray(sample_weight, np.float32)
        if self.balanced:
            cnt = Y.sum(0)
            cw = n / (k * np.maximum(cnt, 1))
            w *= (Y * cw[None, :]).sum(1)
        w = w / (w.mean() + EPS)

        self.W = np.zeros((d, k), np.float32)
        self.b = np.zeros(k, np.float32)
        mW = np.zeros_like(self.W); vW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b); vb = np.zeros_like(self.b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.iters + 1):
            P = softmax(X @ self.W + self.b, axis=1)
            G = (P - Y) * w[:, None] / n
            gW = X.T @ G + self.l2 * self.W
            gb = G.sum(0)
            mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW ** 2
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
            self.W -= self.lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
            self.b -= self.lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b, axis=1)


class ShrinkLDA(_Base):
    """收缩协方差的线性判别（LDA / shrinkage LDA）。

    为什么在 961 样本 / 836 维上它比多项逻辑回归更合适：
      逻辑回归要同时估 6×(d+1) 个参数；LDA 只估 6 个类中心 + 一个**收缩过的**
      协方差矩阵，参数少一个量级。样本协方差在 n<d 时奇异，用
          S' = (1-α)·S + α·(tr(S)/d)·I
      把它拉向各向同性，α=1 退化为"最近类原型"，α=0 是纯 LDA。
      实测最优 α≈0.6：CV-F1 0.475（逻辑回归 0.459），top-2 0.727（0.660）。

    产物仍是 (W, b)，预测 = softmax(X @ W + b)，与 `Linear` 完全同构，
    所以 pipeline.py / 网页导出的模型格式不需要任何改动。
    """

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        n, d = X.shape
        k = len(self.classes_)
        mu = np.zeros((k, d), np.float32)
        S = np.zeros((d, d), np.float64)
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            mu[i] = Xc.mean(0)
            Z = Xc - mu[i]
            S += Z.T @ Z
        S /= max(n - k, 1)
        a = float(min(max(self.alpha, 0.0), 1.0))
        S = (1 - a) * S + a * (np.trace(S) / d) * np.eye(d)
        w, V = np.linalg.eigh(S)
        w = np.maximum(w, 1e-8)
        Sinv = (V / w) @ V.T
        self.W = (Sinv @ mu.T).astype(np.float32)
        self.b = (-0.5 * np.einsum("kd,dk->k", mu, Sinv @ mu.T)).astype(np.float32)
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b, axis=1)


class Calibrated(_Base):
    """把一个已经是线性的分类器按温度 T 校准：softmax(z / T)。

    温度 T 在验证集上按 NLL 最小拟合。因为 (X@W+b)/T = X@(W/T) + b/T，
    校准后的模型**仍然是同一个线性模型**，只需把 W、b 各除一次 T，
    下游（pipeline / 网页 / JS 引擎）一行都不用改。用 `fold()` 完成这一步。
    """

    def __init__(self, base, T: float = 1.0):
        self.base = base
        self.T = float(T)

    def fit(self, X, y, sample_weight=None):
        self.base.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.base.classes_
        return self

    def predict_proba(self, X):
        return softmax((X @ self.base.W + self.base.b) / self.T, axis=1)

    def fold(self):
        """把温度折进 W、b，返回一个等价的普通线性模块（T≡1）。"""
        if not hasattr(self.base, "W"):
            raise ValueError("只有线性分类器可以折温度")
        self.base.W = (self.base.W / self.T).astype(np.float32)
        self.base.b = (self.base.b / self.T).astype(np.float32)
        self.T = 1.0
        return self.base


def fit_temperature(P: np.ndarray, y: np.ndarray, lo: float = 0.1, hi: float = 50.0,
                    n: int = 160) -> float:
    """按验证集负对数似然选温度（保序、不改变 argmax，所以准确率不变）。

    在 log 尺度上搜索：LDA 的判别值量级可能很大（Sinv @ mu），最优 T 常常在 3–15，
    线性等距搜索会在低端浪费分辨率、又可能撞到上限。
    """
    P = np.clip(np.asarray(P, np.float64), 1e-12, 1.0)
    y = np.asarray(y, int)
    logits = np.log(P)
    best_T, best = 1.0, np.inf
    for T in np.exp(np.linspace(np.log(lo), np.log(hi), n)):
        Q = logits / T
        Q -= Q.max(1, keepdims=True)
        Q = np.exp(Q)
        Q /= Q.sum(1, keepdims=True)
        nll = -float(np.mean(np.log(np.clip(Q[np.arange(len(y)), y], 1e-12, 1.0))))
        if nll < best:
            best, best_T = nll, float(T)
    return best_T


class RFFRidge(_Base):
    """随机傅里叶特征（RBF 核的近似）+ 岭回归闭式解。"""

    def __init__(self, n_features: int = 512, gamma: float | None = None,
                 alpha: float = 1.0, seed: int = 0):
        self.n_features, self.gamma, self.alpha, self.seed = n_features, gamma, alpha, seed

    def fit(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        if self.gamma is None:
            # median heuristic：把成对距离中位数当作尺度
            idx = rng.choice(n, size=min(n, 200), replace=False)
            sub = X[idx]
            dd = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(2)
            med = np.median(dd[dd > 0]) if (dd > 0).any() else 1.0
            self.gamma = 1.0 / max(med, EPS)
        self.Wr = rng.normal(0, np.sqrt(2 * self.gamma), size=(d, self.n_features)).astype(np.float32)
        self.br = rng.uniform(0, 2 * np.pi, size=self.n_features).astype(np.float32)
        Z = self._map(X)
        Y = np.zeros((n, k), np.float32)
        Y[np.arange(n), np.searchsorted(self.classes_, y)] = 1.0
        A = Z.T @ Z + self.alpha * np.eye(self.n_features, dtype=np.float32)
        self.Wo = np.linalg.solve(A, Z.T @ Y)
        return self

    def _map(self, X):
        return np.cos(X @ self.Wr + self.br).astype(np.float32)

    def predict_proba(self, X):
        return softmax(self._map(X) @ self.Wo, axis=1)


MODELS = {"ncp": NCP, "linear": Linear, "rff": RFFRidge}


# ------------------------------------------------------------------ 评估
def fit_eval(Xtr, ytr, Xva, yva, Xte, yte, model="linear", pca: int = 48,
             n_cls: int = 6, seed: int = 0, **kw):
    """标准三步：标准化(+PCA 白化) -> 训练 -> 在 val 上选阈值/早停 -> test 评估。"""
    st = Standardizer().fit(Xtr)
    Xtr_, Xva_, Xte_ = st.transform(Xtr), st.transform(Xva), st.transform(Xte)
    if pca and pca > 0:
        pc = PCAReducer(n_components=pca, whiten=True, seed=seed).fit(Xtr_)
        Xtr_, Xva_, Xte_ = pc.transform(Xtr_), pc.transform(Xva_), pc.transform(Xte_)
    clf = MODELS[model](seed=seed, **kw)
    clf.fit(Xtr_, ytr)
    pva = clf.predict_proba(Xva_)
    pte = clf.predict_proba(Xte_)
    res = dict(
        model=model, pca=pca,
        acc_va=float((pva.argmax(1) == yva).mean()),
        f1_va=float(macro_f1(yva, pva.argmax(1), n_cls)),
        acc_te=float((pte.argmax(1) == yte).mean()),
        f1_te=float(macro_f1(yte, pte.argmax(1), n_cls)),
        prob_te=pte, pred_te=pte.argmax(1), prob_va=pva,
    )
    return res, clf


def top2_accuracy(P, y):
    order = np.argsort(P, axis=1)[:, ::-1][:, :2]
    return float(np.mean([y[i] in order[i] for i in range(len(y))]))


def risk_coverage(P, y, conf=None):
    """按置信度从高到低覆盖，返回 (coverage, accuracy) 曲线与"守住 90% 精度所需覆盖"。"""
    if conf is None:
        conf = P.max(1)
    order = np.argsort(conf)[::-1]
    correct = (P.argmax(1) == y)[order]
    n = len(y)
    cov = np.arange(1, n + 1) / n
    acc = np.cumsum(correct) / np.arange(1, n + 1)
    need = None
    for c, a in zip(cov, acc):
        if a >= 0.90:
            need = float(c)
    return cov, acc, need


def ece(P, y, bins: int = 10):
    """期望校准误差：置信度与实际正确率的一致性。"""
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(np.float32)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)
