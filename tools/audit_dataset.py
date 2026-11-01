# -*- coding: utf-8 -*-
"""audit_dataset.py —— 数据体检 + 标注一致性（把"标签噪声的天花板"量出来）。

为什么要做这一步
----------------
data_share 里最容易被忽略的资产是那个 `control/` 目录：**10 张图、8 个人各标一遍**。
它给出的不是训练数据，而是"人能做到多一致"——也就是任何模型在类别上的**性能上限**。
如果 8 个人对"这张图的端部构成"只有 80% 一致，那么模型 90% 的准确率就没有意义（或者
说明测试集里有泄漏）。同理，原始 annotations/ 里有 1 张图被两个人标过，可以量出
单标注引入的噪声。

检查项
------
  A. 数量与结构：文件/框/窗口/类别/切分是否自洽
  B. 重复：字节级 md5、感知哈希 dHash 近似重复；是否跨切分泄漏
  C. 分组：切分是否按原图分组（同一原图的多个窗口不能跨 train/test）
  D. 对照图一致性：类别存在性（Fleiss κ）、框 IoU、端点距离
  E. 多标注图冲突：同一张图被多人标注时类别是否一致

输出：runs/audit.json、runs/audit.md
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromterm.common import DATA, RUNS, ensure, save_json, setup_console
from chromterm.data import CLASSES, load_det_index, load_index

DS = os.path.join(DATA, "dataset")
CTRL = os.path.join(DATA, "control")


# ------------------------------------------------------------------ A 结构
def check_structure() -> dict:
    idx = load_index()
    det = load_det_index()
    n_boxes = int(sum(len(b) for b in det["boxes"]))
    rep = dict(
        n_windows=int(len(idx)), n_images=int(idx["image"].nunique()),
        n_det_images=int(len(det)), n_det_boxes=n_boxes,
        classes={c: int((idx["cls"] == c).sum()) for c in CLASSES},
        splits={s: int((idx["split"] == s).sum()) for s in ("train", "val", "test")},
        windows_match_boxes=bool(n_boxes == len(idx)),
        sources={str(k): int(v) for k, v in idx["source"].value_counts().items()},
        size_classes={str(k): int(v) for k, v in idx["size_class"].value_counts().items()},
    )
    # 每张图几个窗口
    per = idx.groupby("image").size()
    rep["windows_per_image"] = {str(k): int(v) for k, v in per.value_counts().sort_index().items()}
    return rep


# ------------------------------------------------------------------ B 重复
def _md5(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash(p: str, s: int = 8) -> np.ndarray:
    a = np.asarray(Image.open(p).convert("L").resize((s + 1, s), Image.BILINEAR), np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def check_duplicates(max_ham: int = 4) -> dict:
    idx = load_index()
    det = load_det_index()
    # 整图：md5 + dHash
    md5s = defaultdict(list)
    hs, names, splits = {}, [], {}
    for r in det.itertuples(index=False):
        md5s[_md5(r.image)].append(r.name)
        hs[r.name] = _dhash(r.image)
        names.append(r.name)
        splits[r.name] = r.split
    dup_md5 = {k: v for k, v in md5s.items() if len(v) > 1}
    H = np.stack([hs[n] for n in names])
    D = (H[:, None, :] != H[None, :, :]).sum(2)
    np.fill_diagonal(D, 99)
    pairs = [(names[i], names[j], int(D[i, j])) for i, j in np.argwhere(D <= max_ham) if i < j]
    cross = [p for p in pairs if splits[p[0]] != splits[p[1]]]
    # 窗口：md5
    wmd5 = defaultdict(list)
    for r in idx.itertuples(index=False):
        wmd5[_md5(r.path)].append((r.name, r.split, r.cls))
    wdup = {k: v for k, v in wmd5.items() if len(v) > 1}
    return dict(
        det_md5_dups=len(dup_md5), det_md5_examples=list(dup_md5.values())[:5],
        det_near_dups=len(pairs), det_near_dup_examples=[list(p) for p in cross[:8]],
        det_near_dups_cross_split=len(cross),
        window_md5_dups=len(wdup), window_dups_cross_split=sum(
            1 for v in wdup.values() if len({x[1] for x in v}) > 1),
        window_dups_class_conflict=sum(1 for v in wdup.values() if len({x[2] for x in v}) > 1),
    )


def check_grouping() -> dict:
    """同一原图的多个窗口是否可能跨切分（正确做法是按原图分组）。"""
    idx = load_index()
    by_img = idx.groupby("image")["split"].nunique()
    return dict(images_multi_split=int((by_img > 1).sum()),
                images=len(by_img),
                split_by_image=bool((by_img == 1).all()))


# ------------------------------------------------------------------ D 一致性
def _load_shapes(json_path: str):
    d = json.load(open(json_path, encoding="utf-8"))
    boxes, tips = [], []
    for sh in d.get("shapes", []):
        pts = np.asarray(sh.get("points") or [], np.float32)
        if sh.get("shape_type") in ("rectangle", "rotation") and pts.shape[0] >= 3:
            boxes.append((sh.get("label"), pts))
        elif sh.get("shape_type") == "point" and pts.shape[0] >= 1:
            tips.append(pts[0])
    return boxes, tips


def _box_poly_iou(p: np.ndarray, q: np.ndarray) -> float:
    """两个凸四边形的 IoU（用 Sutherland–Hodgman 裁剪 + 鞋带公式）。"""
    def clip(poly, a, b):
        out = []
        n = len(poly)
        for i in range(n):
            cur, nxt = poly[i], poly[(i + 1) % n]
            sc = (b[0] - a[0]) * (cur[1] - a[1]) - (b[1] - a[1]) * (cur[0] - a[0])
            sn = (b[0] - a[0]) * (nxt[1] - a[1]) - (b[1] - a[1]) * (nxt[0] - a[0])
            if sc >= 0:
                out.append(cur)
            if (sc >= 0) != (sn >= 0):
                t = sc / (sc - sn + 1e-12)
                out.append(cur + t * (nxt - cur))
        return out

    def area(poly):
        if len(poly) < 3:
            return 0.0
        s = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5

    # 保证同向
    def signed(poly):
        s = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            s += x1 * y2 - x2 * y1
        return s
    P, Q = p.copy(), q.copy()
    if signed(P) < 0:
        P = P[::-1]
    if signed(Q) < 0:
        Q = Q[::-1]
    inter = P
    for i in range(len(Q)):
        a, b = Q[i], Q[(i + 1) % len(Q)]
        inter = clip(inter, a, b)
        if len(inter) == 0:
            return 0.0
    ai = area(inter)
    return float(ai / (area(P) + area(Q) - ai + 1e-9))


def fleiss_kappa(M: np.ndarray) -> float:
    """Fleiss' κ。M 为 items × categories 的计数矩阵。"""
    M = np.asarray(M, np.float64)
    n_items = M.shape[0]
    if n_items < 2:
        return float("nan")
    n_raters = M.sum(1)
    p_j = M.sum(0) / M.sum()
    P_i = (np.sum(M * (M - 1), axis=1)) / (n_raters * (n_raters - 1) + 1e-12)
    P_bar, P_e = P_i.mean(), np.sum(p_j ** 2)
    return float((P_bar - P_e) / (1 - P_e + 1e-12))


def check_control() -> dict:
    """对照图：8 位标注者的一致性 = 任务的人为上限。"""
    if not os.path.isdir(os.path.join(CTRL, "annotations")):
        return dict(available=False)
    persons = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CTRL, "annotations", "p*")))
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(CTRL, "images", "*"))[0:1]) and \
        sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(CTRL, "images")))

    presence = {(st, c): np.zeros(len(persons)) for st in stems for c in CLASSES}
    ious, tip_d = [], []
    per_image = {}
    for j, per in enumerate(persons):
        for st in stems:
            p = os.path.join(CTRL, "annotations", per, st + ".json")
            if not os.path.exists(p):
                continue
            boxes, tips = _load_shapes(p)
            labels = [b[0] for b in boxes]
            for c in CLASSES:
                presence[(st, c)][j] = 1.0 if c in labels else 0.0
            per_image.setdefault(st, []).append((labels, boxes, tips))
    # 同一张图内两两比
    for st, recs in per_image.items():
        for i in range(len(recs)):
            for k in range(i + 1, len(recs)):
                li, bi, ti = recs[i]
                lj, bj, tj = recs[k]
                # 框：按中心距离贪心匹配
                ci = np.array([b[1].mean(0) for b in bi]) if bi else np.zeros((0, 2))
                cj = np.array([b[1].mean(0) for b in bj]) if bj else np.zeros((0, 2))
                used = set()
                for a in range(len(bi)):
                    best, bj_idx = None, None
                    for b in range(len(bj)):
                        if b in used:
                            continue
                        d = np.linalg.norm(ci[a] - cj[b])
                        if best is None or d < best:
                            best, bj_idx = d, b
                    if bj_idx is not None and best < 80:
                        used.add(bj_idx)
                        ious.append(_box_poly_iou(bi[a][1], bj[bj_idx][1]))
                # 端点：最近距离
                for t in ti:
                    if len(tj):
                        tip_d.append(float(np.min(np.linalg.norm(tj - t, axis=1))))
    # 类别存在性 κ（10 图 × 6 类 = 60 个 item，8 位标注者）
    M = np.array([[presence[(st, c)].sum(), len(persons) - presence[(st, c)].sum()]
                  for st in stems for c in CLASSES])
    kappa_presence = fleiss_kappa(M)
    # "该图有哪些端部"的类别集合一致率（多数票 vs 每人）
    maj_agree = []
    for st in stems:
        cnt = Counter()
        for rec in per_image.get(st, []):
            for lab in set(rec[0]):
                cnt[lab] += 1
        if not cnt:
            continue
        maj = {c for c, v in cnt.items() if v >= len(per_image[st]) / 2}
        for rec in per_image[st]:
            maj_agree.append(float(set(rec[0]) == maj))
    n_boxes = [len(rec[0]) for recs in per_image.values() for rec in recs]
    return dict(available=True, persons=persons, n_images=len(stems),
                annotators=len(persons),
                kappa_class_presence=kappa_presence,
                exact_set_agreement=float(np.mean(maj_agree)) if maj_agree else None,
                box_iou_median=float(np.median(ious)) if ious else None,
                box_iou_p10=float(np.percentile(ious, 10)) if ious else None,
                n_box_pairs=len(ious),
                tip_dist_median_px=float(np.median(tip_d)) if tip_d else None,
                tip_dist_p90_px=float(np.percentile(tip_d, 90)) if tip_d else None,
                boxes_per_annotator=dict(Counter(n_boxes)))


def check_multi_annotation() -> dict:
    """训练集里被多人标注过的图：类别是否冲突（= 单标注噪声）。"""
    rows = []
    for per_dir in sorted(glob.glob(os.path.join(DATA, "annotations", "p*"))) + \
            glob.glob(os.path.join(DATA, "annotations", "own")):
        for p in glob.glob(os.path.join(per_dir, "*.json")):
            st = os.path.splitext(os.path.basename(p))[0]
            boxes, _ = _load_shapes(p)
            rows.append((st, os.path.basename(per_dir), tuple(sorted(b[0] for b in boxes))))
    df = pd.DataFrame(rows, columns=["stem", "person", "labels"])
    cnt = df.groupby("stem").size()
    multi = cnt[cnt > 1].index
    agree = []
    for st in multi:
        sets = df[df.stem == st]["labels"].tolist()
        agree.append(len(set(sets)) == 1)
    return dict(n_annotation_files=int(len(df)), n_images=int(df.stem.nunique()),
                images_multi_annotated=int(len(multi)),
                multi_full_agreement=float(np.mean(agree)) if agree else None,
                examples=[[st, df[df.stem == st]["person"].tolist(),
                           df[df.stem == st]["labels"].tolist()] for st in list(multi)[:6]])


# ------------------------------------------------------------------ 报告
def write_markdown(rep: dict, path: str) -> None:
    L = []
    A = rep["structure"]
    L.append("# data_share 数据体检报告\n")
    L.append("> 由 `src/audit_dataset.py` 自动生成。数字均可复现。\n")
    L.append("## A. 结构\n")
    L.append(f"- 端部窗口 **{A['n_windows']}** 个，来自 **{A['n_images']}** 张原图")
    L.append(f"- 检测数据集 **{A['n_det_images']}** 张图、**{A['n_det_boxes']}** 个框"
             f"（与窗口数一致：{A['windows_match_boxes']}）")
    L.append(f"- 类别：{A['classes']}")
    L.append(f"- 切分：{A['splits']}")
    L.append(f"- 来源批次：{A['sources']}")
    L.append(f"- 每图窗口数：{A['windows_per_image']}\n")
    B = rep["duplicates"]
    L.append("## B. 重复与泄漏\n")
    L.append(f"- 整图 md5 重复组：**{B['det_md5_dups']}**")
    L.append(f"- 整图 dHash 近似重复对（≤4 bit）：**{B['det_near_dups']}**，"
             f"其中跨切分 **{B['det_near_dups_cross_split']}**")
    L.append(f"- 窗口 md5 重复组：**{B['window_md5_dups']}**（跨切分 {B['window_dups_cross_split']}，"
             f"跨类冲突 {B['window_dups_class_conflict']}）\n")
    C = rep["grouping"]
    L.append("## C. 分组\n")
    L.append(f"- 按原图分组切分：**{C['split_by_image']}**"
             f"（跨切分的原图数 {C['images_multi_split']}）\n")
    D = rep["control"]
    L.append("## D. 对照图一致性（人为上限）\n")
    if D.get("available"):
        L.append(f"- {D['n_images']} 张图 × {D['annotators']} 位标注者")
        L.append(f"- 类别存在性 Fleiss κ = **{D['kappa_class_presence']:.3f}**")
        L.append(f"- 与多数票完全一致的标注者比例 = **{D['exact_set_agreement']:.2f}**")
        L.append(f"- 框 IoU 中位 **{D['box_iou_median']:.3f}**（p10 {D['box_iou_p10']:.3f}，"
                 f"{D['n_box_pairs']} 对）")
        L.append(f"- 端点距离中位 **{D['tip_dist_median_px']:.1f} px**"
                 f"（p90 {D['tip_dist_p90_px']:.1f} px）")
        L.append(f"- 每人标注框数分布：{D['boxes_per_annotator']}\n")
    E = rep["multi_annotation"]
    L.append("## E. 多标注图冲突\n")
    L.append(f"- 标注文件 {E['n_annotation_files']} 个，覆盖 {E['n_images']} 张图；"
             f"被多人标注 {E['images_multi_annotated']} 张")
    if E.get("multi_full_agreement") is not None:
        L.append(f"- 这些图里“多人标注完全一致”的比例 = **{E['multi_full_agreement']:.2f}**\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    setup_console()
    ensure(RUNS)
    print("A. 结构 ...", flush=True)
    rep = dict(structure=check_structure())
    print("B. 重复 ...", flush=True)
    rep["duplicates"] = check_duplicates()
    print("C. 分组 ...", flush=True)
    rep["grouping"] = check_grouping()
    print("D. 对照图一致性 ...", flush=True)
    rep["control"] = check_control()
    print("E. 多标注冲突 ...", flush=True)
    rep["multi_annotation"] = check_multi_annotation()

    save_json(rep, os.path.join(RUNS, "audit.json"))
    write_markdown(rep, os.path.join(RUNS, "audit.md"))
    print(json.dumps(rep, ensure_ascii=False, indent=2)[:3000])
    print("\n写入 runs/audit.json、runs/audit.md")


if __name__ == "__main__":
    main()
