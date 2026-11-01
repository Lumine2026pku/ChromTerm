# -*- coding: utf-8 -*-
"""common.py —— 公共基础：路径、读图、插值、形态学、连通域、指标、画图。

只依赖 numpy + Pillow（本环境的运行时不带 scipy / skimage / cv2 / torch），
所以所有算子都在这里用 numpy 重新实现，保持"零重依赖"。

设计原则：
  * 所有函数只接受 / 返回 numpy.float32 数组，灰度统一 0..255（或 OD 的 0..~1.5）；
  * 不修改输入数组；
  * 每个算子都写清"和 cv2/skimage 里哪个函数等价"，便于日后替换。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EPS = 1e-6

# ------------------------------------------------------------------ 路径
# 仓库布局：
#   <repo>/chromterm/*.py         代码
#   <repo>/models/model_best.npz  权重（随仓库分发，30 KB）
#   <repo>/index.html + assets/   前端（GitHub Pages 入口就在根目录）
#   <repo>/examples/              示例图
#   <repo>/runs/  <repo>/_cache/  运行产物（.gitignore，不进仓库）
#   <repo>/data_share/            **不随仓库分发**的数据集；训练/评估才需要
HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/chromterm
ROOT = os.path.dirname(HERE)                               # <repo>
WEB = ROOT            # index.html 在仓库根目录（GitHub Pages 要求）
MODELS = os.path.join(ROOT, "models")
EXAMPLES = os.path.join(ROOT, "examples")
DOCS = os.path.join(ROOT, "docs")
RUNS = os.path.join(ROOT, "runs")
CACHE = os.path.join(ROOT, "_cache")
# 数据集不进仓库：默认 <repo>/data_share，也可用环境变量 CHROMTERM_DATA 指到别处。
# 另外三个回退位置是为了"仓库被放在课题组的共享目录里"这种常见情形——那时
# dataset 可能在仓库的上一级或上两级的 data_share / 其他组员工作/data_share 里。
def _find_data() -> str:
    parent = os.path.dirname(ROOT)
    cands = [os.environ.get("CHROMTERM_DATA") or "",
             os.environ.get("CHROM_DATA") or "",
             os.path.join(ROOT, "data_share"),
             os.path.join(ROOT, "其他组员工作", "data_share"),
             os.path.join(parent, "data_share"),
             os.path.join(parent, "其他组员工作", "data_share")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "dataset")):
            return c
    return os.path.join(ROOT, "data_share")


DATA = _find_data()
MODEL_FILE = os.path.join(MODELS, "model_best.npz")
METRICS_FILE = os.path.join(MODELS, "metrics.json")


def ensure(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def setup_console() -> None:
    """Windows 控制台默认 GBK，打印中文会炸；统一改成 utf-8。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ------------------------------------------------------------------ 读图
def load_rgb(path: str) -> np.ndarray:
    """读成 HxWx3 float32（0..255）。16 位图按比例压回 0..255。"""
    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I", "F"):
            a = np.asarray(im, dtype=np.float64)
            hi = float(a.max()) or 1.0
            a = a * (255.0 / hi)
            return np.repeat(a[:, :, None], 3, axis=2).astype(np.float32)
        return np.asarray(im.convert("RGB"), dtype=np.float32)


def save_rgb(arr: np.ndarray, path: str) -> None:
    a = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


# ------------------------------------------------------------------ 重采样
def resize_bilinear(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """等价 cv2.resize(..., INTER_LINEAR)，但用 Pillow 的 C 实现，快且无依赖。"""
    if img.ndim == 2:
        im = Image.fromarray(img.astype(np.float32), mode="F")
        out = im.resize((int(out_w), int(out_h)), Image.BILINEAR)
        return np.asarray(out, dtype=np.float32)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")
    out = im.resize((int(out_w), int(out_h)), Image.BILINEAR)
    return np.asarray(out, dtype=np.float32)


def _bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """在 (xs, ys) 浮点坐标上做双线性采样，越界按边缘复制。

    这是整套旋转 / 反向映射的唯一底层算子：
    cv2.warpAffine(..., flags=INTER_LINEAR | BORDER_REPLICATE) 的等价物。
    """
    h, w = img.shape[:2]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (xs - x0)[..., None] if img.ndim == 3 else (xs - x0)
    fy = (ys - y0)[..., None] if img.ndim == 3 else (ys - y0)

    def g(yy, xx):
        return img[yy, xx]

    top = g(y0, x0) * (1 - fx) + g(y0, x1) * fx
    bot = g(y1, x0) * (1 - fx) + g(y1, x1) * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def rotate_about(img: np.ndarray, angle_deg: float, center=None, out_shape=None,
                 scale: float = 1.0) -> np.ndarray:
    """绕 center 旋转 angle_deg（+ 使图像内容逆时针转，与 PIL rotate 同向）。

    实测约定（见 _cache/probe_rot.py）：估计出的臂轴角 φ 对应的"转正"操作是
    rotate_about(img, -φ)。即 rotate_about(·, θ) 把内容转 +θ。
    """
    if out_shape is None:
        out_shape = img.shape[:2]
    oh, ow = out_shape
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
    cx, cy = center
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # 输出网格 -> 输入坐标（逆向映射）
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float32)
    dx = (xx - (ow - 1) / 2.0) / scale
    dy = (yy - (oh - 1) / 2.0) / scale
    sx = ca * dx + sa * dy + cx
    sy = -sa * dx + ca * dy + cy
    return _bilinear_sample(img, sx, sy)


def rotate_expand(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """旋转换算到"刚好装下整幅图"的更大画布上，避免四个角被裁掉。"""
    h, w = img.shape[:2]
    a = np.deg2rad(abs(angle_deg))
    ow = int(np.ceil(w * np.cos(a) + h * np.sin(a)))
    oh = int(np.ceil(w * np.sin(a) + h * np.cos(a)))
    return rotate_about(img, angle_deg, center=((w - 1) / 2.0, (h - 1) / 2.0),
                        out_shape=(oh, ow))


# ------------------------------------------------------------------ 1D 滤波
def gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 1e-6:
        return np.array([1.0], dtype=np.float32)
    r = max(1, int(round(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def conv1d_reflect(img: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """沿指定轴做 1D 卷积，边界用 reflect（cv2.BORDER_REFLECT101 的近似）。"""
    if kernel.size == 1 or img.shape[axis] == 1:
        return img.astype(np.float32, copy=True)
    r = kernel.size // 2
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    ip = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    n = img.shape[axis]
    for i, kv in enumerate(kernel):
        sl = [slice(None)] * img.ndim
        sl[axis] = slice(i, i + n)
        out += kv * ip[tuple(sl)]
    return out


def gaussian_blur(img: np.ndarray, sigma: float, sigma_y: float | None = None) -> np.ndarray:
    if sigma_y is None:
        sigma_y = sigma
    out = conv1d_reflect(img.astype(np.float32), gaussian_kernel1d(float(sigma)), 1)
    return conv1d_reflect(out, gaussian_kernel1d(float(sigma_y)), 0)


def gaussian_blur_aniso(img: np.ndarray, sigma_x: float, sigma_y: float,
                        angle_deg: float) -> np.ndarray:
    """各向异性高斯：先按 angle_deg 转正，沿 x 用 sigma_x、沿 y 用 sigma_y，再转回去。

    等价于"各向异性扩散 / 方向自适应平滑"，但实现上是精确的旋转高斯。
    angle_deg 为长轴方向（沿 x 轴正方向 = 0°。
    """
    h, w = img.shape[:2]
    rotd = rotate_about(img, angle_deg, out_shape=(h, w), scale=1.0)
    b = gaussian_blur(rotd, sigma_x, sigma_y)
    return rotate_about(b, -angle_deg, out_shape=(h, w), scale=1.0)


# ------------------------------------------------------------------ 形态学
def _sliding_extreme_1d(a: np.ndarray, r: int, mode: str, axis: int) -> np.ndarray:
    """半径 r 的 1D 滑动 max/min，O(N)（van Herk / Gil-Werman 前缀-后缀分块算法）。

    窗口宽 k = 2r+1，两端各补 k-1 个边界值后按 k 分块：
        out[i] = max(块内后缀极值[i+r], 块内前缀极值[i+3r])
    与逐点暴力结果逐元素相同（见 _cache/selftest_common.py）。
    """
    if r <= 0:
        return a.astype(np.float32, copy=True)
    a = np.moveaxis(a.astype(np.float32), axis, 0)
    n = a.shape[0]
    k = 2 * r + 1
    rest = a.shape[1:]

    pad = [(0, 0)] * a.ndim
    pad[0] = (k - 1, k - 1)
    ap = np.pad(a, pad, mode="edge")
    rem = (-ap.shape[0]) % k
    if rem:
        pad2 = [(0, 0)] * ap.ndim
        pad2[0] = (0, rem)
        ap = np.pad(ap, pad2, mode="edge")

    nb = ap.shape[0] // k
    blocks = ap.reshape((nb, k) + rest)
    fn = np.maximum if mode == "max" else np.minimum
    pref = fn.accumulate(blocks, axis=1).reshape((nb * k,) + rest)
    suff = fn.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].reshape((nb * k,) + rest)

    idx = np.arange(n, dtype=np.int64)
    out = fn(suff[idx + r], pref[idx + 3 * r])
    return np.moveaxis(out, 0, axis)


def max_filter(img: np.ndarray, radius: int) -> np.ndarray:
    """方形结构元膨胀（可分：先横行再竖列）。"""
    out = _sliding_extreme_1d(img, radius, "max", axis=1)
    return _sliding_extreme_1d(out, radius, "max", axis=0)


def min_filter(img: np.ndarray, radius: int) -> np.ndarray:
    out = _sliding_extreme_1d(img, radius, "min", axis=1)
    return _sliding_extreme_1d(out, radius, "min", axis=0)


def morph_close(img: np.ndarray, radius: int) -> np.ndarray:
    """闭运算 = 先膨胀后腐蚀；用来估计"没有染色体时背景该是什么颜色"。"""
    return min_filter(max_filter(img, radius), radius)


def morph_open(img: np.ndarray, radius: int) -> np.ndarray:
    return max_filter(min_filter(img, radius), radius)


def label_components(mask: np.ndarray, connectivity: int = 8):
    """连通域标记：行程编码 + 并查集（scanline CCL）。

    纯 Python 逐像素 BFS 在 1600×1200 上要几十秒；按"行行程"合并后，
    行程数通常只有几百到几千，1600×1200 只需 ~30 ms，且结果是精确的。

    返回 (labels int32, sizes dict)。等价 cv2.connectedComponentsWithStats。
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    if not mask.any():
        return labels, {}

    # 1) 每行行程 [start, end)
    padded = np.zeros((h, w + 2), dtype=bool)
    padded[:, 1:-1] = mask
    d = np.diff(padded.astype(np.int8), axis=1)
    s_list, e_list = [], []
    for r in range(h):
        s = np.nonzero(d[r] == 1)[0]
        if s.size == 0:
            s_list.append(np.zeros(0, np.int32))
            e_list.append(np.zeros(0, np.int32))
        else:
            e = np.nonzero(d[r] == -1)[0]
            s_list.append(s.astype(np.int32))
            e_list.append(e.astype(np.int32))

    run_row, run_s, run_e = [], [], []
    for r in range(h):
        for k in range(len(s_list[r])):
            run_row.append(r)
            run_s.append(int(s_list[r][k]) - 1)
            run_e.append(int(e_list[r][k]) - 1)
    n = len(run_row)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 2) 相邻两行的行程重叠 -> 合并
    row_start = np.cumsum([0] + [len(s) for s in s_list])
    for r in range(h - 1):
        a0, a1 = row_start[r], row_start[r + 1]
        b0, b1 = row_start[r + 1], row_start[r + 2]
        if a0 == a1 or b0 == b1:
            continue
        sa = np.asarray(run_s[a0:a1]); ea = np.asarray(run_e[a0:a1])
        sb = np.asarray(run_s[b0:b1]); eb = np.asarray(run_e[b0:b1])
        touch = 1 if connectivity == 8 else 0
        for i in range(len(sa)):
            # 下一行中与 [sa-touch, ea+touch) 有交叠的行程：eb > sa-touch 且 sb < ea+touch
            j = int(np.searchsorted(eb, sa[i] - touch, side="right"))
            while j < len(sb) and sb[j] < ea[i] + touch:
                union(a0 + i, b0 + j)
                j += 1

    # 3) 落到图上
    remap = {}
    nxt = 1
    out_sizes = {}
    for r in range(h):
        for i in range(row_start[r], row_start[r + 1]):
            r_ = find(i)
            lab = remap.get(r_)
            if lab is None:
                lab = nxt
                remap[r_] = lab
                nxt += 1
            labels[r, run_s[i]:run_e[i]] = lab
            out_sizes[lab] = out_sizes.get(lab, 0) + (run_e[i] - run_s[i])
    return labels, out_sizes


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """填孔：从图像边框泛洪背景，未被泛洪到的背景像素即为孔洞。"""
    mask = np.asarray(mask, dtype=bool)
    bg = ~mask
    labels, _ = label_components(bg, connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    outside = np.isin(labels, border) & bg
    return mask | (bg & ~outside)


def largest_component(mask: np.ndarray, connectivity: int = 8):
    labels, sizes = label_components(mask, connectivity)
    if not sizes:
        return np.zeros_like(mask, dtype=bool), labels
    keep = max(sizes, key=sizes.get)
    return labels == keep, labels


def dilate_disc(mask: np.ndarray, radius: int) -> np.ndarray:
    """圆盘膨胀（用方形 max 过滤近似 + 距离约束）。细长结构上用方形已足够。"""
    return max_filter(mask.astype(np.float32), radius) > 0.5


# ------------------------------------------------------------------ 阈值
def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    """在值域上做 Otsu；返回阈值两侧类间方差最大的分割点。"""
    v = np.asarray(x, dtype=np.float64).ravel()
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return lo
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b))
    return float(centers[k])


def mad_sigma(x: np.ndarray) -> float:
    """中位数绝对偏差换算成 σ（1.4826 * MAD），对异常值稳健。"""
    v = np.asarray(x, dtype=np.float64).ravel()
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


# ------------------------------------------------------------------ 指标
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int) -> float:
    f1s = []
    for c in range(n_cls):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int) -> np.ndarray:
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, names) -> list:
    rows = []
    n_cls = len(names)
    for c in range(n_cls):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(cls=names[c], support=int(np.sum(y_true == c)), tp=tp,
                         precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4)))
    return rows


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric, n_boot: int = 1000,
                 seed: int = 0) -> tuple[float, float, float]:
    """bootstrap 95% 置信区间（点估计, 下界, 上界）。"""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric(y_true[idx], y_pred[idx]))
    vals = np.sort(np.asarray(vals))
    return float(metric(y_true, y_pred)), float(vals[int(0.025 * n_boot)]), float(vals[int(0.975 * n_boot)])


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------------ 画图
def _font(size: int = 14):
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf"):
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_text_safe(d: ImageDraw.ImageDraw, xy, text, fill=(0, 0, 0), size: int = 14):
    d.text(xy, text, fill=fill, font=_font(size))


def to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        a = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(a, mode="L").convert("RGB")
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")


def normalise_for_display(x: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    """把任意浮点图拉伸到 0..255 便于看（按分位数裁剪，抗离群值）。"""
    v = np.asarray(x, dtype=np.float32).ravel()
    lo, hi = np.percentile(v, lo_q), np.percentile(v, hi_q)
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.float32)


def heatmap_to_rgb(x: np.ndarray) -> np.ndarray:
    """把 0..1 的标量图映射成伪彩（黑-紫-橙-白），用于 OD / 染色量可视化。"""
    t = np.clip(np.asarray(x, dtype=np.float32), 0, 1)[..., None]
    knots = np.array([
        [0, 0, 0], [60, 20, 110], [170, 40, 90], [245, 140, 40], [255, 250, 225],
    ], dtype=np.float32) / 255.0
    pos = np.linspace(0, 1, len(knots))
    rgb = np.zeros(t.shape[:-1] + (3,), dtype=np.float32)
    for c in range(3):
        rgb[..., c] = np.interp(t[..., 0], pos, knots[:, c])
    return rgb * 255.0


def hstack_panels(panels, gap: int = 6, bg=(255, 255, 255)) -> Image.Image:
    """把若干 PIL 图横向拼起来（高度对齐到最大者）。"""
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    return canvas


def vstack_panels(panels, gap: int = 6, bg=(255, 255, 255)) -> Image.Image:
    w = max(p.width for p in panels)
    h = sum(p.height for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    y = 0
    for p in panels:
        canvas.paste(p, (0, y))
        y += p.height + gap
    return canvas


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
