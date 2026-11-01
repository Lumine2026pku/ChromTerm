/*!
 * ChromTerm · 浏览器端推理引擎（单末端）
 * ------------------------------------------------------------------
 * 这是 chromterm/（Python）推理链的 JS 移植，用来让 GitHub Pages 这种
 * 纯静态托管也能跑模型：没有后端、没有 WASM、没有第三方库。
 *
 * 与 Python 版的对应关系：
 *   stainMap()   ← prep_od.stain_map        背景闭运算 → 光密度 → 染料方向 → 染色量
 *   featPyr()    ← features.feat_pyr        多尺度池化 32×10 + 16×5
 *   featBand()   ← features.feat_band       带纹剖面（原始 / 去趋势 / 一阶导）
 *   featAuto()   ← features.feat_auto       自相关（lag 2–40）
 *   featHog()    ← features.feat_hog        梯度方向直方图 6×3×9
 *   featWidth()  ← features.feat_width      粗细剖面
 *   predict()    ← model.Linear + Standardizer
 *
 * 几何量测块（geo，11 维）在网页端**不实现**：用训练集均值代入，
 * 标准化后恰为 0（等于"不知道就不贡献"）。实测 test 准确率 0.5133 vs 完整
 * 0.5067，差在噪声内，所以静态网页不需要移植骨架化那一整套。
 *
 * 关键细节都按 Python/Pillow 的语义实现（bilinear resize 的滤波器缩放、
 * np.gradient 的边界、np.percentile 的线性插值……），否则特征会漂。
 */

'use strict';

const EPS = 1e-6;

/* ------------------------------------------------------------------ 基础 */

function f32(n) { return new Float32Array(n); }

/** 索引按 numpy 'reflect' 方式折叠（不重复边缘像素） */
function reflectIndex(i, n) {
  if (n === 1) return 0;
  while (i < 0 || i >= n) {
    if (i < 0) i = -i;
    if (i >= n) i = 2 * n - 2 - i;
  }
  return i;
}

/** numpy.percentile(a, q)，线性插值（a 会被复制排序，不修改入参） */
function percentile(arr, q) {
  const n = arr.length;
  if (n === 0) return NaN;
  if (n === 1) return arr[0];
  const a = Float64Array.from(arr);
  a.sort();
  const rank = (n - 1) * (q / 100);
  const lo = Math.floor(rank);
  const hi = Math.min(lo + 1, n - 1);
  const f = rank - lo;
  return a[lo] + f * (a[hi] - a[lo]);
}

function median(arr) { return percentile(arr, 50); }

function madSigma(arr) {
  const m = median(arr);
  const dev = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) dev[i] = Math.abs(arr[i] - m);
  return 1.4826 * median(dev);
}

/* --------------------------------------------------- 一维卷积 / 高斯 */

function gaussianKernel1d(sigma) {
  if (sigma <= 1e-6) return Float32Array.from([1]);
  const r = Math.max(1, Math.round(3 * sigma));
  const k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-0.5 * (i / sigma) ** 2);
    k[i + r] = v; s += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return k;
}

/** 沿 x（axis=1）做 1D 卷积，反射边界；img 为 {w,h,data} */
function conv1dX(img, kernel) {
  const { w, h, data } = img;
  const r = (kernel.length - 1) >> 1;
  const out = f32(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = 0; k < kernel.length; k++) {
        const xx = reflectIndex(x + k - r, w);
        acc += kernel[k] * data[row + xx];
      }
      out[row + x] = acc;
    }
  }
  return { w, h, data: out };
}

/** 沿 y（axis=0）做 1D 卷积 */
function conv1dY(img, kernel) {
  const { w, h, data } = img;
  const r = (kernel.length - 1) >> 1;
  const out = f32(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = 0; k < kernel.length; k++) {
        const yy = reflectIndex(y + k - r, h);
        acc += kernel[k] * data[yy * w + x];
      }
      out[row + x] = acc;
    }
  }
  return { w, h, data: out };
}

/** gaussian_blur(img, sx, sy)：先 x 后 y（与 Python 一致） */
function gaussianBlur(img, sx, sy) {
  if (sy === undefined) sy = sx;
  let out = img;
  if (sx > 1e-6) out = conv1dX(out, gaussianKernel1d(sx));
  if (sy > 1e-6) out = conv1dY(out, gaussianKernel1d(sy));
  return out;
}

/* --------------------------------------------------- Pillow 式双线性缩放 */

/** 计算 Pillow `resize(..., BILINEAR)` 的一维权重（滤波器支撑按缩放比例放大） */
function resampleWeights(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1.0);
  const support = filterscale;          // 双线性：支撑 = 1 * filterscale
  const table = [];
  for (let o = 0; o < outSize; o++) {
    const center = (o + 0.5) * scale;
    let xmin = Math.floor(center - support + 0.5);
    let xmax = Math.ceil(center + support - 0.5);
    if (xmin < 0) xmin = 0;
    if (xmax > inSize) xmax = inSize;
    const idx = [];
    const wts = [];
    let sum = 0;
    for (let i = xmin; i < xmax; i++) {
      const t = (i + 0.5 - center) / filterscale;
      const v = Math.max(0, 1 - Math.abs(t));
      idx.push(i); wts.push(v); sum += v;
    }
    if (sum === 0) {                     // 极端情况：退化为最近邻
      const i = Math.min(inSize - 1, Math.max(0, Math.floor(center)));
      idx.length = 0; wts.length = 0;
      idx.push(i); wts.push(1); sum = 1;
    }
    for (let i = 0; i < wts.length; i++) wts[i] /= sum;
    table.push({ idx, wts });
  }
  return table;
}

/** 单通道双线性缩放（等价 Image.resize(size, BILINEAR)） */
function resize1(img, outW, outH) {
  const { w, h, data } = img;
  if (w === outW && h === outH) return { w, h, data: Float32Array.from(data) };
  const wx = resampleWeights(w, outW);
  const tmp = f32(outW * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < outW; x++) {
      const { idx, wts } = wx[x];
      let acc = 0;
      for (let k = 0; k < idx.length; k++) acc += wts[k] * data[row + idx[k]];
      tmp[y * outW + x] = acc;
    }
  }
  const wy = resampleWeights(h, outH);
  const out = f32(outW * outH);
  for (let y = 0; y < outH; y++) {
    const { idx, wts } = wy[y];
    for (let x = 0; x < outW; x++) {
      let acc = 0;
      for (let k = 0; k < idx.length; k++) acc += wts[k] * tmp[idx[k] * outW + x];
      out[y * outW + x] = acc;
    }
  }
  return { w: outW, h: outH, data: out };
}

/** 三通道缩放 */
function resizeRGB(planes, outW, outH) {
  return planes.map((p) => resize1(p, outW, outH));
}

/* --------------------------------------------------- 形态学（边缘填充） */

function extreme1dX(img, radius, isMax) {
  const { w, h, data } = img;
  const out = f32(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let v = isMax ? -Infinity : Infinity;
      for (let k = -radius; k <= radius; k++) {
        let xx = x + k;
        if (xx < 0) xx = 0; else if (xx >= w) xx = w - 1;   // numpy 'edge'
        const t = data[row + xx];
        if (isMax ? t > v : t < v) v = t;
      }
      out[row + x] = v;
    }
  }
  return { w, h, data: out };
}

function extreme1dY(img, radius, isMax) {
  const { w, h, data } = img;
  const out = f32(w * h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let v = isMax ? -Infinity : Infinity;
      for (let k = -radius; k <= radius; k++) {
        let yy = y + k;
        if (yy < 0) yy = 0; else if (yy >= h) yy = h - 1;
        const t = data[yy * w + x];
        if (isMax ? t > v : t < v) v = t;
      }
      out[y * w + x] = v;
    }
  }
  return { w, h, data: out };
}

function maxFilter(img, radius) {
  return extreme1dY(extreme1dX(img, radius, true), radius, true);
}

function minFilter(img, radius) {
  return extreme1dY(extreme1dX(img, radius, false), radius, false);
}

/** 闭运算 = 膨胀后腐蚀（估计"没有染色体时背景该是什么颜色"） */
function morphClose(img, radius) { return minFilter(maxFilter(img, radius), radius); }

/* --------------------------------------------------- 光密度 / 染色量 */

/**
 * RGB 三通道（0..255 的 Float32Array）→ 染色量
 * 返回 { w, h, s (原始染色量), sn (0..1 归一化), scale, fg, purity }
 */
function stainMap(planes, w, h) {
  const radius = Math.max(24, Math.round(0.18 * Math.min(h, w)));
  const bg = planes.map((p) => morphClose(p, radius));

  const n = w * h;
  const od = [f32(n), f32(n), f32(n)];
  for (let c = 0; c < 3; c++) {
    const I = planes[c].data, B = bg[c].data, O = od[c];
    for (let i = 0; i < n; i++) {
      const i0 = Math.max(B[i], 1);
      const v = -Math.log10(Math.max(I[i], 1) / i0);
      O[i] = Math.max(v, 0.02);
    }
  }

  // ---- 染料方向：OD 模长前 8% 的像素、按模长加权幂迭代 ----
  const norms = f32(n);
  for (let i = 0; i < n; i++) {
    norms[i] = Math.hypot(od[0][i], od[1][i], od[2][i]);
  }
  const thr = percentile(norms, 92);
  const sel = [];
  for (let i = 0; i < n; i++) if (norms[i] >= thr) sel.push(i);

  let dye, purity = 0;
  if (sel.length < 20) {
    const k = 1 / Math.sqrt(3);
    dye = [k, k, k];
  } else {
    let d = [0, 0, 0];
    for (const i of sel) {
      const wgt = norms[i];
      d[0] += od[0][i] * wgt; d[1] += od[1][i] * wgt; d[2] += od[2][i] * wgt;
    }
    let nn = Math.hypot(d[0], d[1], d[2]) + EPS;
    d = d.map((v) => v / nn);
    for (let it = 0; it < 3; it++) {
      const c = [0, 0, 0];
      for (const i of sel) {
        const wgt = norms[i];
        const dot = od[0][i] * d[0] + od[1][i] * d[1] + od[2][i] * d[2];
        c[0] += od[0][i] * wgt * dot;
        c[1] += od[1][i] * wgt * dot;
        c[2] += od[2][i] * wgt * dot;
      }
      const cn = Math.hypot(c[0], c[1], c[2]);
      if (cn < EPS) break;
      d = c.map((v) => v / cn);
    }
    if (d[0] + d[1] + d[2] < 0) d = d.map((v) => -v);
    dye = d;
    let acc = 0;
    for (const i of sel) {
      const dot = Math.abs(od[0][i] * d[0] + od[1][i] * d[1] + od[2][i] * d[2]);
      acc += dot / (norms[i] + EPS);
    }
    purity = acc / sel.length;
  }

  // ---- 投影成标量，减背景底噪，按 p99.5 归一 ----
  const s = f32(n);
  for (let i = 0; i < n; i++) {
    const v = od[0][i] * dye[0] + od[1][i] * dye[1] + od[2][i] * dye[2];
    s[i] = Math.max(v, 0);
  }
  const level = percentile(s, 10);
  for (let i = 0; i < n; i++) s[i] = Math.max(s[i] - level, 0);
  let scale = percentile(s, 99.5);
  if (scale < 1e-3) scale = 1;
  const sn = f32(n);
  let fg = 0;
  for (let i = 0; i < n; i++) {
    const v = Math.min(s[i] / scale, 2);
    sn[i] = v;
    if (v > 0.35) fg++;
  }
  return { w, h, s, sn, scale, fg: fg / n, purity };
}

/* --------------------------------------------------- 特征块 */

/** 区域平均池化：先按格边长做高斯预平滑，再双线性缩放 */
function pool2d(img, gw, gh) {
  const { w, h } = img;
  const sy = Math.max(h / (2 * gh), 0.5);
  const sx = Math.max(w / (2 * gw), 0.5);
  return resize1(gaussianBlur(img, sx, sy), gw, gh);
}

function featPyr(sn) {
  const a = pool2d(sn, 32, 10).data;
  const b = pool2d(sn, 16, 5).data;
  const out = f32(a.length + b.length);
  out.set(a, 0); out.set(b, a.length);
  return out;
}

/** 沿臂方向的染色剖面：中间 60% 行上逐列取 60 分位 */
function rowProfile(sn, q) {
  const { w, h, data } = sn;
  const r0 = Math.trunc(0.2 * h), r1 = Math.trunc(0.8 * h);
  const out = f32(w);
  const buf = new Float64Array(Math.max(1, r1 - r0));
  for (let x = 0; x < w; x++) {
    let k = 0;
    for (let y = r0; y < r1; y++) buf[k++] = data[y * w + x];
    out[x] = percentile(buf.subarray(0, k), q * 100);
  }
  return out;
}

function resample1d(x, n) {
  const m = x.length;
  const out = f32(n);
  if (m === 0) return out;
  if (m === 1) { out.fill(x[0]); return out; }
  for (let i = 0; i < n; i++) {
    const t = (n === 1) ? 0 : (i / (n - 1)) * (m - 1);
    const lo = Math.floor(t), hi = Math.min(lo + 1, m - 1);
    out[i] = x[lo] + (t - lo) * (x[hi] - x[lo]);
  }
  return out;
}

function gradient1d(x) {
  const n = x.length, d = f32(n);
  if (n === 1) { d[0] = 0; return d; }
  d[0] = x[1] - x[0];
  d[n - 1] = x[n - 1] - x[n - 2];
  for (let i = 1; i < n - 1; i++) d[i] = (x[i + 1] - x[i - 1]) / 2;
  return d;
}

function featBand(sn) {
  const p = rowProfile(sn, 0.6);
  const sigma = Math.max(2, p.length / 24);
  const r = Math.max(1, Math.round(3 * sigma));
  const k = gaussianKernel1d(sigma);
  const low = f32(p.length);
  for (let i = 0; i < p.length; i++) {
    let acc = 0;
    for (let j = 0; j < k.length; j++) acc += k[j] * p[reflectIndex(i + j - r, p.length)];
    low[i] = acc;
  }
  const hp = f32(p.length);
  for (let i = 0; i < p.length; i++) hp[i] = p[i] - low[i];
  const d = gradient1d(hp);
  const out = f32(3 * 64);
  out.set(resample1d(p, 64), 0);
  out.set(resample1d(hp, 64), 64);
  out.set(resample1d(d, 64), 128);
  return out;
}

function featAuto(sn, maxLag) {
  maxLag = maxLag || 40;
  const p0 = rowProfile(sn, 0.6);
  const n = p0.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += p0[i];
  mean /= n;
  let sd = 0;
  for (let i = 0; i < n; i++) sd += (p0[i] - mean) ** 2;
  sd = Math.sqrt(sd / n) + EPS;
  const p = f32(n);
  for (let i = 0; i < n; i++) p[i] = (p0[i] - mean) / sd;
  const ac = new Float64Array(maxLag + 2);
  for (let lag = 0; lag <= maxLag; lag++) {
    let acc = 0;
    for (let i = 0; i + lag < n; i++) acc += p[i] * p[i + lag];
    ac[lag] = acc;
  }
  const a0 = ac[0] + EPS;
  const out = f32(maxLag - 1);
  for (let i = 0; i < maxLag - 1; i++) {
    const lag = i + 2;
    out[i] = lag <= maxLag ? ac[lag] / a0 : 0;
  }
  return out;
}

/** np.gradient 的 2D 版本：返回 [gy, gx] */
function gradient2d(img) {
  const { w, h, data } = img;
  const gy = f32(w * h), gx = f32(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const ym = Math.max(0, y - 1), yp = Math.min(h - 1, y + 1);
      const xm = Math.max(0, x - 1), xp = Math.min(w - 1, x + 1);
      if (h === 1) gy[i] = 0;
      else if (y === 0) gy[i] = data[(y + 1) * w + x] - data[i];
      else if (y === h - 1) gy[i] = data[i] - data[(y - 1) * w + x];
      else gy[i] = (data[yp * w + x] - data[ym * w + x]) / 2;
      if (w === 1) gx[i] = 0;
      else if (x === 0) gx[i] = data[y * w + x + 1] - data[i];
      else if (x === w - 1) gx[i] = data[i] - data[y * w + x - 1];
      else gx[i] = (data[y * w + xp] - data[y * w + xm]) / 2;
    }
  }
  return { gy, gx };
}

function featHog(sn, cx, cy, bins) {
  cx = cx || 6; cy = cy || 3; bins = bins || 9;
  const b = gaussianBlur(sn, 1.0, 1.0);
  const { gy, gx } = gradient2d(b);
  const { w, h } = sn;
  const ys = linspaceInt(0, h, cy + 1);
  const xs = linspaceInt(0, w, cx + 1);
  const out = f32(cy * cx * bins);
  let o = 0;
  for (let i = 0; i < cy; i++) {
    for (let j = 0; j < cx; j++) {
      const acc = new Float64Array(bins);
      for (let y = ys[i]; y < Math.max(ys[i] + 1, ys[i + 1]); y++) {
        for (let x = xs[j]; x < Math.max(xs[j] + 1, xs[j + 1]); x++) {
          const k = y * w + x;
          const mag = Math.hypot(gx[k], gy[k]);
          let ang = Math.atan2(gy[k], gx[k]) % Math.PI;
          if (ang < 0) ang += Math.PI;
          const idx = Math.min(bins - 1, Math.max(0, Math.trunc((ang / Math.PI) * bins)));
          acc[idx] += mag;
        }
      }
      let nrm = 0;
      for (let k = 0; k < bins; k++) nrm += acc[k] * acc[k];
      nrm = Math.sqrt(nrm) + EPS;
      for (let k = 0; k < bins; k++) out[o++] = acc[k] / nrm;
    }
  }
  return out;
}

/** np.linspace(a,b,n).astype(int) —— 注意 astype 是向零截断 */
function linspaceInt(a, b, n) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const v = n === 1 ? a : a + (b - a) * i / (n - 1);
    out[i] = Math.trunc(v);
  }
  return out;
}

function featWidth(sn, thr, n) {
  thr = thr === undefined ? 0.35 : thr;
  n = n || 32;
  const { w, h, data } = sn;
  const prof = f32(w);
  for (let x = 0; x < w; x++) {
    let c = 0;
    for (let y = 0; y < h; y++) if (data[y * w + x] > thr) c++;
    prof[x] = c;
  }
  const pos = [];
  for (let x = 0; x < w; x++) if (prof[x] > 0) pos.push(prof[x]);
  const med = pos.length ? median(Float32Array.from(pos)) : 1;
  for (let x = 0; x < w; x++) prof[x] /= Math.max(med, EPS);
  return resample1d(prof, n);
}

/* --------------------------------------------------- 归一化窗口 */

/** 居中裁成 2.2:1，再缩放到 246×112（等价 Python 的 _to_window） */
function toWindow(planes, w, h) {
  const target = 246 / 112;
  let x0 = 0, y0 = 0, cw = w, ch = h;
  if (w / h > target) { cw = Math.trunc(h * target); x0 = Math.trunc((w - cw) / 2); }
  else { ch = Math.trunc(w / target); y0 = Math.trunc((h - ch) / 2); }
  const crop = planes.map((p) => cropPlane(p, x0, y0, cw, ch));
  const win = resizeRGB(crop, 246, 112);
  return { win, box: { x0, y0, cw, ch } };
}

function cropPlane(img, x0, y0, cw, ch) {
  const { w, data } = img;
  const out = f32(cw * ch);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) out[y * cw + x] = data[(y0 + y) * w + (x0 + x)];
  }
  return { w: cw, h: ch, data: out };
}

/* --------------------------------------------------- 分类器 */

function predict(x, model) {
  const k = model.classes.length;
  const logits = new Float64Array(k);
  const mu = model.mu, sd = model.sd, W = model.W, b = model.b;
  for (let c = 0; c < k; c++) {
    let acc = b[c];
    const wc = W[c];
    for (let i = 0; i < x.length; i++) acc += ((x[i] - mu[i]) / sd[i]) * wc[i];
    logits[c] = acc;
  }
  let mx = -Infinity;
  for (let c = 0; c < k; c++) mx = Math.max(mx, logits[c]);
  let sum = 0;
  const p = new Float64Array(k);
  for (let c = 0; c < k; c++) { p[c] = Math.exp(logits[c] - mx); sum += p[c]; }
  for (let c = 0; c < k; c++) p[c] /= sum;
  return p;
}

/* --------------------------------------------------- 主入口 */

/** 交错 RGB（长度 w*h*3，0..255）→ 三个平面对象 {w,h,data} */
function planesFromRGB(rgb, w, h) {
  const n = w * h;
  const r = f32(n), g = f32(n), b = f32(n);
  for (let i = 0; i < n; i++) {
    r[i] = rgb[i * 3];
    g[i] = rgb[i * 3 + 1];
    b[i] = rgb[i * 3 + 2];
  }
  return [{ w, h, data: r }, { w, h, data: g }, { w, h, data: b }];
}

/**
 * @param {Float32Array|Uint8ClampedArray} rgb  交错 RGB，长度 w*h*3（0..255）
 * @param {number} w @param {number} h
 * @param {object} model   assets/model/model_best.json
 * @returns {object} 判读结果（与 Python 单末端模式同构，去掉几何字段）
 */
function analyze(rgb, w, h, model) {
  const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());

  const planes = planesFromRGB(rgb, w, h);
  const { win, box } = toWindow(planes, w, h);
  const m = stainMap(win, 246, 112);                 // 与 Python 一致：在 246×112 上估背景
  const sn = resize1({ w: m.w, h: m.h, data: m.sn }, 192, 64);

  const parts = [];
  parts.push(featPyr(sn));
  parts.push(featBand(sn));
  parts.push(featAuto(sn, 40));
  parts.push(featHog(sn, 6, 3, 9));
  parts.push(featWidth(sn, 0.35, 32));
  parts.push(Float32Array.from(model.geo_mean));      // 几何块：用训练均值代入（贡献 0）

  const x = new Float32Array(model.dim);
  let o = 0;
  for (const p of parts) { x.set(p, o); o += p.length; }

  // ---- 有没有染色体末端？两道门：染色量太小 / 带纹自相关太弱 ----
  const band = rowProfile(sn, 0.6);
  const acPeak = autocorrPeak(band, 4, 24);
  const fg = m.fg;
  const noSignal = (m.scale < 0.02) || (fg < 0.02) || (acPeak < 0.40);

  const probs = predict(x, model);
  const order = Array.from(probs.keys()).sort((a, b2) => probs[b2] - probs[a]);
  const cls = model.classes[order[0]];
  const conf = probs[order[0]];
  const conf2 = probs[order[1]];
  const dt = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;

  // 端部位置（原图坐标）：染色量质心，仅用于在图上画一个点
  let sx = 0, sy = 0, sw = 0;
  for (let y = 0; y < 112; y++) {
    for (let xx = 0; xx < 246; xx++) {
      const v = m.sn[y * 246 + xx];
      if (v > 0.35) { sx += xx * v; sy += y * v; sw += v; }
    }
  }
  let wx = 123, wy = 56;
  if (sw > 0) { wx = sx / sw; wy = sy / sw; }
  const marker = {
    x: box.x0 + (wx / 246) * box.cw,
    y: box.y0 + (wy / 112) * box.ch,
  };

  return {
    ok: !noSignal,
    mode: noSignal ? 'none' : 'single',
    reason: noSignal
      ? (m.scale < 0.02 ? '几乎没有染色信号（对比度太低）'
        : fg < 0.02 ? '画面里没有明显的染色区域'
          : '没有找到带纹结构（自相关太弱），可能不是染色体')
      : '',
    classes: model.classes,
    probs: Array.from(probs),
    cls, conf, second: model.classes[order[1]], conf2,
    margin: conf - conf2,
    ac_peak: acPeak, fg, scale: m.scale,
    box,                                   // 原图里的取值框（单末端=居中裁的区域）
    marker,                                // 原图里的端部位置（点）
    window: { w: 246, h: 112, sn: Array.from(m.sn) },   // 供缩略图渲染
    elapsed_ms: Math.round(dt),
  };
}

/** 自相关在 lag∈[lo,hi] 上的最大峰（与 Python ac_peak 一致） */
function autocorrPeak(p, lo, hi) {
  const n = p.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += p[i];
  mean /= n;
  let sd = 0;
  for (let i = 0; i < n; i++) sd += (p[i] - mean) ** 2;
  sd = Math.sqrt(sd / n) + EPS;
  const q = new Float32Array(n);
  for (let i = 0; i < n; i++) q[i] = (p[i] - mean) / sd;
  let a0 = 0;
  for (let i = 0; i < n; i++) a0 += q[i] * q[i];
  a0 += EPS;
  let best = 0;
  for (let lag = lo; lag <= Math.min(hi, n - 1); lag++) {
    let acc = 0;
    for (let i = 0; i + lag < n; i++) acc += q[i] * q[i + lag];
    best = Math.max(best, acc / a0);
  }
  return best;
}

/* ES module 导出（浏览器用 <script type="module"> 引入；Node 校验时复制为 .mjs） */
export { analyze, stainMap, resize1, resizeRGB, featPyr, featBand, featAuto,
         featHog, featWidth, rowProfile, predict, toWindow };
