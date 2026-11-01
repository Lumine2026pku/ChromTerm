/*!
 * ChromTerm 前端：上传 → 判读 → 渲染结论与叠加标注
 * ------------------------------------------------------------------
 * 两套后端自动切换：
 *   · 静态模式（GitHub Pages）：浏览器内用 assets/js/engine.js 跑单末端推理
 *   · 服务模式（本地 python app.py）：调用 /api/analyze，支持整片多末端
 *
 * 标注叠加（SVG）的几个坑都在这里处理了：
 *   1) 图片还没 decode 完就画 → SVG 尺寸为 0，什么也看不见 → 统一 await 图片就绪；
 *   2) 上传后换图时旧标注还在 → 先清空再画；
 *   3) 浏览器把大图缩到 1600 先推理 → 标注坐标必须按 原图/推理图 的比例换算；
 *   4) 窗口缩放/滚动 → 用 ResizeObserver 保持 SVG 尺寸与图片一致。
 */
'use strict';

const $ = (id) => document.getElementById(id);

const state = {
  api: false,             // 是否有本地 Python 服务
  model: null,            // 静态模式：assets/model/model_best.json
  engine: null,           // 静态模式：动态 import 的引擎
  mode: 'single',         // single | multi
  data: null,             // 最近一次判读结果（统一结构）
  image: null,            // 当前 <img> 的自然尺寸等
  busy: false,
  objectUrl: null,
};

const QUALITY_TEXT = { good: '可直接采信', fair: '需人工确认', poor: '低置信', cut: '端部贴边' };
const MAX_INFER_SIDE = 1600;          // 推理前把图缩到这个边长以内（与后端一致）
const API_TIMEOUT = 1500;

/* ------------------------------------------------------------------ 启动 */

async function boot() {
  const info = await probeApi();
  if (info) {
    state.api = true;
    $('modeBadge').classList.add('ready');
    $('modeText').textContent = '服务模式 · 多末端可用';
    $('modeMulti').disabled = false;
    const m = info.model || {};
    fillModelInfo(m);
  } else {
    state.api = false;
    $('modeBadge').classList.add('static');
    $('modeText').textContent = '静态模式 · 单末端';
    try {
      state.model = await fetchJson('assets/model/model_best.json');
      fillModelInfo(state.model.metrics || {});
    } catch (e) {
      showError('模型文件加载失败：' + e.message);
      return;
    }
  }
  bindUI();
  // ?mode=single|multi 可指定模式；?demo=term|photo|blank 便于演示与截图自检
  const q = new URLSearchParams(location.search);
  if (q.get('mode') === 'multi' && state.api) setMode('multi');
  const demo = q.get('demo');
  if (demo) setTimeout(() => loadDemo(demo), 300);
}

async function probeApi() {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), API_TIMEOUT);
    const r = await fetch('api/info', { signal: ctrl.signal, cache: 'no-store' });
    clearTimeout(t);
    if (!r.ok) return null;
    const j = await r.json();
    return j && j.ok ? j : null;
  } catch (e) {
    return null;                       // 静态托管（GitHub Pages）会 404 / 拒绝
  }
}

async function fetchJson(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(url + ' → HTTP ' + r.status);
  return r.json();
}

function fillModelInfo(m) {
  const acc = m.test_acc != null ? (m.test_acc * 100).toFixed(1) + '%' : '—';
  const top2 = m.test_top2 != null ? (m.test_top2 * 100).toFixed(1) + '%' : '—';
  $('perfBadge').hidden = false;
  $('perfBadge').textContent = `测试集 top-1 ${acc} · top-2 ${top2}`;
  $('idleModel').textContent =
    `模型：光密度归一化 + 带纹/形态特征 + 6 类逻辑回归（测试集 150 个端部窗口：` +
    `top-1 ${acc}、top-2 ${top2}）。结果请当作辅助判读。`;
  $('footModel').textContent =
    `模型 ${m.feature_set || 'OD + 带纹特征'} · 指标来源 models/metrics.json`;
}

/* ------------------------------------------------------------------ 交互 */

function bindUI() {
  const drop = $('drop');
  const file = $('file');
  $('pick').addEventListener('click', () => file.click());
  file.addEventListener('change', () => { if (file.files[0]) send(file.files[0]); });

  ['dragenter', 'dragover'].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) send(f);
  });
  window.addEventListener('paste', (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (const it of items) {
      if (it.type && it.type.startsWith('image/')) { send(it.getAsFile()); break; }
    }
  });

  $('exampleTerm').addEventListener('click', () => loadDemo('term'));
  $('examplePhoto').addEventListener('click', () => loadDemo('photo'));
  const exTif = $('exampleTif');
  if (exTif) exTif.addEventListener('click', () => loadDemo('tiff'));

  $('modeSingle').addEventListener('click', () => setMode('single'));
  $('modeMulti').addEventListener('click', () => setMode('multi'));

  $('zoomFit').addEventListener('click', () => setZoom(false));
  $('zoom100').addEventListener('click', () => setZoom(true));
  $('modal').addEventListener('click', () => $('modal').classList.add('hidden'));
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('modal').classList.add('hidden'); });

  const ro = new ResizeObserver(() => syncOverlaySize());
  ro.observe($('frame'));
  window.addEventListener('resize', syncOverlaySize);
  // 图片真正解码完成后，如果已有判读结果就重画一次标注（这是"标注不显示"的兜底）
  $('photo').addEventListener('load', () => {
    if (state.data) drawOverlay(state.data);
  });
}

function setMode(mode) {
  if (mode === 'multi' && !state.api) return;
  state.mode = mode;
  $('modeSingle').classList.toggle('active', mode === 'single');
  $('modeMulti').classList.toggle('active', mode === 'multi');
  $('modeSingle').setAttribute('aria-selected', mode === 'single');
  $('modeMulti').setAttribute('aria-selected', mode === 'multi');
  if (state.image && state.data) analyzeCurrent();
}

function setZoom(actual) {
  $('stage').classList.toggle('actual', actual);
  $('zoomFit').classList.toggle('active', !actual);
  $('zoom100').classList.toggle('active', actual);
  requestAnimationFrame(syncOverlaySize);
}

async function loadDemo(kind) {
  if (state.busy) return;
  try {
    if (kind === 'blank') {
      const c = document.createElement('canvas');
      c.width = 900; c.height = 600;
      const g = c.getContext('2d');
      g.fillStyle = '#f4f4f4'; g.fillRect(0, 0, c.width, c.height);
      const blob = await new Promise((res) => c.toBlob(res, 'image/png'));
      send(new File([blob], 'blank.png', { type: 'image/png' }));
      return;
    }
    const map = {
      photo: ['examples/example_photo.jpg', 'example_photo.jpg'],
      term: ['examples/example_terminal.jpg', 'example_terminal.jpg'],
      tiff: ['examples/example_terminal.tif', 'example_terminal.tif'],
    };
    const [url, name] = map[kind] || map.term;
    const r = await fetch(url, { cache: 'force-cache' });
    if (!r.ok) throw new Error(url + ' → HTTP ' + r.status);
    const blob = await r.blob();
    const type = name.endsWith('.tif') ? 'image/tiff' : (blob.type || 'image/jpeg');
    send(new File([blob], name, { type }));
  } catch (e) {
    showError('示例图加载失败：' + e.message + '（本地请用 python app.py 或 http 服务器打开本页）');
  }
}

/* ------------------------------------------------------------------ 读图 */

function waitImage(img) {
  return new Promise((resolve, reject) => {
    if (img.complete && img.naturalWidth > 0) return resolve();
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('图片解码失败'));
  });
}

/**
 * 读图并显示。
 *
 * 浏览器（Chrome/Edge/Firefox）的 <img> 不支持 image/tiff，直接 src=tif 会触发 error
 * —— 这正是 .tif 上传「解码失败」的原因。这里加一个分支：图片解码失败且字节头是 TIFF 时，
 * 用 assets/js/tiff.js 在浏览器里解码，再转成 PNG dataURL 交给后续流程（显示 / 叠加 / 推理）。
 * 服务模式其实也可以让 Python(PIL) 读，但显示同样需要这一份解码结果。
 */
async function decodeImage(file) {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  const img = $('photo');
  img.src = state.objectUrl;
  let note = '';
  try {
    await waitImage(img);
  } catch (err) {
    // 不是图片 / 浏览器不支持该格式：先看是不是 TIFF
    const buf = await file.arrayBuffer();
    const { isTiff, decodeTiff } = await import('./js/tiff.js');
    if (!isTiff(buf)) throw err;
    let dec;
    try {
      dec = await decodeTiff(buf);
    } catch (e2) {
      throw new Error(`TIFF 解码失败：${e2.message}；请另存为 PNG/JPG，或用 python app.py 的服务模式判读。`);
    }
    if (dec.width * dec.height > 4e7) {
      throw new Error(`这张 TIFF 太大（${dec.width}×${dec.height}），请先裁剪或另存为 PNG/JPG`);
    }
    const c = document.createElement('canvas');
    c.width = dec.width; c.height = dec.height;
    c.getContext('2d').putImageData(new ImageData(dec.data, dec.width, dec.height), 0, 0);
    img.src = c.toDataURL('image/png');
    await waitImage(img);
    note = ' · TIFF 已在浏览器内解码';
  }
  state.image = { w: img.naturalWidth, h: img.naturalHeight, file };
  $('viewer').classList.remove('hidden');
  $('imgMeta').textContent =
    `${file.name || '粘贴的图片'} · ${img.naturalWidth}×${img.naturalHeight} · ` +
    `${(file.size / 1024).toFixed(0)} KB${note}`;
  syncOverlaySize();
  return img;
}

/** 把图片画到 canvas，最长边不超过 MAX_INFER_SIDE，返回 {canvas, scale} */
function toCanvas(img) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const k = Math.min(1, MAX_INFER_SIDE / Math.max(w, h));
  const cw = Math.max(1, Math.round(w * k)), ch = Math.max(1, Math.round(h * k));
  const c = document.createElement('canvas');
  c.width = cw; c.height = ch;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(img, 0, 0, cw, ch);
  return { canvas: c, w: cw, h: ch, scale: cw / w };
}

function windowThumb(img, box, size) {
  const c = document.createElement('canvas');
  c.width = 246; c.height = 112;
  const g = c.getContext('2d');
  g.imageSmoothingQuality = 'high';
  g.drawImage(img, box.x0, box.y0, box.cw, box.ch, 0, 0, 246, 112);
  return c.toDataURL('image/png');
}

/* ------------------------------------------------------------------ 推理 */

async function send(file) {
  if (state.busy) return;
  if (!/^image\//.test(file.type) && !/\.(jpe?g|png|tif?f|bmp|webp)$/i.test(file.name || '')) {
    showError('请选择图片文件（JPG / PNG / TIF）。');
    return;
  }
  state.busy = true;
  $('idle').classList.add('hidden');
  $('result').classList.add('hidden');
  $('error').classList.add('hidden');
  $('loading').classList.remove('hidden');
  $('elapsed').textContent = '';
  const t0 = performance.now();
  const timer = setInterval(() => {
    $('elapsed').textContent = ` 已用 ${((performance.now() - t0) / 1000).toFixed(1)}s`;
  }, 200);

  try {
    let img = null;
    try {
      img = await decodeImage(file);
    } catch (err) {
      // 浏览器解不开（例如分块 / JPEG 压缩的 TIFF）：服务模式下仍可让 Python 判读
      if (!state.api) throw err;
      state.file = file;
      state.image = null;
      $('viewer').classList.add('hidden');
      $('imgMeta').textContent = `${file.name || '图片'} · 浏览器无法显示，交由本地服务判读`;
    }
    state.file = file;
    await analyzeCurrent(img);
  } catch (e) {
    showError(String(e.message || e));
  } finally {
    clearInterval(timer);
    state.busy = false;
    $('loading').classList.add('hidden');
  }
}

async function analyzeCurrent(img) {
  $('loading').classList.remove('hidden');
  $('result').classList.add('hidden');
  $('error').classList.add('hidden');
  try {
    if (state.api) {
      state.data = await analyzeViaApi(state.file, state.mode);
    } else {
      if (!img) throw new Error('这张图片浏览器解不开，无法在本地推理；请另存为 PNG/JPG，或用 python app.py 的服务模式。');
      state.data = await analyzeInBrowser(img);
    }
    render(state.data);
  } catch (e) {
    showError(String(e.message || e));
  } finally {
    $('loading').classList.add('hidden');
  }
}

/* --- 服务模式：把原图字节直接发给本地 Python --- */
async function analyzeViaApi(file, mode) {
  const t0 = performance.now();
  const r = await fetch(`api/analyze?mode=${encodeURIComponent(mode)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream', 'X-Filename': file.name || '' },
    body: file,
  });
  const j = await r.json().catch(() => null);
  if (!j) throw new Error('服务端返回了无法解析的内容');
  if (j.error) throw new Error(j.error);
  const out = { ...j, elapsed_ms: Math.round(performance.now() - t0), engine: 'server' };
  return out;
}

/* --- 静态模式：浏览器内单末端推理 --- */
async function analyzeInBrowser(img) {
  if (!state.model) state.model = await fetchJson('assets/model/model_best.json');
  if (!state.engine) state.engine = await import('./js/engine.js');

  const { canvas, w, h, scale } = toCanvas(img);
  const ctx = canvas.getContext('2d');
  const data = ctx.getImageData(0, 0, w, h).data;      // RGBA
  const rgb = new Float32Array(w * h * 3);
  for (let i = 0, j = 0; i < w * h; i++) {
    rgb[j++] = data[i * 4];
    rgb[j++] = data[i * 4 + 1];
    rgb[j++] = data[i * 4 + 2];
  }
  const t0 = performance.now();
  const res = state.engine.analyze(rgb, w, h, state.model);
  const dt = Math.round(performance.now() - t0);

  // 坐标从"推理图"换回"原图"
  const inv = 1 / scale;
  const box = {
    x0: res.box.x0 * inv, y0: res.box.y0 * inv,
    cw: res.box.cw * inv, ch: res.box.ch * inv,
  };
  const classes = res.classes;
  const probs = classes.map((c, i) => ({ cls: c, p: res.probs[i] }))
    .sort((a, b) => b.p - a.p);

  const term = {
    id: 1,
    cls: res.cls, conf: res.conf, probs,
    second: res.second, conf2: res.conf2, margin: res.margin,
    x: res.marker.x * inv, y: res.marker.y * inv,
    quad: null,
    box, className: 'box',
    stats: { scale: res.scale, fg: res.fg, ac_peak: res.ac_peak },
    quality: res.conf >= 0.55 ? 'good' : res.conf >= 0.35 ? 'fair' : 'poor',
    flags: [],
    crop_png: res.ok ? windowThumb(img, box, 246) : null,
    geo: null,
  };
  return {
    ok: res.ok,
    mode: res.mode,
    reason: res.reason,
    image: { w: img.naturalWidth, h: img.naturalHeight },
    terminals: res.ok ? [term] : [],
    verdict: res.ok ? {
      cls: res.cls, conf: res.conf, second: res.second, margin: res.margin,
      terminal_id: 1, reliable: res.conf >= 0.45,
      text: res.conf >= 0.45 ? '整张图按一个末端判读' : '置信度偏低，建议人工确认',
    } : null,
    elapsed_ms: dt,
    engine: 'browser',
  };
}

/* ------------------------------------------------------------------ 渲染 */

function render(j) {
  $('result').classList.remove('hidden');
  $('raw').textContent = JSON.stringify(j, null, 2);
  // 先等图片就绪再画，避免 SVG 尺寸为 0（这正是"标注有时不显示"的根因）
  drawOverlay(j);
  renderVerdict(j);
  renderTerminals(j);
  const how = j.engine === 'browser' ? '浏览器内推理' : '本地服务';
  const ms = j.elapsed_ms && j.elapsed_ms > 0 ? ` ${j.elapsed_ms} ms` : '';
  $('imgMeta').textContent =
    $('imgMeta').textContent.replace(/ · (判读|浏览器内推理|本地服务).*$/, '') + ` · ${how}${ms}`;
}

function renderVerdict(j) {
  const box = $('verdict');
  if (!j.ok || !j.terminals || !j.terminals.length) {
    box.className = 'verdict none';
    box.innerHTML = `
      <div class="vlabel">判读结论</div>
      <div class="vmain"><span class="vclass">无</span></div>
      <div class="vtext">画面里没有找到可识别的染色体末端。</div>
      ${j.reason ? `<div class="warn">· ${esc(j.reason)}</div>` : ''}
      <div class="warn">建议：确认对焦与曝光；末端应是一条比背景明显更「染」、带横纹的臂；也可以先裁出单个末端再上传。</div>`;
    return;
  }
  const v = j.verdict || {};
  const conf = ((v.conf || 0) * 100).toFixed(1);
  const warn = [];
  if (!v.reliable) warn.push('置信度偏低，第一名与第二名很接近，请人工确认。');
  if (v.cls === 'other') warn.push('判为 other：更像断裂端 / 着丝粒端 / 模糊区，而不是五条主臂之一。');
  if (j.terminals.length > 1) warn.push(`共找到 ${j.terminals.length} 个候选末端，全部列在下方。`);
  const top = j.terminals[0];
  const topFlags = [...(top.flags || [])];
  if (topFlags.length) warn.push(`第 1 个末端带有质检标记：${topFlags.join('；')}`);
  if (j.mode === 'single' && j.image && Math.max(j.image.w, j.image.h) > 900) {
    warn.push('看起来像整片照片：单末端判读只取了画面中央的一块，建议裁出单个末端，或切到「整片多末端」。');
  }
  box.className = 'verdict';
  box.innerHTML = `
    <div class="vlabel">最可能的结论</div>
    <div class="vmain">
      <span class="vclass">${esc(v.cls || '—')}</span>
      <span class="vconf">置信度 ${conf}%</span>
      ${j.terminals.length > 1 ? `<span class="vconf">（第 ${v.terminal_id || 1} 个末端）</span>` : ''}
    </div>
    <div class="vtext">${esc(v.text || '')}</div>
    ${warn.map((w) => `<div class="warn">· ${esc(w)}</div>`).join('')}`;
}

function renderTerminals(j) {
  const list = $('termList');
  list.innerHTML = '';
  if (!j.terminals || !j.terminals.length) return;
  for (const t of j.terminals) {
    const card = document.createElement('div');
    card.className = 'term';
    card.dataset.id = t.id;
    const conf = ((t.conf || 0) * 100).toFixed(1);
    const q = t.quality || 'fair';
    const probs = (t.probs || []).slice(0, 6).map((p) =>
      (typeof p === 'string' ? { cls: p, p: 0 } : p));
    const geo = [];
    if (t.geo) {
      geo.push(['长度', fmt(t.geo.length_px, 'px')],
               ['臂宽', fmt(t.geo.width_px, 'px')],
               ['端部朝向', t.geo.tip_angle_deg == null ? '—' : `${Math.round(t.geo.tip_angle_deg)}°`],
               ['带数', fmt(t.geo.n_bands)],
               ['方向一致性', t.geo.coherence == null ? '—' : t.geo.coherence.toFixed(2)],
               ['近端在画面内', t.geo.prox_in_frame ? '是' : '否']);
    } else if (t.stats) {
      geo.push(['染色量强度', t.stats.scale == null ? '—' : t.stats.scale.toFixed(2)],
               ['前景占比', t.stats.fg == null ? '—' : (t.stats.fg * 100).toFixed(0) + '%'],
               ['带纹自相关', t.stats.ac_peak == null ? '—' : t.stats.ac_peak.toFixed(2)]);
    }
    geo.push(['margin', t.margin == null ? '—' : t.margin.toFixed(2)]);
    const flags = t.flags || [];
    card.innerHTML = `
      <div class="term-head">
        <div class="idx">${t.id}</div>
        <div>
          <div class="term-cls">${esc(t.cls)}</div>
          <div class="term-conf">置信度 ${conf}% · 第二名 ${esc(t.second || '—')} ${((t.conf2 || 0) * 100).toFixed(1)}%</div>
        </div>
        <span class="chip ${q}">${QUALITY_TEXT[q] || q}</span>
      </div>
      <div class="term-body">
        ${t.crop_png ? `<img class="crop" src="${t.crop_png}" alt="末端 ${t.id} 的窗口">` : ''}
        <div class="detail">
          <div class="probs">
            ${probs.map((p, i) => `
              <div class="prow ${i === 0 ? 'p1' : ''}">
                <span class="pname">${esc(p.cls)}</span>
                <span class="ptrack"><span class="pbar" style="width:${Math.max(1, (p.p || 0) * 100)}%"></span></span>
                <span class="pval">${((p.p || 0) * 100).toFixed(1)}%</span>
              </div>`).join('')}
          </div>
          <div class="geo">
            ${geo.map(([k, v]) => `<div><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join('')}
          </div>
          ${flags.length ? `<div class="flags">${flags.map((f) => `<span class="flag">${esc(f)}</span>`).join('')}</div>` : ''}
          ${!t.geo && t.stats ? '<p class="fine" style="margin:6px 0 0">静态模式只做分类；长度 / 臂宽 / 端部朝向等几何量测需要本地服务（python app.py）。</p>' : ''}
        </div>
      </div>`;
    const crop = card.querySelector('.crop');
    if (crop) {
      crop.addEventListener('click', () => {
        $('modalImg').src = t.crop_png;
        $('modalCap').textContent = `末端 ${t.id}：${t.cls}（${conf}%）` +
          ' · 这是模型实际看到的窗口（含掩膜/中线/端部）';
        $('modal').classList.remove('hidden');
      });
    }
    card.addEventListener('mouseenter', () => highlight(t.id, true));
    card.addEventListener('mouseleave', () => highlight(t.id, false));
    list.appendChild(card);
  }
}

function fmt(v, unit) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return `${v}${unit ? ' ' + unit : ''}`;
}

function esc(s) {
  return String(s === null || s === undefined ? '' : s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ------------------------------------------------------------------ 叠加标注 */

function syncOverlaySize() {
  const img = $('photo'), svg = $('overlay');
  if (!img || !svg) return;
  const r = img.getBoundingClientRect();
  if (r.width > 0 && r.height > 0) {
    svg.setAttribute('width', Math.round(r.width));
    svg.setAttribute('height', Math.round(r.height));
  }
}

function drawOverlay(j) {
  const svg = $('overlay');
  const img = $('photo');
  svg.innerHTML = '';
  if (!j || !img || !img.naturalWidth) return;
  const W = img.naturalWidth, H = img.naturalHeight;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  syncOverlaySize();
  if (!j.terminals || !j.terminals.length) return;

  const NS = 'http://www.w3.org/2000/svg';
  const R = Math.max(6, Math.min(W, H) * 0.016);
  const FS = Math.max(16, Math.min(W, H) * 0.030);

  j.terminals.forEach((t) => {
    const g = document.createElementNS(NS, 'g');
    const cut = (t.flags || []).some((f) => String(f).includes('贴边'));

    // 取值框：多末端是四边形，单末端是中心裁剪矩形
    let pts = null;
    if (t.quad && t.quad.length === 4) pts = t.quad;
    else if (t.box) {
      const b = t.box;
      pts = [[b.x0, b.y0], [b.x0 + b.cw, b.y0], [b.x0 + b.cw, b.y0 + b.ch], [b.x0, b.y0 + b.ch]];
    }
    if (pts) {
      const poly = document.createElementNS(NS, 'polygon');
      poly.setAttribute('points', pts.map((p) => p.join(',')).join(' '));
      poly.setAttribute('class', 'box' + (cut ? ' cut' : ''));
      poly.dataset.id = t.id;
      poly.addEventListener('click', () => highlight(t.id, true, true));
      g.appendChild(poly);
    }

    // 末端位置（点）
    if (Number.isFinite(t.x) && Number.isFinite(t.y)) {
      const c = document.createElementNS(NS, 'circle');
      c.setAttribute('cx', t.x);
      c.setAttribute('cy', t.y);
      c.setAttribute('r', R);
      c.setAttribute('class', 'dot');
      c.dataset.id = t.id;
      c.addEventListener('click', () => highlight(t.id, true, true));
      g.appendChild(c);

      if (j.terminals.length > 1) {
        const tx = document.createElementNS(NS, 'text');
        tx.setAttribute('x', t.x + R * 1.2);
        tx.setAttribute('y', t.y + R * 0.9);
        tx.setAttribute('class', 'lbl');
        tx.setAttribute('font-size', FS);
        tx.setAttribute('stroke-width', FS * 0.35);
        tx.textContent = String(t.id);
        g.appendChild(tx);
      }
    }
    svg.appendChild(g);
  });

  // 尺寸同步兜底（换图重画在 bindUI 里统一挂了一次 load 监听，避免监听器累积）
  requestAnimationFrame(syncOverlaySize);
}

function highlight(id, on, scroll) {
  document.querySelectorAll(`#overlay [data-id="${id}"]`).forEach((el) =>
    el.classList.toggle('hl', on));
  const card = document.querySelector(`.term[data-id="${id}"]`);
  if (card) {
    card.classList.toggle('hl', on);
    if (scroll) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

function showError(msg) {
  $('error').textContent = '× ' + msg;
  $('error').classList.remove('hidden');
  $('loading').classList.add('hidden');
  $('idle').classList.remove('hidden');
  $('result').classList.add('hidden');
}

window.addEventListener('DOMContentLoaded', boot);
