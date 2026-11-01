/*!
 * ChromTerm · 浏览器端 TIFF 解码（小体积、只做显微镜图片常见的那几种）
 * ------------------------------------------------------------------
 * 为什么需要它：Chrome / Edge / Firefox 的 <img> 不支持 image/tiff（只有 Safari 能解），
 * 所以 GitHub Pages 的纯前端模式遇到 .tif 会直接报「图片解码失败」。
 * 这里补一个最小可用的 TIFF 解码器，覆盖实验室显微镜导出的常见形态：
 *
 *   字节序    II / MM
 *   压缩      1 不压缩、5 LZW、8 / 32946 Deflate、32773 PackBits
 *   位深      8 / 16 位（16 位按整幅 min–max 映射到 8 位）
 *   采样      灰度、RGB、带 alpha（忽略 alpha）、调色板
 *   存储      按条带（strip）存储；chunky 或 planar
 *   预测器    1 无、2 水平差分（LZW/Deflate 常见）
 *
 * 明确不支持（会抛出可读的错误，而不是静默出错）：
 *   分块存储（tiled，tag 322/323/324）、JPEG 压缩（tag 259 = 7）、CMYK。
 *   这类文件请另存为 PNG/JPG，或用本地服务模式（python app.py，PIL 能读）。
 *
 * 返回 { width, height, data: Uint8ClampedArray(RGBA) }，可直接 putImageData。
 */

'use strict';

const T = {
  ImageWidth: 256, ImageLength: 257, BitsPerSample: 258, Compression: 259,
  Photometric: 262, StripOffsets: 273, SamplesPerPixel: 277, RowsPerStrip: 278,
  StripByteCounts: 279, PlanarConfig: 284, Predictor: 317, ColorMap: 320,
  TileWidth: 322, TileLength: 323, TileOffsets: 324, TileByteCounts: 325,
  SampleFormat: 339,
};

const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8 };

function assertMagic(buf) {
  const b = new Uint8Array(buf);
  const ii = b[0] === 0x49 && b[1] === 0x49 && b[2] === 0x2a && b[3] === 0x00;
  const mm = b[0] === 0x4d && b[1] === 0x4d && b[2] === 0x00 && b[3] === 0x2a;
  if (!ii && !mm) throw new Error('不是 TIFF 文件（缺少 II*\\0 或 MM\\0* 头）');
  return ii ? 'little' : 'big';
}

/** 判断一段字节是不是 TIFF（给 app.js 用，不抛异常） */
export function isTiff(buf) {
  try { assertMagic(buf); return true; } catch (e) { return false; }
}

/** 读一个 IFD，返回 { tags: Map<tag, values[]>, next } */
function readIFD(view, order, offset) {
  const le = order === 'little';
  if (offset + 2 > view.byteLength) throw new Error('IFD 偏移越界');
  const count = view.getUint16(offset, le);
  const tags = new Map();
  for (let i = 0; i < count; i++) {
    const p = offset + 2 + i * 12;
    if (p + 12 > view.byteLength) throw new Error('IFD 条目越界');
    const tag = view.getUint16(p, le);
    const type = view.getUint16(p + 2, le);
    const n = view.getUint32(p + 4, le);
    const size = (TYPE_SIZE[type] || 1) * n;
    const valueOffset = size <= 4 ? p + 8 : view.getUint32(p + 8, le);
    const vals = [];
    for (let k = 0; k < n; k++) {
      const q = valueOffset + k * (TYPE_SIZE[type] || 1);
      if (q + (TYPE_SIZE[type] || 1) > view.byteLength) { vals.push(0); continue; }
      switch (type) {
        case 1: case 7: case 6: vals.push(view.getUint8(q)); break;
        case 3: case 8: vals.push(view.getUint16(q, le)); break;
        case 4: case 9: case 13: vals.push(view.getUint32(q, le)); break;
        case 5: vals.push(view.getUint32(q, le) / (view.getUint32(q + 4, le) || 1)); break;
        case 11: vals.push(view.getFloat32(q, le)); break;
        case 12: vals.push(view.getFloat64(q, le)); break;
        default: vals.push(view.getUint8(q));
      }
    }
    tags.set(tag, vals);
  }
  const next = view.getUint32(offset + 2 + count * 12, le);
  return { tags, next };
}

/** TIFF 版 LZW（MSB 打包、码长提前一位增长） */
function lzwDecode(src, expected) {
  const out = new Uint8Array(expected);
  let o = 0;
  let bitBuf = 0, bitCnt = 0, pos = 0;
  const read = (n) => {
    while (bitCnt < n) {
      if (pos >= src.length) return -1;
      bitBuf = ((bitBuf << 8) | src[pos++]) >>> 0;
      bitCnt += 8;
    }
    const v = (bitBuf >>> (bitCnt - n)) & ((1 << n) - 1);
    bitCnt -= n;
    return v;
  };
  let dict = [];
  const reset = () => {
    dict = [];
    for (let i = 0; i < 256; i++) dict[i] = [i];
    dict[256] = null; dict[257] = null;
    return 258;
  };
  let next = reset();
  let bits = 9, prev = -1;
  for (;;) {
    const code = read(bits);
    if (code < 0) break;
    if (code === 256) { next = reset(); bits = 9; prev = -1; continue; }
    if (code === 257) break;
    let entry;
    if (code < next && dict[code]) entry = dict[code];
    else if (prev >= 0 && dict[prev]) entry = dict[prev].concat(dict[prev][0]);
    else throw new Error('LZW 数据损坏');
    for (let i = 0; i < entry.length && o < expected; i++) out[o++] = entry[i];
    if (prev >= 0 && dict[prev] && next < 4096) {
      dict[next++] = dict[prev].concat(entry[0]);
      if (next === (1 << bits) - 1 && bits < 12) bits++;
    }
    prev = code;
    if (o >= expected) break;
  }
  return out;
}

function packBitsDecode(src, expected) {
  const out = new Uint8Array(expected);
  let o = 0, i = 0;
  while (i < src.length && o < expected) {
    const n = (src[i++] << 24) >> 24;          // signed
    if (n >= 0) {
      for (let k = 0; k <= n && i < src.length && o < expected; k++) out[o++] = src[i++];
    } else if (n !== -128) {
      const v = src[i++];
      for (let k = 0; k < 1 - n && o < expected; k++) out[o++] = v;
    }
  }
  return out;
}

async function inflate(src) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('当前浏览器不支持 DecompressionStream，无法解压 Deflate 压缩的 TIFF');
  }
  const tryOnce = async (fmt) => {
    const ds = new DecompressionStream(fmt);
    const buf = await new Response(new Blob([src]).stream().pipeThrough(ds)).arrayBuffer();
    return new Uint8Array(buf);
  };
  try { return await tryOnce('deflate'); } catch (e) { return await tryOnce('deflate-raw'); }
}

/** 水平差分还原（Predictor = 2） */
function undoLsbPredictor(bytes, width, rows, samples, bps) {
  const bytesPerSample = bps / 8;
  const rowBytes = width * samples * bytesPerSample;
  for (let r = 0; r < rows; r++) {
    const base = r * rowBytes;
    if (bps === 8) {
      for (let x = samples; x < width * samples; x++) {
        bytes[base + x] = (bytes[base + x] + bytes[base + x - samples]) & 0xff;
      }
    } else {
      for (let x = samples; x < width * samples; x++) {
        const cur = base + x * 2;
        const prv = cur - samples * 2;
        const v = ((bytes[cur] | (bytes[cur + 1] << 8)) +
                   (bytes[prv] | (bytes[prv + 1] << 8))) & 0xffff;
        bytes[cur] = v & 0xff;
        bytes[cur + 1] = (v >> 8) & 0xff;
      }
    }
  }
}

/**
 * 解码一张 TIFF（只取第一页）。
 * @param {ArrayBuffer} buf
 * @returns {Promise<{width:number,height:number,data:Uint8ClampedArray}>}
 */
export async function decodeTiff(buf) {
  const order = assertMagic(buf);
  const le = order === 'little';
  const view = new DataView(buf);
  const ifdOffset = view.getUint32(4, le);
  const { tags } = readIFD(view, order, ifdOffset);

  const get1 = (t, d) => (tags.has(t) ? tags.get(t)[0] : d);
  const getN = (t, d) => (tags.has(t) ? tags.get(t) : d);

  const width = get1(T.ImageWidth, 0);
  const height = get1(T.ImageLength, 0);
  if (!width || !height) throw new Error('TIFF 缺少宽高');
  if (tags.has(T.TileWidth)) throw new Error('暂不支持分块（tiled）TIFF，请另存为 PNG/JPG');
  const compression = get1(T.Compression, 1);
  if (compression === 7) throw new Error('暂不支持 JPEG 压缩的 TIFF，请另存为 PNG/JPG');
  const photo = get1(T.Photometric, 1);
  const spp = get1(T.SamplesPerPixel, photo === 2 ? 3 : 1);
  const bits = getN(T.BitsPerSample, [8]);
  const bps = bits[0];
  if (bps !== 8 && bps !== 16) throw new Error(`暂不支持 ${bps} 位 TIFF`);
  const format = get1(T.SampleFormat, 1);
  if (format !== 1 && format !== 2) throw new Error('暂不支持浮点 / 有符号 TIFF');
  const planar = get1(T.PlanarConfig, 1);
  const predictor = get1(T.Predictor, 1);
  const rowsPerStrip = Math.min(get1(T.RowsPerStrip, height) || height, height);
  const stripOffsets = getN(T.StripOffsets, []);
  const stripBytes = getN(T.StripByteCounts, []);
  if (!stripOffsets.length) throw new Error('TIFF 缺少条带偏移（可能不是 strip 存储）');

  const bytesPerSample = bps / 8;
  const stripChannels = planar === 2 ? 1 : spp;    // 一条条带里每像素的采样数
  const planeRowBytes = width * stripChannels * bytesPerSample;   // 一条扫描线的字节数
  const stripsPerPlane = Math.ceil(height / rowsPerStrip);
  const nPlanes = planar === 2 ? spp : 1;

  // 每个条带解压后的原始字节
  const planes = [];
  for (let plane = 0; plane < nPlanes; plane++) {
    const out = new Uint8Array(width * height * stripChannels * bytesPerSample);
    for (let s = 0; s < stripsPerPlane; s++) {
      const idx = plane * stripsPerPlane + s;
      const off = stripOffsets[idx];
      if (off === undefined) continue;
      const len = (stripBytes[idx] !== undefined)
        ? stripBytes[idx]
        : buf.byteLength - off;
      const src = new Uint8Array(buf, off, Math.min(len, buf.byteLength - off));
      const rows = Math.min(rowsPerStrip, height - s * rowsPerStrip);
      const expect = width * rows * stripChannels * bytesPerSample;
      let raw;
      if (compression === 1) raw = src;
      else if (compression === 5) raw = lzwDecode(src, expect);
      else if (compression === 8 || compression === 32946) raw = await inflate(src);
      else if (compression === 32773) raw = packBitsDecode(src, expect);
      else throw new Error(`暂不支持 TIFF 压缩方式 ${compression}`);
      if (predictor === 2) undoLsbPredictor(raw, width, rows, stripChannels, bps);
      out.set(raw.subarray(0, Math.min(expect, raw.length)), s * rowsPerStrip * planeRowBytes);
    }
    planes.push(out);
  }

  // 组装 RGBA
  const data = new Uint8ClampedArray(width * height * 4);
  const chPer = planar === 2 ? 1 : spp;          // 每个平面内每像素的采样数
  const sampleAt = (plane, x, y, c) => {
    const p = planes[plane];
    if (!p) return 0;
    const i = (y * width + x) * chPer + c;
    if (bps === 8) return p[i] || 0;
    const q = i * 2;
    return le ? (p[q] | (p[q + 1] << 8)) : ((p[q] << 8) | p[q + 1]);
  };

  // 16 位：整幅 min–max 映射到 8 位（比直接 >>8 更稳，OD 归一化本身会再处理）
  let lo = 0, hi = 65535;
  if (bps === 16) {
    lo = Infinity; hi = -Infinity;
    for (const p of planes) {
      for (let q = 0; q + 1 < p.length; q += 2) {
        const v = le ? (p[q] | (p[q + 1] << 8)) : ((p[q] << 8) | p[q + 1]);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!(hi > lo)) { lo = 0; hi = 1; }
  }
  const to8 = (v) => (bps === 8 ? v : Math.round((v - lo) * 255 / (hi - lo)));

  const cmap = tags.has(T.ColorMap) ? tags.get(T.ColorMap) : null;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      let r, g, b;
      if (photo === 2 && spp >= 3) {                 // RGB(A)
        if (planar === 2) {
          r = to8(sampleAt(0, x, y, 0));
          g = to8(sampleAt(1, x, y, 0));
          b = to8(sampleAt(2, x, y, 0));
        } else {
          r = to8(sampleAt(0, x, y, 0));
          g = to8(sampleAt(0, x, y, 1));
          b = to8(sampleAt(0, x, y, 2));
        }
      } else if (photo === 3 && cmap) {              // 调色板
        const idx = sampleAt(0, x, y, 0);
        const n = cmap.length / 3;
        r = Math.round(cmap[idx] / 257);
        g = Math.round(cmap[n + idx] / 257);
        b = Math.round(cmap[2 * n + idx] / 257);
      } else {                                       // 灰度
        let v = to8(sampleAt(0, x, y, 0));
        if (photo === 0) v = 255 - v;                // WhiteIsZero
        r = g = b = v;
      }
      data[o] = r; data[o + 1] = g; data[o + 2] = b; data[o + 3] = 255;
    }
  }
  return { width, height, data };
}
