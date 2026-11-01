# -*- coding: utf-8 -*-
"""webapp.py —— 本地网页服务：上传一张染色体图 -> 判读结果。

只依赖 Python 标准库（http.server）+ 本项目模块，不需要 Flask / FastAPI。

用法（在仓库根目录）：
    python app.py                     # 默认 http://127.0.0.1:8770
    python app.py --port 9000 --open   # 换端口并自动打开浏览器
    python app.py --host 0.0.0.0       # 同一局域网其它设备也能访问（有风险，默认关闭）

接口：
    GET  /                  -> index.html（仓库根目录）
    GET  /assets/<file>     -> 前端与模型 JSON（/static/ 兼容旧路径）
    GET  /api/info          -> 模型信息与测试集指标
    GET  /api/example       -> 示例照片（?kind=blank 返回一张空白图，用来演示「无」）
    POST /api/analyze       -> body 为图片字节，返回 JSON 判读结果
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from .common import (EXAMPLES, METRICS_FILE, MODEL_FILE, MODELS, ROOT, RUNS, WEB,
                     load_json, setup_console, to_pil)
from .detect_terminals import analyse_image
from .pipeline import Predictor
from .viz import overlay_png

MAX_UPLOAD = 40 * 1024 * 1024          # 40 MB
MAX_SIDE = 2600                        # 超过就先缩到这个边长，保护内存
EXAMPLE = os.path.join(EXAMPLES, "example_photo.jpg")

STATE: dict = {}


# ------------------------------------------------------------------ 启动
def init_state():
    STATE["predictor"] = Predictor(MODEL_FILE)
    STATE["started"] = time.time()
    info = dict(model=STATE["predictor"].model,
                feature_set=STATE["predictor"].feature_set,
                n_classes=len(STATE["predictor"].classes),
                classes=STATE["predictor"].classes)
    # 指标随仓库分发在 models/metrics.json，改了模型就同步改它
    if os.path.exists(METRICS_FILE):
        sm = load_json(METRICS_FILE)
        info.update(test_acc=sm.get("test_acc"), test_macro_f1=sm.get("test_macro_f1"),
                    test_top2=sm.get("test_top2"), n_test=sm.get("n_test"),
                    coverage_at_90=sm.get("coverage_at_90"),
                    loso_acc=sm.get("loso_acc"))
        ctl = sm.get("human") or {}
        info.update(kappa=ctl.get("kappa"), human_agreement=ctl.get("agreement"))
    STATE["info"] = info


# ------------------------------------------------------------------ 工具
def png_data_url(img: np.ndarray, scale: int = 2, max_w: int = 560) -> str:
    im = to_pil(img)
    if im.width < max_w:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _to_window(rgb: np.ndarray) -> np.ndarray:
    """把任意端部图整理成 246×112：先按 2.2:1 居中裁，再缩放。"""
    h, w = rgb.shape[:2]
    target = 246 / 112
    if w / h > target:
        cw = int(h * target)
        x0 = (w - cw) // 2
        rgb = rgb[:, x0:x0 + cw]
    else:
        ch = int(w / target)
        y0 = (h - ch) // 2
        rgb = rgb[y0:y0 + ch, :]
    from .common import resize_bilinear
    return resize_bilinear(rgb, 246, 112)


def _ctype(rel: str) -> str:
    ext = os.path.splitext(rel)[1].lower()
    return {".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".mjs": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml", ".ico": "image/x-icon",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            }.get(ext, "application/octet-stream")


def _num(v, nd=1):
    try:
        import math
        v = float(v)
        if not math.isfinite(v):
            return None
    except Exception:
        return None
    return round(v, nd)


def _sanitise(result: dict, k: float = 1.0) -> dict:
    """把 analyse_image 的结果压成 JSON 安全的形状（去掉 numpy 数组、补缩略图与 geo 块）。

    k：服务端若把大图缩小过，坐标要乘以 1/scale 换回原图（k>1）。
    """
    h, w = result["image"]["w"], result["image"]["h"]
    out = dict(ok=result["ok"], mode=result["mode"], image=dict(w=w, h=h),
               flags=result.get("flags", []), verdict=result.get("verdict"),
               reason=(result.get("flags") or [None])[0],
               n_terminals=len(result.get("terminals", [])),
               elapsed_s=result.get("elapsed_s"))
    terms = []
    for t in result.get("terminals", []):
        crop = t.pop("crop", None)
        probs = t.get("probs") or {}
        ordered = sorted(probs.items(), key=lambda kv: -kv[1])
        t["probs"] = [dict(cls=key, p=round(float(v), 4)) for key, v in ordered]
        t["crop_png"] = png_data_url(crop, 2, 520) if crop is not None else None
        if t.get("quad"):
            t["quad"] = [[round(float(x) * k, 1), round(float(y) * k, 1)]
                         for x, y in t["quad"]]
        if t.get("box"):
            b = t["box"]
            t["box"] = dict(x0=round(b["x0"] * k, 1), y0=round(b["y0"] * k, 1),
                            cw=round(b["cw"] * k, 1), ch=round(b["ch"] * k, 1))
        t["x"] = round(float(t.get("x", 0)) * k, 1)
        t["y"] = round(float(t.get("y", 0)) * k, 1)
        # 质检标记合并成一个列表（前端只认 flags）
        t["flags"] = [f for f in (list(t.get("detect_flags") or []) +
                                  list(t.get("window_flags") or [])) if f]
        t.pop("detect_flags", None)
        t.pop("window_flags", None)
        t["geo"] = dict(length_px=_num(t.pop("length_px", None)),
                        width_px=_num(t.pop("width_px", None)),
                        tip_angle_deg=_num(t.pop("tip_angle_deg", None)),
                        n_bands=_num(t.pop("n_bands", None)),
                        coherence=_num(t.pop("coherence", None), 2),
                        axis_deg=_num(t.pop("axis_deg", None)),
                        prox_in_frame=bool(t.pop("prox_in_frame", False)))
        terms.append(t)
    out["terminals"] = terms
    out["model"] = STATE["info"]
    return out


# ------------------------------------------------------------------ 请求处理
class Handler(BaseHTTPRequestHandler):
    server_version = "ChromTerm/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[web] " + (fmt % args) + "\n")

    # ---- 基础响应 ----
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _file(self, path: str, ctype: str):
        if not os.path.isfile(path):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    # ---- 路由 ----
    def do_GET(self):
        p = self.path.split("?")[0]
        try:
            if p in ("/", "/index.html"):
                self._file(os.path.join(ROOT, "index.html"), "text/html; charset=utf-8")
            elif p.startswith(("/assets/", "/examples/", "/docs/", "/static/")):
                rel = p[len("/static/"):] if p.startswith("/static/") else p.lstrip("/")
                self._file(os.path.join(ROOT, *rel.split("/")), _ctype(rel))
            elif p == "/api/info":
                self._json(dict(ok=True, model=STATE["info"]))
            elif p == "/api/example":
                kind = ""
                if "?" in self.path:
                    from urllib.parse import parse_qs
                    kind = (parse_qs(self.path.split("?", 1)[1]).get("kind") or [""])[0]
                if kind == "blank":          # 用来演示「无」的状态
                    arr = np.full((600, 900, 3), 244, np.uint8)
                    buf = io.BytesIO()
                    Image.fromarray(arr).save(buf, format="PNG")
                    self._send(200, buf.getvalue(), "image/png")
                elif os.path.exists(EXAMPLE):
                    self._file(EXAMPLE, "image/jpeg")
                else:
                    self._json(dict(ok=False, error="示例图不存在"), 404)
            elif p == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")
        except Exception:
            traceback.print_exc()
            self._json(dict(ok=False, error="服务端异常"), 500)

    def do_POST(self):
        p = self.path.split("?")[0]
        if p != "/api/analyze":
            self._json(dict(ok=False, error="未知接口"), 404)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                self._json(dict(ok=False, error="没有收到图片数据"), 400)
                return
            if n > MAX_UPLOAD:
                self._json(dict(ok=False, error="图片太大（>40 MB），请先压缩"), 413)
                return
            data = self.rfile.read(n)
            t0 = time.time()
            try:
                im = Image.open(io.BytesIO(data))
                im.load()
                im = im.convert("RGB")
            except Exception:
                self._json(dict(ok=False, error="无法解析这张图片，请换 JPG/PNG/TIF"), 400)
                return
            w0, h0 = im.size
            scale = 1.0
            if max(w0, h0) > MAX_SIDE:
                scale = MAX_SIDE / max(w0, h0)
                im = im.resize((int(w0 * scale), int(h0 * scale)), Image.LANCZOS)
            rgb = np.asarray(im, dtype=np.float32)

            # mode=single（默认，精度最高）/ multi（整片多末端）/ auto
            from urllib.parse import parse_qs
            q = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            mode = (q.get("mode") or ["single"])[0]
            if mode == "auto":
                mode = None
            if mode not in (None, "single", "multi"):
                mode = "single"

            result = analyse_image(rgb, STATE["predictor"], max_terminals=6,
                                   mode=mode, verbose=False)
            # 服务端把大图缩小过 -> 坐标要换算回原图，否则前端画偏
            payload = _sanitise(result, 1.0 / scale if scale < 1.0 else 1.0)
            payload["file"] = dict(name=self.headers.get("X-Filename", ""),
                                   w=w0, h=h0, bytes=len(data))
            payload["server_s"] = round(time.time() - t0, 2)
            self._json(payload)
        except Exception:
            traceback.print_exc()
            self._json(dict(ok=False, error="分析失败，请检查图片或查看服务端日志"), 500)


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(ROOT, "index.html")):
        raise SystemExit(f"找不到入口 {os.path.join(ROOT, 'index.html')}（index.html 必须在仓库根目录）")
    print("加载模型 ...", end="", flush=True)
    init_state()
    print(f" 完成（{STATE['info']['feature_set']} / {STATE['info']['model']}）")
    print(f"测试集参考：acc={STATE['info'].get('test_acc')}  "
          f"macro-F1={STATE['info'].get('test_macro_f1')}  "
          f"top-2={STATE['info'].get('test_top2')}")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"\n打开 {url}  （Ctrl+C 停止）")
    if args.open:
        import webbrowser
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
