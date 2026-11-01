# -*- coding: utf-8 -*-
"""冒烟测试：不依赖数据集，只用 examples/ 里的图。

    pip install pytest
    python -m pytest -q

覆盖：
  * 包能导入、权重能加载
  * 单张端部小图 -> 恰好 1 个末端（网页默认模式）
  * mode="multi" 整片照片 -> 至少 1 个末端，字段齐全、坐标落在图内
  * 空白图 -> 「无」
  * HTTP 服务：/（根目录 index.html）、/assets/*、/api/info、/api/analyze 端到端
  * 网页端资产自洽：模型 JSON 的维数、geo_mean 长度、有限性，引擎与入口文件存在
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chromterm.common import EXAMPLES, MODEL_FILE  # noqa: E402
from chromterm.detect_terminals import analyse_image  # noqa: E402
from chromterm.pipeline import Predictor  # noqa: E402

PHOTO = os.path.join(EXAMPLES, "example_photo.jpg")
WINDOW = os.path.join(EXAMPLES, "example_terminal.jpg")
WEB_MODEL = os.path.join(ROOT, "assets", "model", "model_best.json")
ENGINE = os.path.join(ROOT, "assets", "js", "engine.js")
INDEX = os.path.join(ROOT, "index.html")


def load(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


@pytest.fixture(scope="module")
def predictor():
    assert os.path.exists(MODEL_FILE), "models/model_best.npz 缺失"
    return Predictor()


def test_model_loads(predictor):
    assert predictor.model in ("linear", "ncp", "rff")
    assert set(predictor.classes) == {"2L", "2R", "3L", "3R", "X", "other"}


def test_web_assets():
    """网页端资产要自洽：维数对上、没有 NaN、入口与引擎都在。"""
    assert os.path.isfile(INDEX), "index.html 必须在仓库根目录（GitHub Pages 入口）"
    html = open(INDEX, encoding="utf-8").read()
    assert "assets/app.js" in html and "assets/style.css" in html
    assert os.path.isfile(ENGINE)
    m = json.load(open(WEB_MODEL, encoding="utf-8"))
    assert m["dim"] == 836
    assert len(m["mu"]) == m["dim"] and len(m["sd"]) == m["dim"]
    assert len(m["geo_mean"]) == 11, "geo 块是 11 维（不是缓存里 geo_names 的 12 列）"
    assert len(m["W"]) == 6 and all(len(row) == m["dim"] for row in m["W"])
    for arr in [m["mu"], m["sd"], m["geo_mean"]] + m["W"]:
        assert np.isfinite(np.asarray(arr, float)).all(), "模型 JSON 里不能有 NaN/Inf"


def test_single_window(predictor):
    res = analyse_image(load(WINDOW), predictor)
    assert res["ok"] is True
    assert res["mode"] == "single"
    assert res["n_terminals"] == 1
    t = res["terminals"][0]
    assert t["cls"] in predictor.classes
    assert 0.0 <= t["conf"] <= 1.0
    assert abs(float(sum(t["probs"].values())) - 1.0) < 0.05      # 6 类概率和为 1
    assert t["quality"] in ("good", "fair", "poor", "cut")
    # 单末端模式的坐标必须在原图坐标系内（否则前端会画到图外）
    h, w = load(WINDOW).shape[:2]
    assert 0 <= t["x"] <= w and 0 <= t["y"] <= h
    assert t["quad"] is None and t["box"] is not None


def test_single_mode_on_photo(predictor):
    """大图也能强制走单末端：坐标要换算回原图，box 是居中的 2.2:1 裁剪。"""
    rgb = load(PHOTO)
    h, w = rgb.shape[:2]
    res = analyse_image(rgb, predictor, mode="single")
    assert res["mode"] == "single" and res["n_terminals"] == 1
    t = res["terminals"][0]
    assert 0 <= t["x"] <= w and 0 <= t["y"] <= h
    b = t["box"]
    assert 0 <= b["x0"] <= w and 0 <= b["y0"] <= h
    assert 0 < b["cw"] <= w and 0 < b["ch"] <= h


def test_photo_multi(predictor):
    rgb = load(PHOTO)
    h, w = rgb.shape[:2]
    res = analyse_image(rgb, predictor, max_terminals=6, mode="multi")
    assert res["ok"] is True
    assert res["n_terminals"] >= 1
    for t in res["terminals"]:
        assert 0 <= t["x"] <= w and 0 <= t["y"] <= h
        if t["quad"]:
            assert len(t["quad"]) == 4
            for x, y in t["quad"]:
                assert -2 <= x <= w + 2 and -2 <= y <= h + 2
    assert res["verdict"]["terminal_id"] == res["terminals"][0]["id"]


def test_blank_is_none(predictor):
    blank = np.full((400, 600, 3), 244.0, dtype=np.float32)
    res = analyse_image(blank, predictor, mode="single")
    assert res["ok"] is False
    assert res["n_terminals"] == 0


def test_tiff_example_is_valid():
    """仓库里的示例 TIFF 必须能被读、且与同名 JPG 是同一张图（允许压缩误差）。"""
    tif = os.path.join(EXAMPLES, "example_terminal.tif")
    assert os.path.isfile(tif), "示例 TIFF 缺失"
    a = np.asarray(Image.open(tif).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(WINDOW).convert("RGB"), dtype=np.float32)
    assert a.shape == b.shape == (112, 246, 3)
    assert np.abs(a - b).mean() < 3.0


def test_tiff_decoder_matches_pil(tmp_path):
    """浏览器端的 TIFF 解码器要和 PIL 对得上（Node 缺失时跳过）。"""
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("没有 node，跳过 JS TIFF 解码器校验")
    engine = os.path.join(ROOT, "assets", "js", "tiff.js")
    tif = os.path.join(EXAMPLES, "example_terminal.tif")
    script = tmp_path / "check.mjs"
    script.write_text(
        "import fs from 'node:fs';\n"
        "const { pathToFileURL } = await import('node:url');\n"
        "const mod = await import(pathToFileURL(process.argv[2]).href);\n"
        "const b = fs.readFileSync(process.argv[3]);\n"
        "const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);\n"
        "const d = await mod.decodeTiff(ab);\n"
        "process.stdout.write(JSON.stringify({ w: d.width, h: d.height, px: Array.from(d.data) }));\n",
        encoding="utf-8")
    out = subprocess.check_output([node, str(script), engine, tif], timeout=120)
    dec = json.loads(out.decode("utf-8"))
    ref = np.asarray(Image.open(tif).convert("RGBA"), dtype=np.uint8)
    assert (dec["w"], dec["h"]) == (ref.shape[1], ref.shape[0])
    js = np.asarray(dec["px"], dtype=np.uint8).reshape(ref.shape)
    assert int(np.abs(js.astype(int) - ref.astype(int)).max()) <= 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_http_endpoints():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "app.py"),
                             "--port", str(port)], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):                     # 最多等 30 s（首次要加载模型）
            try:
                with urllib.request.urlopen(base + "/api/info", timeout=2) as r:
                    info = json.loads(r.read().decode("utf-8"))
                break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.fail("服务没起来")

        assert info["ok"] is True
        assert info["model"]["n_classes"] == 6

        with urllib.request.urlopen(base + "/", timeout=5) as r:
            html = r.read().decode("utf-8")
        assert "端部判读" in html and "assets/app.js" in html
        with urllib.request.urlopen(base + "/assets/js/engine.js", timeout=5) as r:
            assert b"export" in r.read()

        data = open(PHOTO, "rb").read()
        for mode, expect_multi in (("single", False), ("multi", True)):
            req = urllib.request.Request(f"{base}/api/analyze?mode={mode}", data=data,
                                         method="POST",
                                         headers={"Content-Type": "application/octet-stream"})
            with urllib.request.urlopen(req, timeout=300) as r:
                res = json.loads(r.read().decode("utf-8"))
            assert res["ok"] is True
            assert res["n_terminals"] >= 1
            t = res["terminals"][0]
            assert t["crop_png"].startswith("data:image/png;base64,")
            assert "geo" in t and "flags" in t
            if expect_multi:
                assert res["mode"] == "multi"
            else:
                assert res["mode"] == "single" and res["n_terminals"] == 1
                assert t["box"] is not None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
