# ChromTerm · 果蝇唾腺染色体端部判读

**上传一张端部小图（推荐）或整片显微照片 → 找出染色体末端 → 给出 6 类（2L / 2R / 3L / 3R / X / other）的概率、置信度与「最可能的结论」；没有可识别末端时明确输出「无」。**

**在线版（GitHub Pages，纯前端、浏览器内推理）**：`https://<你的用户名>.github.io/ChromTerm/`
**本地版（可选后端，支持整片多末端）**：`python app.py --open`

推理全部手写：**numpy + Pillow**（Python）/ **纯 JavaScript**（浏览器），
没有 torch / OpenCV / SciPy / scikit-learn，也没有 Flask / FastAPI，**克隆下来就能跑**。

![判读界面](docs/web_result.png)

---

## 目录

- [两种运行方式](#两种运行方式)
- [输入什么、输出什么](#输入什么输出什么)
- [快速开始](#快速开始)
- [发布到 GitHub Pages](#发布到-github-pages)
- [HTTP 接口（本地服务）](#http-接口本地服务)
- [模型与实测指标](#模型与实测指标)
- [目录结构](#目录结构)
- [训练 / 复现（可选）](#训练--复现可选)
- [常见问题](#常见问题)
- [局限](#局限)

## 两种运行方式

| | 静态模式（GitHub Pages） | 服务模式（本地 `python app.py`） |
|---|---|---|
| 推理在哪 | 浏览器里，`assets/js/engine.js` | 本地 Python，`chromterm/` |
| 单末端判读 | 支持 | 支持（Python 实现） |
| 整片多末端 | 不可用（Pages 只有静态文件） | 实验性：骨架端点找 1–6 个候选 |
| 几何量测（长度/臂宽/端部朝向） | 不提供，只做分类 | 提供 |
| 隐私 | 图片不出浏览器 | 图片不出本机 |

页面会自动探测 `/api/info`：**能连上后端就走服务模式，连不上就退回静态模式**（并禁用「整片多末端」）。

## 输入什么、输出什么

**输入**：JPG / PNG / TIF。推荐 **单个端部**（如 246×112 的小图）——整张当作一个末端，精度最高；
整片照片也能判读，但多条臂在染色中心粘连时容易选错，页面会明确提示。

**输出（JSON，界面只是它的可视化）**：

```jsonc
{
  "ok": true,
  "mode": "single",
  "n_terminals": 1,
  "image": {"w": 246, "h": 112},
  "verdict": {"cls": "2R", "conf": 0.836, "second": "other", "margin": 0.78,
              "terminal_id": 1, "reliable": true, "text": "整张图按一个末端判读"},
  "terminals": [{
      "id": 1,
      "x": 130.2, "y": 58.9,                            // 端部位置（原图坐标）
      "box": {"x0": 0, "y0": 0, "cw": 246, "ch": 111},  // 模型实际看到的窗口
      "quad": null,                                     // 多末端模式给四角多边形
      "cls": "2R", "conf": 0.836,
      "probs": [{"cls": "2R", "p": 0.836}],             // 6 类全给
      "second": "other", "conf2": 0.052, "margin": 0.784,
      "geo": {"length_px": 58.5, "width_px": 31.7, "tip_angle_deg": 108.2,
              "n_bands": 3.0, "coherence": 1.0, "prox_in_frame": true},
      "flags": ["近端不在画面内(长度不完整)"],
      "quality": "good",                                // good / fair / poor / cut
      "crop_png": "data:image/png;base64,..."           // 判读用的窗口（含叠加）
  }]
}
```

没有可识别末端时返回 `{"ok": false, "mode": "none", "n_terminals": 0, "reason": "..."}`，界面显示「判读结论：无」。

## 快速开始

要求 **Python ≥ 3.9**（只有服务模式需要；静态网页纯前端）。

```bash
git clone https://github.com/<your-name>/ChromTerm.git
cd ChromTerm
pip install -r requirements.txt      # numpy / pandas / Pillow
python app.py --open                 # → http://127.0.0.1:8770
```

只想看网页、不想装 Python：

```bash
python -m http.server 8000           # 浏览器打开 http://127.0.0.1:8000/
# 或者直接把仓库推到 GitHub，用 Pages 打开
```

> 直接双击 `index.html`（`file://`）会因浏览器的 ES module / CORS 限制而白屏，请用 http 方式打开。

页面上：

- 「示例：端部小图」 → `examples/example_terminal.jpg`，得到 2R（约 84%）
- 「示例：整片照片」 → `examples/example_photo.jpg`，单末端模式会提示「像整片照片」；服务模式下切「整片多末端」可得 6 个候选
- `?demo=term` / `?demo=photo` / `?demo=blank` 便于演示与截图自检；`?mode=multi` 直接进多末端

## 发布到 GitHub Pages

仓库根目录已经有 `index.html`，**不需要构建**。两种做法：

**A. 分支直接发布（最简单）**

1. 推送到 GitHub（`main` 分支）。
2. Settings → Pages → Build and deployment → Source 选 **Deploy from a branch** → 分支 `main`、目录 `/ (root)` → Save。
3. 一两分钟后打开 `https://<你的用户名>.github.io/ChromTerm/`。

**B. 用附带的工作流发布（只发布网页文件）**

仓库已包含 `.github/workflows/pages.yml`：把 `index.html + assets/ + examples/` 收进 `_site` 再上传，
不会把 Python 代码与文档一起发出去。使用前把 Settings → Pages → Source 改成 **GitHub Actions**。

## HTTP 接口（本地服务）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 前端页面（仓库根目录的 index.html） |
| GET | `/assets/<file>`、`/examples/<file>` | 前端、模型 JSON、示例图 |
| GET | `/api/info` | 模型信息 + 测试集指标 |
| GET | `/api/example` | 示例照片（`?kind=blank` 返回空白图，演示「无」） |
| POST | `/api/analyze?mode=single\|multi` | body 为图片字节 → 判读 JSON |

```bash
# 单末端（默认）
curl -s -X POST --data-binary @examples/example_terminal.jpg \
     -H "Content-Type: application/octet-stream" \
     "http://127.0.0.1:8770/api/analyze?mode=single"

# 整片多末端
curl -s -X POST --data-binary @examples/example_photo.jpg \
     -H "Content-Type: application/octet-stream" \
     "http://127.0.0.1:8770/api/analyze?mode=multi"
```

耗时参考（CPU）：端部小图 **0.1–0.3 s**（浏览器内约 40 ms 推理 + 读图）；1600×1335 整片照片 **3–5 s**。

## 模型与实测指标

**染色量 OD 图 → 6 类特征块（多尺度池化 / 带纹剖面 / 自相关 / 梯度方向直方图 / 粗细剖面 / 几何量测）
→ 多项逻辑回归（L2=1.0，836 维，无 PCA）**，权重 30 KB。

在 961 个端部窗口（432 张原图、5 个成像批次）上，官方 test（150 个窗口）：

| 指标 | Python（服务） | 浏览器（静态） |
|---|---|---|
| top-1 准确率 | **50.7%**（95% CI ±8 点） | **50.0%** |
| macro-F1 | **0.497** | 0.489 |
| top-2 准确率 | **66.0%** | — |
| 重复 5 折分组 CV（15 折） | macro-F1 **0.459 ± 0.039** | — |
| 留一来源批次（LOSO） | 平均 **0.533**（单批次 0.42–0.73） | — |
| 按置信度接受 60% / 20% 的窗口 | 63.3% / 76.7% | — |
| 人为上限（10 张对照图 × 8 人标注） | κ=0.67、集合完全一致 0.65 | — |

浏览器版与 Python 版在 150 个窗口上只差 **1 个样本**：整套特征（OD、闭运算、Pillow 式双线性缩放、
带纹剖面、HOG、自相关）都按 Python/Pillow 的语义逐行移植（见 `assets/js/engine.js` 顶部注释）。
浏览器版**不做几何量测块**（geo），用训练均值代入——实测 test 准确率 0.5133 vs 完整 0.5067，差在噪声内，
所以静态网页不需要移植骨架化那一整套。

**怎么读**：6 类单标签 + 单标注训练集本身带噪声（κ=0.67），50% 的 top-1 不是「模型不行」，
而是这个任务在现有标注下的真实水平；真正实用的是 **top-2 66% + 置信度拒识 + 人工确认**。
逐类看：2L 与 X 最好，2R/3L/3R 互相混淆最严重，other 最差（F1≈0.35）。

细节、消融与负结果见 [`docs/说明_模型改进.docx`](docs/说明_模型改进.docx)：
光密度归一化 +0.17 macro-F1；几何转正与端部锚定裁剪**不加分反而降点**；几何特征块贡献≈0；
随机特征 + 岭回归不如线性。

## 目录结构

```
ChromTerm/
├── index.html                GitHub Pages 入口（静态版，浏览器内推理）
├── assets/
│   ├── style.css  app.js     前端（原生，无构建）
│   ├── js/engine.js          浏览器端推理引擎（Python 推理链的 JS 移植）
│   └── model/model_best.json 网页用权重（836 维 + 指标，75 KB）
├── app.py                    本地服务启动器（python app.py）
├── chromterm/                Python 包：同一套推理 + 多末端 + HTTP 服务
│   ├── common.py prep_od.py geometry.py normalize.py features.py model.py
│   ├── detect_terminals.py   单 / 多末端流程
│   ├── pipeline.py           Predictor（加载模型、端到端）
│   ├── webapp.py             标准库 HTTP 服务
│   └── viz.py cli.py
├── models/
│   ├── model_best.npz        Python 权重（30 KB）
│   └── metrics.json          指标（界面与报告共用同一来源）
├── examples/                 示例图（整片 + 端部窗口）
├── docs/                     截图与完整报告
├── tools/                    训练 / 评估 / 体检 / 导出网页模型
└── tests/                    冒烟测试（含「网页资产自洽」检查）
```

## 训练 / 复现（可选）

推理不需要数据集；下面的脚本需要把数据集放到 `data_share/`（或用 `CHROMTERM_DATA` 指向别处），
格式与 `dataset/level1`、`dataset/det` 一致（YOLO 框 + 246×112 端部窗口）。

```bash
export CHROMTERM_DATA=/path/to/data_share      # Windows: set CHROMTERM_DATA=...
python tools/build_cache.py          # 预处理缓存（961 窗口，5–15 分钟）
python tools/train_classifier.py     # 消融 + 评估 + 写 models/model_best.npz（单线程 BLAS，可复现）
python tools/export_web_model.py     # 由 npz 生成 assets/model/model_best.json（网页用）
python tools/audit_dataset.py        # 数据体检与 8 人标注一致性
```

`tests/` 里的测试不依赖数据集（只用 `examples/`），`python -m pytest -q` 即可。

## 常见问题

**双击 index.html 白屏**：浏览器不允许 `file://` 下的 ES module，用 `python -m http.server` 或 Pages 打开。
**GitHub Pages 上「整片多末端」是灰的**：静态托管没有后端，只能跑单末端；整片需要本地 `python app.py`。
**标注框/点偶尔不显示**：已在 `assets/app.js` 里修掉——图片解码完成前不画（否则 SVG 尺寸为 0）、
换图先清空、用 `ResizeObserver` 同步 SVG 尺寸、大图按 `原图/推理图` 比例换算坐标。
**端口被占用**：`python app.py --port 9000`。
**一张图识别出很多候选但有的不像末端**：多末端是**启发式**（骨架端点 + 打分），不是训练出来的检测器。

## 局限

- **研究 / 教学用途**，不是医疗器械，不能用于诊断。
- 单末端是主路径；整片多末端为实验性候选，粘连 / 遮挡处会误检或漏检。
- 6 类之间本身难分（2R/3L/3R 在端部几个臂宽内高度相似），请以 top-2 + 人工确认为准。
- 跨批次泛化有限：LOSO 单批次最低 0.42。

## 致谢与引用

- 数据与标注口径来自项目组 8 位同学的 `data_share` 数据集与《端部图像处理流程说明》。
- 若用于发表，请引用本仓库与上述数据集说明。参见 `CITATION.cff`。

## License

MIT（见 [LICENSE](LICENSE)）。数据集不随仓库分发，其使用请遵循原提供方约定。
