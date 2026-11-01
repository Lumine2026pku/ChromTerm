#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ChromTerm 启动器。

    python app.py                      # http://127.0.0.1:8770
    python app.py --open               # 顺便打开浏览器
    python app.py --port 9000          # 换端口
    python app.py --host 0.0.0.0       # 局域网可访问（默认只监听本机）

不需要安装成包：脚本会把仓库根目录加进 sys.path。
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chromterm.webapp import main  # noqa: E402

if __name__ == "__main__":
    main()
