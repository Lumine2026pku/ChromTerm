# -*- coding: utf-8 -*-
"""build_cache.py —— 预处理缓存：对 961 个端部窗口跑 OD + 几何 + 三种归一化。

用法：
    python src/build_cache.py            # 全量（约 2–3 分钟）
    python src/build_cache.py --limit 40 # 先跑 40 个试一下

产物：_cache/level1_features.npz、_cache/level1_features.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromterm.common import ensure, CACHE, setup_console
from chromterm.data import build_cache


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个窗口（调试）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure(CACHE)
    build_cache(force=args.force, limit=args.limit)
    print("done")


if __name__ == "__main__":
    main()
