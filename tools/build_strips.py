#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_strips.py —— 把端部窗口按中线拉直成 48×96 规范化长条并缓存。

产物：`_cache/strips.npz`

用法：python tools/build_strips.py [--limit N] [--long-edge 320]
"""
import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromterm.build_strips import build  # noqa: E402
from chromterm.common import setup_console  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="拉直成规范化长条并缓存")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个窗口（调试用）")
    args = ap.parse_args()
    setup_console()
    build(limit=args.limit or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
