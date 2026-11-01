#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_profiles.py —— 对 961 个端部窗口跑一遍沿中线采样并缓存。

产物：`_cache/profiles.npz`、`_cache/profiles_bands.csv`、`_cache/profiles.csv`

用法：python tools/build_profiles.py [--limit N]
"""
import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromterm.build_profiles import build  # noqa: E402
from chromterm.common import setup_console  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="沿中线采样并缓存")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个窗口（调试用）")
    args = ap.parse_args()
    setup_console()
    build(limit=args.limit or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
