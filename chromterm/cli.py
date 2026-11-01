# -*- coding: utf-8 -*-
"""命令行入口：安装后 `chromterm --open` 等价于 `python app.py --open`。"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    from .webapp import main as server_main

    # webapp.main() 自己解析 argparse；这里只把参数转发过去
    if argv is not None:
        sys.argv = [sys.argv[0]] + list(argv)
    server_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
