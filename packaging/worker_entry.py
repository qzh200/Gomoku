"""引擎运行时入口：GomokuWorkers.exe --worker <name> 时运行对应引擎。

源码包内包装器 / 发行包内 exe 都通过 `--worker` 选择引擎，因此启动器只需
在 `exec -a` 不可用的 Windows 上多传一个参数，不必为每个引擎单独打包。
"""

from __future__ import annotations

import sys

WORKERS = {
    "classical": "classical_worker.__main__",
    "llm": "llm_worker.__main__",
    "az": "az_worker.__main__",
}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] != "--worker":
        print("用法: GomokuWorkers.exe --worker {classical|llm|az} [--env-file PATH]", file=sys.stderr)
        return 2
    name = argv[2] if len(argv) > 2 else ""
    module = WORKERS.get(name)
    if module is None:
        print(f"未知的引擎: {name!r}", file=sys.stderr)
        return 2
    rest = argv[3:]
    import importlib

    entry = importlib.import_module(module)
    return int(entry.main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
