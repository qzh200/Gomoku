"""The entry point of the classical worker.

Usage:
    python -m classical_worker
    python -m classical_worker --env-file classical_worker/.env
"""

from __future__ import annotations

import argparse
import sys

from aiworker import ConfigError, MyGoClient, Worker, load_dotenv, worker_dotenv_candidates

from .config import Settings
from .engine import ClassicalEngine

# The name of the package, used to find its .env file.
PACKAGE = "classical_worker"


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the command line."""
    parser = argparse.ArgumentParser(description="MyGo 传统算法 Worker（窗口棋型 + PVS + VCF）")
    parser.add_argument(
        "--env-file",
        default=None,
        help="参数文件路径，默认按 classical_worker/.env、.env、当前目录/.env 依次查找",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Read the parameters and play the turns until the process is stopped."""
    args = build_parser().parse_args(argv)
    for candidate in worker_dotenv_candidates(PACKAGE, args.env_file):
        load_dotenv(candidate)
        if candidate.is_file():
            print(f"参数文件: {candidate}", flush=True)
            break
    else:
        if args.env_file:
            print(f"找不到参数文件: {args.env_file}", file=sys.stderr)
            return 2

    try:
        settings = Settings.from_env()
    except ConfigError as error:
        print(f"参数有误: {error}", file=sys.stderr)
        return 2

    engine = ClassicalEngine(
        depth=settings.depth,
        width=settings.width,
        time_budget=settings.time_budget,
        vcf_budget=settings.vcf_budget,
        vct_budget=settings.vct_budget,
        tt_entries=settings.tt_entries,
    )
    worker = Worker(
        MyGoClient(settings.connection.mygo_url, settings.connection.engine_token),
        engine,
        settings.connection.worker_id,
        wait_seconds=settings.connection.wait_seconds,
        extend_seconds=settings.connection.extend_seconds,
    )
    try:
        worker.run()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
