from __future__ import annotations

import argparse
import sys

from aiworker import ConfigError, MyGoClient, Worker, load_dotenv, worker_dotenv_candidates

from .config import Settings
from .engine import AlphaZeroEngine


PACKAGE = "az_worker"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MyGo AlphaZero Worker（神经网络策略 + MCTS）")
    parser.add_argument(
        "--env-file",
        default=None,
        help="参数文件路径，默认按 az_worker/.env、.env、当前目录/.env 依次查找",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
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

    try:
        engine = AlphaZeroEngine(
            model_path=settings.model_path,
            device=settings.device,
            simulations=settings.simulations,
            batch_size=settings.batch_size,
            time_budget=settings.time_budget,
            temperature=settings.temperature,
            vcf_budget=settings.vcf_budget,
            vct_budget=settings.vct_budget,
            solver_depth=settings.solver_depth,
            solver_nodes=settings.solver_nodes,
            radius=settings.candidate_radius,
        )
    except ValueError as error:
        print(f"权重不可用: {error}", file=sys.stderr)
        return 2
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
