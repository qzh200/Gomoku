from __future__ import annotations

import argparse
import sys

from aiworker import ConfigError, MyGoClient, Worker, load_dotenv, worker_dotenv_candidates

from .config import Settings
from .engine import LLMEngine
from .openai_client import OpenAIClient


PACKAGE = "llm_worker"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MyGo LLM 引擎 Worker")
    parser.add_argument(
        "--env-file",
        default=None,
        help="参数文件路径，默认按 llm_worker/.env、.env、当前目录/.env 依次查找",
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

    client = OpenAIClient(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        disable_thinking=settings.disable_thinking,
    )
    engine = LLMEngine(client, max_attempts=settings.max_attempts)
    worker = Worker(
        MyGoClient(settings.mygo_url, settings.engine_token),
        engine,
        settings.worker_id,
        wait_seconds=settings.wait_seconds,
        extend_seconds=settings.extend_seconds,
    )
    try:
        worker.run()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
