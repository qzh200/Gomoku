from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MYGO_URL = "http://127.0.0.1:8080/api/v1"
DEFAULT_RUNTIME_CONFIG = "configs/engines.runtime.json"
DEFAULT_WAIT_SECONDS = 20
DEFAULT_EXTEND_SECONDS = 120


ENV_MYGO_URL = "MYGO_URL"
ENV_ENGINE_TOKEN = "MYGO_ENGINE_TOKEN"
ENV_ENGINE_ID = "MYGO_ENGINE_ID"
ENV_WORKER_ID = "MYGO_WORKER_ID"
ENV_RUNTIME_CONFIG = "MYGO_RUNTIME_CONFIG"
ENV_WAIT_SECONDS = "MYGO_WAIT_SECONDS"
ENV_EXTEND_SECONDS = "MYGO_LEASE_EXTEND_SECONDS"


class ConfigError(RuntimeError):
    pass


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        key, separator, value = stripped.partition("=")
        if not separator:
            raise ConfigError(f"第 {number} 行不是 KEY=VALUE 的形式: {line!r}")
        key = key.strip()
        if not key:
            raise ConfigError(f"第 {number} 行的键为空: {line!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = parse_dotenv(path.read_text(encoding="utf-8"))
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


def worker_dotenv_candidates(package: str, explicit: str | None) -> list[Path]:
    if explicit:
        return [Path(explicit)]
    here = Path(__file__).resolve().parent.parent
    return [here / package / ".env", here / ".env", Path.cwd() / ".env"]


def runtime_config_candidates(explicit: str | None = None) -> list[Path]:
    if explicit:
        return [Path(explicit)]
    root = Path(__file__).resolve().parent.parent
    return [root / DEFAULT_RUNTIME_CONFIG, Path.cwd() / DEFAULT_RUNTIME_CONFIG]


def read_runtime_token(path: Path, engine_id: str) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    engines = payload.get("engines")
    if not isinstance(engines, list):
        return ""
    for engine in engines:
        if not isinstance(engine, dict):
            continue
        if engine.get("engine_id") != engine_id:
            continue
        token = engine.get("token")
        if isinstance(token, str) and token.strip():
            return token.strip()
    return ""


def find_engine_token(engine_id: str, explicit: str | None = None) -> str:
    for candidate in runtime_config_candidates(explicit):
        token = read_runtime_token(candidate, engine_id)
        if token:
            return token
    return ""


def resolve_engine_token(source: dict[str, str], engine_id: str) -> str:
    token = source.get(ENV_ENGINE_TOKEN, "").strip()
    if token:
        return token
    token = find_engine_token(engine_id, source.get(ENV_RUNTIME_CONFIG, "").strip() or None)
    if token:
        return token
    raise ConfigError(
        f"拿不到 {engine_id} 引擎的 Bearer Token："
        f"先启动 MyGo（它会写 {DEFAULT_RUNTIME_CONFIG}），"
        f"或直接设置 {ENV_ENGINE_TOKEN}"
    )


@dataclass(frozen=True)
class Connection:
    mygo_url: str
    engine_token: str
    engine_id: str
    worker_id: str
    wait_seconds: int
    extend_seconds: int

    @classmethod
    def read(cls, source: dict[str, str], default_engine_id: str) -> "Connection":
        def text(name: str, fallback: str) -> str:
            value = source.get(name, "").strip()
            return value or fallback

        def integer(name: str, fallback: int) -> int:
            raw = source.get(name, "").strip()
            if not raw:
                return fallback
            try:
                return int(raw)
            except ValueError as error:
                raise ConfigError(f"环境变量 {name} 不是整数: {raw!r}") from error

        engine_id = text(ENV_ENGINE_ID, default_engine_id)
        return cls(
            mygo_url=text(ENV_MYGO_URL, DEFAULT_MYGO_URL).rstrip("/"),
            engine_token=resolve_engine_token(source, engine_id),
            engine_id=engine_id,
            worker_id=text(ENV_WORKER_ID, f"{engine_id}-worker-01"),
            wait_seconds=integer(ENV_WAIT_SECONDS, DEFAULT_WAIT_SECONDS),
            extend_seconds=integer(ENV_EXTEND_SECONDS, DEFAULT_EXTEND_SECONDS),
        )
