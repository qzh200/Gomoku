"""The settings every MyGo worker shares.

The values come from the process environment; a ``.env`` file fills the
environment first. The bearer token of the engine is not expected in the
file: MyGo writes it to the runtime registry when it starts, and the worker
reads it from there.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# The address of a local MyGo and the file it writes the tokens to.
DEFAULT_MYGO_URL = "http://127.0.0.1:8080/api/v1"
DEFAULT_RUNTIME_CONFIG = "configs/engines.runtime.json"
DEFAULT_WAIT_SECONDS = 20
DEFAULT_EXTEND_SECONDS = 120

# The names of the environment variables of the shared settings.
ENV_MYGO_URL = "MYGO_URL"
ENV_ENGINE_TOKEN = "MYGO_ENGINE_TOKEN"
ENV_ENGINE_ID = "MYGO_ENGINE_ID"
ENV_WORKER_ID = "MYGO_WORKER_ID"
ENV_RUNTIME_CONFIG = "MYGO_RUNTIME_CONFIG"
ENV_WAIT_SECONDS = "MYGO_WAIT_SECONDS"
ENV_EXTEND_SECONDS = "MYGO_LEASE_EXTEND_SECONDS"


class ConfigError(RuntimeError):
    """A setting is missing or cannot be read."""


def parse_dotenv(text: str) -> dict[str, str]:
    """Read the ``KEY=VALUE`` pairs of a .env file.

    Blank lines and lines starting with ``#`` are ignored. A value may be
    quoted; the quotes are removed. ``export`` in front of a key is allowed.
    """
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
    """Fill the environment from a .env file.

    The environment wins over the file, so a parameter given on the command
    line of the shell is not overwritten.
    """
    if not path.is_file():
        return {}
    values = parse_dotenv(path.read_text(encoding="utf-8"))
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


def worker_dotenv_candidates(package: str, explicit: str | None) -> list[Path]:
    """Return the .env files of one worker package, in order."""
    if explicit:
        return [Path(explicit)]
    here = Path(__file__).resolve().parent.parent
    return [here / package / ".env", here / ".env", Path.cwd() / ".env"]


def runtime_config_candidates(explicit: str | None = None) -> list[Path]:
    """Return the files MyGo writes the engine tokens to, in order."""
    if explicit:
        return [Path(explicit)]
    root = Path(__file__).resolve().parent.parent
    return [root / DEFAULT_RUNTIME_CONFIG, Path.cwd() / DEFAULT_RUNTIME_CONFIG]


def read_runtime_token(path: Path, engine_id: str) -> str:
    """Read the bearer token of an engine from the runtime registry.

    MyGo writes that file when it starts, so the worker learns its token
    without anybody copying it. An empty string means the file holds no token
    for the engine.
    """
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
    """Return the token of the engine, from the runtime registry."""
    for candidate in runtime_config_candidates(explicit):
        token = read_runtime_token(candidate, engine_id)
        if token:
            return token
    return ""


def resolve_engine_token(source: dict[str, str], engine_id: str) -> str:
    """Return the token of the engine, or raise when there is none.

    An explicit ``MYGO_ENGINE_TOKEN`` wins; otherwise the runtime registry of
    MyGo is read, which is what makes the token automatic.
    """
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
    """How a worker reaches MyGo."""

    mygo_url: str
    engine_token: str
    engine_id: str
    worker_id: str
    wait_seconds: int
    extend_seconds: int

    @classmethod
    def read(cls, source: dict[str, str], default_engine_id: str) -> "Connection":
        """Read the connection settings of a worker."""

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
