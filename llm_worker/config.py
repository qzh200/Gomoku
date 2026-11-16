"""Settings of the LLM worker.

The settings come from the process environment. A ``.env`` file fills the
environment first, so the parameters of the engine live in one place instead
of in the command line.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# The defaults of the settings. The engine runs against a local MyGo and the
# public OpenAI endpoint until the .env file says otherwise.
DEFAULT_MYGO_URL = "http://127.0.0.1:8080/api/v1"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT = 60.0
DEFAULT_WAIT_SECONDS = 20
DEFAULT_EXTEND_SECONDS = 120
DEFAULT_ATTEMPTS = 2

# The names of the environment variables of every setting.
ENV_MYGO_URL = "MYGO_URL"
ENV_ENGINE_TOKEN = "MYGO_ENGINE_TOKEN"
ENV_ENGINE_ID = "MYGO_ENGINE_ID"
ENV_WORKER_ID = "MYGO_WORKER_ID"
ENV_RUNTIME_CONFIG = "MYGO_RUNTIME_CONFIG"
ENV_BASE_URL = "OPENAI_BASE_URL"
ENV_API_KEY = "OPENAI_API_KEY"
ENV_MODEL = "OPENAI_MODEL"
ENV_TEMPERATURE = "OPENAI_TEMPERATURE"
ENV_MAX_TOKENS = "OPENAI_MAX_TOKENS"
ENV_TIMEOUT = "OPENAI_TIMEOUT"
ENV_WAIT_SECONDS = "MYGO_WAIT_SECONDS"
ENV_EXTEND_SECONDS = "MYGO_LEASE_EXTEND_SECONDS"
ENV_ATTEMPTS = "OPENAI_MAX_ATTEMPTS"
ENV_DISABLE_THINKING = "OPENAI_DISABLE_THINKING"

# The engine this worker plays for, and the file MyGo writes the tokens to.
DEFAULT_ENGINE_ID = "llm"
DEFAULT_RUNTIME_CONFIG = "configs/engines.runtime.json"


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


def dotenv_candidates(explicit: str | None) -> list[Path]:
    """Return the .env files to try, in order."""
    if explicit:
        return [Path(explicit)]
    here = Path(__file__).resolve().parent
    return [here / ".env", here.parent / ".env", Path.cwd() / ".env"]


def runtime_config_candidates(explicit: str | None = None) -> list[Path]:
    """Return the files MyGo writes the engine tokens to, in order."""
    if explicit:
        return [Path(explicit)]
    here = Path(__file__).resolve().parent
    return [
        here.parent / DEFAULT_RUNTIME_CONFIG,
        Path.cwd() / DEFAULT_RUNTIME_CONFIG,
    ]


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


@dataclass(frozen=True)
class Settings:
    """The parameters of one LLM worker."""

    mygo_url: str
    engine_token: str
    engine_id: str
    worker_id: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    wait_seconds: int
    extend_seconds: int
    max_attempts: int
    disable_thinking: bool

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        """Read the settings from the environment."""
        source = os.environ if env is None else env

        def text(name: str, fallback: str) -> str:
            value = source.get(name, "").strip()
            return value or fallback

        def number(name: str, fallback: float) -> float:
            raw = source.get(name, "").strip()
            if not raw:
                return fallback
            try:
                return float(raw)
            except ValueError as error:
                raise ConfigError(f"环境变量 {name} 不是数字: {raw!r}") from error

        def integer(name: str, fallback: int) -> int:
            raw = source.get(name, "").strip()
            if not raw:
                return fallback
            try:
                return int(raw)
            except ValueError as error:
                raise ConfigError(f"环境变量 {name} 不是整数: {raw!r}") from error

        def flag(name: str) -> bool:
            return text(name, "").lower() in ("1", "true", "yes", "on")

        engine_id = text(ENV_ENGINE_ID, DEFAULT_ENGINE_ID)
        token = source.get(ENV_ENGINE_TOKEN, "").strip()
        if not token:
            # The token comes from the file MyGo wrote when it started.
            token = find_engine_token(
                engine_id,
                source.get(ENV_RUNTIME_CONFIG, "").strip() or None,
            )
        if not token:
            raise ConfigError(
                f"拿不到 {engine_id} 引擎的 Bearer Token："
                f"先启动 MyGo（它会写 {DEFAULT_RUNTIME_CONFIG}），"
                f"或直接设置 {ENV_ENGINE_TOKEN}"
            )
        api_key = source.get(ENV_API_KEY, "").strip()
        if not api_key:
            raise ConfigError(f"缺少 {ENV_API_KEY}：它是模型服务的 API Key")

        return cls(
            mygo_url=text(ENV_MYGO_URL, DEFAULT_MYGO_URL).rstrip("/"),
            engine_token=token,
            engine_id=engine_id,
            worker_id=text(ENV_WORKER_ID, f"{engine_id}-worker-01"),
            base_url=text(ENV_BASE_URL, DEFAULT_BASE_URL).rstrip("/"),
            api_key=api_key,
            model=text(ENV_MODEL, DEFAULT_MODEL),
            temperature=number(ENV_TEMPERATURE, DEFAULT_TEMPERATURE),
            max_tokens=integer(ENV_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            timeout=number(ENV_TIMEOUT, DEFAULT_TIMEOUT),
            wait_seconds=integer(ENV_WAIT_SECONDS, DEFAULT_WAIT_SECONDS),
            extend_seconds=integer(ENV_EXTEND_SECONDS, DEFAULT_EXTEND_SECONDS),
            max_attempts=integer(ENV_ATTEMPTS, DEFAULT_ATTEMPTS),
            disable_thinking=flag(ENV_DISABLE_THINKING),
        )
