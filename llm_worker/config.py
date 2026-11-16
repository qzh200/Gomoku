"""Settings of the LLM worker.

The settings come from the process environment. A ``.env`` file fills the
environment first, so the parameters of the engine live in one place instead
of in the command line.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The defaults of the settings. The engine runs against a local MyGo and the
# public OpenAI endpoint until the .env file says otherwise.
DEFAULT_MYGO_URL = "http://127.0.0.1:8080/api/v1"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 200
DEFAULT_TIMEOUT = 60.0
DEFAULT_WAIT_SECONDS = 20
DEFAULT_EXTEND_SECONDS = 120
DEFAULT_ATTEMPTS = 2

# The names of the environment variables of every setting.
ENV_MYGO_URL = "MYGO_URL"
ENV_ENGINE_TOKEN = "MYGO_ENGINE_TOKEN"
ENV_WORKER_ID = "MYGO_WORKER_ID"
ENV_BASE_URL = "OPENAI_BASE_URL"
ENV_API_KEY = "OPENAI_API_KEY"
ENV_MODEL = "OPENAI_MODEL"
ENV_TEMPERATURE = "OPENAI_TEMPERATURE"
ENV_MAX_TOKENS = "OPENAI_MAX_TOKENS"
ENV_TIMEOUT = "OPENAI_TIMEOUT"
ENV_WAIT_SECONDS = "MYGO_WAIT_SECONDS"
ENV_EXTEND_SECONDS = "MYGO_LEASE_EXTEND_SECONDS"
ENV_ATTEMPTS = "OPENAI_MAX_ATTEMPTS"


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


@dataclass(frozen=True)
class Settings:
    """The parameters of one LLM worker."""

    mygo_url: str
    engine_token: str
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

        token = source.get(ENV_ENGINE_TOKEN, "").strip()
        if not token:
            raise ConfigError(
                f"缺少 {ENV_ENGINE_TOKEN}：它是 llm 引擎的 Bearer Token，见 configs/engines.json"
            )
        api_key = source.get(ENV_API_KEY, "").strip()
        if not api_key:
            raise ConfigError(f"缺少 {ENV_API_KEY}：它是模型服务的 API Key")

        return cls(
            mygo_url=text(ENV_MYGO_URL, DEFAULT_MYGO_URL).rstrip("/"),
            engine_token=token,
            worker_id=text(ENV_WORKER_ID, "llm-worker-01"),
            base_url=text(ENV_BASE_URL, DEFAULT_BASE_URL).rstrip("/"),
            api_key=api_key,
            model=text(ENV_MODEL, DEFAULT_MODEL),
            temperature=number(ENV_TEMPERATURE, DEFAULT_TEMPERATURE),
            max_tokens=integer(ENV_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            timeout=number(ENV_TIMEOUT, DEFAULT_TIMEOUT),
            wait_seconds=integer(ENV_WAIT_SECONDS, DEFAULT_WAIT_SECONDS),
            extend_seconds=integer(ENV_EXTEND_SECONDS, DEFAULT_EXTEND_SECONDS),
            max_attempts=integer(ENV_ATTEMPTS, DEFAULT_ATTEMPTS),
        )
