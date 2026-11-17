from __future__ import annotations

import os
from dataclasses import dataclass

from aiworker.settings import ConfigError, Connection


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT = 60.0
DEFAULT_ATTEMPTS = 2


ENV_BASE_URL = "OPENAI_BASE_URL"
ENV_API_KEY = "OPENAI_API_KEY"
ENV_MODEL = "OPENAI_MODEL"
ENV_TEMPERATURE = "OPENAI_TEMPERATURE"
ENV_MAX_TOKENS = "OPENAI_MAX_TOKENS"
ENV_TIMEOUT = "OPENAI_TIMEOUT"
ENV_ATTEMPTS = "OPENAI_MAX_ATTEMPTS"
ENV_DISABLE_THINKING = "OPENAI_DISABLE_THINKING"


DEFAULT_ENGINE_ID = "llm"


@dataclass(frozen=True)
class Settings:
    mygo_url: str
    engine_token: str
    worker_id: str
    wait_seconds: int
    extend_seconds: int
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    max_attempts: int
    disable_thinking: bool

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        source = os.environ if env is None else env
        connection = Connection.read(source, DEFAULT_ENGINE_ID)

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

        api_key = source.get(ENV_API_KEY, "").strip()
        if not api_key:
            raise ConfigError(f"缺少 {ENV_API_KEY}：它是模型服务的 API Key")

        return cls(
            mygo_url=connection.mygo_url,
            engine_token=connection.engine_token,
            worker_id=connection.worker_id,
            wait_seconds=connection.wait_seconds,
            extend_seconds=connection.extend_seconds,
            base_url=text(ENV_BASE_URL, DEFAULT_BASE_URL).rstrip("/"),
            api_key=api_key,
            model=text(ENV_MODEL, DEFAULT_MODEL),
            temperature=number(ENV_TEMPERATURE, DEFAULT_TEMPERATURE),
            max_tokens=integer(ENV_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            timeout=number(ENV_TIMEOUT, DEFAULT_TIMEOUT),
            max_attempts=integer(ENV_ATTEMPTS, DEFAULT_ATTEMPTS),
            disable_thinking=flag(ENV_DISABLE_THINKING),
        )
