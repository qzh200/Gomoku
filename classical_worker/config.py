"""Settings of the classical worker.

The connection to MyGo comes from the shared settings of ``aiworker``; the
search parameters live here, with a .env file in front of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from aiworker.settings import ConfigError, Connection

# The defaults of the search.
DEFAULT_DEPTH = 6
DEFAULT_WIDTH = 10
DEFAULT_TIME_BUDGET = 2.0

# The names of the environment variables of the search.
ENV_DEPTH = "CLASSICAL_DEPTH"
ENV_WIDTH = "CLASSICAL_WIDTH"
ENV_TIME_BUDGET = "CLASSICAL_TIME_BUDGET"

# The engine this worker plays for.
DEFAULT_ENGINE_ID = "classical"


@dataclass(frozen=True)
class Settings:
    """The parameters of one classical worker."""

    connection: Connection
    depth: int
    width: int
    time_budget: float

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        """Read the settings from the environment."""
        source = os.environ if env is None else env
        connection = Connection.read(source, DEFAULT_ENGINE_ID)

        def integer(name: str, fallback: int) -> int:
            raw = source.get(name, "").strip()
            if not raw:
                return fallback
            try:
                return int(raw)
            except ValueError as error:
                raise ConfigError(f"环境变量 {name} 不是整数: {raw!r}") from error

        def number(name: str, fallback: float) -> float:
            raw = source.get(name, "").strip()
            if not raw:
                return fallback
            try:
                return float(raw)
            except ValueError as error:
                raise ConfigError(f"环境变量 {name} 不是数字: {raw!r}") from error

        return cls(
            connection=connection,
            depth=integer(ENV_DEPTH, DEFAULT_DEPTH),
            width=integer(ENV_WIDTH, DEFAULT_WIDTH),
            time_budget=number(ENV_TIME_BUDGET, DEFAULT_TIME_BUDGET),
        )
