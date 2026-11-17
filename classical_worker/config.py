from __future__ import annotations

import os
from dataclasses import dataclass

from aiworker.settings import ConfigError, Connection


DEFAULT_DEPTH = 8
DEFAULT_WIDTH = 12
DEFAULT_TIME_BUDGET = 2.0
DEFAULT_VCF_BUDGET = 0.10
DEFAULT_TT_ENTRIES = 200_000


ENV_DEPTH = "CLASSICAL_DEPTH"
ENV_WIDTH = "CLASSICAL_WIDTH"
ENV_TIME_BUDGET = "CLASSICAL_TIME_BUDGET"
ENV_VCF_BUDGET = "CLASSICAL_VCF_BUDGET"
ENV_TT_ENTRIES = "CLASSICAL_TT_ENTRIES"


DEFAULT_ENGINE_ID = "classical"


@dataclass(frozen=True)
class Settings:
    connection: Connection
    depth: int
    width: int
    time_budget: float
    vcf_budget: float
    tt_entries: int

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
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
            vcf_budget=number(ENV_VCF_BUDGET, DEFAULT_VCF_BUDGET),
            tt_entries=integer(ENV_TT_ENTRIES, DEFAULT_TT_ENTRIES),
        )
