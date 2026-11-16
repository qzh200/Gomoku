from __future__ import annotations

import os
from dataclasses import dataclass

from aiworker.settings import ConfigError, Connection


DEFAULT_MODEL_PATH = "models/az/current.pt"
DEFAULT_DEVICE = "auto"
DEFAULT_SIMULATIONS = 400
DEFAULT_C_PUCT = 1.5
DEFAULT_BATCH_SIZE = 32
DEFAULT_CANDIDATE_RADIUS = 1
DEFAULT_TEMPERATURE = 0.0


ENV_MODEL_PATH = "AZ_MODEL_PATH"
ENV_DEVICE = "AZ_DEVICE"
ENV_SIMULATIONS = "AZ_SIMULATIONS"
ENV_C_PUCT = "AZ_C_PUCT"
ENV_BATCH_SIZE = "AZ_BATCH_SIZE"
ENV_CANDIDATE_RADIUS = "AZ_CANDIDATE_RADIUS"
ENV_TEMPERATURE = "AZ_TEMPERATURE"


DEFAULT_ENGINE_ID = "az"


@dataclass(frozen=True)
class Settings:
    connection: Connection
    model_path: str
    device: str
    simulations: int
    c_puct: float
    batch_size: int
    candidate_radius: int
    temperature: float

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

        def text(name: str, fallback: str) -> str:
            return source.get(name, "").strip() or fallback

        return cls(
            connection=connection,
            model_path=text(ENV_MODEL_PATH, DEFAULT_MODEL_PATH),
            device=text(ENV_DEVICE, DEFAULT_DEVICE),
            simulations=integer(ENV_SIMULATIONS, DEFAULT_SIMULATIONS),
            c_puct=number(ENV_C_PUCT, DEFAULT_C_PUCT),
            batch_size=integer(ENV_BATCH_SIZE, DEFAULT_BATCH_SIZE),
            candidate_radius=integer(ENV_CANDIDATE_RADIUS, DEFAULT_CANDIDATE_RADIUS),
            temperature=number(ENV_TEMPERATURE, DEFAULT_TEMPERATURE),
        )
