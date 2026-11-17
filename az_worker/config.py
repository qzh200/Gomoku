from __future__ import annotations

import os
from dataclasses import dataclass

from aiworker.settings import ConfigError, Connection


DEFAULT_MODEL_PATH = "models/az/current.pt"
DEFAULT_DEVICE = "auto"
DEFAULT_SIMULATIONS = 100_000
DEFAULT_BATCH_SIZE = 32
DEFAULT_TIME_BUDGET = 2.0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_VCF_BUDGET = 0.25
DEFAULT_VCT_BUDGET = 0.25
DEFAULT_SOLVER_DEPTH = 10
DEFAULT_SOLVER_NODES = 4_000
DEFAULT_CANDIDATE_RADIUS = 0


ENV_MODEL_PATH = "AZ_MODEL_PATH"
ENV_DEVICE = "AZ_DEVICE"
ENV_SIMULATIONS = "AZ_SIMULATIONS"
ENV_BATCH_SIZE = "AZ_BATCH_SIZE"
ENV_TIME_BUDGET = "AZ_TIME_BUDGET"
ENV_TEMPERATURE = "AZ_TEMPERATURE"
ENV_VCF_BUDGET = "AZ_VCF_BUDGET"
ENV_VCT_BUDGET = "AZ_VCT_BUDGET"
ENV_SOLVER_DEPTH = "AZ_SOLVER_DEPTH"
ENV_SOLVER_NODES = "AZ_SOLVER_NODES"
ENV_CANDIDATE_RADIUS = "AZ_CANDIDATE_RADIUS"


DEFAULT_ENGINE_ID = "az"


@dataclass(frozen=True)
class Settings:
    connection: Connection
    model_path: str
    device: str
    simulations: int
    batch_size: int
    time_budget: float
    temperature: float
    vcf_budget: float
    vct_budget: float
    solver_depth: int
    solver_nodes: int
    candidate_radius: int

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
            batch_size=integer(ENV_BATCH_SIZE, DEFAULT_BATCH_SIZE),
            time_budget=number(ENV_TIME_BUDGET, DEFAULT_TIME_BUDGET),
            temperature=number(ENV_TEMPERATURE, DEFAULT_TEMPERATURE),
            vcf_budget=number(ENV_VCF_BUDGET, DEFAULT_VCF_BUDGET),
            vct_budget=number(ENV_VCT_BUDGET, DEFAULT_VCT_BUDGET),
            solver_depth=integer(ENV_SOLVER_DEPTH, DEFAULT_SOLVER_DEPTH),
            solver_nodes=integer(ENV_SOLVER_NODES, DEFAULT_SOLVER_NODES),
            candidate_radius=integer(ENV_CANDIDATE_RADIUS, DEFAULT_CANDIDATE_RADIUS),
        )
