"""The worker protocol and the shared settings of the MyGo engines.

``protocol`` claims a turn, keeps its lease alive, submits the move and drops
a result that came too late. ``settings`` reads the parameters and the bearer
token that MyGo publishes.

An engine plugs into the protocol by answering ``choose_move``; the two
engines of this repository (``llm_worker`` and ``classical_worker``) do that.
"""

from .protocol import Claim, Engine, LeaseKeeper, MyGoClient, Worker, WorkerError
from .settings import (
    ConfigError,
    Connection,
    find_engine_token,
    load_dotenv,
    parse_dotenv,
    read_runtime_token,
    worker_dotenv_candidates,
)

__all__ = [
    "Claim",
    "ConfigError",
    "Connection",
    "Engine",
    "LeaseKeeper",
    "MyGoClient",
    "Worker",
    "WorkerError",
    "find_engine_token",
    "load_dotenv",
    "parse_dotenv",
    "read_runtime_token",
    "worker_dotenv_candidates",
]
