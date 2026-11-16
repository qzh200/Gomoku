"""The LLM engine of MyGo.

The package plays the turns of a Gomoku game with a language model. It talks
to the MyGo API over the worker protocol and to the model over the OpenAI
chat completions protocol.

The parts are separate on purpose:

* ``config`` reads the parameters of the engine from the environment;
* ``board`` draws the board and reads the coordinates of a move;
* ``openai_client`` speaks the OpenAI protocol;
* ``engine`` asks the model for a move and checks the answer;
* ``worker`` plays the turns the server hands out.
"""

from .config import ConfigError, Settings, load_dotenv, parse_dotenv
from .engine import EngineError, LLMEngine
from .openai_client import APIError, ChatMessage, OpenAIClient
from .worker import Claim, LeaseKeeper, MyGoClient, Worker, WorkerError

__all__ = [
    "APIError",
    "ChatMessage",
    "Claim",
    "ConfigError",
    "EngineError",
    "LeaseKeeper",
    "LLMEngine",
    "MyGoClient",
    "OpenAIClient",
    "Settings",
    "Worker",
    "WorkerError",
    "load_dotenv",
    "parse_dotenv",
]
