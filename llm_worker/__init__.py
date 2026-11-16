"""The LLM engine of MyGo.

The package plays the turns of a Gomoku game with a language model. It talks
to the MyGo API over the worker protocol of ``aiworker`` and to the model over
the OpenAI chat completions protocol.

The parts are separate on purpose:

* ``config`` reads the parameters of the model from the environment;
* ``board`` draws the board and reads the coordinates of a move;
* ``openai_client`` speaks the OpenAI protocol;
* ``engine`` asks the model for a move and checks the answer.

The turn itself (claim, lease, submit) lives in ``aiworker.protocol``.
"""

from .config import Settings
from .engine import EngineError, LLMEngine
from .openai_client import APIError, Answer, ChatMessage, OpenAIClient

__all__ = [
    "APIError",
    "Answer",
    "ChatMessage",
    "EngineError",
    "LLMEngine",
    "OpenAIClient",
    "Settings",
]
