"""The AlphaZero engine of MyGo: a network that scores the moves and a search.

The package has four parts:

* ``board`` and ``features``: the board and the planes the network reads;
* ``network``: the policy and value network;
* ``mcts``: the search over the moves;
* ``engine``: the ``choose_move`` the worker protocol expects.

``selfplay`` holds the training: the games the network plays against itself
and the batches it learns from.
"""

from .board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board, other
from .engine import AlphaZeroEngine, EngineError
from .features import PLANE_COUNT, encode, legal_mask
from .mcts import MCTS, Node, SearchResult
from .network import GomokuNet

__all__ = [
    "BLACK",
    "BOARD_SIZE",
    "EMPTY",
    "MCTS",
    "PLANE_COUNT",
    "WHITE",
    "AlphaZeroEngine",
    "Board",
    "EngineError",
    "GomokuNet",
    "Node",
    "SearchResult",
    "encode",
    "legal_mask",
    "other",
]
