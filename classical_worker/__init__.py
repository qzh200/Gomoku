"""The classical engine of MyGo: negamax, alpha-beta and a greedy ordering.

The package holds no protocol and no network code: it reads a board and
answers a cell.

The parts are separate on purpose:

* ``position`` keeps the board and the score of its runs, and plays and takes
  back stones;
* ``search`` is the negamax search with alpha-beta pruning and the greedy
  move ordering;
* ``engine`` turns the board of a turn into a move;
* ``config`` reads the parameters of the search from the environment.

The turn itself (claim, lease, submit) lives in ``aiworker.protocol``.
"""

from .config import Settings
from .engine import ClassicalEngine, EngineError
from .position import BOARD_SIZE, BLACK, EMPTY, WHITE, Position, coordinate_of, other
from .search import Searcher, SearchTimeout

__all__ = [
    "BLACK",
    "BOARD_SIZE",
    "ClassicalEngine",
    "EMPTY",
    "EngineError",
    "Position",
    "SearchTimeout",
    "Searcher",
    "Settings",
    "WHITE",
    "coordinate_of",
    "other",
]
