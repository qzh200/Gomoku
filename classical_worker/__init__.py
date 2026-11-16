"""The classical engine of MyGo: threat search, PVS and a transposition table.

The package holds no protocol and no network code: it reads a board and
answers a cell.

The parts are separate on purpose:

* ``patterns`` scores six-cell windows and knows the Gomoku shapes;
* ``position`` keeps the board, its window score and its Zobrist hash, and
  plays and takes back stones;
* ``threats`` classifies threats, produces forced moves and runs VCF/VCT;
* ``tt`` is the transposition table;
* ``search`` is the iterative deepening principal variation search;
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
