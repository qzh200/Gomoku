"""The classical engine: negamax, alpha-beta and a greedy ordering.

The engine owns no state of the game: it reads the board of a turn, searches
it and answers a cell. The turn itself is played by ``aiworker.protocol``.
"""

from __future__ import annotations

from .position import Position
from .search import Searcher


class EngineError(RuntimeError):
    """The engine could not find a move."""


class ClassicalEngine:
    """Picks a move with a negamax search over the greedy candidate moves."""

    def __init__(self, depth: int = 6, width: int = 10, time_budget: float = 2.0) -> None:
        self.depth = depth
        self.width = width
        self.time_budget = time_budget

    def choose_move(
        self,
        board: list[list[int]],
        history: list[dict],
        player: int,
        last_move: dict | None = None,
        config: dict | None = None,
    ) -> tuple[int, int]:
        """Return the row and the column of the move to play.

        The settings of the game win over the ones of the worker, so a game
        may ask for a deeper search without restarting the process.
        """
        settings = config or {}
        searcher = Searcher(
            depth=int(settings.get("depth", self.depth)),
            width=int(settings.get("width", self.width)),
            time_budget=float(settings.get("time_budget", self.time_budget)),
        )
        move = searcher.best_move(Position(board), player)
        if move is None:
            raise EngineError("棋盘已满，没有可落子的位置")
        return move
