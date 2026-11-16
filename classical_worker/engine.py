"""The classical engine: iterative deepening PVS, threats and forced wins.

The engine owns no state of the game: it reads the board of a turn, searches
it and answers a cell. The turn itself is played by ``aiworker.protocol``.
"""

from __future__ import annotations

from .position import Position
from .search import Searcher


class EngineError(RuntimeError):
    """The engine could not find a move."""


class ClassicalEngine:
    """Picks a move with a principal variation search over threat-ordered moves."""

    def __init__(
        self,
        depth: int = 8,
        width: int = 12,
        time_budget: float = 2.0,
        vcf_budget: float = 0.10,
        vct_budget: float = 0.0,
        tt_entries: int = 200_000,
    ) -> None:
        self.depth = depth
        self.width = width
        self.time_budget = time_budget
        self.searcher = Searcher(
            depth=depth,
            width=width,
            time_budget=time_budget,
            vcf_budget=vcf_budget,
            vct_budget=vct_budget,
            tt_entries=tt_entries,
        )

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
        move = self.searcher.best_move(
            Position(board),
            player,
            depth=int(settings.get("depth", self.depth)),
            width=int(settings.get("width", self.width)),
            time_budget=float(settings.get("time_budget", self.time_budget)),
        )
        if move is None:
            raise EngineError("棋盘已满，没有可落子的位置")
        return move
