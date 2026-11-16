from __future__ import annotations

from .position import Position
from .search import Searcher


class EngineError(RuntimeError):
    pass


class ClassicalEngine:
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
