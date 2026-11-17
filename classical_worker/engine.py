from __future__ import annotations

from .config import (
    DEFAULT_DEPTH,
    DEFAULT_TIME_BUDGET,
    DEFAULT_TT_ENTRIES,
    DEFAULT_VCF_BUDGET,
    DEFAULT_WIDTH,
)
from .position import Position
from .search import Searcher


class EngineError(RuntimeError):
    pass


class ClassicalEngine:
    def __init__(
        self,
        depth: int = DEFAULT_DEPTH,
        width: int = DEFAULT_WIDTH,
        time_budget: float = DEFAULT_TIME_BUDGET,
        vcf_budget: float = DEFAULT_VCF_BUDGET,
        tt_entries: int = DEFAULT_TT_ENTRIES,
    ) -> None:
        self.depth = depth
        self.width = width
        self.time_budget = time_budget
        self.searcher = Searcher(
            depth=depth,
            width=width,
            time_budget=time_budget,
            vcf_budget=vcf_budget,
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
        # history / last_move 是 Worker 协议要求的参数，本引擎用不到：棋局状态全在 board 里。
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
