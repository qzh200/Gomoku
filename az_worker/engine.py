from __future__ import annotations

import time
from pathlib import Path

import torch

from .board import Board, other
from .config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SIMULATIONS,
    DEFAULT_SOLVER_DEPTH,
    DEFAULT_SOLVER_NODES,
    DEFAULT_VCF_BUDGET,
    DEFAULT_VCT_BUDGET,
)
from .mcts import MCTS, row_col_of
from .network import GomokuNet, load_model
from .vcf import VCF, VCT, winning_move


DEFAULT_TEMPERATURE = 0.0


class EngineError(RuntimeError):
    pass


class AlphaZeroEngine:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: str = "auto",
        simulations: int = DEFAULT_SIMULATIONS,
        batch_size: int = 32,
        time_budget: float | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        vcf_budget: float = DEFAULT_VCF_BUDGET,
        vct_budget: float = DEFAULT_VCT_BUDGET,
        solver_depth: int = DEFAULT_SOLVER_DEPTH,
        solver_nodes: int = DEFAULT_SOLVER_NODES,
        seed: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.simulations = max(1, simulations)
        self.temperature = temperature
        self.time_budget = time_budget
        self.vcf_budget = max(0.0, vcf_budget)
        self.vct_budget = max(0.0, vct_budget)
        self.solver_depth = solver_depth
        self.solver_nodes = solver_nodes

        if self.model_path.is_file():
            self.model = load_model(str(model_path), self.device)
        else:
            self.model = GomokuNet().to(self.device)
        self.model.eval()

        self.searcher = MCTS(
            self.model,
            self.device,
            batch_size=batch_size,
            time_budget=time_budget,
            seed=seed,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def choose_move(
        self,
        board: list[list[int]],
        history: list[dict],
        player: int,
        last_move: dict | None = None,
        config: dict | None = None,
    ) -> tuple[int, int]:
        settings = config or {}
        started = time.monotonic()
        position = Board(board)
        if position.is_full():
            raise EngineError("棋盘已满，没有可落子的位置")
        if not position.stones:
            return (7, 7)

        simulations = int(settings.get("simulations", self.simulations))
        temperature = float(settings.get("temperature", self.temperature))

        last_cell = None
        if last_move:
            last_cell = (int(last_move["row"]), int(last_move["col"]))

        forced = self._forced_move(position, player)
        if forced is not None:
            return forced

        if self.vcf_budget > 0:
            move = winning_move(
                position,
                player,
                self.vcf_budget,
                VCF,
                self.solver_depth,
                self.solver_nodes,
            )
            if move is not None:
                return move
        if self.vct_budget > 0:
            move = winning_move(
                position,
                player,
                self.vct_budget,
                VCT,
                self.solver_depth,
                self.solver_nodes,
            )
            if move is not None:
                return move

        budget = self.time_budget
        if budget is not None:
            budget = max(0.05, budget - (time.monotonic() - started))
        result = self.searcher.search(
            position,
            player,
            simulations=max(1, simulations),
            add_noise=False,
            last_move=last_cell,
            time_budget=budget,
        )
        move = (
            result.sampled_move(temperature, self.searcher.rng)
            if temperature > 0
            else result.best_move()
        )
        if move is None:
            raise EngineError("搜索没有找到落点")
        return row_col_of(move)

    @staticmethod
    def _forced_move(position: Board, player: int) -> tuple[int, int] | None:
        cells = position.empty_cells()
        for cell in cells:
            if position.wins_with(cell[0], cell[1], player):
                return cell
        for cell in cells:
            if position.wins_with(cell[0], cell[1], other(player)):
                return cell
        return None
