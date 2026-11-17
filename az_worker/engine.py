from __future__ import annotations

from pathlib import Path

import torch

from .board import Board, other
from .mcts import DEFAULT_CANDIDATE_WIDTH, MCTS, row_col_of
from .network import GomokuNet, load_model


DEFAULT_MODEL_PATH = "models/az/current.pt"


DEFAULT_SIMULATIONS = 100_000


DEFAULT_TEMPERATURE = 0.0


class EngineError(RuntimeError):
    pass


class AlphaZeroEngine:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: str = "auto",
        simulations: int = DEFAULT_SIMULATIONS,
        c_puct: float = 1.5,
        batch_size: int = 32,
        candidate_radius: int = 1,
        candidate_width: int = DEFAULT_CANDIDATE_WIDTH,
        time_budget: float | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.simulations = max(1, simulations)
        self.temperature = temperature
        self.time_budget = time_budget

        if self.model_path.is_file():
            self.model = load_model(str(model_path), self.device)
        else:
            self.model = GomokuNet().to(self.device)
        self.model.eval()

        self.searcher = MCTS(
            self.model,
            self.device,
            c_puct=c_puct,
            batch_size=batch_size,
            candidate_radius=candidate_radius,
            candidate_width=candidate_width,
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

        result = self.searcher.search(
            position,
            player,
            simulations=max(1, simulations),
            add_noise=False,
            last_move=last_cell,
            time_budget=self.time_budget,
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
        candidates = position.candidates(2)
        own = [cell for cell in candidates if position.wins_with(*cell, player)]
        if own:
            return own[0]
        foe = [cell for cell in candidates if position.wins_with(*cell, other(player))]
        if foe:
            return foe[0]
        return None
