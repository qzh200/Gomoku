"""The AlphaZero engine: a search over the moves the network prefers.

The engine keeps one network and one searcher. The network is loaded from a
file, so the engine starts fast and the weights survive a restart. A game
may ask for more simulations than the worker was started with, but not fewer.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .board import Board, other
from .mcts import MCTS, row_col_of
from .network import GomokuNet, load_model

# The file the weights are read from when the worker says nothing else.
DEFAULT_MODEL_PATH = "models/az/current.pt"

# The number of simulations of one move.
DEFAULT_SIMULATIONS = 400

# The temperature of the move selection: zero plays the best move.
DEFAULT_TEMPERATURE = 0.0


class EngineError(RuntimeError):
    """The engine could not find a move."""


class AlphaZeroEngine:
    """Picks a move with a search over the moves the network scores."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: str = "auto",
        simulations: int = DEFAULT_SIMULATIONS,
        c_puct: float = 1.5,
        batch_size: int = 32,
        candidate_radius: int = 1,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.simulations = max(1, simulations)
        self.temperature = temperature
        self.candidate_radius = candidate_radius

        if self.model_path.is_file():
            self.model = load_model(str(self.model_path), self.device)
        else:
            # An engine without weights still plays; the search is only as
            # good as the network, and a fresh network knows nothing.
            self.model = GomokuNet().to(self.device)
        self.model.eval()

        self.searcher = MCTS(
            self.model,
            self.device,
            c_puct=c_puct,
            batch_size=batch_size,
            candidate_radius=candidate_radius,
            seed=seed,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Return the device the network runs on."""
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
        """Return the row and the column of the move to play."""
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

        # A five is a five, whatever the network thinks: the search only
        # sees the position it is given, so the two obvious moves are taken
        # before it. This keeps an early network from throwing a game away.
        forced = self._forced_move(position, player)
        if forced is not None:
            return forced

        result = self.searcher.search(
            position,
            player,
            simulations=max(1, simulations),
            add_noise=False,
            last_move=last_cell,
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
        """Return the move that wins now or blocks a five of the foe.

        A five of the foe is a cell where the foe would make five in one
        move. One such cell has to be blocked. Several of them mean the
        position is lost: the foe has an open four or a double four, and one
        move cannot stop both of them. The block is played anyway, because it
        is the least bad move, the same way the classical engine plays it.
        """
        candidates = position.candidates(2)
        own = [cell for cell in candidates if position.wins_with(*cell, player)]
        if own:
            return own[0]
        foe = [cell for cell in candidates if position.wins_with(*cell, other(player))]
        if foe:
            return foe[0]
        return None
