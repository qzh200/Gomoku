from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .board import BLACK, Board, other
from .features import encode, legal_mask
from .mcts import MCTS, row_col_of
from .network import GomokuNet, load_model, save


TEMPERATURE_MOVES = 12


_WORKER: dict = {}


@dataclass
class Example:
    planes: np.ndarray
    policy: np.ndarray
    legal: np.ndarray
    value: float = 0.0


@dataclass
class GameResult:
    winner: int | None
    moves: int
    examples: list[Example]


class SelfPlay:
    def __init__(
        self,
        model: GomokuNet,
        device: torch.device,
        simulations: int = 400,
        batch_size: int = 32,
        temperature_moves: int = TEMPERATURE_MOVES,
        dirichlet_alpha: float = 0.05,
        noise_weight: float = 0.25,
        time_budget: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.simulations = simulations
        self.temperature_moves = temperature_moves
        self.rng = random.Random(seed)
        self.searcher = MCTS(
            model,
            device,
            dirichlet_alpha=dirichlet_alpha,
            noise_weight=noise_weight,
            batch_size=batch_size,
            time_budget=time_budget,
            seed=seed,
        )

    def play_game(self) -> GameResult:
        board = Board()
        player = BLACK
        examples: list[Example] = []
        sides: list[int] = []
        last_move: tuple[int, int] | None = None
        winner: int | None = None

        while not board.is_full():
            temperature = 1.0 if len(examples) < self.temperature_moves else 0.0
            result = self.searcher.search(
                board,
                player,
                simulations=self.simulations,
                add_noise=True,
                last_move=last_move,
            )
            move = result.sampled_move(temperature, self.rng)
            if move is None:
                break
            row, col = row_col_of(move)
            examples.append(
                Example(
                    planes=encode(board, player, last_move),
                    policy=result.policy,
                    legal=legal_mask(board),
                )
            )
            sides.append(player)
            last_move = (row, col)
            if board.play(row, col, player):
                winner = player
                break
            player = other(player)

        for example, side in zip(examples, sides):
            if winner is None:
                example.value = 0.0
            else:
                example.value = 1.0 if winner == side else -1.0

        return GameResult(winner, len(examples), examples)


def save_checkpoint(model: GomokuNet, path: str | Path, meta: dict | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    save(model, str(target), meta)


def worker_init(device: str, temperature_moves: int = TEMPERATURE_MOVES) -> None:
    torch.set_num_threads(1)
    _WORKER.clear()
    _WORKER["device"] = torch.device(device)
    _WORKER["temperature_moves"] = temperature_moves


def checkpoint_stamp(path: str | Path) -> int:
    return Path(path).stat().st_mtime_ns


def worker_play(task: tuple[str, int, int, int, int]) -> GameResult:
    model_path, stamp, simulations, batch_size, seed = task
    if _WORKER.get("path") != model_path or _WORKER.get("stamp") != stamp:
        model = load_model(model_path, _WORKER["device"])
        model.eval()
        _WORKER["path"] = model_path
        _WORKER["stamp"] = stamp
        _WORKER["model"] = model
    player = SelfPlay(
        _WORKER["model"],
        _WORKER["device"],
        simulations=simulations,
        batch_size=batch_size,
        temperature_moves=_WORKER["temperature_moves"],
        seed=seed,
    )
    return player.play_game()
