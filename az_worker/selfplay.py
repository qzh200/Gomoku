from __future__ import annotations

import random
from dataclasses import dataclass
from multiprocessing import Pipe
from pathlib import Path

import numpy as np
import torch

from .board import BLACK, Board, other
from .features import encode, legal_mask
from .inference import RemoteEvaluator
from .mcts import MCTS, cell_of, row_col_of
from .network import ACTION_COUNT, GomokuNet, load_model, save


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
        evaluator=None,
        solver=None,
        radius: int = 0,
    ) -> None:
        self.simulations = simulations
        self.temperature_moves = temperature_moves
        self.solver = solver
        self.rng = random.Random(seed)
        self.searcher = MCTS(
            model,
            device,
            dirichlet_alpha=dirichlet_alpha,
            noise_weight=noise_weight,
            batch_size=batch_size,
            time_budget=time_budget,
            seed=seed,
            evaluator=evaluator,
            radius=radius,
        )

    def play_game(self) -> GameResult:
        board = Board()
        player = BLACK
        examples: list[Example] = []
        sides: list[int] = []
        last_move: tuple[int, int] | None = None
        winner: int | None = None

        while not board.is_full():
            move = None
            policy = None
            if self.solver is not None:
                forced = self.solver.move(board, player)
                if forced is not None:
                    row, col = forced
                    move = cell_of(row, col)
                    policy = np.zeros(ACTION_COUNT, dtype=np.float32)
                    policy[move] = 1.0
            if move is None:
                temperature = 1.0 if len(examples) < self.temperature_moves else 0.0
                result = self.searcher.search(
                    board,
                    player,
                    simulations=self.simulations,
                    add_noise=True,
                    last_move=last_move,
                )
                move = result.sampled_move(temperature, self.rng)
                policy = result.policy
            if move is None:
                break
            row, col = row_col_of(move)
            examples.append(
                Example(
                    planes=encode(board, player, last_move),
                    policy=policy,
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


def save_checkpoint(
    model: GomokuNet,
    path: str | Path,
    meta: dict | None = None,
    extra: dict | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    save(model, str(target), meta, extra)


def worker_init(
    device: str,
    temperature_moves: int = TEMPERATURE_MOVES,
    requests=None,
    solver_args=None,
    radius: int = 0,
) -> None:
    torch.set_num_threads(1)
    _WORKER.clear()
    _WORKER["temperature_moves"] = temperature_moves
    _WORKER["radius"] = max(0, int(radius))
    _WORKER["models"] = {}
    _WORKER["evaluator"] = None
    _WORKER["solver"] = _make_solver(solver_args)
    if device == "shared":
        if requests is None:
            raise RuntimeError("共享推理模式缺少请求队列")
        parent, child = Pipe()
        requests.put(parent)
        _WORKER["device"] = torch.device("cpu")
        _WORKER["evaluator"] = RemoteEvaluator(child)
    else:
        _WORKER["device"] = torch.device(device)


def _make_solver(solver_args):
    if not solver_args:
        return None
    from .vcf import PlaySolver

    return PlaySolver(*solver_args)


def checkpoint_stamp(path: str | Path) -> int:
    return Path(path).stat().st_mtime_ns


def worker_play(task: tuple[str, int, int, int, int]) -> GameResult:
    model_path, stamp, simulations, batch_size, seed = task
    evaluator = _WORKER.get("evaluator")
    model = None
    if evaluator is None:
        cache = _WORKER["models"]
        entry = cache.get("model")
        if entry is None or entry[0] != model_path or entry[1] != stamp:
            model = load_model(model_path, _WORKER["device"])
            model.eval()
            cache["model"] = (model_path, stamp, model)
        else:
            model = entry[2]
    player = SelfPlay(
        model,
        _WORKER["device"],
        simulations=simulations,
        batch_size=batch_size,
        temperature_moves=_WORKER["temperature_moves"],
        seed=seed,
        evaluator=evaluator,
        solver=_WORKER.get("solver"),
        radius=_WORKER.get("radius", 0),
    )
    return player.play_game()
