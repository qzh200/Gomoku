from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .board import BLACK, Board, other
from .features import encode, legal_mask
from .mcts import MCTS, row_col_of
from .network import GomokuNet, save, train_tensors


DEFAULT_TEMPERATURE_MOVES = 12


DEFAULT_AUGMENT = True


@dataclass
class Example:
    planes: np.ndarray
    policy: np.ndarray
    legal: np.ndarray
    value: float = 0.0

    teacher_cell: int = -1


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
        c_puct: float = 1.5,
        simulations: int = 400,
        batch_size: int = 32,
        candidate_radius: int = 1,
        temperature_moves: int = DEFAULT_TEMPERATURE_MOVES,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        seed: int | None = None,
    ) -> None:
        self.simulations = simulations
        self.candidate_radius = candidate_radius
        self.temperature_moves = temperature_moves
        self.rng = random.Random(seed)
        self.searcher = MCTS(
            model,
            device,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            noise_weight=dirichlet_weight,
            batch_size=batch_size,
            candidate_radius=candidate_radius,
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
                    legal=legal_mask(board, self.candidate_radius),
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


def save_checkpoint(model: GomokuNet, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    save(model, str(target))


def train_gpu(
    model: GomokuNet,
    optimizer: torch.optim.Optimizer,
    dataset,
    steps: int = 200,
    batch_size: int = 1024,
    augment_data: bool = DEFAULT_AUGMENT,
    rng: random.Random | None = None,
    log: bool = False,
) -> list[float]:
    picker = rng or random.Random(0)
    losses: list[float] = []
    for _ in range(steps):
        planes, policy, legal, value = dataset.sample(
            batch_size,
            picker,
            augment=augment_data,
        )
        loss, policy_loss, value_loss = train_tensors(
            model,
            optimizer,
            planes,
            policy,
            value,
            legal,
        )
        losses.append(loss)
        if log and len(losses) % 20 == 0:
            print(
                f"  损失 {loss:.4f}（策略 {policy_loss:.4f}，价值 {value_loss:.4f}）",
                flush=True,
            )
    return losses


def evaluate(
    model: GomokuNet,
    examples: list[Example],
    device: torch.device,
    batch_size: int = 128,
) -> tuple[float, float]:
    if not examples:
        return float("nan"), float("nan")
    policy_loss = 0.0
    value_loss = 0.0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            planes = torch.from_numpy(
                np.stack([example.planes for example in batch]).astype(np.float32)
            ).to(device)
            policy_target = torch.from_numpy(
                np.stack([example.policy for example in batch]).astype(np.float32)
            ).to(device)
            value_target = torch.from_numpy(
                np.array([example.value for example in batch], dtype=np.float32)
            ).to(device)
            mask = torch.from_numpy(
                np.stack([example.legal for example in batch]).astype(np.float32)
            ).to(device)
            logits, predicted = model(planes)
            logits = logits.masked_fill(mask == 0, -1e9)
            log_probabilities = torch.log_softmax(logits, dim=1)
            target = policy_target * mask
            target = target / torch.clamp(target.sum(dim=1, keepdim=True), min=1e-8)
            policy_loss += float(
                (-(target * log_probabilities).sum(dim=1).mean()).item() * len(batch)
            )
            value_loss += float(((predicted - value_target) ** 2).mean().item()) * len(batch)
    count = len(examples)
    return policy_loss / count, value_loss / count
