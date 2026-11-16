from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .board import BLACK, Board, other
from .features import TRANSFORMS, encode, legal_mask, transform_planes, transform_policy
from .mcts import MCTS, row_col_of


DEFAULT_TEMPERATURE_MOVES = 12


DEFAULT_AUGMENT = True


@dataclass
class Example:
    planes: list[list[list[float]]]
    policy: list[float]
    legal: list[float]
    value: float = 0.0

    teacher_cell: int = -1


@dataclass
class ReplayBuffer:
    examples: list[Example] = field(default_factory=list)
    capacity: int = 60_000

    def add(self, example: Example) -> None:
        self.examples.append(example)
        if len(self.examples) > self.capacity:
            del self.examples[: len(self.examples) - self.capacity]

    def sample(self, count: int, rng: random.Random) -> list[Example]:
        if not self.examples:
            return []
        count = min(count, len(self.examples))
        return rng.sample(self.examples, count)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.examples, str(target))

    def load(self, path: str | Path) -> None:
        source = Path(path)
        if not source.is_file():
            return
        self.examples = list(torch.load(str(source), weights_only=False))


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
        self.model = model
        self.device = device
        self.simulations = simulations
        self.batch_size = batch_size
        self.candidate_radius = candidate_radius
        self.temperature_moves = temperature_moves
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
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


def augment(
    example: Example, rng: random.Random
) -> tuple[list[list[list[float]]], list[float], list[float]]:
    index = rng.randrange(len(TRANSFORMS))
    if index == 0:
        return example.planes, example.policy, example.legal
    planes = transform_planes(example.planes, index)
    policy = transform_policy(example.policy, index)
    legal = transform_policy(example.legal, index)
    return planes, policy, legal


def train(
    model: GomokuNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    steps: int = 200,
    batch_size: int = 256,
    augment_data: bool = DEFAULT_AUGMENT,
    rng: random.Random | None = None,
    log: bool = False,
) -> list[float]:
    picker = rng or random.Random(0)
    losses: list[float] = []
    for _ in range(steps):
        batch = buffer.sample(batch_size, picker)
        if not batch:
            break
        planes = []
        policies = []
        values = []
        legal = []
        for example in batch:
            if augment_data:
                item_planes, item_policy, item_legal = augment(example, picker)
            else:
                item_planes, item_policy, item_legal = (
                    example.planes,
                    example.policy,
                    example.legal,
                )
            planes.append(item_planes)
            policies.append(item_policy)
            legal.append(item_legal)
            values.append(example.value)
        loss, policy_loss, value_loss = train_batch(
            model, optimizer, planes, policies, values, legal, device
        )
        losses.append(loss)
        if log and len(losses) % 20 == 0:
            print(
                f"  损失 {loss:.4f}（策略 {policy_loss:.4f}，价值 {value_loss:.4f}）",
                flush=True,
            )
    return losses


def save_checkpoint(model: GomokuNet, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    save(model, str(target))


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
            planes = torch.tensor(
                [example.planes for example in batch], dtype=torch.float32, device=device
            )
            policy_target = torch.tensor(
                [example.policy for example in batch], dtype=torch.float32, device=device
            )
            value_target = torch.tensor(
                [example.value for example in batch], dtype=torch.float32, device=device
            )
            mask = torch.tensor(
                [example.legal for example in batch], dtype=torch.float32, device=device
            )
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
