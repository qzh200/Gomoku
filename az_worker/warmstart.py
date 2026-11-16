from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

from classical_worker.position import BLACK, WHITE, Position, other
from classical_worker.search import Searcher
from classical_worker.threats import ordered_moves

from .board import BOARD_SIZE, Board
from .features import encode, legal_mask


DEFAULT_GAMES = 100
DEFAULT_DEPTH = 4
DEFAULT_WIDTH = 12
DEFAULT_TIME = 0.05
DEFAULT_CHANNELS = 64
DEFAULT_BLOCKS = 6
DEFAULT_MODEL = "models/az/warmstart.pt"


DEFAULT_TEACHER_MOVES = 8


DEFAULT_DECAY = 2.0




VALUE_OUTCOME = "outcome"
VALUE_EVAL = "eval"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用传统引擎当老师，预热神经网络")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="棋局数")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="传统引擎的搜索深度")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="传统引擎的宽度")
    parser.add_argument("--time", type=float, default=DEFAULT_TIME, help="传统引擎每手秒数")
    parser.add_argument("--teacher-moves", type=int, default=DEFAULT_TEACHER_MOVES, help="老师排名的着法数")
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY, help="着法权重的衰减")
    parser.add_argument("--score-scale", type=float, default=DEFAULT_SCORE_SCALE, help="分数到价值的换算")
    parser.add_argument(
        "--value-target",
        choices=(VALUE_OUTCOME, VALUE_EVAL),
        default=DEFAULT_VALUE_TARGET,
        help="价值目标：outcome 是整局结果（与自对弈一致），eval 是传统引擎的评估分",
    )
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help="网络通道数")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS, help="残差块数量")
    parser.add_argument("--steps", type=int, default=None, help="训练步数，默认按样本数定")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="权重文件路径")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    return parser


def teacher_policy(ranked: list[tuple[int, int]], decay: float) -> list[float]:
    policy = [0.0] * (BOARD_SIZE * BOARD_SIZE)
    for index, (row, col) in enumerate(ranked):
        policy[row * BOARD_SIZE + col] = math.exp(-index / decay)
    return policy


def collect_examples(
    games: int,
    depth: int,
    width: int,
    time_budget: float,
    teacher_moves: int,
    decay: float,
    score_scale: float = DEFAULT_SCORE_SCALE,
    value_target: str = VALUE_OUTCOME,
) -> list[Example]:
    examples: list[Example] = []
    for _ in range(games):




def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed if args.seed is not None else 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(args.model)
    if model_path.is_file():
        model = load_model(str(model_path), device)
        print(f"读取权重: {model_path}", flush=True)
    else:
        model = GomokuNet(channels=args.channels, blocks=args.blocks).to(device)
        print(f"没有权重文件，从随机网络开始: {model_path}", flush=True)

    examples = collect_examples(
        args.games,
        args.depth,
        args.width,
        args.time,
        args.teacher_moves,
        args.decay,
        args.score_scale,
        args.value_target,
    )


    held_out = max(1, len(examples) // 10)
    validation = examples[-held_out:]
    examples = examples[:-held_out]
    print(f"样本: {len(examples)}（另留 {len(validation)} 个做验证）", flush=True)
    values = [example.value for example in examples]
    print(
        f"价值范围: {min(values):.3f} 到 {max(values):.3f}，"
        f"平均 {sum(values) / len(values):.3f}",
        flush=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = args.steps if args.steps is not None else max(1, len(examples) // 32) * 6

            model,
            optimizer,
            planes,
            legal,
        )
        if (step + 1) % 50 == 0:
            agreement = agreement_of(model, validation, device)
            print(
                f"  第 {step + 1}/{steps} 步，损失 {loss:.4f}"
                f"（策略 {policy_loss:.4f}，价值 {value_loss:.4f}），"
                f"验证首选一致 {agreement:.0%}",
                flush=True,
            )

    agreement = agreement_of(model, validation, device)
    print(f"验证首选一致率: {agreement:.0%}", flush=True)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    save(model, str(model_path))
    print(f"预热权重已写入: {model_path}", flush=True)
    return 0


def agreement_of(model: GomokuNet, examples: list[Example], device: torch.device) -> float:
    if not examples:
        return 0.0
    matches = 0
    for start in range(0, len(examples), 128):
        batch = examples[start : start + 128]
        policies, _ = evaluate(
            model,
            [example.planes for example in batch],
            [example.legal for example in batch],
            device,
        )
        for policy, example in zip(policies, batch):
            best = max(range(len(policy)), key=lambda cell: policy[cell])
            if example.teacher_cell >= 0 and best == example.teacher_cell:
                matches += 1
    return matches / len(examples)


if __name__ == "__main__":
    raise SystemExit(main())
